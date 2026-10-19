"""The owner envelope (hash-pinned caps) and the hash-chained audit log."""
from __future__ import annotations

import gzip
import json
from decimal import Decimal
from pathlib import Path

import pytest

from loci_litellm.audit_chain import GENESIS, AuditChain, AuditUnavailable, entry_hash
from loci_litellm.envelope import (
    ALARM_INVALID,
    ALARM_MISMATCH,
    ALARM_MISSING,
    ALARM_UNPINNED,
    DEFAULT_CAPS,
    EnvelopeInvalid,
    EnvelopeWatcher,
    file_digest,
    parse_envelope,
    record_pin,
    sign_pin,
)
from tests._governor_fixtures import BASE_TS, FakeClock

D = Decimal
KEY = b"owner-key-material"
DAY = 86400.0

ENVELOPE = """\
[governor]
enforce = true
day_timezone = "America/New_York"

[caps]
day_soft = 30
day_hard = 45
rolling_24h = 60
hour_downgrade = 10
hour_reject = 25
interactive_stop = 75
high_cost_share = "0.25"

[clients.scripts]
secret_sha256 = "%s"
cap_day = 12.5
interactive = false
rw = true
"""
SECRET_HASH = "ab" * 32


def write_envelope(tmp_path: Path, text: str = ENVELOPE % SECRET_HASH, name: str = "envelope.toml") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def pin(tmp_path: Path, envelope: Path) -> Path:
    pin_file = tmp_path / "envelope.pin"
    pin_file.write_text(file_digest(str(envelope)) + "\n", encoding="utf-8")
    return pin_file


def watcher(tmp_path: Path, envelope: Path, pin_file: Path, clock: FakeClock, audit: AuditChain | None = None,
            key: bytes | None = None) -> EnvelopeWatcher:
    return EnvelopeWatcher(str(envelope), str(pin_file), audit, key, clock=clock)


# --------------------------------------------------------------------------- envelope
def test_a_pinned_envelope_supplies_the_caps_the_enforce_flag_and_the_clients(tmp_path: Path) -> None:
    env = write_envelope(tmp_path)
    state = watcher(tmp_path, env, pin(tmp_path, env), FakeClock()).state()
    assert (state.level, state.alarm, state.source, state.trusted) == ("L0", "", "envelope", True)
    assert state.enforce is True and state.day_timezone == "America/New_York"
    assert (state.caps.day_soft, state.caps.day_hard, state.caps.high_cost_share) == (D("30"), D("45"), D("0.25"))
    assert state.caps.hour_reject == D("25") and state.caps.prompt_tokens_max == 60000
    client = state.clients["scripts"]
    assert (client.cap_day, client.interactive, client.rw, client.secret_sha256) == (D("12.5"), False, True, SECRET_HASH)


def test_one_changed_byte_gives_defaults_level_l3_and_a_hash_mismatch_alarm(tmp_path: Path) -> None:
    env = write_envelope(tmp_path)
    pin_file = pin(tmp_path, env)
    clock = FakeClock()
    w = watcher(tmp_path, env, pin_file, clock)
    assert w.state().caps.day_hard == D("45")
    env.write_text(ENVELOPE.replace("day_hard = 45", "day_hard = 46") % SECRET_HASH, encoding="utf-8")
    clock.advance(60)
    state = w.state()
    assert (state.level, state.alarm, state.source, state.trusted) == ("L3", ALARM_MISMATCH, "defaults", False)
    assert state.caps == DEFAULT_CAPS and state.caps.day_hard == D("50")
    assert (state.enforce, dict(state.clients)) == (False, {})
    assert w.alarms == {ALARM_MISMATCH: 1}


def test_a_matching_signed_audit_entry_keeps_the_level_and_a_wrong_signature_does_not(tmp_path: Path) -> None:
    env = write_envelope(tmp_path)
    pin_file = pin(tmp_path, env)
    clock = FakeClock()
    audit = AuditChain(str(tmp_path / "audit"), clock=clock)
    w = watcher(tmp_path, env, pin_file, clock, audit, KEY)
    env.write_text(ENVELOPE.replace("day_hard = 45", "day_hard = 46") % SECRET_HASH, encoding="utf-8")
    digest = file_digest(str(env))
    record_pin(audit, b"some-other-key", digest)              # signed by the wrong key
    assert w.state().alarm == ALARM_MISMATCH
    clock.advance(60)
    record_pin(audit, KEY, digest)
    state = w.state()
    assert (state.level, state.alarm, state.caps.day_hard, state.sha256) == ("L0", "", D("46"), digest)


def test_a_signed_entry_for_another_digest_or_a_broken_chain_does_not_count(tmp_path: Path) -> None:
    env = write_envelope(tmp_path)
    pin_file = pin(tmp_path, env)
    clock = FakeClock()
    audit = AuditChain(str(tmp_path / "audit"), clock=clock)
    record_pin(audit, KEY, "0" * 64)
    env.write_text(ENVELOPE.replace("day_hard = 45", "day_hard = 46") % SECRET_HASH, encoding="utf-8")
    assert watcher(tmp_path, env, pin_file, clock, audit, KEY).state().alarm == ALARM_MISMATCH
    digest = file_digest(str(env))
    record_pin(audit, KEY, digest)
    ok = watcher(tmp_path, env, pin_file, clock, audit, KEY).state()
    assert ok.level == "L0"
    lines = audit.active_path.read_text(encoding="utf-8").splitlines()
    lines[0] = lines[0].replace("0" * 64, "1" * 64, 1)           # rewrite history: the chain no longer verifies
    audit.active_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    broken = watcher(tmp_path, env, pin_file, clock, audit, KEY).state()
    assert (broken.level, broken.alarm) == ("L3", ALARM_MISMATCH)


def test_missing_and_unpinned_envelopes_fail_closed_with_their_own_alarms(tmp_path: Path) -> None:
    clock = FakeClock()
    gone = watcher(tmp_path, tmp_path / "nope.toml", tmp_path / "nope.pin", clock).state()
    assert (gone.level, gone.alarm, gone.source, gone.caps) == ("L3", ALARM_MISSING, "defaults", DEFAULT_CAPS)
    env = write_envelope(tmp_path)
    unpinned = watcher(tmp_path, env, tmp_path / "no.pin", clock).state()
    assert (unpinned.level, unpinned.alarm, unpinned.enforce) == ("L3", ALARM_UNPINNED, False)
    assert watcher(tmp_path, env, pin(tmp_path, env), clock).state().level == "L0"


def test_the_pin_is_compared_case_insensitively_and_ignores_trailing_text(tmp_path: Path) -> None:
    env = write_envelope(tmp_path)
    pin_file = tmp_path / "envelope.pin"
    pin_file.write_text(file_digest(str(env)).upper() + "  envelope.toml\n", encoding="utf-8")
    assert watcher(tmp_path, env, pin_file, FakeClock()).state().level == "L0"


def test_the_envelope_is_rechecked_every_sixty_seconds_not_every_call(tmp_path: Path) -> None:
    env = write_envelope(tmp_path)
    clock = FakeClock()
    w = watcher(tmp_path, env, pin(tmp_path, env), clock)
    w.state()
    clock.advance(59.9)
    w.state()
    assert w.checks == 1
    clock.advance(0.1)
    w.state()
    assert w.checks == 2


@pytest.mark.parametrize("text, why", [
    ("[caps]\nday_hard = -1\n", "day_hard must be a finite number >= 0"),
    ("[caps]\nday_soft = 60\nday_hard = 50\n", "day_soft <= day_hard"),
    ("[caps]\nday_hard = true\n", "day_hard must be a number"),
    ("[caps]\nday_hard = 50\nrolling_24h = 40\n", "day_hard <= rolling_24h"),
    ("[caps]\nhour_downgrade = 30\nhour_reject = 25\n", "hour_downgrade <= hour_reject"),
    ("[caps]\nmystery = 1\n", "unknown cap 'mystery'"),
    ("[caps]\nprompt_tokens_max = 1.5\n", "prompt_tokens_max must be an integer"),
    ("[governor]\nenforce = \"yes\"\n", "governor.enforce must be true or false"),
    ("[clients.x]\nsecret_sha256 = \"short\"\n", "64-hex secret_sha256"),
    ("[caps\n", "envelope does not parse"),
])
def test_a_malformed_envelope_is_rejected_with_the_reason(text: str, why: str) -> None:
    with pytest.raises(EnvelopeInvalid, match=why):
        parse_envelope(text.encode("utf-8"))


def test_a_malformed_but_pinned_envelope_is_invalid_not_trusted(tmp_path: Path) -> None:
    env = write_envelope(tmp_path, "[caps]\nday_hard = -1\n")
    state = watcher(tmp_path, env, pin(tmp_path, env), FakeClock()).state()
    assert (state.level, state.alarm, state.caps) == ("L3", ALARM_INVALID, DEFAULT_CAPS)


def test_json_envelopes_work_and_money_stays_decimal(tmp_path: Path) -> None:
    caps, enforce, tz, clients = parse_envelope(b'{"caps": {"day_hard": 45.1, "day_soft": "30.05"}, "governor": {"enforce": true}}', ".json")
    assert (caps.day_hard, caps.day_soft, enforce, tz, clients) == (D("45.1"), D("30.05"), True, "local", {})
    assert isinstance(caps.day_hard, Decimal)
    defaults = parse_envelope(b"", ".toml")
    assert (defaults[0], defaults[1], defaults[2]) == (DEFAULT_CAPS, False, "local")


def test_defaults_are_the_owner_decision_numbers() -> None:
    assert (DEFAULT_CAPS.day_soft, DEFAULT_CAPS.day_hard, DEFAULT_CAPS.rolling_24h) == (D("30"), D("50"), D("60"))
    assert (DEFAULT_CAPS.hour_downgrade, DEFAULT_CAPS.hour_reject, DEFAULT_CAPS.interactive_stop) == (D("10"), D("25"), D("75"))
    assert sign_pin(KEY, "ab") == sign_pin(KEY, "ab") != sign_pin(KEY, "ac")


# --------------------------------------------------------------------------- audit chain
def make_chain(tmp_path: Path, n: int = 5, **kwargs: object) -> AuditChain:
    chain = AuditChain(str(tmp_path / "audit"), **kwargs)  # type: ignore[arg-type]
    for i in range(n):
        chain.append("event", {"i": i})
    return chain


def test_a_clean_chain_verifies_and_links_each_entry_to_the_previous(tmp_path: Path) -> None:
    chain = make_chain(tmp_path)
    check = chain.verify()
    assert (check.ok, check.entries, check.break_index, check.start_seq) == (True, 5, None, 0)
    entries = list(chain.entries())
    assert entries[0]["prev"] == GENESIS
    assert [e["prev"] for e in entries[1:]] == [e["hash"] for e in entries[:-1]]
    assert entries[2]["hash"] == entry_hash(entries[1]["hash"], 2, entries[2]["ts"], "event", {"i": 2})


def test_modifying_a_line_returns_the_exact_break_index(tmp_path: Path) -> None:
    chain = make_chain(tmp_path)
    lines = chain.active_path.read_text(encoding="utf-8").splitlines()
    lines[2] = lines[2].replace('"i":2', '"i":9')
    chain.active_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    check = chain.verify()
    assert (check.ok, check.break_index, check.entries, check.reason) == (False, 2, 2, "entry hash does not match its content")


def test_truncating_a_line_returns_its_index_and_dropping_trailing_lines_is_detected(tmp_path: Path) -> None:
    chain = make_chain(tmp_path)
    lines = chain.active_path.read_text(encoding="utf-8").splitlines()
    chain.active_path.write_text("\n".join(lines[:3]) + "\n" + lines[3][:40] + "\n", encoding="utf-8")
    cut = chain.verify()
    assert (cut.ok, cut.break_index) == (False, 3) and cut.reason.startswith("unparseable entry")
    chain.active_path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")            # whole trailing lines dropped
    dropped = chain.verify()
    assert (dropped.ok, dropped.break_index, dropped.reason) == (False, 3, "chain is shorter than its head anchor (truncated)")


def test_reordering_or_deleting_a_middle_line_breaks_at_that_index(tmp_path: Path) -> None:
    chain = make_chain(tmp_path)
    lines = chain.active_path.read_text(encoding="utf-8").splitlines()
    chain.active_path.write_text("\n".join([lines[0], lines[2], lines[1], *lines[3:]]) + "\n", encoding="utf-8")
    assert (chain.verify().ok, chain.verify().break_index) == (False, 1)
    chain.active_path.write_text("\n".join([lines[0], lines[1], *lines[3:]]) + "\n", encoding="utf-8")
    assert (chain.verify().ok, chain.verify().break_index) == (False, 2)
    chain.active_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert chain.verify().ok is True                                                     # restored: positive twin


def test_a_new_chain_object_continues_the_old_one(tmp_path: Path) -> None:
    make_chain(tmp_path, 3)
    again = AuditChain(str(tmp_path / "audit"))
    entry = again.append("event", {"i": 3})
    assert entry["seq"] == 3 and again.verify().entries == 4
    assert list(again.entries())[3]["prev"] == list(again.entries())[2]["hash"]


def test_a_write_failure_raises_audit_unavailable(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")
    with pytest.raises(AuditUnavailable, match="audit append failed"):
        AuditChain(str(blocker / "audit")).append("event", {})
    with pytest.raises(AuditUnavailable, match="audit append failed"):
        make_chain(tmp_path, 1).append("event", {"bad": object()})
    corrupt = make_chain(tmp_path / "second", 1)
    corrupt.active_path.write_text("garbage\n", encoding="utf-8")
    with pytest.raises(AuditUnavailable, match="tail unreadable"):
        AuditChain(str(tmp_path / "second" / "audit")).append("event", {})


def test_rotation_keeps_one_chain_across_segments(tmp_path: Path) -> None:
    clock = FakeClock()
    chain = make_chain(tmp_path, 12, segment_bytes=600, clock=clock)
    assert [p.name for p in chain.segments()] == [f"audit.{first:012d}.1760000000.jsonl" for first in (0, 3, 6)]
    check = chain.verify()
    assert (check.ok, check.entries, check.start_seq) == (True, 12, 0)
    assert [e["seq"] for e in chain.entries()] == list(range(12))


def test_gzip_after_seven_days_and_delete_after_sixty(tmp_path: Path) -> None:
    clock = FakeClock(BASE_TS)
    chain = AuditChain(str(tmp_path / "audit"), segment_bytes=300, clock=clock)
    for i in range(6):
        chain.append("event", {"i": i})
        clock.advance(DAY)                                        # one entry a day, one segment per ~2 entries
    names = [p.name for p in chain.segments()]
    assert names and not any(n.endswith(".gz") for n in names)
    clock.advance(3 * DAY)
    summary = chain.maintenance()
    gz_names = [p.name for p in chain.segments()]
    assert summary["gzipped"] and all(n.endswith(".gz") for n in gz_names[:len(summary["gzipped"])])
    assert summary["deleted"] == []
    first_gz = next(p for p in chain.segments() if p.name.endswith(".gz"))
    with gzip.open(first_gz, "rt", encoding="utf-8") as handle:
        assert json.loads(handle.readline())["seq"] == 0
    assert chain.verify().ok is True
    clock.advance(55 * DAY)                                       # now > 60 days past the oldest segments
    gone = chain.maintenance()
    assert gone["deleted"] and chain.verify().ok is True
    kinds = [e["kind"] for e in chain.entries()]
    assert kinds[-1] == "retention" and "retention" in kinds
    assert chain.verify().start_seq > 0


def test_total_size_cap_deletes_the_oldest_closed_segments_first(tmp_path: Path) -> None:
    clock = FakeClock()
    chain = make_chain(tmp_path, 14, segment_bytes=400, clock=clock, max_total_bytes=1500)
    before = [p.name for p in chain.segments()]
    assert len(before) == 6
    result = chain.maintenance()
    after = [p.name for p in chain.segments()]
    assert result["deleted"] == before[:len(result["deleted"])] and result["deleted"]
    assert result["deleted"] == before[:4]
    assert after == [f"audit.{first:012d}.1760000000.jsonl" for first in (8, 10, 12)]   # the retention entry opens segment 12
    assert [e["kind"] for e in chain.entries()][-1] == "retention"
    assert chain.verify().ok is True
    assert chain.total_bytes() == 1643
    untouched = make_chain(tmp_path / "other", 14, segment_bytes=400, clock=clock)
    assert untouched.maintenance() == {"gzipped": [], "deleted": []}
