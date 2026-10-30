"""Dedicated tests for the hash-chained audit log: tamper/truncation detection, retention maintenance,
and the fsync + maintenance-throttle policy added for the proxy hot path.

Mutation proofs (run by hand, each reverted):
  - append(): dropped `self._write_head()` -> test_dropping_the_last_line_is_detected_by_the_head_anchor
    FAILED (truncation no longer detected). Restored -> green.
  - _should_fsync(): `return True` unconditionally -> test_fsync_interval_batches_within_the_window FAILED
    (fsync count == entries, not 1). Restored -> green.
  - maybe_maintain(): removed the `_next_maint_ts` guard -> test_maybe_maintain_runs_at_most_once_per_interval
    FAILED (ran per append). Restored -> green.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from loci_litellm.audit_chain import AuditChain, AuditUnavailable


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _chain(tmp_path: Path, **kw) -> tuple[AuditChain, Clock]:
    clock = Clock()
    # fsync off by default here so maintenance/verify tests do not pay per-write fsync; durability is
    # covered by the dedicated fsync tests below. maintenance_interval high by default so append()'s own
    # auto-maintenance does not pre-empt the explicit maintenance() call under test (the throttle itself
    # has its own dedicated test that sets a real interval).
    chain = AuditChain(str(tmp_path), clock=clock, fsync_mode=kw.pop("fsync_mode", "never"),
                       maintenance_interval_s=kw.pop("maintenance_interval_s", 10 ** 12), **kw)
    return chain, clock


def test_append_then_verify_is_ok_and_counts_every_entry(tmp_path: Path) -> None:
    chain, _ = _chain(tmp_path)
    for i in range(3):
        chain.append("event", {"i": i})
    check = chain.verify()
    assert check.ok is True and check.entries == 3 and check.break_index is None and check.start_seq == 0


def test_rewriting_an_entry_breaks_verify_at_that_exact_index(tmp_path: Path) -> None:
    chain, _ = _chain(tmp_path)
    for i in range(3):
        chain.append("event", {"i": i})
    # positive twin: untouched chain verifies
    assert chain.verify().ok is True

    lines = chain.active_path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["data"] = {"i": 99}            # rewrite seq 1's payload, leave its stored hash
    lines[1] = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    chain.active_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    check = chain.verify()
    assert check.ok is False and check.break_index == 1
    assert check.reason == "entry hash does not match its content"


def test_dropping_the_last_line_is_detected_by_the_head_anchor(tmp_path: Path) -> None:
    chain, _ = _chain(tmp_path)
    for i in range(3):
        chain.append("event", {"i": i})
    lines = chain.active_path.read_text(encoding="utf-8").splitlines()
    chain.active_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")  # drop seq 2

    check = chain.verify()
    assert check.ok is False and "truncated" in check.reason


def test_maintenance_gzips_old_segments_deletes_ancient_ones_and_records_retention(tmp_path: Path) -> None:
    # tiny segments so each append rotates; a controllable clock to age the closed segments.
    chain, clock = _chain(tmp_path, segment_bytes=1, gzip_after_days=7.0, delete_after_days=60.0)
    chain.append("event", {"n": 0})      # seq 0 -> active
    clock.t += 10 * 86400                 # +10 days
    chain.append("event", {"n": 1})      # rotation closes the seq-0 segment, writes seq 1
    clock.t += 100 * 86400                # now the first closed segment is ~110 days old

    before = {p.name for p in chain.segments()}
    result = chain.maintenance()

    # the ancient closed segment was deleted (>60d); a retention entry records it; chain still verifies
    assert result["deleted"], f"expected a deletion, segments were {before}"
    assert any(e["kind"] == "retention" for e in chain.entries())
    assert chain.verify().ok is True


def test_maintenance_gzips_the_mid_aged_segment(tmp_path: Path) -> None:
    chain, clock = _chain(tmp_path, segment_bytes=1, gzip_after_days=7.0, delete_after_days=60.0)
    chain.append("event", {"n": 0})
    clock.t += 10 * 86400                 # 10 days: older than gzip (7d), younger than delete (60d)
    chain.append("event", {"n": 1})       # closes the seq-0 segment

    result = chain.maintenance()

    assert result["gzipped"] and not result["deleted"]
    assert any(p.name.endswith(".gz") for p in chain.segments())


def test_maybe_maintain_runs_at_most_once_per_interval(tmp_path: Path, monkeypatch) -> None:
    chain, clock = _chain(tmp_path, maintenance_interval_s=3600.0)
    calls = {"n": 0}
    monkeypatch.setattr(chain, "maintenance", lambda: calls.__setitem__("n", calls["n"] + 1) or {})

    for i in range(5):                    # five appends inside one interval
        chain.append("event", {"i": i})
    assert calls["n"] == 1                 # maintenance fired once (on the first append), then throttled

    clock.t += 3601                        # past the interval
    chain.append("event", {"i": 5})
    assert calls["n"] == 2                 # fires again


def test_fsync_interval_batches_within_the_window(tmp_path: Path, monkeypatch) -> None:
    import loci_litellm.audit_chain as ac
    fsyncs = {"n": 0}
    monkeypatch.setattr(ac.os, "fsync", lambda fd: fsyncs.__setitem__("n", fsyncs["n"] + 1))
    clock = Clock()
    chain = AuditChain(str(tmp_path), clock=clock, fsync_mode="interval", fsync_interval_s=1.0)
    for i in range(5):                    # clock not advanced: all within one 1s window
        chain.append("event", {"i": i})
    assert fsyncs["n"] == 1                # only the first append fsynced


def test_fsync_always_syncs_every_append(tmp_path: Path, monkeypatch) -> None:
    # positive twin to the interval test: "always" must fsync each write
    import loci_litellm.audit_chain as ac
    fsyncs = {"n": 0}
    monkeypatch.setattr(ac.os, "fsync", lambda fd: fsyncs.__setitem__("n", fsyncs["n"] + 1))
    chain = AuditChain(str(tmp_path), clock=Clock(), fsync_mode="always")
    for i in range(5):
        chain.append("event", {"i": i})
    assert fsyncs["n"] == 5


def test_fsync_never_does_not_sync(tmp_path: Path, monkeypatch) -> None:
    import loci_litellm.audit_chain as ac
    fsyncs = {"n": 0}
    monkeypatch.setattr(ac.os, "fsync", lambda fd: fsyncs.__setitem__("n", fsyncs["n"] + 1))
    chain = AuditChain(str(tmp_path), clock=Clock(), fsync_mode="never")
    for i in range(3):
        chain.append("event", {"i": i})
    assert fsyncs["n"] == 0


def test_a_gzipped_segment_is_still_read_and_verified(tmp_path: Path) -> None:
    chain, clock = _chain(tmp_path, segment_bytes=1, gzip_after_days=7.0, delete_after_days=60.0)
    chain.append("event", {"n": 0})
    clock.t += 10 * 86400
    chain.append("event", {"n": 1})
    chain.maintenance()                   # gzips the first closed segment
    # the gzipped segment is transparently read back, and the full chain still verifies
    assert any(p.name.endswith(".gz") for p in chain.segments())
    kinds = [e["kind"] for e in chain.entries()]
    assert kinds.count("event") == 2 and chain.verify().ok is True
