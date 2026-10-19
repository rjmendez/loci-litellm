"""Governor v0 (G1 spend, G2 hourly, G8 STOP/PAUSE): decisions, reservation arithmetic, fail policy, counter."""
from __future__ import annotations

import json
import os
import random
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from loci_litellm import governor as gov
from loci_litellm.audit_chain import AuditChain
from loci_litellm.governor import (
    Arm,
    Catalog,
    GovernorRejection,
    Request,
    Snapshot,
    SpendCounter,
    StopFiles,
    decide,
    local_day,
    parse_mode,
    sanitize_max_tokens,
)
from tests._governor_fixtures import (
    BASE_TS,
    BIG,
    D,
    FakeClock,
    RecordingStore,
    body,
    caps,
    catalog,
    make_governor,
    trusted,
)

NY = ZoneInfo("America/New_York")


def snap(**kw: str) -> Snapshot:
    return Snapshot(**{k: D(v) for k, v in kw.items()})


# --------------------------------------------------------------------------- G1: caps and arithmetic
def test_boundary_exactly_at_the_hard_cap_is_allowed_and_one_cent_over_is_denied() -> None:
    c = caps(day_soft=D("40"), day_hard=D("40"))
    req = Request(model="big", prompt_tokens=1000, max_tokens=1, has_tools=True)      # reserves exactly 0.02
    at_cap = decide(req, snap(day_total="39.98"), c, catalog())
    over = decide(req, snap(day_total="39.99"), c, catalog())
    assert (at_cap.action, at_cap.reservation) == ("allow", D("0.02"))
    assert (over.action, over.code, over.status, over.cap) == ("reject", "budget_day", 429, D("40"))


def test_governor_denies_at_39_99_and_reserves_at_39_00_with_a_recording_store(tmp_path: Path) -> None:
    c = caps(day_soft=D("40"), day_hard=D("40.00"))
    denied_store = RecordingStore(snap=snap(day_total="39.99"))
    rejection = make_governor(tmp_path, denied_store, c, cat=catalog(BIG)).process(body(tokens=1000, tools=True))
    assert isinstance(rejection, GovernorRejection)
    assert (rejection.status_code, rejection.code, rejection.body["error"]["cap"]) == (429, "budget_day", "40.00")
    assert (denied_store.denials, denied_store.reservations) == (["budget_day"], [])

    ok_store = RecordingStore(snap=snap(day_total="39.00"))
    assert make_governor(tmp_path, ok_store, c, cat=catalog(BIG)).process(body(tokens=1000, tools=True)) is None
    assert (ok_store.denials, ok_store.reservations) == ([], [(D("0.02"), "anonymous", False)])


def test_reservation_arithmetic_is_exact_decimal() -> None:
    arm = Arm("a", "alias", D("3"), D("15"), supports_tools=True)
    cat = Catalog([arm], num_retries=0)
    out = decide(Request(model="alias", prompt_tokens=41_000, max_tokens=4096), Snapshot(), caps(), cat)
    # 41000 * 3 + 4096 * 15 = 123000 + 61440 = 184440 micro-dollars per million = 0.18444
    assert out.reservation == D("0.184440")
    assert isinstance(out.reservation, Decimal)


def test_reservation_uses_the_dearest_arm_in_the_chain_times_every_attempt() -> None:
    arms = [Arm(f"id{p}", name, D(p), D(p), supports_tools=True)
            for p, name in ((1, "a"), (3, "b"), (5, "c"), (9, "d"))]
    cat = Catalog(arms, fallbacks={"a": ["b", "c", "d"]}, num_retries=2)
    out = decide(Request(model="a", prompt_tokens=100_000, max_tokens=10_000), Snapshot(), caps(), cat)
    # worst arm costs (100000 + 10000) * 9 / 1e6 = 0.99 per attempt; three attempts
    assert out.reservation == D("2.970000")
    single = Catalog(arms[:1], num_retries=2)
    assert decide(Request(model="a", prompt_tokens=100_000, max_tokens=10_000), Snapshot(), caps(), single).reservation == D("0.330000")


def test_reservation_is_decimal_arithmetic_not_float_rounding() -> None:
    arm = Arm("a", "alias", D("1.1"), D("0"), supports_tools=True)
    out = decide(Request(model="alias", prompt_tokens=10_970, max_tokens=1), Snapshot(), caps(), Catalog([arm], num_retries=0))
    assert out.reservation == D("0.012067")          # 10970 * 1.1 = 12067 exactly; a float product rounds up to 0.012068


def test_p95_completion_caps_the_reserved_completion_for_that_arm() -> None:
    arm = Arm("a", "alias", D("0"), D("10"), supports_tools=True)
    capped = Catalog([arm], num_retries=0, p95_completion={"a": 100})
    uncapped = Catalog([arm], num_retries=0)
    req = Request(model="alias", prompt_tokens=0, max_tokens=1000)
    assert decide(req, Snapshot(), caps(), capped).reservation == D("0.001000")
    assert decide(req, Snapshot(), caps(), uncapped).reservation == D("0.010000")


def test_settlement_is_the_sum_of_attempts_and_trues_up_the_reservation(tmp_path: Path) -> None:
    clock = FakeClock()
    counter = SpendCounter(str(tmp_path / "counter.json"), clock=clock, tz=NY)
    counter.reserve("c1", D("2.97"), clock(), "anonymous")
    assert counter.snapshot(clock()).day_total == D("2.97")
    counter.settle("c1", D("0.50"), clock(), final=False)
    counter.settle("c1", D("0.50"), clock(), final=False)
    assert counter.snapshot(clock()).day_total == D("2.97") + D("1.00")      # reservation still held, two attempts booked
    counter.settle("c1", D("0.70"), clock(), final=True)
    after = counter.snapshot(clock())
    assert (after.day_total, after.rolling_24h, after.hour) == (D("1.70"), D("1.70"), D("1.70"))
    assert counter.stats()["inflight"] == 0


def test_a_call_that_reports_no_cost_is_booked_at_its_reservation(tmp_path: Path) -> None:
    clock = FakeClock()
    counter = SpendCounter(str(tmp_path / "counter.json"), clock=clock, tz=NY)
    counter.reserve("c1", D("0.40"), clock())
    assert counter.settle("c1", None, clock()) == D("0.40")
    assert counter.stats()["settled_at_reservation"] == 1
    counter.reserve("c2", D("0.40"), clock())
    assert counter.settle("c2", D("0"), clock()) == D("0")          # a reported zero is a real zero
    assert counter.snapshot(clock()).day_total == D("0.40")


# --------------------------------------------------------------------------- G1/G2: the other lines
CAPS = caps(day_soft=D("30"), day_hard=D("50"), rolling_24h=D("60"), hour_downgrade=D("10"),
            hour_reject=D("25"), interactive_stop=D("75"))


def test_tool_carrying_request_over_the_hard_cap_is_rejected_and_a_tool_free_one_is_downgraded() -> None:
    req = Request(model="big", prompt_tokens=150_000, max_tokens=1, has_tools=True)    # reserves 3.00 (small: 0.30)
    near = snap(day_total="49")
    rejected = decide(req, near, CAPS, catalog())
    assert (rejected.action, rejected.code, rejected.status, rejected.cap) == ("reject", "budget_day", 429, D("50"))
    free = decide(replace(req, has_tools=False), snap(day_total="49"), CAPS, catalog())
    assert (free.action, free.code, free.target_model, free.target_arm, free.reservation) == \
        ("downgrade", "budget_day", "small", "arm-small", D("0.300000"))
    still_over = decide(replace(req, has_tools=False), snap(day_total="49.5"), caps(day_soft=D("30"), day_hard=D("50")), catalog(BIG))
    assert (still_over.action, still_over.code) == ("reject", "budget_day")


def test_hard_cap_and_rolling_cap_report_their_own_codes() -> None:
    req = Request(model="big", prompt_tokens=1000, max_tokens=1, has_tools=True)         # reserves 0.02
    assert decide(req, snap(day_total="10", rolling_24h="59.99"), CAPS, catalog()).code == "budget_rolling"
    assert decide(req, snap(day_total="49.99", rolling_24h="49.99"), CAPS, catalog()).code == "budget_day"
    assert decide(req, snap(day_total="10", rolling_24h="59.98"), CAPS, catalog()).action == "allow"


def test_hourly_trigger_downgrades_and_the_higher_line_rejects_non_interactive_callers() -> None:
    req = Request(model="big", prompt_tokens=1000, max_tokens=1, has_tools=True)         # reserves 0.02
    below = decide(req, snap(hour="9.97"), CAPS, catalog())
    assert (below.action, below.code, below.explore_allowed) == ("allow", "", True)
    trigger = decide(req, snap(hour="9.99"), CAPS, catalog())
    assert (trigger.action, trigger.code, trigger.cap, trigger.target_model) == \
        ("downgrade", "budget_hour_downgrade", D("10"), "small")
    assert trigger.explore_allowed is False
    reject = decide(req, snap(hour="24.99"), CAPS, catalog())
    assert (reject.action, reject.code, reject.status, reject.cap) == ("reject", "budget_hour", 429, D("25"))


def test_soft_day_cap_downgrades_without_exploration() -> None:
    req = Request(model="big", prompt_tokens=1000, max_tokens=1)
    assert decide(req, snap(day_total="29.98"), CAPS, catalog()).action == "allow"
    soft = decide(req, snap(day_total="29.99"), CAPS, catalog())
    assert (soft.action, soft.code, soft.cap, soft.target_model, soft.explore_allowed) == \
        ("downgrade", "budget_soft", D("30"), "small", False)


def test_interactive_caller_is_downgraded_never_rejected_until_the_absolute_stop() -> None:
    inter = Request(model="big", prompt_tokens=1000, max_tokens=1, has_tools=True, interactive=True)
    at_hard = decide(inter, snap(day_total="49.99"), CAPS, catalog())
    assert (at_hard.action, at_hard.code, at_hard.target_model) == ("downgrade", "budget_day", "small")
    hourly = decide(inter, snap(hour="24.99"), CAPS, catalog())
    assert (hourly.action, hourly.code) == ("downgrade", "budget_hour_downgrade")
    just_under_stop = decide(inter, snap(day_total="74.98"), CAPS, catalog())
    assert just_under_stop.action == "downgrade"
    stop = decide(inter, snap(day_total="74.99"), CAPS, catalog())
    assert (stop.action, stop.code, stop.status, stop.cap) == ("reject", "budget_abs_stop", 429, D("75"))
    # the same state for a non-interactive caller is a rejection at the hard cap already
    assert decide(replace(inter, interactive=False), snap(day_total="49.99"), CAPS, catalog()).action == "reject"


def test_interactive_caller_on_the_cheapest_arm_is_allowed_and_flagged_not_dropped() -> None:
    inter = Request(model="small", prompt_tokens=1000, max_tokens=1, interactive=True)
    out = decide(inter, snap(day_total="49.999"), CAPS, catalog())
    assert (out.action, out.code, out.notes, out.explore_allowed) == ("allow", "", ("no_cheaper_arm",), False)


# --------------------------------------------------------------------------- fail policy
def test_unreadable_ledger_applies_the_provisional_cap_and_a_readable_one_does_not(tmp_path: Path) -> None:
    clock = FakeClock()
    unreadable = tmp_path / "ledger_is_a_directory"
    unreadable.mkdir()
    bad = SpendCounter(str(tmp_path / "bad.json"), str(unreadable), clock=clock, tz=NY)
    degraded = bad.snapshot(clock())
    assert degraded.degraded is True
    c = caps(day_hard=D("50"), day_soft=D("50"), provisional_day=D("25"))
    req = Request(model="big", prompt_tokens=100_000, max_tokens=1, has_tools=True)     # reserves 2.00
    out = decide(req, replace(degraded, day_total=D("23.01")), c, catalog())
    assert (out.action, out.code, out.cap, out.notes) == ("reject", "budget_day", D("25"), ("ledger_degraded", "provisional_cap"))
    assert decide(req, replace(degraded, day_total=D("22.99")), c, catalog()).action == "allow"
    good = SpendCounter(str(tmp_path / "good.json"), str(tmp_path / "missing_ledger.jsonl"), clock=clock, tz=NY)
    assert good.snapshot(clock()).degraded is False
    assert decide(req, replace(good.snapshot(clock()), day_total=D("23.01")), c, catalog()).action == "allow"


def test_corrupt_counter_is_reseeded_from_the_ledger_and_not_degraded(tmp_path: Path) -> None:
    clock = FakeClock()
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text(json.dumps({"ts": BASE_TS - 60, "cost_usd": 1.25}) + "\n", encoding="utf-8")
    counter_file = tmp_path / "counter.json"
    counter_file.write_text("{not json", encoding="utf-8")
    counter = SpendCounter(str(counter_file), str(ledger), clock=clock, tz=NY)
    assert counter.snapshot(clock()).day_total == D("1.25")
    assert counter.degraded is False
    nothing = tmp_path / "nothing.json"
    nothing.write_text("{not json", encoding="utf-8")
    assert SpendCounter(str(nothing), None, clock=clock, tz=NY).snapshot(clock()).degraded is True


def test_exception_in_the_g1_arithmetic_is_503_in_enforce_and_never_a_pass_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_a: object, **_k: object) -> None:
        raise ZeroDivisionError("injected")

    monkeypatch.setattr(gov, "decide", explode)
    store = RecordingStore()
    g = make_governor(tmp_path, store, CAPS)
    data = body()
    rejection = g.process(data)
    assert isinstance(rejection, GovernorRejection)
    assert (rejection.status_code, rejection.code) == (503, "budget_gate_error")
    assert (store.reservations, g.stats()["fail_closed_events"]) == ([], 1)
    assert data["model"] == "big"


def test_exception_in_the_g1_arithmetic_in_observe_mode_is_recorded_and_lets_the_call_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(*_a: object, **_k: object) -> None:
        raise ZeroDivisionError("injected")

    monkeypatch.setattr(gov, "decide", explode)
    events: list = []
    g = make_governor(tmp_path, RecordingStore(), CAPS, mode="observe", events=events)
    assert g.process(body()) is None
    assert [(e["event"], e["code"]) for e in events] == [("would_block", "budget_gate_error")]


def test_optimisation_failure_passes_through_and_is_counted(tmp_path: Path) -> None:
    calls: list = []

    def broken(data: dict, ctx: object) -> None:
        calls.append(1)
        raise KeyError("grounding trim failed")

    store = RecordingStore()
    g = make_governor(tmp_path, store, CAPS, optimisers=(broken,))
    assert g.process(body()) is None
    assert (len(calls), g.stats()["fail_open_events"], len(store.reservations)) == (1, 1, 1)


def test_more_than_five_fail_open_events_in_60s_flip_to_fail_closed_and_recover(tmp_path: Path) -> None:
    clock = FakeClock()

    def broken(data: dict, ctx: object) -> None:
        raise KeyError("optimiser down")

    store = RecordingStore()
    g = make_governor(tmp_path, store, CAPS, clock=clock, optimisers=(broken,))
    for _ in range(5):                                  # five events: still open
        assert g.process(body()) is None
        clock.advance(1)
    sixth = g.process(body())                           # the sixth within 60 s trips it
    assert isinstance(sixth, GovernorRejection) and (sixth.status_code, sixth.code) == (503, "governor_degraded")
    assert g.stats()["fail_closed_breaker"] is True
    assert len(store.reservations) == 5
    clock.advance(61)                                   # events age out: one new failure is fail-open again
    assert g.process(body()) is None
    assert g.stats()["fail_closed_breaker"] is False


# --------------------------------------------------------------------------- fuzz
def test_fuzz_no_request_passes_without_a_reservation(tmp_path: Path) -> None:
    rng = random.Random(20261018)
    max_tokens_values = [None, -1, "x", 10**9, "missing", 0, 4096, True, float("nan"), "512", 3.9]
    models = ["big", "small", "no-such-model", ""]
    store = RecordingStore()
    g = make_governor(tmp_path, store, caps(day_hard=D("1000000"), day_soft=D("1000000")), cat=catalog())
    outcomes = 0
    for i in range(300):
        data: dict = {"model": rng.choice(models), "messages": rng.choice([[], [{"role": "user", "content": ""}], [{"role": "user", "content": "y" * rng.randint(0, 5000)}], [{"role": "user", "content": None}]])}
        choice = rng.choice(max_tokens_values)
        if choice != "missing":
            data["max_tokens"] = choice
        before = len(store.reservations)
        result = g.process(data)
        outcomes += 1
        if result is None:
            assert len(store.reservations) == before + 1, f"request {i} passed unreserved: {data!r}"
            if data["model"] in ("no-such-model", ""):
                assert store.reservations[-1][0] > D("0")
        else:
            assert len(store.reservations) == before
    assert (outcomes, len(store.reservations)) == (300, 300)
    assert all(isinstance(amount, Decimal) for amount, _, _ in store.reservations)


def test_unknown_model_and_missing_price_reserve_the_unknown_price_not_zero() -> None:
    out = decide(Request(model="not-in-catalog", prompt_tokens=1000, max_tokens=100), Snapshot(), caps(), Catalog([], num_retries=0))
    assert out.reservation == D("0.022500")                  # (1000 * 15 + 100 * 75) / 1e6
    unpriced = Catalog([Arm("u", "unpriced", None, None)], num_retries=0)
    assert decide(Request(model="unpriced", prompt_tokens=1000, max_tokens=100), Snapshot(), caps(), unpriced).reservation == D("0.022500")
    free = Catalog([Arm("l", "local", D("0"), D("0"))], num_retries=0)
    assert decide(Request(model="local", prompt_tokens=1000, max_tokens=100), Snapshot(), caps(), free).reservation == D("0")


def test_sanitize_max_tokens_exact_values() -> None:
    ceiling = 32768
    assert [sanitize_max_tokens(v, ceiling) for v in (None, -1, "x", 10**9, 0, True, float("nan"), float("inf"))] == [ceiling] * 8
    assert [sanitize_max_tokens(v, ceiling) for v in (1, 4096, "512", 3.9, ceiling)] == [1, 4096, 512, 3, ceiling]


# --------------------------------------------------------------------------- counter file
def test_counter_file_is_written_through_temp_and_rename_and_stays_under_4_kib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    path = tmp_path / "counter.json"
    counter = SpendCounter(str(path), clock=clock, tz=NY)
    replaced: list = []
    real_replace = os.replace

    def recording_replace(src: str, dst: str) -> None:
        replaced.append((Path(src).name, Path(dst).name))
        real_replace(src, dst)

    monkeypatch.setattr(gov.os, "replace", recording_replace)
    counter.settle("c0", D("1.5"), clock())
    assert replaced == [("counter.json.tmp", "counter.json")]
    assert not (tmp_path / "counter.json.tmp").exists()
    for i in reversed(range(96)):                              # fill every 15-minute bucket of the rolling day
        counter.settle(f"c{i}", D("1234.567891"), clock() - i * 900, client=f"client-{i % 20}")
    assert path.stat().st_size < 4096
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["v"] == 1 and len(doc["buckets"]) == 96


def test_cap_does_not_reset_when_the_ledger_rotates(tmp_path: Path) -> None:
    clock = FakeClock()
    ledger = tmp_path / "ledger.jsonl"
    counter_file = tmp_path / "counter.json"
    first = SpendCounter(str(counter_file), str(ledger), clock=clock, tz=NY)
    first.settle("c1", D("39.00"), clock())
    ledger.write_text(json.dumps({"ts": BASE_TS - 30, "cost_usd": 30.0}) + "\n", encoding="utf-8")
    os.replace(ledger, str(ledger) + ".1")                     # rotation: the live ledger is now empty/absent
    ledger.write_text("", encoding="utf-8")
    after = SpendCounter(str(counter_file), str(ledger), clock=clock, tz=NY).snapshot(clock())
    assert (after.day_total, after.rolling_24h) == (D("39"), D("39"))
    # without the counter file the seed still sees the rotated generation
    counter_file.unlink()
    reseeded = SpendCounter(str(counter_file), str(ledger), clock=clock, tz=NY).snapshot(clock())
    assert (reseeded.day_total, reseeded.rolling_24h) == (D("30"), D("30"))


def test_seed_sums_both_ledger_generations_and_skips_bad_rows(tmp_path: Path) -> None:
    clock = FakeClock()
    ledger = tmp_path / "ledger.jsonl"
    rows_old = [{"ts": BASE_TS - 600, "cost_usd": 2.5}, {"ts": BASE_TS - 100000, "cost_usd": 9.0}]
    rows_new = [{"ts": BASE_TS - 60, "cost_usd": 0.25}]
    Path(str(ledger) + ".1").write_text("\n".join(json.dumps(r) for r in rows_old) + "\nnot json\n", encoding="utf-8")
    ledger.write_text("\n".join(json.dumps(r) for r in rows_new) + "\n", encoding="utf-8")
    counter = SpendCounter(str(tmp_path / "c.json"), str(ledger), clock=clock, tz=NY)
    shot = counter.snapshot(clock())
    assert (shot.day_total, shot.rolling_24h, shot.hour) == (D("2.75"), D("2.75"), D("2.75"))
    assert (counter.stats()["ledger_bad_rows"], counter.stats()["seeded_from_ledger"]) == (1, 1)


def test_local_day_is_dst_aware_and_rolls_the_counter(tmp_path: Path) -> None:
    def utc(y: int, m: int, d: int, h: int, mi: int) -> float:
        return datetime(y, m, d, h, mi, tzinfo=timezone.utc).timestamp()

    # fall back 2026-11-01: a 25-hour local day
    assert local_day(utc(2026, 11, 1, 4, 30), NY) == "2026-11-01"        # 00:30 EDT
    assert local_day(utc(2026, 11, 2, 4, 30), NY) == "2026-11-01"        # 23:30 EST, 24 h later
    assert local_day(utc(2026, 11, 2, 5, 30), NY) == "2026-11-02"        # 00:30 EST
    # spring forward 2026-03-08: a 23-hour local day
    assert local_day(utc(2026, 3, 8, 6, 30), NY) == "2026-03-08"         # 01:30 EST
    assert local_day(utc(2026, 3, 8, 7, 30), NY) == "2026-03-08"         # 03:30 EDT
    assert local_day(utc(2026, 3, 9, 3, 59), NY) == "2026-03-08"         # 23:59 EDT
    assert local_day(utc(2026, 3, 9, 4, 0), NY) == "2026-03-09"
    clock = FakeClock(utc(2026, 11, 2, 4, 30))
    counter = SpendCounter(str(tmp_path / "c.json"), clock=clock, tz=NY)
    counter.settle("a", D("7"), clock())
    clock.advance(3600)                                                   # 00:30 EST next day
    shot = counter.snapshot(clock())
    assert (shot.day_total, shot.rolling_24h) == (D("0"), D("7"))


# --------------------------------------------------------------------------- G8: STOP and PAUSE
def test_stop_file_answers_503_after_the_one_second_cache_while_zero_tier_and_health_pass(tmp_path: Path) -> None:
    clock = FakeClock()
    store = RecordingStore()
    g = make_governor(tmp_path, store, CAPS, clock=clock)
    assert g.process(body()) is None                                      # t=0: no STOP file
    (tmp_path / "STOP").write_text("", encoding="utf-8")
    clock.advance(0.5)
    assert g.process(body()) is None                                      # still inside the 1 s cache
    clock.advance(0.5)
    stopped = g.process(body())
    assert isinstance(stopped, GovernorRejection) and (stopped.status_code, stopped.code) == (503, "stopped")
    assert stopped.headers == {"Retry-After": "30"}
    assert g.process(body(metadata={"loci_tier": "ZERO"})) is None
    assert g.process(body(), call_type="health") is None
    assert len(store.reservations) == 2                                   # the two early calls only
    (tmp_path / "STOP").unlink()
    clock.advance(1)
    assert g.process(body()) is None                                      # positive twin: removed, passes


def test_pause_halts_learning_and_exploration_but_not_cheap_routing(tmp_path: Path) -> None:
    clock = FakeClock()
    store = RecordingStore(snap=snap(day_total="29.99"))                  # over the soft cap: a downgrade applies
    g = make_governor(tmp_path, store, CAPS, clock=clock)
    assert g.learning_allowed() is True
    (tmp_path / "PAUSE").write_text("", encoding="utf-8")
    clock.advance(2)
    data = body(tokens=1000)
    assert g.process(data) is None
    assert g.learning_allowed() is False
    assert data["metadata"]["loci_paused"] is True
    assert data["metadata"]["loci_explore_allowed"] is False
    assert data["model"] == "small"                                       # cheap routing still works
    (tmp_path / "PAUSE").unlink()
    clock.advance(2)
    fresh = body(tokens=1000)
    assert g.process(fresh) is None and "loci_paused" not in fresh["metadata"]
    assert g.learning_allowed() is True


def test_stop_file_with_an_undeterminable_state_counts_as_present(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    clock = FakeClock()
    files = StopFiles(str(tmp_path / "STOP"), str(tmp_path / "PAUSE"), clock=clock)
    assert files.stopped() is False
    real_stat = Path.stat

    def locked(self: Path, *a: object, **k: object):
        if self.name == "STOP":
            raise PermissionError("denied")
        return real_stat(self, *a, **k)

    monkeypatch.setattr(Path, "stat", locked)
    clock.advance(2)
    assert files.stopped() is True and files.stat_errors == 1


# --------------------------------------------------------------------------- observe is the default
def test_mode_parsing_defaults_to_observe() -> None:
    assert [parse_mode(v) for v in (None, "", "observe", "block", "enforce-ish", "1")] == ["observe"] * 6
    assert [parse_mode(v) for v in ("enforce", " ENFORCE ", "Enforce")] == ["enforce"] * 3


def test_observe_allows_the_over_cap_request_and_writes_the_exact_would_block_event(tmp_path: Path) -> None:
    store = RecordingStore(snap=snap(day_total="49.99"))
    events: list = []
    data = body(tokens=1000, tools=True)
    g = make_governor(tmp_path, store, CAPS, mode="", events=events)
    assert g.process(data) is None
    assert data["model"] == "big"
    assert [(e["event"], e["code"], e["cap"], e["mode"]) for e in events] == [("would_block", "budget_day", "50", "observe")]
    assert store.denials == ["budget_day"]
    assert len(store.reservations) == 1                                   # observe still books the spend
    enforced = make_governor(tmp_path, RecordingStore(snap=snap(day_total="49.99")), CAPS, mode="enforce")
    assert isinstance(enforced.process(body(tokens=1000, tools=True)), GovernorRejection)


def test_observe_mode_never_edits_the_request(tmp_path: Path) -> None:
    store = RecordingStore(snap=snap(day_total="49.99"))
    data = body(tokens=1000)                                              # tool-free: enforce would downgrade it
    before = json.dumps(data, sort_keys=True)
    events: list = []
    g = make_governor(tmp_path, store, CAPS, mode="observe", events=events)
    assert g.process(data) is None
    assert json.dumps({k: v for k, v in data.items() if k != "metadata"}, sort_keys=True) == before
    assert events[0]["action"] == "downgrade" and events[0]["event"] == "would_block"


def test_enforce_needs_the_envelope_entry(tmp_path: Path) -> None:
    over = snap(day_total="49.99")
    unpinned = replace(trusted(CAPS), level="L3", alarm="envelope_unpinned", source="defaults")
    g = make_governor(tmp_path, RecordingStore(snap=over), CAPS, mode="enforce", env=unpinned)
    assert g.process(body(tokens=1000, tools=True)) is None               # asked to enforce, no envelope: observe
    assert g.stats()["enforce_refused_by_envelope"] == 1
    off = make_governor(tmp_path, RecordingStore(snap=over), CAPS, mode="enforce", env=trusted(CAPS, enforce=False))
    assert off.process(body(tokens=1000, tools=True)) is None
    on = make_governor(tmp_path, RecordingStore(snap=over), CAPS, mode="enforce", env=trusted(CAPS, enforce=True))
    assert isinstance(on.process(body(tokens=1000, tools=True)), GovernorRejection)


def test_once_enforcing_a_failed_envelope_check_keeps_enforcing_on_the_defaults(tmp_path: Path) -> None:
    state = {"env": trusted(CAPS)}
    store = RecordingStore(snap=snap(day_total="49.99"))
    clock = FakeClock()
    g = gov.Governor(store=store, envelope=lambda: state["env"], catalog=lambda: catalog(), clock=clock,
                     stop=StopFiles(str(tmp_path / "STOP"), str(tmp_path / "PAUSE"), clock=clock), mode="enforce")
    assert isinstance(g.process(body(tokens=1000, tools=True)), GovernorRejection)
    state["env"] = gov.EnvelopeState(caps=caps(day_hard=D("50")), enforce=False, day_timezone="local", level="L3",
                                     alarm="caps_hash_mismatch", source="defaults")
    again = g.process(body(tokens=1000, tools=True))
    assert isinstance(again, GovernorRejection) and again.code == "budget_day"
    assert g.stats()["enforce_on_defaults"] == 1


# --------------------------------------------------------------------------- audit and the callback
def test_audit_failure_is_audit_unavailable_for_rw_agents_only(tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x", encoding="utf-8")
    broken = AuditChain(str(blocker / "audit"))
    store = RecordingStore()
    g = make_governor(tmp_path, store, CAPS, audit=broken)
    rw = g.process(body(metadata={"loci_rw": True}))
    assert isinstance(rw, GovernorRejection) and (rw.status_code, rw.code) == (503, "audit_unavailable")
    assert store.reservations == []
    assert g.process(body()) is None                                      # a read-only agent is not affected
    assert g.stats()["audit_failures"] == 1
    healthy = make_governor(tmp_path, RecordingStore(), CAPS, audit=AuditChain(str(tmp_path / "audit")))
    assert healthy.process(body(metadata={"loci_rw": True})) is None


def test_would_block_events_land_in_the_hash_chain(tmp_path: Path) -> None:
    audit = AuditChain(str(tmp_path / "audit"))
    g = make_governor(tmp_path, RecordingStore(snap=snap(day_total="49.99")), CAPS, mode="observe", audit=audit)
    g.process(body(tokens=1000, tools=True))
    entries = list(audit.entries())
    assert [(e["kind"], e["data"]["code"]) for e in entries] == [("would_block", "budget_day")]
    assert audit.verify().ok is True


@pytest.mark.asyncio
async def test_callback_raises_the_rejection_and_settles_from_the_log_event(tmp_path: Path) -> None:
    clock = FakeClock()
    counter = SpendCounter(str(tmp_path / "counter.json"), clock=clock, tz=NY)
    g = make_governor(tmp_path, counter, CAPS, clock=clock)
    cb = gov.LociGovernor(g)
    data = body(tokens=1000, tools=True)
    assert await cb.async_pre_call_hook(None, None, data, "acompletion") is None
    call_id = data["metadata"]["loci_governor_id"]
    assert counter.reserved(call_id) == D("0.02")
    kwargs = {"litellm_params": {"metadata": {"loci_governor_id": call_id}}, "standard_logging_object": {"response_cost": 0.0125}}
    await cb.async_log_success_event(kwargs, None, None, None)
    assert counter.reserved(call_id) is None
    assert counter.snapshot(clock()).day_total == D("0.0125")
    counter.settle("x", D("60"), clock())
    with pytest.raises(GovernorRejection, match="budget_day"):
        await cb.async_pre_call_hook(None, None, body(tokens=1000, tools=True), "acompletion")
    failing = body(tokens=1000, tools=True)
    clock.advance(86400 * 2)
    assert await cb.async_pre_call_hook(None, None, failing, "acompletion") is None
    await cb.async_post_call_failure_hook(failing, RuntimeError("upstream"), None)
    assert counter.reserved(failing["metadata"]["loci_governor_id"]) is None


def test_default_instance_does_no_io_on_import_and_stays_inert_without_the_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_GOVERNOR_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("LOCI_GOVERNOR", raising=False)
    monkeypatch.setenv("LOCI_GOVERNOR_LEDGER", str(tmp_path / "ledger.jsonl"))
    cb = gov.LociGovernor()
    assert not (tmp_path / "state").exists()
    assert cb._governor is None
    data = {"model": "anything", "messages": [{"role": "user", "content": "hi"}]}
    assert cb.governor.process(data) is None                              # observe, no envelope: nothing is refused
    assert data["model"] == "anything"
    assert data["metadata"]["loci_governor_mode"] == "observe"


def test_catalog_from_model_list_reads_prices_tools_and_fallbacks_without_names() -> None:
    model_list = [
        {"model_name": "alpha", "model_info": {"id": "id-a", "input_cost_per_token": 0.000003, "output_cost_per_token": 0.000015,
                                                 "supports_function_calling": True, "max_input_tokens": 1000},
         "litellm_params": {"model": "p/m", "api_base": "https://api.example.test"}},
        {"model_name": "beta", "model_info": {"id": "id-b"}, "litellm_params": {"model": "q/n", "api_base": "http://127.0.0.1:11434"}},
    ]
    cat = Catalog.from_model_list(model_list, fallbacks={"alpha": ["beta"]}, num_retries=1)
    alpha, beta = cat.groups()["alpha"][0], cat.groups()["beta"][0]
    assert (alpha.arm_id, alpha.in_per_mtok, alpha.out_per_mtok, alpha.supports_tools, alpha.max_input_tokens, alpha.local) == \
        ("id-a", D("3"), D("15"), True, 1000, False)
    assert (beta.in_per_mtok, beta.supports_tools, beta.local, beta.priced) == (None, False, True, False)
    assert [a.arm_id for a in cat.chain("alpha")] == ["id-a", "id-b"]
