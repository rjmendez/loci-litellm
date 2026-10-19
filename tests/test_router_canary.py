"""Router canary: share and strata, frozen snapshot, pass rule, degeneracy verdict, rollback, quality breaker."""
from __future__ import annotations

import copy
import functools
import json
import random
from datetime import datetime
from decimal import Decimal

import pytest

from loci_litellm import colony_router as cr
from loci_litellm.colony_router import (
    AuditLog, Caps, IncumbentSnapshot, LociColonyRouter, QualityBreaker, SliceStats, SnapshotRejected, Verdict,
    Watchdog, evaluate_pass, freeze_snapshot, judge, session_bucket, wilson_lower_bound,
)
from tests.test_colony_router import MODEL_LIST, make_router, req, train

CAPS = Caps()
ALL_GATES = {name: True for name in cr.REQUIRED_EXTERNAL_GATES}


def snapshot(mapping: dict | None = None) -> IncumbentSnapshot:
    return IncumbentSnapshot.load(freeze_snapshot(CAPS, mapping or {"*": "arm-b"}), CAPS)


def canary(share: float = 0.5, **kw):
    """A canary router whose best proven arm is arm-c while the incumbent is arm-b."""
    defaults = dict(canary_share=share, snapshot=snapshot(), spend_fn=lambda: {})
    defaults.update(kw)
    router = make_router(**defaults)
    train(router, "arm-c", 0.95)
    return router


@functools.lru_cache(maxsize=None)
def sessions(slice_name: str, share: float, count: int = 400) -> tuple:
    def bucket_in(b: float) -> bool:
        if slice_name == "control":
            return b < 0.10
        if slice_name == "learned":
            return 0.10 <= b < 0.10 + share
        return b >= 0.10 + share

    found, i = [], 0
    while len(found) < count:
        sid = f"canary-{slice_name}-{i}"
        if bucket_in(session_bucket(sid)):
            found.append(sid)
        i += 1
    return tuple(found)


# --- inert by default ---

def test_by_default_the_learned_path_is_inert_in_every_slice_and_stratum() -> None:
    router = make_router(snapshot=snapshot(), spend_fn=lambda: {})
    train(router, "arm-c", 0.95)

    decisions = [
        router.decide(req(session_id=sid, **flags))
        for sid in sessions("learned", 0.5, 100)
        for flags in ({}, {"has_tools": True}, {"delegated_write": True})
    ]

    assert {(d.acted, d.model, d.route_reason) for d in decisions} == {(False, None, "canary_off")}
    assert router.stats()["acted"] == 0
    assert router.canary_share == 0.0


def test_the_environment_default_has_no_canary_and_no_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LOCI_ROUTER_CANARY_SHARE", "LOCI_ROUTER_SNAPSHOT", "LOCI_ROUTER_LEARN_DIR", "LOCI_ROUTER_PAUSE_FILE"):
        monkeypatch.delenv(name, raising=False)

    router = cr._router_from_env()

    assert (router.canary_share, router.snapshot, router.state) == (0.0, None, "active")
    assert (router.stats()["invalid_canary_share"], router.stats()["snapshot_rejected"]) == (0, 0)


def test_the_canary_share_from_the_environment_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_ROUTER_SNAPSHOT", raising=False)
    monkeypatch.setenv("LOCI_ROUTER_CANARY_SHARE", "0.5")
    assert cr._router_from_env().canary_share == 0.5
    for bad in ("0.3", "1", "abc", "-0.1"):
        monkeypatch.setenv("LOCI_ROUTER_CANARY_SHARE", bad)
        router = cr._router_from_env()
        assert (router.canary_share, router.stats()["invalid_canary_share"]) == (0.0, 1)


def test_a_share_without_a_valid_snapshot_does_not_act(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("LOCI_ROUTER_CANARY_SHARE", "0.5")
    monkeypatch.delenv("LOCI_ROUTER_SNAPSHOT", raising=False)
    missing = cr._router_from_env()
    missing.spend_fn = lambda: {}
    missing.arms_fn = lambda: MODEL_LIST
    bad = tmp_path / "snapshot.json"
    bad.write_text("{not json", encoding="utf-8")
    monkeypatch.setenv("LOCI_ROUTER_SNAPSHOT", str(bad))
    rejected = cr._router_from_env()
    rejected.spend_fn = lambda: {}
    rejected.arms_fn = lambda: MODEL_LIST

    sid = sessions("learned", 0.5, 1)[0]
    assert missing.decide(req(session_id=sid)).route_reason == "no_snapshot"
    assert rejected.decide(req(session_id=sid)).route_reason == "no_snapshot"
    assert rejected.stats()["snapshot_rejected"] == 1
    assert rejected.stats()["acted"] == 0


def test_an_unsupported_canary_share_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="canary_share must be one of"):
        make_router(canary_share=0.3)
    assert make_router(canary_share=0.9).canary_share == 0.9


# --- share, control slice, strata ---

@pytest.mark.parametrize("share, learned, holdout", [(0.1, 1023, 7946), (0.5, 5007, 3962), (0.9, 8969, 0)])
def test_control_stays_ten_percent_while_the_learned_share_grows(share: float, learned: int, holdout: int) -> None:
    router = make_router(canary_share=share)
    counts = {"control": 0, "learned": 0, "holdout": 0}
    for i in range(10_000):
        counts[router._slice(req(session_id=f"session-{i}"))[0]] += 1

    assert counts == {"control": 1031, "learned": learned, "holdout": holdout}
    assert abs(counts["control"] / 10_000 - 0.10) <= 0.005


def test_the_router_acts_only_on_its_share() -> None:
    router = canary(0.5)

    outcome = {}
    for slice_name in ("control", "learned", "holdout"):
        decisions = [router.decide(req(session_id=sid)) for sid in sessions(slice_name, 0.5, 100)]
        outcome[slice_name] = (
            {(d.acted, d.slice) for d in decisions}, {d.model for d in decisions if d.model is not None},
            sum(d.acted for d in decisions),
        )

    assert outcome["control"] == ({(False, "control")}, set(), 0)
    assert outcome["holdout"] == ({(False, "holdout")}, set(), 0)
    assert outcome["learned"][0] == {(True, "learned")}
    assert outcome["learned"][2] == 100
    assert "arm-c" in outcome["learned"][1] <= {"arm-a", "arm-b", "arm-c", "arm-d"}
    assert router.stats()["acted"] == 100


def test_control_and_holdout_sessions_give_the_reason_they_were_left_alone() -> None:
    router = canary(0.5)

    assert router.decide(req(session_id=sessions("control", 0.5, 1)[0])).route_reason == "slice_control"
    assert router.decide(req(session_id=sessions("holdout", 0.5, 1)[0])).route_reason == "slice_holdout"
    assert router.decide(req(session_id="")).route_reason == "slice_control"


def test_tool_carrying_and_delegated_write_requests_are_never_routed_by_the_learned_rule() -> None:
    router = canary(0.9)

    tools = [router.decide(req(session_id=sid, has_tools=True)) for sid in sessions("learned", 0.9, 300)]
    writes = [router.decide(req(session_id=sid, delegated_write=True)) for sid in sessions("learned", 0.9, 300)]
    plain = [router.decide(req(session_id=sid)) for sid in sessions("learned", 0.9, 300)]

    assert {(d.acted, d.model, d.route_reason) for d in tools} == {(False, None, "ineligible_tools")}
    assert {(d.acted, d.model, d.route_reason) for d in writes} == {(False, None, "ineligible_delegated_write")}
    assert {d.acted for d in plain} == {True}
    assert router.stats()["acted"] == 300


def test_cached_prefix_and_later_turns_are_still_routed_greedily_with_zero_epsilon() -> None:
    router = canary(0.9)
    for flags, reason in (({"cached_prefix": True}, "cached_prefix"), ({"first_turn": False}, "not_first_turn")):
        decisions = [router.decide(req(session_id=sid, **flags)) for sid in sessions("learned", 0.9, 100)]
        assert {(d.acted, d.model, d.epsilon, d.epsilon_reason, d.propensity) for d in decisions} == {
            (True, "arm-c", 0.0, reason, 1.0)
        }


def test_canary_exploration_is_three_percent_with_the_exact_propensity() -> None:
    router = canary(0.9, rng=random.Random(99))

    rows = [router.decide(req(session_id=sid)) for sid in sessions("learned", 0.9, 400) * 3]

    assert {d.epsilon for d in rows} == {0.03}
    assert {(d.chosen_arm == d.greedy_arm, round(d.propensity, 9)) for d in rows} == {(True, 0.9775), (False, 0.0075)}
    assert any(not d.chosen_arm == d.greedy_arm for d in rows)
    assert {d.model for d in rows} == {d.chosen_group for d in rows}


def test_slices_that_are_not_acted_on_keep_the_shadow_epsilon() -> None:
    router = canary(0.5)

    assert {router.decide(req(session_id=sid)).epsilon for sid in sessions("holdout", 0.5, 50)} == {0.05}


@pytest.mark.parametrize("kwargs, reason", [
    (dict(spend_fn=None), "spend_unavailable"),
    (dict(pause_fn=lambda: True), "pause"),
    (dict(level_fn=lambda: 2), "governor_level"),
    (dict(snapshot=None), "no_snapshot"),
])
def test_the_canary_fails_closed(kwargs: dict, reason: str) -> None:
    router = canary(0.5, **kwargs)

    decision = router.decide(req(session_id=sessions("learned", 0.5, 1)[0]))

    assert (decision.acted, decision.model, decision.route_reason) == (False, None, reason)


def test_governor_level_one_still_routes_but_never_explores() -> None:
    router = canary(0.5, level_fn=lambda: 1)

    decision = router.decide(req(session_id=sessions("learned", 0.5, 1)[0]))

    assert (decision.acted, decision.model, decision.epsilon, decision.epsilon_reason) == (True, "arm-c", 0.0, "governor_level")


def test_no_safe_arm_means_no_action_even_in_the_learned_slice() -> None:
    router = canary(0.5, spend_fn=lambda: {p: Decimal("100") for p in ("prov-a", "prov-b", "prov-c", "prov-d")})

    decision = router.decide(req(session_id=sessions("learned", 0.5, 1)[0]))

    assert (decision.acted, decision.model, decision.reason, decision.route_reason) == (False, None, "no_safe_arm", "no_safe_arm")


# --- frozen incumbent snapshot ---

def test_a_frozen_snapshot_loads_with_its_digest_and_maps_cells_to_arms() -> None:
    frozen = freeze_snapshot(CAPS, {"summarize|-": "arm-a", "*": "arm-b"})

    loaded = IncumbentSnapshot.load(frozen, CAPS)

    assert loaded.sha256 == frozen["sha256"]
    assert (loaded.arm_for("summarize|-"), loaded.arm_for("other|-"), loaded.version) == ("arm-a", "arm-b", 1)
    assert IncumbentSnapshot(1, {}, {"x|-": "a"}, "").arm_for("y|-") is None


def test_a_snapshot_loads_from_a_file(tmp_path) -> None:
    path = tmp_path / "snap.json"
    path.write_text(json.dumps(freeze_snapshot(CAPS, {"*": "arm-b"})), encoding="utf-8")

    assert IncumbentSnapshot.load(str(path), CAPS).arm_for("any|-") == "arm-b"
    with pytest.raises(SnapshotRejected, match="snapshot unreadable"):
        IncumbentSnapshot.load(str(tmp_path / "missing.json"), CAPS)


def test_an_equal_cap_written_differently_is_accepted() -> None:
    frozen = freeze_snapshot(CAPS, {"*": "arm-b"})
    frozen["caps"] = {**frozen["caps"], "provider_share": "0.450", "daily_hard_cap_usd": "50.00"}
    body = {k: frozen[k] for k in ("version", "caps", "mapping")}
    frozen["sha256"] = cr._snapshot_digest(body)

    assert IncumbentSnapshot.load(frozen, CAPS).arm_for("x") == "arm-b"


@pytest.mark.parametrize("key, widened", [
    ("daily_hard_cap_usd", "75"), ("provider_share", "0.60"), ("eps_max", "0.10"), ("control_share", "0.05"),
    ("canary_eps", "0.05"), ("lambda_max", "9.0"), ("min_arm_n", "10"),
])
def test_a_snapshot_that_changes_any_cap_is_rejected_even_with_a_valid_digest(key: str, widened: str) -> None:
    frozen = freeze_snapshot(CAPS, {"*": "arm-b"})
    frozen["caps"][key] = widened
    frozen["sha256"] = cr._snapshot_digest({k: frozen[k] for k in ("version", "caps", "mapping")})

    with pytest.raises(SnapshotRejected, match="cap"):
        IncumbentSnapshot.load(frozen, CAPS)


def test_the_cap_error_names_the_cap_and_both_values() -> None:
    frozen = freeze_snapshot(CAPS, {"*": "arm-b"})
    frozen["caps"]["provider_share"] = "0.60"

    with pytest.raises(SnapshotRejected, match=r"cap 'provider_share' differs from the pinned envelope: 0\.60 != 0\.45"):
        IncumbentSnapshot.load(frozen, CAPS)


def test_missing_unknown_and_non_decimal_caps_are_rejected() -> None:
    missing = freeze_snapshot(CAPS, {"*": "a"})
    del missing["caps"]["eps_max"]
    unknown = freeze_snapshot(CAPS, {"*": "a"})
    unknown["caps"]["extra_cap"] = "1"
    as_float = freeze_snapshot(CAPS, {"*": "a"})
    as_float["caps"]["provider_share"] = 0.45
    garbage = freeze_snapshot(CAPS, {"*": "a"})
    garbage["caps"]["provider_share"] = "lots"

    with pytest.raises(SnapshotRejected, match="cap 'eps_max' is missing"):
        IncumbentSnapshot.load(missing, CAPS)
    with pytest.raises(SnapshotRejected, match="cap 'extra_cap' is not a known cap"):
        IncumbentSnapshot.load(unknown, CAPS)
    with pytest.raises(SnapshotRejected, match="cap 'provider_share' must be a decimal string"):
        IncumbentSnapshot.load(as_float, CAPS)
    with pytest.raises(SnapshotRejected, match="cap 'provider_share' is not a number"):
        IncumbentSnapshot.load(garbage, CAPS)
    with pytest.raises(SnapshotRejected, match="snapshot cap table is missing"):
        IncumbentSnapshot.load({"version": 1}, CAPS)


def test_an_edited_mapping_or_a_bad_body_is_rejected() -> None:
    edited = freeze_snapshot(CAPS, {"*": "arm-b"})
    edited["mapping"]["*"] = "arm-d"
    bad_version = freeze_snapshot(CAPS, {"*": "arm-b"}, version=2)
    bad_mapping = freeze_snapshot(CAPS, {"*": "arm-b"})
    bad_mapping["mapping"] = {"*": 7}

    with pytest.raises(SnapshotRejected, match="sha256 does not match"):
        IncumbentSnapshot.load(edited, CAPS)
    with pytest.raises(SnapshotRejected, match="unsupported snapshot version 2"):
        IncumbentSnapshot.load(bad_version, CAPS)
    with pytest.raises(SnapshotRejected, match="mapping must be a bounded string-to-string object"):
        IncumbentSnapshot.load(bad_mapping, CAPS)
    with pytest.raises(SnapshotRejected, match="not a JSON object"):
        IncumbentSnapshot.load([1, 2], CAPS)


def test_the_snapshot_names_the_incumbent_arm_for_a_cell() -> None:
    router = make_router(snapshot=snapshot({"unclassified|-": "arm-a"}))

    decision = router.decide(req(requested="arm-b"))

    assert decision.incumbent_arm == "arm-a"
    assert decision.greedy_arm == "arm-a"


# --- verdicts and the pass rule ---

def stats(**over) -> SliceStats:
    base = dict(
        n=200, completed=190, refused_or_empty=10, accepted=170, cost_usd=Decimal("100"), escalations=10,
        p95_latency_s=2.0, cache_read_tokens=500, prompt_tokens=1000, labelled=1000, quality_ok=900, mean_reward=0.5,
    )
    base.update(over)
    return SliceStats(**base)


INCUMBENT = stats()
CHEAPER_BUT_REFUSES = stats(completed=140, refused_or_empty=60, accepted=100, cost_usd=Decimal("60"), quality_ok=700, mean_reward=0.6)
CHEAPER_AND_FINE = stats(cost_usd=Decimal("80"), quality_ok=890, mean_reward=0.6)


def test_a_cheaper_candidate_that_refuses_is_degenerate_and_fails_the_pass_rule() -> None:
    assert judge(CHEAPER_BUT_REFUSES, INCUMBENT) == cr.VerdictResult(
        Verdict.DEGENERATE, ("completion", "refusal", "cheap_by_refusing", "reward_up_quality_down")
    )
    assert evaluate_pass(CHEAPER_BUT_REFUSES, INCUMBENT, ALL_GATES) == cr.PassResult(
        False, ("degenerate", "completion", "refusal", "cheap_by_refusing", "reward_up_quality_down")
    )


def test_a_cheaper_candidate_that_stays_correct_is_healthy_and_passes() -> None:
    assert judge(CHEAPER_AND_FINE, INCUMBENT) == cr.VerdictResult(Verdict.HEALTHY, ())
    assert evaluate_pass(CHEAPER_AND_FINE, INCUMBENT, ALL_GATES) == cr.PassResult(True, ())


def test_refusing_more_while_costing_more_is_not_cheap_by_refusing() -> None:
    candidate = stats(refused_or_empty=40, completed=190, cost_usd=Decimal("120"))

    assert judge(candidate, INCUMBENT) == cr.VerdictResult(Verdict.DEGENERATE, ("refusal",))


@pytest.mark.parametrize("completed, expected", [(188, ()), (187, ("completion",))])
def test_completion_boundary_is_exactly_one_percentage_point(completed: int, expected: tuple) -> None:
    assert judge(stats(completed=completed), INCUMBENT).reasons == expected


@pytest.mark.parametrize("refused, expected", [(12, ()), (13, ("refusal",))])
def test_refusal_boundary_is_exactly_one_percentage_point(refused: int, expected: tuple) -> None:
    assert judge(stats(refused_or_empty=refused), INCUMBENT).reasons == expected


@pytest.mark.parametrize("escalations, expected", [(14, ()), (30, ("escalation_drift",))])
def test_escalation_drift_beyond_two_standard_errors(escalations: int, expected: tuple) -> None:
    assert judge(stats(escalations=escalations), INCUMBENT).reasons == expected


def test_fewer_escalations_than_the_incumbent_is_not_drift() -> None:
    assert judge(stats(escalations=0), INCUMBENT).reasons == ()


def test_reward_up_with_quality_down_is_degenerate_and_the_twin_is_not() -> None:
    assert judge(stats(mean_reward=0.9, quality_ok=850), INCUMBENT).reasons == ("reward_up_quality_down",)
    assert judge(stats(mean_reward=0.9, quality_ok=900), INCUMBENT).reasons == ()
    assert judge(stats(mean_reward=0.4, quality_ok=850), INCUMBENT).reasons == ()


@pytest.mark.parametrize("n, verdict", [(29, Verdict.TOO_EARLY), (30, Verdict.HEALTHY)])
def test_burn_in_boundary(n: int, verdict: Verdict) -> None:
    candidate = stats(n=n, completed=n, refused_or_empty=0, escalations=0)
    incumbent = stats(n=n, completed=n, refused_or_empty=0, escalations=0)

    assert judge(candidate, incumbent).verdict is verdict
    failures = evaluate_pass(candidate, incumbent, ALL_GATES).failures
    assert (failures == ("too_early",)) is (n == 29)
    assert ("too_early" in failures) is (n == 29)


@pytest.mark.parametrize("field, value", [
    ("p95_latency_s", float("nan")), ("p95_latency_s", float("inf")), ("mean_reward", float("nan")),
    ("cost_usd", Decimal("NaN")), ("cost_usd", Decimal("Infinity")),
])
def test_non_finite_numbers_are_degenerate_before_anything_else(field: str, value) -> None:
    assert judge(stats(**{field: value}), INCUMBENT) == cr.VerdictResult(Verdict.DEGENERATE, ("nonfinite",))
    assert judge(stats(), stats(**{field: value})) == cr.VerdictResult(Verdict.DEGENERATE, ("nonfinite",))


def test_cost_per_accepted_must_be_down_at_least_ten_percent_exactly() -> None:
    assert evaluate_pass(stats(cost_usd=Decimal("90")), INCUMBENT, ALL_GATES).passed is True
    assert evaluate_pass(stats(cost_usd=Decimal("90.01")), INCUMBENT, ALL_GATES) == cr.PassResult(False, ("cost_per_accepted",))
    assert evaluate_pass(stats(cost_usd=Decimal("100"), accepted=0), INCUMBENT, ALL_GATES).failures == ("no_accepted",)


def test_the_quality_lower_bound_must_stay_above_minus_eight_points() -> None:
    assert evaluate_pass(stats(cost_usd=Decimal("80"), quality_ok=890), INCUMBENT, ALL_GATES).passed is True
    assert evaluate_pass(stats(cost_usd=Decimal("80"), quality_ok=800), INCUMBENT, ALL_GATES) == cr.PassResult(
        False, ("quality_lower_bound",)
    )
    assert evaluate_pass(stats(cost_usd=Decimal("80"), labelled=0, quality_ok=0), INCUMBENT, ALL_GATES).failures == ("no_labels",)


def test_the_bound_uses_the_standard_error_not_just_the_point_difference() -> None:
    # point difference -0.06 is above -0.08 but its one-sided 95% lower bound (-0.0846) is below it
    assert evaluate_pass(stats(cost_usd=Decimal("80"), quality_ok=840), INCUMBENT, ALL_GATES).failures == ("quality_lower_bound",)
    # point difference -0.05 has a lower bound of -0.0743
    assert evaluate_pass(stats(cost_usd=Decimal("80"), quality_ok=850), INCUMBENT, ALL_GATES).passed is True


@pytest.mark.parametrize("over, failure", [
    (dict(p95_latency_s=2.5), None), (dict(p95_latency_s=2.5000001), "p95_latency"),
    (dict(cache_read_tokens=470), None), (dict(cache_read_tokens=469), "cache_read_share"),
    (dict(cap_breaches=0), None), (dict(cap_breaches=1), "cap_breach"),
    (dict(attributed_escalations=0), None), (dict(attributed_escalations=1), "governor_escalation"),
])
def test_each_remaining_pass_criterion_has_an_exact_boundary(over: dict, failure) -> None:
    result = evaluate_pass(stats(cost_usd=Decimal("80"), **over), INCUMBENT, ALL_GATES)

    assert result.failures == (() if failure is None else (failure,))
    assert result.passed is (failure is None)


def test_external_gates_fail_closed_when_missing_or_false() -> None:
    assert evaluate_pass(CHEAPER_AND_FINE, INCUMBENT).failures == tuple(f"external:{n}" for n in cr.REQUIRED_EXTERNAL_GATES)
    one_false = {**ALL_GATES, "judge_gate": False}
    assert evaluate_pass(CHEAPER_AND_FINE, INCUMBENT, one_false).failures == ("external:judge_gate",)
    assert evaluate_pass(CHEAPER_AND_FINE, INCUMBENT, {**ALL_GATES, "holm": 1}).failures == ("external:holm",)


# --- watchdog and rollback ---

def test_a_degenerate_verdict_rolls_back_to_the_incumbent_within_one_tick(tmp_path) -> None:
    marker = tmp_path / "learn" / "ROLLED_BACK"
    router = canary(0.5, rollback_marker=str(marker))
    sid = sessions("learned", 0.5, 1)[0]
    assert router.decide(req(session_id=sid)).acted is True
    watchdog = Watchdog(router, lambda: (CHEAPER_BUT_REFUSES, INCUMBENT))

    result = watchdog.tick(1000.0)

    assert result.verdict is Verdict.DEGENERATE
    assert router.state == "rolled_back"
    assert router.rollback_reason == "degenerate:completion,refusal,cheap_by_refusing,reward_up_quality_down"
    after = router.decide(req(session_id=sid))
    assert (after.acted, after.model, after.route_reason) == (False, None, "rolled_back")
    assert [(e["kind"], e["reason"], e["snapshot_sha"]) for e in router.audit.entries] == [
        ("rollback", router.rollback_reason, router.snapshot.sha256)
    ]
    assert json.loads(marker.read_text(encoding="utf-8")) == {"ts": 1000.0, "reason": router.rollback_reason}
    assert router.stats()["rollbacks"] == 1


def test_a_healthy_or_too_early_verdict_leaves_the_canary_running() -> None:
    sid = sessions("learned", 0.5, 1)[0]
    for pair, verdict in (
        ((CHEAPER_AND_FINE, INCUMBENT), Verdict.HEALTHY),
        ((stats(n=10), stats(n=10)), Verdict.TOO_EARLY),
    ):
        router = canary(0.5)
        assert Watchdog(router, lambda pair=pair: pair).tick(5.0).verdict is verdict
        assert (router.state, router.decide(req(session_id=sid)).acted) == ("active", True)
        assert router.stats()["rollbacks"] == 0


def test_a_rollback_is_final_a_later_healthy_tick_does_not_resume() -> None:
    router = canary(0.5)
    pairs = [(CHEAPER_BUT_REFUSES, INCUMBENT), (CHEAPER_AND_FINE, INCUMBENT)]
    watchdog = Watchdog(router, lambda: pairs.pop(0))
    watchdog.tick(1.0)

    assert watchdog.tick(2.0).verdict is Verdict.HEALTHY

    assert router.state == "rolled_back"
    assert router.decide(req(session_id=sessions("learned", 0.5, 1)[0])).acted is False
    assert (router.stats()["rollbacks"], len(router.audit.entries)) == (1, 1)
    assert router.rollback("again") is False


def test_a_rolled_back_router_stays_rolled_back_after_a_restart(tmp_path) -> None:
    marker = str(tmp_path / "ROLLED_BACK")
    first = canary(0.5, rollback_marker=marker)
    first.rollback("degenerate:test", 3.0)

    restarted = canary(0.5, rollback_marker=marker)
    fresh = canary(0.5, rollback_marker=str(tmp_path / "other-marker"))

    sid = sessions("learned", 0.5, 1)[0]
    assert (restarted.state, restarted.rollback_reason) == ("rolled_back", "marker")
    assert restarted.decide(req(session_id=sid)).route_reason == "rolled_back"
    assert (fresh.state, fresh.decide(req(session_id=sid)).acted) == ("active", True)


def test_a_marker_that_cannot_be_written_is_counted_and_the_rollback_still_holds(tmp_path) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("x", encoding="utf-8")
    router = canary(0.5, rollback_marker=str(blocker / "ROLLED_BACK"))

    assert router.rollback("degenerate:test", 1.0) is True

    assert (router.state, router.stats()["marker_write_errors"]) == ("rolled_back", 1)


def test_the_watchdog_ticks_at_most_once_per_interval() -> None:
    router = canary(0.5)
    calls = []
    watchdog = Watchdog(router, lambda: calls.append(1) or (CHEAPER_AND_FINE, INCUMBENT))

    assert watchdog.maybe_tick(0.0).verdict is Verdict.HEALTHY
    assert watchdog.maybe_tick(299.0) is None
    assert watchdog.maybe_tick(300.0).verdict is Verdict.HEALTHY
    assert len(calls) == 2
    assert router.stats()["watchdog_ticks"] == 2


def test_missing_stats_and_a_failing_stats_source_are_counted_not_read_as_healthy() -> None:
    router = canary(0.5)
    assert Watchdog(router, lambda: None).tick(1.0) is None
    failing = Watchdog(router, lambda: (_ for _ in ()).throw(RuntimeError("ledger offline")))
    assert failing.tick(2.0) is None

    stats_ = router.stats()
    assert (stats_["watchdog_no_data"], stats_["watchdog_errors"], stats_["watchdog_ticks"], router.state) == (1, 1, 2, "active")


# --- quality breaker ---

def test_wilson_lower_bound_known_values() -> None:
    assert abs(wilson_lower_bound(90, 100) - 0.8256) < 5e-5
    assert wilson_lower_bound(0, 0) == 0.0
    assert wilson_lower_bound(0, 10) == 0.0 or abs(wilson_lower_bound(0, 10)) < 1e-12


def _fill(breaker: QualityBreaker, cell: str, down: tuple, base: tuple) -> None:
    for ok_count, n, downgraded in ((*down, True), (*base, False)):
        for i in range(n):
            breaker.observe(cell, downgraded, i < ok_count)


def test_a_quality_breaker_trip_disables_the_downgrade_and_writes_an_audit_entry(tmp_path) -> None:
    audit = AuditLog(str(tmp_path / "audit.jsonl"))
    breaker = QualityBreaker(audit)
    _fill(breaker, "summarize|-", down=(20, 30), base=(27, 30))

    assert breaker.downgrade_allowed("summarize|-", 100.0) is True
    assert breaker.check("summarize|-", 100.0) is True

    assert breaker.downgrade_allowed("summarize|-", 100.0) is False
    assert breaker.downgrade_allowed("other|-", 100.0) is True
    entry = audit.entries[0]
    assert (entry["kind"], entry["cell"], entry["n_downgraded"], entry["n_baseline"], entry["disabled_until"]) == (
        "quality_breaker_trip", "summarize|-", 30, 30, 100.0 + 86400.0
    )
    assert abs(entry["lb_downgraded"] - 0.4877972581) < 1e-9
    assert abs(entry["lb_baseline"] - 0.7437855868) < 1e-9
    assert json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0]) == json.loads(json.dumps(entry))
    assert breaker.check("summarize|-", 200.0) is False  # already disabled: no second entry
    assert len(audit.entries) == 1


def test_the_downgrade_comes_back_after_24_hours_with_a_fresh_sample() -> None:
    breaker = QualityBreaker(AuditLog())
    _fill(breaker, "c|-", down=(20, 30), base=(27, 30))
    breaker.check("c|-", 0.0)

    assert breaker.downgrade_allowed("c|-", 86399.9) is False
    assert breaker.downgrade_allowed("c|-", 86400.0) is True
    assert breaker.check("c|-", 86400.0) is False  # counts were cleared


def test_the_breaker_needs_30_results_on_each_side_and_a_real_gap() -> None:
    few = QualityBreaker(AuditLog())
    _fill(few, "c|-", down=(10, 29), base=(27, 30))
    enough = QualityBreaker(AuditLog())
    _fill(enough, "c|-", down=(10, 30), base=(27, 30))
    close = QualityBreaker(AuditLog())
    _fill(close, "c|-", down=(26, 30), base=(27, 30))
    few_base = QualityBreaker(AuditLog())
    _fill(few_base, "c|-", down=(10, 30), base=(25, 29))

    assert few.check("c|-", 0.0) is False
    assert enough.check("c|-", 0.0) is True
    assert close.check("c|-", 0.0) is False
    assert few_base.check("c|-", 0.0) is False
    assert QualityBreaker(AuditLog()).check("never-seen|-", 0.0) is False


def test_a_tripped_breaker_keeps_the_learned_router_from_downgrading_the_cell() -> None:
    models = MODEL_LIST
    mapping = {"*": "arm-a"}  # incumbent arm-a; arm-d (free) would be a downgrade, arm-b and arm-c are not
    sid = sessions("learned", 0.5, 1)[0]

    def routed(trip: bool):
        router = make_router(canary_share=0.5, snapshot=snapshot(mapping), spend_fn=lambda: {}, arms_fn=lambda: models)
        train(router, "arm-d", 0.95)
        if trip:
            _fill(router.breaker, "unclassified|-", down=(20, 30), base=(27, 30))
            router.breaker.check("unclassified|-", router.clock())
        return router, router.decide(req(requested="arm-a", session_id=sid))

    free, free_decision = routed(trip=False)
    blocked, blocked_decision = routed(trip=True)

    assert (free_decision.acted, free_decision.model) == (True, "arm-d")
    assert (blocked_decision.acted, blocked_decision.model) == (True, "arm-a")
    assert blocked_decision.excluded == {"arm-d": "breaker"}
    assert blocked_decision.safe_arms == ["arm-a", "arm-b", "arm-c"]
    assert [e["kind"] for e in blocked.audit.entries] == ["quality_breaker_trip"]
    assert len(free.audit.entries) == 0


# --- the callback acts only when the router says so ---

def _data(sid: str, **extra) -> dict:
    data = {
        "model": "arm-b", "litellm_call_id": "call-1", "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"session_id": sid},
    }
    data.update(extra)
    return data


@pytest.mark.asyncio
async def test_the_callback_routes_the_learned_slice_and_leaves_everything_else_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ROUTER_SHADOW", "1")
    callback = LociColonyRouter(canary(0.5))
    learned = _data(sessions("learned", 0.5, 1)[0])
    control = _data(sessions("control", 0.5, 1)[0])
    with_tools = _data(sessions("learned", 0.5, 1)[0], tools=[{"type": "function", "function": {"name": "f"}}])
    untouched = [copy.deepcopy(control), copy.deepcopy(with_tools)]

    for data in (learned, control, with_tools):
        assert await callback.async_pre_call_hook(None, None, data, "acompletion") is None

    assert learned["model"] == "arm-c"
    decision_id = callback.router.recorder.rows()[0]["decision_id"]
    assert learned["metadata"] == {"session_id": sessions("learned", 0.5, 1)[0], "loci_router": {"decision_id": decision_id, "slice": "learned"}}
    assert [control, with_tools] == untouched


@pytest.mark.asyncio
async def test_the_callback_never_routes_when_the_canary_share_is_zero_or_shadow_is_off(monkeypatch: pytest.MonkeyPatch) -> None:
    sid = sessions("learned", 0.5, 1)[0]
    monkeypatch.setenv("LOCI_ROUTER_SHADOW", "1")
    off = LociColonyRouter(canary(0.0))
    data = _data(sid)
    before = copy.deepcopy(data)
    await off.async_pre_call_hook(None, None, data, "acompletion")
    assert data == before

    monkeypatch.delenv("LOCI_ROUTER_SHADOW")
    disabled = LociColonyRouter(canary(0.5))
    data = _data(sid)
    before = copy.deepcopy(data)
    await disabled.async_pre_call_hook(None, None, data, "acompletion")
    assert data == before
    assert disabled.router.stats()["decisions"] == 0


@pytest.mark.asyncio
async def test_a_watchdog_tick_on_an_outcome_event_rolls_the_router_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ROUTER_SHADOW", "1")
    router = canary(0.5)
    callback = LociColonyRouter(router, Watchdog(router, lambda: (CHEAPER_BUT_REFUSES, INCUMBENT)))
    sid = sessions("learned", 0.5, 1)[0]
    first = _data(sid)
    await callback.async_pre_call_hook(None, None, first, "acompletion")
    assert first["model"] == "arm-c"

    await callback.async_log_success_event(
        {"litellm_call_id": "call-1", "litellm_params": {"metadata": {"model_group": "arm-c"}}},
        None, datetime.now(), datetime.now(),
    )

    assert router.state == "rolled_back"
    second = _data(sid)
    await callback.async_pre_call_hook(None, None, second, "acompletion")
    assert second["model"] == "arm-b"
