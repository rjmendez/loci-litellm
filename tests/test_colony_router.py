"""Colony router in shadow: arms, propensity, Thompson shadow rows, control slice, strata, recorder, callback."""
from __future__ import annotations

import copy
import functools
import itertools
import json
import random
import statistics
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from loci_litellm import colony_router as cr
from loci_litellm.colony_router import (
    Caps, CellStore, ColonyRouter, LagrangeController, LociColonyRouter, RouteRequest, ShadowRecorder,
    enumerate_arms, propensity, request_from_data, session_bucket,
)


def deployment(name: str, provider: str, price_in: str = "0.000001", price_out: str = "0.000002", **info) -> dict:
    model_info = {"input_cost_per_token": float(price_in), "output_cost_per_token": float(price_out), **info}
    return {"model_name": name, "litellm_params": {"model": f"{provider}/{name}-impl"}, "model_info": model_info}


MODEL_LIST = [
    deployment("arm-a", "prov-a", "0.000001", "0.000002"),
    deployment("arm-b", "prov-b", "0.000003", "0.000015"),
    deployment("arm-c", "prov-c", "0.000002", "0.000008"),
    deployment("arm-d", "prov-d", "0", "0"),
]


def make_router(**kw) -> ColonyRouter:
    defaults = dict(
        arms_fn=lambda: MODEL_LIST, rng=random.Random(1), thompson_rng=random.Random(2),
        thompson_rate=0.0, clock=lambda: 1000.0,
    )
    defaults.update(kw)
    return ColonyRouter(**defaults)


@functools.lru_cache(maxsize=None)
def session(control: bool, skip: int = 0, salt: str = cr.DEFAULT_SALT) -> str:
    found = (
        f"sess-{i}" for i in itertools.count()
        if (session_bucket(f"sess-{i}", salt) < 0.10) is control
    )
    return next(itertools.islice(found, skip, None))


def req(**kw) -> RouteRequest:
    defaults = dict(requested="arm-b", session_id=session(False), call_id="")
    defaults.update(kw)
    return RouteRequest(**defaults)


def train(router: ColonyRouter, arm: str, quality: float, times: int = 40, task: str = "unclassified") -> None:
    for _ in range(times):
        router.cells.observe(arm, task, "-", quality)


# --- arm enumeration ---

def test_arms_come_from_model_info_with_decimal_prices() -> None:
    arm_set = enumerate_arms(MODEL_LIST)

    assert [(a.arm_id, a.group, a.provider, a.unit_cost, a.supports_tools) for a in arm_set.arms] == [
        ("arm-a", "arm-a", "prov-a", Decimal("0.000003"), True),
        ("arm-b", "arm-b", "prov-b", Decimal("0.000018"), True),
        ("arm-c", "arm-c", "prov-c", Decimal("0.00001"), True),
        ("arm-d", "arm-d", "prov-d", Decimal("0"), True),
    ]
    assert arm_set.excluded == ()


def test_a_seat_billed_provider_is_never_an_arm_and_a_gateway_one_is() -> None:
    seat = deployment("seat-model", "github_copilot")
    gateway = deployment("gw-model", "prov-x")

    arm_set = enumerate_arms([seat, gateway])

    assert [a.arm_id for a in arm_set.arms] == ["gw-model"]
    assert arm_set.excluded == (("seat-model", "seat_billed"),)


def test_other_non_arm_entries_are_excluded_with_a_reason() -> None:
    entries = [
        deployment("flagged", "prov-x", seat_billed=True),
        deployment("opted-out", "prov-x", loci_router_arm=False),
        deployment("embedder", "prov-x", mode="embedding"),
        {"litellm_params": {"model": "prov-x/y"}},
        deployment("chat-ok", "prov-x", mode="chat"),
    ]

    arm_set = enumerate_arms(entries)

    assert [a.arm_id for a in arm_set.arms] == ["chat-ok"]
    assert arm_set.excluded == (
        ("flagged", "seat_billed"), ("opted-out", "opted_out"), ("embedder", "not_chat"), ("(unnamed)", "no_model_name"),
    )


def test_the_exclusion_list_can_be_extended_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ROUTER_EXCLUDE_PROVIDERS", "prov-x, other")

    arm_set = enumerate_arms([deployment("x1", "prov-x"), deployment("y1", "prov-y")])

    assert [a.arm_id for a in arm_set.arms] == ["y1"]


def test_deployments_sharing_a_name_become_distinct_arms_and_ids_win_over_names() -> None:
    entries = [deployment("same", "prov-a"), deployment("same", "prov-b"), deployment("named", "prov-a", id="dep-9")]

    arm_set = enumerate_arms(entries)

    assert [(a.arm_id, a.group, a.provider) for a in arm_set.arms] == [
        ("same", "same", "prov-a"), ("same#2", "same", "prov-b"), ("dep-9", "named", "prov-a"),
    ]


def test_tool_support_and_a_non_list_model_list() -> None:
    arm_set = enumerate_arms([deployment("t", "p", supports_function_calling=False), deployment("u", "p")])

    assert [(a.arm_id, a.supports_tools) for a in arm_set.arms] == [("t", False), ("u", True)]
    assert enumerate_arms(None).arms == ()
    assert enumerate_arms({"not": "a list"}).arms == ()


# --- exact propensity ---

def test_propensity_formula() -> None:
    assert abs(propensity(0.05, 4, True) - 0.9625) < 1e-9
    assert abs(propensity(0.05, 4, False) - 0.0125) < 1e-9
    assert propensity(0.0, 4, True) == 1.0


def test_logged_propensity_equals_the_closed_form_for_every_row() -> None:
    router = make_router(epsilon=0.05, rng=random.Random(20261018))
    rows = []
    for i in range(2000):
        rows.append(router.decide(req(session_id=session(False, skip=i % 50))).to_row())

    explored = 0
    for row in rows:
        greedy = row["chosen_arm"] == row["greedy_arm"]
        expected = (1 - 0.05) * (1.0 if greedy else 0.0) + 0.05 / 4
        assert abs(row["propensity"] - expected) < 1e-9
        assert row["epsilon"] == 0.05
        assert row["greedy_arm"] == "arm-b"
        explored += row["explore"]
    non_greedy = [r for r in rows if r["chosen_arm"] != r["greedy_arm"]]
    assert {round(r["propensity"], 9) for r in non_greedy} == {0.0125}
    assert {round(r["propensity"], 9) for r in rows if r["chosen_arm"] == r["greedy_arm"]} == {0.9625}
    assert (explored, len(non_greedy)) == (101, 82)


def test_the_logged_propensities_of_all_arms_sum_to_one() -> None:
    eps, n = 0.05, 4
    assert abs(propensity(eps, n, True) + (n - 1) * propensity(eps, n, False) - 1.0) < 1e-9


def test_epsilon_is_capped_by_the_pinned_envelope() -> None:
    assert make_router(epsilon=0.9).epsilon == 0.05
    assert make_router(epsilon=-1.0).epsilon == 0.0
    assert make_router(epsilon=0.03).epsilon == 0.03


def test_the_default_exploration_rate_is_exactly_five_percent_even_under_a_looser_envelope() -> None:
    # The envelope clamps a too-high rate, which hides the default itself: raising EPS_MAX to 0.06 passed every test.
    # A looser envelope shows the real default.
    loose = cr.Caps(eps_max=Decimal("0.5"))
    assert make_router(caps=loose).epsilon == 0.05
    assert make_router(caps=loose, epsilon=0.3).epsilon == 0.3          # twin: an explicit rate under that envelope is honoured
    assert make_router().epsilon == 0.05                                # the pinned envelope agrees


def test_an_unparseable_epsilon_in_the_environment_falls_back_to_the_default(monkeypatch) -> None:
    # a looser envelope, so the fallback value itself is visible instead of being clamped to 0.05 either way
    monkeypatch.setattr(cr.Caps, "from_env", classmethod(lambda cls: cls(eps_max=Decimal("0.5"))))
    monkeypatch.setenv("LOCI_ROUTER_EPSILON", "not-a-number")
    assert cr._router_from_env().epsilon == 0.05
    monkeypatch.setenv("LOCI_ROUTER_EPSILON", "0.3")                    # twin: a valid value is used, not the default
    assert cr._router_from_env().epsilon == 0.3
    monkeypatch.delenv("LOCI_ROUTER_EPSILON")
    assert cr._router_from_env().epsilon == 0.05


# --- Thompson shadow rows ---

def test_thompson_row_has_a_monte_carlo_propensity_from_200_draws() -> None:
    router = make_router(thompson_rate=1.0, thompson_rng=random.Random(7))
    for arm in ("arm-a", "arm-b", "arm-d"):
        train(router, arm, 0.0)
    train(router, "arm-c", 1.0)

    row = router.decide(req()).to_row()

    assert row["thompson"] == {"arm": "arm-c", "propensity": 1.0, "draws": 200}


def test_thompson_propensity_matches_an_independent_recount_of_the_draws() -> None:
    router = make_router(thompson_rate=1.0, thompson_rng=random.Random(11))

    row = router.decide(req()).to_row()

    oracle = random.Random(11)
    oracle.random()  # the sampling-rate gate consumes one value first
    picks = []
    for _ in range(200):
        draws = [(oracle.betavariate(1.0, 1.0), arm) for arm in ("arm-a", "arm-b", "arm-c", "arm-d")]
        picks.append(max(draws, key=lambda d: d[0])[1])
    assert row["thompson"]["arm"] == picks[0]
    assert row["thompson"]["propensity"] == picks.count(picks[0]) / 200
    assert row["thompson"]["draws"] == 200
    assert 1 <= round(row["thompson"]["propensity"] * 200) <= 200


def test_thompson_draws_have_a_floor_of_200() -> None:
    assert make_router(thompson_draws=50).thompson_draws == 200
    assert make_router(thompson_draws=500).thompson_draws == 500


def test_no_thompson_row_when_the_sampling_rate_is_zero_or_the_session_is_control() -> None:
    assert make_router(thompson_rate=0.0).decide(req()).thompson is None
    assert make_router(thompson_rate=1.0).decide(req(session_id=session(True))).thompson is None
    assert make_router(thompson_rate=1.0).decide(req()).thompson is not None


def test_thompson_never_changes_the_epsilon_greedy_choice() -> None:
    plain = make_router(rng=random.Random(5), thompson_rate=0.0)
    shadowed = make_router(rng=random.Random(5), thompson_rate=1.0)

    a = [(d.chosen_arm, d.propensity) for d in (plain.decide(req(session_id=session(False, i))) for i in range(300))]
    b = [(d.chosen_arm, d.propensity) for d in (shadowed.decide(req(session_id=session(False, i))) for i in range(300))]

    assert a == b


# --- control slice ---

def test_control_slice_is_exactly_ten_percent_of_10000_seeded_sessions() -> None:
    control = sum(session_bucket(f"session-{i}") < 0.10 for i in range(10_000))

    assert control == 1031
    assert abs(control / 10_000 - 0.10) <= 0.005


def test_the_same_session_always_maps_the_same_way_across_calls_and_routers() -> None:
    ids = [f"session-{i}" for i in range(200)]
    first = [session_bucket(s) < 0.10 for s in ids]

    assert [session_bucket(s) < 0.10 for s in ids] == first
    r1, r2 = make_router(), make_router()
    for s in ids[:40]:
        flags = {r.decide(req(session_id=s)).control for r in (r1, r2, r1, r2, r1)}
        assert len(flags) == 1


def test_control_is_per_session_not_per_call() -> None:
    router = make_router(epsilon=0.05)
    for control in (True, False):
        sid = session(control)
        flags = [router.decide(req(session_id=sid)).control for _ in range(60)]
        assert flags == [control] * 60


def test_the_salt_changes_the_assignment() -> None:
    ids = [f"session-{i}" for i in range(500)]

    default = [session_bucket(s) < 0.10 for s in ids]
    salted = [session_bucket(s, "another-salt") < 0.10 for s in ids]

    assert default != salted


def test_control_sessions_get_the_incumbent_with_propensity_one_and_still_log_the_counterfactual() -> None:
    router = make_router()
    train(router, "arm-c", 0.95)

    row = router.decide(req(session_id=session(True))).to_row()

    assert (row["control"], row["control_reason"]) == (True, "slice")
    assert (row["chosen_arm"], row["propensity"], row["epsilon"], row["epsilon_reason"]) == ("arm-b", 1.0, 0.0, "control")
    assert row["greedy_arm"] == "arm-c"
    assert row["would_change"] is False


def test_a_request_without_a_session_id_is_control_and_counted() -> None:
    router = make_router()

    row = router.decide(req(session_id="")).to_row()

    assert (row["control"], row["control_reason"], row["session_hash"]) == (True, "no_session", "")
    assert router.stats()["no_session"] == 1
    twin = router.decide(req(session_id=session(False))).to_row()
    assert (twin["control"], twin["control_reason"]) == (False, "")
    assert router.stats()["no_session"] == 1


def test_the_logged_session_is_a_hash_not_the_raw_id() -> None:
    row = make_router().decide(req(session_id="secret-session-token")).to_row()

    assert "secret-session-token" not in json.dumps(row)
    assert row["session_hash"] == cr.session_hash("secret-session-token")
    assert len(row["session_hash"]) == 12


# --- epsilon is zero in the guarded strata ---

STRATA = [
    ("tools", dict(has_tools=True)),
    ("cached_prefix", dict(cached_prefix=True)),
    ("delegated_write", dict(delegated_write=True)),
    ("not_first_turn", dict(first_turn=False)),
]


@pytest.mark.parametrize("reason, flags", STRATA)
def test_epsilon_is_zero_in_each_guarded_stratum_and_nothing_explores(reason: str, flags: dict) -> None:
    router = make_router(epsilon=0.05, rng=random.Random(3))

    assert router.epsilon_for(req(**flags), 4, False) == (0.0, reason)
    rows = [router.decide(req(session_id=session(False, i), **flags)).to_row() for i in range(300)]
    assert {(r["explore"], r["propensity"], r["epsilon"], r["epsilon_reason"]) for r in rows} == {(False, 1.0, 0.0, reason)}


def test_epsilon_is_zero_at_governor_level_one_pause_and_low_resource() -> None:
    assert make_router(level_fn=lambda: 1).epsilon_for(req(), 4, False) == (0.0, "governor_level")
    assert make_router(level_fn=lambda: 3).epsilon_for(req(), 4, False) == (0.0, "governor_level")
    assert make_router(pause_fn=lambda: True).epsilon_for(req(), 4, False) == (0.0, "pause")
    assert make_router(resource_fn=lambda: 0.34).epsilon_for(req(), 4, False) == (0.0, "resource")


def test_epsilon_is_positive_when_no_guard_applies_including_the_boundaries() -> None:
    assert make_router(level_fn=lambda: 0, resource_fn=lambda: 0.35).epsilon_for(req(), 4, False) == (0.05, "")
    assert make_router().epsilon_for(req(), 2, False) == (0.05, "")
    assert make_router().epsilon_for(req(), 1, False) == (0.0, "single_arm")


# --- safe set ---

def test_provider_share_cap_is_a_constraint_on_the_safe_set_at_exactly_45_percent() -> None:
    limit = Decimal("22.50")  # 45% of the 50 daily hard cap
    assert Caps().provider_limit_usd() == limit
    at_cap = make_router(spend_fn=lambda: {"prov-a": limit}).decide(req()).to_row()
    just_under = make_router(spend_fn=lambda: {"prov-a": Decimal("22.49")}).decide(req()).to_row()

    assert at_cap["safe_arms"] == ["arm-b", "arm-c", "arm-d"]
    assert at_cap["excluded"] == {"arm-a": "provider_share_cap"}
    assert just_under["safe_arms"] == ["arm-a", "arm-b", "arm-c", "arm-d"]
    assert just_under["excluded"] == {}
    assert at_cap["share_cap"] == "applied"


def test_a_missing_spend_view_is_reported_not_silent() -> None:
    router = make_router(spend_fn=None)

    row = router.decide(req()).to_row()

    assert row["share_cap"] == "unavailable"
    assert router.stats()["spend_unavailable"] == 1
    failing = make_router(spend_fn=lambda: (_ for _ in ()).throw(RuntimeError("ledger offline")))
    assert failing.decide(req()).to_row()["share_cap"] == "unavailable"


def test_an_arm_in_an_open_failure_quorum_leaves_the_safe_set() -> None:
    router = make_router()
    for caller in ("c1", "c2", "c1"):
        router.observe_result("arm-a", False, caller=caller, status=503)

    row = router.decide(req()).to_row()

    assert row["excluded"] == {"arm-a": "unhealthy"}
    assert row["safe_arms"] == ["arm-b", "arm-c", "arm-d"]


def test_a_single_callers_failures_do_not_remove_an_arm_from_the_safe_set() -> None:
    router = make_router()
    for _ in range(5):
        router.observe_result("arm-a", False, caller="only-one", status=503)

    assert router.decide(req()).to_row()["excluded"] == {}


def test_a_request_with_tools_only_sees_tool_capable_arms() -> None:
    models = MODEL_LIST[:2] + [deployment("arm-x", "prov-x", supports_function_calling=False)]
    router = make_router(arms_fn=lambda: models)

    with_tools = router.decide(req(has_tools=True)).to_row()
    without = router.decide(req(session_id=session(False, 1))).to_row()

    assert with_tools["excluded"] == {"arm-x": "no_tool_support"}
    assert without["excluded"] == {}


def test_no_safe_arm_logs_a_reason_and_picks_nothing() -> None:
    router = make_router(spend_fn=lambda: {p: Decimal("100") for p in ("prov-a", "prov-b", "prov-c", "prov-d")})

    row = router.decide(req()).to_row()

    assert (row["reason"], row["chosen_arm"], row["propensity"], row["safe_arms"]) == ("no_safe_arm", None, None, [])
    assert router.stats()["no_safe_arm"] == 1


def test_unavailable_arms_are_reported_and_cached_for_a_minute() -> None:
    calls = []
    clock = [1000.0]

    def arms():
        calls.append(clock[0])
        return MODEL_LIST

    router = make_router(arms_fn=arms, clock=lambda: clock[0])
    router.decide(req())
    clock[0] += 59.0
    router.decide(req())
    assert calls == [1000.0]
    clock[0] += 1.0
    router.decide(req())
    assert calls == [1000.0, 1060.0]

    down = make_router(arms_fn=lambda: None)
    row = down.decide(req()).to_row()
    assert (row["reason"], row["chosen_arm"]) == ("no_arms", None)
    assert down.stats()["arms_unavailable"] == 1
    assert down.stats()["no_arms"] == 1


# --- greedy rule ---

def test_with_no_labelled_outcomes_the_greedy_arm_is_the_incumbent() -> None:
    assert make_router().decide(req(requested="arm-c")).greedy_arm == "arm-c"


def test_when_the_incumbent_is_unavailable_the_cheapest_safe_arm_is_greedy() -> None:
    router = make_router(spend_fn=lambda: {"prov-b": Decimal("30")})

    decision = router.decide(req(requested="arm-b"))

    assert decision.incumbent_arm is None
    assert decision.greedy_arm == "arm-d"


def test_a_cell_stays_on_its_prior_until_30_labelled_outcomes() -> None:
    router = make_router()
    train(router, "arm-c", 1.0, times=29)
    assert router.decide(req()).greedy_arm == "arm-b"
    train(router, "arm-c", 1.0, times=1)

    assert router.decide(req(session_id=session(False, 1))).greedy_arm == "arm-c"


def test_a_proven_better_arm_beats_the_incumbent_and_a_proven_worse_one_does_not() -> None:
    better = make_router()
    train(better, "arm-c", 0.9)
    worse = make_router()
    train(worse, "arm-c", 0.1)

    assert better.decide(req()).greedy_arm == "arm-c"
    assert worse.decide(req()).greedy_arm == "arm-b"


def test_below_the_quality_floor_a_proven_arm_scores_minus_one_and_loses_to_an_unproven_incumbent() -> None:
    below = make_router(quality_floor=0.7)
    train(below, "arm-c", 0.6)
    above = make_router(quality_floor=0.5)
    train(above, "arm-c", 0.6)

    assert below.decide(req()).greedy_arm == "arm-b"
    assert above.decide(req()).greedy_arm == "arm-c"


def test_the_cost_multiplier_prefers_the_cheaper_of_two_equally_good_arms() -> None:
    def tilted(lam: float) -> str:
        router = make_router(lagrange=LagrangeController(lam=lam))
        train(router, "arm-b", 0.8)
        train(router, "arm-d", 0.8)
        return router.decide(req()).greedy_arm

    assert tilted(0.0) == "arm-b"  # a tie goes to the incumbent
    assert tilted(2.0) == "arm-d"


def test_would_change_compares_the_pick_with_the_requested_model() -> None:
    router = make_router()
    train(router, "arm-c", 0.9)

    row = router.decide(req(session_id=session(False))).to_row()

    assert (row["chosen_group"], row["would_change"]) == ("arm-c", True)
    assert router.stats()["would_change"] == 1


# --- Lagrange multiplier ---

def test_lambda_rises_monotonically_to_the_clamp_at_twice_the_target_and_stays() -> None:
    controller = LagrangeController(lam=0.0)
    seen = [controller.step(Decimal("100"), Decimal("50"), 300.0 * i) for i in range(14)]

    assert seen[:4] == [0.5, 1.0, 1.5, 2.0]
    assert all(b >= a for a, b in zip(seen, seen[1:]))
    assert seen[9:] == [5.0] * 5
    assert max(seen) == 5.0


def test_lambda_holds_at_target_and_falls_below_it() -> None:
    held = LagrangeController(lam=2.0)
    assert [held.step(Decimal("50"), Decimal("50"), 300.0 * i) for i in range(10)] == [2.0] * 10
    below = LagrangeController(lam=0.4)
    assert [below.step(Decimal("0"), Decimal("50"), 300.0 * i) for i in range(3)] == [0.0, 0.0, 0.0]


def test_lambda_updates_at_most_once_per_interval() -> None:
    controller = LagrangeController(lam=0.0)

    assert controller.step(Decimal("100"), Decimal("50"), 0.0) == 0.5
    assert controller.step(Decimal("100"), Decimal("50"), 299.0) == 0.5
    assert controller.step(Decimal("100"), Decimal("50"), 300.0) == 1.0


def test_lambda_rejects_a_non_positive_target() -> None:
    with pytest.raises(ValueError, match="target must be positive"):
        LagrangeController().step(Decimal("1"), Decimal("0"), 0.0)


# --- cell store ---

def test_posterior_update_is_discounted_toward_the_prior() -> None:
    cells = CellStore(min_n=1)
    cells.observe("a", "t", "-", 1.0)
    assert cells.posterior("a", "t", "-") == (2.0, 1.0)
    cells.observe("a", "t", "-", 0.0)

    alpha, beta = cells.posterior("a", "t", "-")
    assert abs(alpha - 1.995) < 1e-12
    assert abs(beta - 2.0) < 1e-12
    assert cells.n("a", "t", "-") == 2


def test_quality_outside_zero_one_is_rejected() -> None:
    with pytest.raises(ValueError, match=r"quality must be within \[0, 1\]"):
        CellStore().observe("a", "t", "-", 1.5)


def test_cells_are_bounded_and_the_eviction_is_counted() -> None:
    cells = CellStore(max_cells=2)
    for arm in ("a", "b", "c"):
        cells.observe(arm, "t", "-", 1.0)

    assert len(cells) == 2
    assert cells.evicted == 1
    assert cells.n("a", "t", "-") == 0


def test_cell_state_at_the_cap_is_far_below_one_mebibyte() -> None:
    cells = CellStore()
    for i in range(cr.MAX_CELLS + 10):
        cells.observe(f"gateway-deployment-{i:05d}", "task-class", "tier", 0.5)

    assert len(cells) == cr.MAX_CELLS
    assert cells.evicted == 10
    assert cells.state_bytes() < 1024 * 1024 // 2


# --- recorder ---

def test_recorder_drops_the_oldest_row_when_full_and_counts_it() -> None:
    recorder = ShadowRecorder()
    for i in range(2005):
        recorder.record({"i": i})

    rows = recorder.rows()
    assert (len(rows), recorder.dropped, recorder.recorded) == (2000, 5, 2005)
    assert (rows[0], rows[-1]) == ({"i": 5}, {"i": 2004})


def test_recorder_appends_json_lines_and_empties_the_queue(tmp_path) -> None:
    target = tmp_path / "learn" / "shadow_rows.jsonl"
    recorder = ShadowRecorder(path=str(target))
    recorder.record({"i": 1, "amount": Decimal("1.50")})
    recorder.record({"i": 2})

    assert recorder.flush() == 2
    assert [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()] == [
        {"i": 1, "amount": "1.50"}, {"i": 2},
    ]
    assert (recorder.pending(), recorder.written, recorder.write_errors) == (0, 2, 0)


def test_a_failed_write_keeps_the_rows_queued_and_is_counted(tmp_path) -> None:
    recorder = ShadowRecorder(path=str(tmp_path))  # a directory: appending to it fails
    recorder.record({"i": 1})

    assert recorder.flush() == 0
    assert (recorder.pending(), recorder.write_errors, recorder.written) == (1, 1, 0)
    assert recorder.rows() == [{"i": 1}]


def test_recorder_without_a_path_flushes_nothing_and_keeps_rows() -> None:
    recorder = ShadowRecorder()
    recorder.record({"i": 1})

    assert recorder.flush() == 0
    assert recorder.pending() == 1


# --- request parsing ---

def test_request_is_described_from_metadata_and_headers() -> None:
    data = {
        "model": "arm-b", "litellm_call_id": "call-1",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}],
        "metadata": {"task_class": "summarize", "headers": {"X-Loci-Session": "s-1", "x-loci-delegation": "rw"}},
    }

    parsed = request_from_data(data)

    assert parsed == RouteRequest(
        requested="arm-b", session_id="s-1", task_class="summarize", tier="-", has_tools=False,
        delegated_write=True, cached_prefix=True, first_turn=True, caller="anon", call_id="call-1",
    )


def test_tools_and_later_turns_are_detected_and_plain_requests_are_not() -> None:
    plain = request_from_data({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    later = request_from_data({
        "model": "m", "tools": [{"type": "function"}],
        "messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}],
    })

    assert (plain.has_tools, plain.first_turn, plain.cached_prefix, plain.delegated_write) == (False, True, False, False)
    assert (later.has_tools, later.first_turn) == (True, False)


# --- the callback is inert by default and shadow-only when on ---

def _callback(arms_calls: list, **kw) -> LociColonyRouter:
    def arms():
        arms_calls.append(1)
        return MODEL_LIST

    return LociColonyRouter(make_router(arms_fn=arms, **kw))


def _data() -> dict:
    return {
        "model": "arm-b", "litellm_call_id": "call-1",
        "messages": [{"role": "user", "content": "hello"}],
        "metadata": {"session_id": session(False), "task_class": "unclassified"},
    }


@pytest.mark.asyncio
async def test_by_default_the_callback_does_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_ROUTER_SHADOW", raising=False)
    calls: list = []
    callback = _callback(calls)
    data = _data()
    before = copy.deepcopy(data)

    result = await callback.async_pre_call_hook(None, None, data, "acompletion")
    await callback.async_log_success_event({"litellm_call_id": "call-1"}, None, datetime.now(), datetime.now())

    assert result is None
    assert data == before
    assert calls == []
    assert callback.router.stats()["decisions"] == 0
    assert callback.router.recorder.rows() == []


@pytest.mark.asyncio
async def test_shadow_logs_the_counterfactual_and_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ROUTER_SHADOW", "1")
    calls: list = []
    callback = _callback(calls)
    train(callback.router, "arm-c", 0.95)
    data = _data()
    before = copy.deepcopy(data)

    result = await callback.async_pre_call_hook(None, None, data, "acompletion")

    assert result is None
    assert data == before  # the routed model, messages and metadata are untouched
    assert calls == [1]
    rows = callback.router.recorder.rows()
    assert len(rows) == 1
    row = rows[0]
    assert (row["kind"], row["requested"], row["chosen_group"], row["would_change"]) == ("decision", "arm-b", "arm-c", True)
    assert row["pre"]["arm-c"] == {"trail": 0.0, "alarm": 0.0, "resource": 1.0}
    assert row["acted"] is False


@pytest.mark.asyncio
async def test_outcome_rows_carry_pre_and_post_action_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ROUTER_SHADOW", "1")
    callback = _callback([])
    await callback.async_pre_call_hook(None, None, _data(), "acompletion")
    decision_id = callback.router.recorder.rows()[0]["decision_id"]
    started = datetime(2026, 1, 1, 12, 0, 0)
    failure = {
        "litellm_call_id": "call-1", "exception": TimeoutError("timed out"),
        "litellm_params": {"metadata": {"model_group": "arm-b"}},
    }

    await callback.async_log_failure_event(failure, None, started, started + timedelta(seconds=2))

    outcome = callback.router.recorder.rows()[-1]
    assert outcome["kind"] == "outcome"
    assert (outcome["decision_id"], outcome["arm"], outcome["ok"], outcome["transient_failure"]) == (decision_id, "arm-b", False, True)
    assert outcome["latency_s"] == 2.0
    assert outcome["pre"] == {"trail": 0.0, "alarm": 0.0, "resource": 1.0}
    assert outcome["post"] == {"trail": 0.0, "alarm": 0.18, "resource": 1.0}


@pytest.mark.asyncio
async def test_a_permanent_failure_leaves_the_post_features_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ROUTER_SHADOW", "1")
    callback = _callback([])
    await callback.async_pre_call_hook(None, None, _data(), "acompletion")
    denied = type("AuthError", (Exception,), {"status_code": 401})("bad key")
    failure = {"litellm_call_id": "call-1", "exception": denied, "litellm_params": {"metadata": {"model_group": "arm-b"}}}

    await callback.async_log_failure_event(failure, None, datetime.now(), datetime.now())

    outcome = callback.router.recorder.rows()[-1]
    assert (outcome["ok"], outcome["transient_failure"], outcome["status"]) == (False, False, None)
    assert outcome["post"] == {"trail": 0.0, "alarm": 0.0, "resource": 1.0}


@pytest.mark.asyncio
async def test_an_outcome_with_no_matching_decision_is_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ROUTER_SHADOW", "1")
    callback = _callback([])
    callback.router.decide(req())  # make the arms known

    await callback.async_log_success_event(
        {"litellm_call_id": "never-seen", "litellm_params": {"metadata": {"model_group": "arm-a"}}},
        None, datetime.now(), datetime.now(),
    )

    stats = callback.router.stats()
    assert (stats["outcomes"], stats["unmatched_outcomes"]) == (1, 1)
    assert callback.router.recorder.rows()[-1]["post"] == {"trail": 0.1, "alarm": 0.0, "resource": 1.0}


@pytest.mark.asyncio
async def test_outcomes_are_flushed_to_the_learn_dir_after_the_call(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("LOCI_ROUTER_SHADOW", "1")
    monkeypatch.setenv("LOCI_ROUTER_LEARN_DIR", str(tmp_path / "learn"))
    monkeypatch.setattr(cr, "proxy_model_list", lambda: MODEL_LIST)
    callback = LociColonyRouter()
    callback.router.arms_fn = lambda: MODEL_LIST

    await callback.async_pre_call_hook(None, None, _data(), "acompletion")
    assert not (tmp_path / "learn").exists()  # nothing is written during the request
    await callback.async_log_success_event(
        {"litellm_call_id": "call-1", "litellm_params": {"metadata": {"model_group": "arm-b"}}},
        None, datetime.now(), datetime.now(),
    )

    lines = (tmp_path / "learn" / "shadow_rows.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["decision", "outcome"]


@pytest.mark.asyncio
async def test_a_failure_inside_the_decision_is_counted_and_the_request_is_unaffected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_ROUTER_SHADOW", "1")
    callback = _callback([])
    broken = {"model": "arm-b", "messages": 5}

    result = await callback.async_pre_call_hook(None, None, broken, "acompletion")

    assert result is None
    assert callback.router.stats()["errors"] == 1
    await callback.async_pre_call_hook(None, None, _data(), "acompletion")
    assert callback.router.stats()["errors"] == 1
    assert callback.router.stats()["decisions"] == 1


def test_an_arms_provider_that_raises_is_reported_as_unavailable() -> None:
    def boom():
        raise RuntimeError("proxy not ready")

    router = make_router(arms_fn=boom)

    assert router.decide(req()).reason == "no_arms"
    assert router.stats()["arms_unavailable"] == 1


# --- budgets ---

def test_a_decision_without_the_thompson_row_takes_well_under_3_ms() -> None:
    many = [deployment(f"arm-{i:02d}", f"prov-{i % 5}") for i in range(8)]
    router = make_router(arms_fn=lambda: many)

    times = [router.decide(req(requested="arm-03", session_id=session(False, i))).decision_us for i in range(200)]

    assert statistics.median(times) < 3000


def test_router_state_stays_under_one_mebibyte_with_every_table_full() -> None:
    router = make_router()
    for i in range(cr.MAX_CELLS + 5):
        router.cells.observe(f"gateway-deployment-{i:05d}", "task-class", "tier", 0.5)
    for i in range(300):
        router.board.observe_success(f"gateway-deployment-{i:05d}", float(i))

    assert router.stats()["state_bytes"] < 1024 * 1024
    assert router.stats()["cells_evicted"] == 5
    assert router.stats()["arms_evicted"] == 300 - 256


def test_caps_default_to_the_decided_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    caps = Caps()
    assert (caps.daily_hard_cap_usd, caps.provider_share, caps.eps_max, caps.control_share, caps.min_arm_n) == (
        Decimal("50"), Decimal("0.45"), Decimal("0.05"), Decimal("0.10"), 30,
    )
    monkeypatch.setenv("LOCI_ROUTER_DAILY_CAP_USD", "40")
    assert Caps.from_env().provider_limit_usd() == Decimal("18.00")
    monkeypatch.setenv("LOCI_ROUTER_DAILY_CAP_USD", "not-a-number")
    assert Caps.from_env().daily_hard_cap_usd == Decimal("50")
