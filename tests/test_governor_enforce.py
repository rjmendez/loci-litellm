"""Governor enforce: 429 bodies, downgrade targets, per-client caps, prompt budget (G3), high-cost share (G4),
concurrency (G7). Enforcement is exercised only with a trusted envelope that enables it."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from loci_litellm.client_keys import CLIENT_HEADER, LEASE_HEADER, SECRET_HEADER, identify
from loci_litellm.envelope import ClientEntry, EnvelopeInvalid, parse_envelope
from loci_litellm.governor import (
    Catalog,
    ConcurrencyLimiter,
    ConcurrencyTimeout,
    GovernorRejection,
    Limits,
    LociGovernor,
    Request,
    Snapshot,
    SpendCounter,
    decide,
    grounding_tokens,
    trim_grounding,
)
from tests._governor_fixtures import (
    D,
    FakeClock,
    RecordingStore,
    body,
    caps,
    make_governor,
    trusted,
)

NY = ZoneInfo("America/New_York")
SECRET_A, SECRET_B = "secret-A-value", "secret-B-value"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


CLIENTS = {
    "svc_a": ClientEntry("svc_a", sha(SECRET_A), cap_day=D("20")),
    "scripts": ClientEntry("scripts", sha(SECRET_B), cap_day=D("5")),
    "owner": ClientEntry("owner", sha(SECRET_A), cap_day=None, interactive=True),
    "delegate": ClientEntry("delegate", sha(SECRET_B), cap_day=None, rw=True),
}


def entry(name: str, in_per_token: float, tools: bool = True, **info: object) -> dict:
    return {"model_name": name, "model_info": {"id": "id-" + name, "input_cost_per_token": in_per_token,
                                                "output_cost_per_token": 0, "supports_function_calling": tools,
                                                "max_input_tokens": 200_000, "loci_local": False, **info},
            "litellm_params": {"model": "vendor/" + name, "api_base": "https://api.example.test"}}


# Alphabetical order is the reverse of price order, so a ranking by name would be wrong.
PREMIUM, MID, CHEAP = "alpha", "bravo", "charlie"
MODEL_LIST = [entry(PREMIUM, 0.00002), entry(MID, 0.00001), entry(CHEAP, 0.000002)]


def cat(model_list: list | None = None, retries: int = 0) -> Catalog:
    return Catalog.from_model_list(model_list or MODEL_LIST, num_retries=retries)


def as_client(data: dict, name: str, secret: str, **more: str) -> dict:
    data["proxy_server_request"] = {"headers": {"X-Loci-Client": name, "X-Loci-Client-Secret": secret, **more}}
    return data


def snap(**kw: str | int) -> Snapshot:
    return Snapshot(**{k: (v if isinstance(v, int) else D(v)) for k, v in kw.items()})


# --------------------------------------------------------------------------- enforce flip and the 429 body
def test_over_cap_tool_request_gets_429_with_reason_and_the_cap_and_a_tool_free_one_goes_to_the_cheapest_arm(tmp_path: Path) -> None:
    c = caps(day_soft=D("30"), day_hard=D("50"))
    store = RecordingStore(snap=snap(day_total="49.99"))
    g = make_governor(tmp_path, store, c, cat=cat())
    rejection = g.process(body(PREMIUM, tokens=1000, tools=True))
    assert isinstance(rejection, GovernorRejection)
    assert rejection.status_code == 429
    assert rejection.body == {"error": {"type": "governor_rejection", "reason": "budget_day", "cap": "50",
                                        "message": "request refused by the spend governor: budget_day (cap 50)"}}
    events: list = []
    free_store = RecordingStore(snap=snap(day_total="49.99"))
    free = make_governor(tmp_path, free_store, c, cat=cat(), events=events)
    data = body(PREMIUM, tokens=1000)
    assert free.process(data) is None
    assert data["model"] == CHEAP and data["metadata"]["loci_governor_downgraded_from"] == PREMIUM
    assert [(e["event"], e["code"], e["target"], e["target_arm"]) for e in events] == [("downgrade", "budget_day", CHEAP, "id-" + CHEAP)]
    assert free_store.reservations == [(D("0.002"), "anonymous", False)]


def test_the_enforce_flip_needs_the_envelope_entry(tmp_path: Path) -> None:
    c = caps(day_soft=D("30"), day_hard=D("50"))
    for env_enforce, expected in ((False, None), (True, 429)):
        g = make_governor(tmp_path, RecordingStore(snap=snap(day_total="49.99")), c, cat=cat(), env=trusted(c, enforce=env_enforce))
        result = g.process(body(PREMIUM, tokens=1000, tools=True))
        assert (result if result is None else result.status_code) == expected


# --------------------------------------------------------------------------- per-client caps
def spend(tmp_path: Path, clock: FakeClock, **by_client: str) -> SpendCounter:
    counter = SpendCounter(str(tmp_path / "counter.json"), clock=clock, tz=NY)
    for name, amount in by_client.items():
        counter.settle("seed-" + name, D(amount), clock(), client=name)
    return counter


def test_a_client_at_its_cap_is_denied_while_another_under_its_cap_passes(tmp_path: Path) -> None:
    clock = FakeClock()
    counter = spend(tmp_path, clock, svc_a="19.99", scripts="1")
    c = caps(day_soft=D("500"), day_hard=D("500"))
    g = make_governor(tmp_path, counter, c, cat=cat(), env=trusted(c, clients=CLIENTS), clock=clock)
    denied = g.process(as_client(body(PREMIUM, tokens=1000, tools=True), "svc_a", SECRET_A))
    assert isinstance(denied, GovernorRejection)
    assert (denied.status_code, denied.code, denied.body["error"]["cap"]) == (429, "budget_client", "20")
    assert g.process(as_client(body(PREMIUM, tokens=1000, tools=True), "scripts", SECRET_B)) is None
    second = tmp_path / "second"
    second.mkdir()
    g2 = make_governor(second, spend(second, clock, svc_a="19.00"), c, cat=cat(), env=trusted(c, clients=CLIENTS), clock=clock)
    assert g2.process(as_client(body(PREMIUM, tokens=1000, tools=True), "svc_a", SECRET_A)) is None
    assert g.stats().get("client_unknown", 0) == 0


@pytest.mark.asyncio
async def test_a_wrong_secret_is_the_anonymous_bucket_and_the_right_secret_is_the_client(tmp_path: Path) -> None:
    clock = FakeClock()
    counter = spend(tmp_path, clock)
    events: list = []
    c = caps()
    g = make_governor(tmp_path, counter, c, cat=cat(), env=trusted(c, clients=CLIENTS), clock=clock, events=events)
    cb = LociGovernor(g)
    for secret, expected in ((SECRET_B, "anonymous"), ("", "anonymous"), (SECRET_A, "svc_a")):
        data = as_client(body(PREMIUM, tokens=1000), "svc_a", secret)
        await cb.async_pre_call_hook(None, None, data, "acompletion")
        assert data["metadata"]["loci_client"] == expected
        kwargs = {"litellm_params": {"metadata": data["metadata"]}, "standard_logging_object": {"response_cost": 0.5}}
        await cb.async_log_success_event(kwargs, None, None, None)
    now = clock()
    assert counter.snapshot(now, "anonymous").client_day == D("1.0")        # two unproved calls
    assert counter.snapshot(now, "svc_a").client_day == D("0.5")
    assert g.stats()["client_unknown"] == 2
    assert [e["client"] for e in events if e["event"] == "client_unknown"] == ["anonymous", "anonymous"]
    assert SECRET_A not in json.dumps(events) and SECRET_B not in json.dumps(events)


def test_the_anonymous_bucket_has_its_own_cap_and_cannot_claim_privileges(tmp_path: Path) -> None:
    clock = FakeClock()
    counter = spend(tmp_path, clock, anonymous="9.99")
    c = caps()
    g = make_governor(tmp_path, counter, c, cat=cat(), env=trusted(c, clients=CLIENTS), clock=clock)
    denied = g.process(as_client(body(PREMIUM, tokens=1000, tools=True), "owner", "wrong"))
    assert isinstance(denied, GovernorRejection)
    assert (denied.code, denied.body["error"]["cap"]) == ("budget_client", "10")
    assert isinstance(g.process(body(PREMIUM, tokens=1000, tools=True)), GovernorRejection)   # no client header: same bucket


def test_identify_decides_by_the_secret_only() -> None:
    ok = identify({CLIENT_HEADER: "svc_a", SECRET_HEADER: SECRET_A}, CLIENTS, D("10"))
    assert (ok.name, ok.known, ok.interactive, ok.rw, ok.cap_day, ok.notes) == ("svc_a", True, False, False, D("20"), ())
    delegate = identify({CLIENT_HEADER: "delegate", SECRET_HEADER: SECRET_B}, CLIENTS, D("10"))
    assert (delegate.name, delegate.rw, delegate.cap_day) == ("delegate", True, None)
    for headers in ({CLIENT_HEADER: "svc_a", SECRET_HEADER: SECRET_B}, {CLIENT_HEADER: "svc_a"},
                    {CLIENT_HEADER: "nobody", SECRET_HEADER: SECRET_A}, {CLIENT_HEADER: "anonymous", SECRET_HEADER: SECRET_A},
                    {CLIENT_HEADER: "svc_a", SECRET_HEADER: 5}):
        who = identify(headers, CLIENTS, D("10"))
        assert (who.name, who.known, who.interactive, who.rw, who.cap_day, who.notes) == ("anonymous", False, False, False, D("10"), ("client_unknown",))
    none = identify({}, CLIENTS, D("10"))
    assert (none.name, none.notes) == ("anonymous", ())


@pytest.mark.parametrize("spent, status", [("19.98", None), ("19.99", 429)])
def test_client_cap_boundary_is_exact(tmp_path: Path, spent: str, status: int | None) -> None:
    clock = FakeClock()
    c = caps(day_soft=D("500"), day_hard=D("500"))
    g = make_governor(tmp_path, spend(tmp_path, clock, svc_a=spent), c, cat=cat(), env=trusted(c, clients=CLIENTS), clock=clock)
    result = g.process(as_client(body(PREMIUM, tokens=1000, tools=True), "svc_a", SECRET_A))       # reserves 0.02 against a cap of 20
    assert (result if result is None else result.status_code) == status


def test_a_client_whose_pinned_secret_is_the_hash_of_nothing_still_needs_a_secret() -> None:
    blank = {"blank": ClientEntry("blank", sha(""))}
    for headers in ({CLIENT_HEADER: "blank"}, {CLIENT_HEADER: "blank", SECRET_HEADER: ""}, {CLIENT_HEADER: "blank", SECRET_HEADER: "  "}):
        who = identify(headers, blank, D("10"))
        assert (who.name, who.known, who.notes) == ("anonymous", False, ("client_unknown",))
    assert identify({CLIENT_HEADER: "blank", SECRET_HEADER: "x"}, {"blank": ClientEntry("blank", sha("x"))}, D("10")).name == "blank"


def test_a_client_cannot_be_named_anonymous_in_the_envelope() -> None:
    with pytest.raises(EnvelopeInvalid, match="reserved for unidentified callers"):
        parse_envelope(('[clients.anonymous]\nsecret_sha256 = "%s"\n' % ("a" * 64)).encode("utf-8"))


# --------------------------------------------------------------------------- interactive callers
def test_an_interactive_turn_is_downgraded_at_the_hourly_trigger_and_an_impersonator_is_rejected(tmp_path: Path) -> None:
    clock = FakeClock()
    c = caps(hour_downgrade=D("10"), hour_reject=D("25"))
    env = trusted(c, clients=CLIENTS)
    owner_counter = spend(tmp_path, clock, owner="24.99")
    g = make_governor(tmp_path, owner_counter, c, cat=cat(), env=env, clock=clock)
    data = as_client(body(PREMIUM, tokens=1000, tools=True), "owner", SECRET_A)
    assert g.process(data) is None
    assert (data["model"], data["metadata"]["loci_client"]) == (CHEAP, "owner")
    other = tmp_path / "other"
    other.mkdir()
    imp_counter = spend(other, clock, owner="24.99")
    g2 = make_governor(other, imp_counter, c, cat=cat(), env=env, clock=clock)
    fake = g2.process(as_client(body(PREMIUM, tokens=1000, tools=True), "owner", "not-the-secret"))
    assert isinstance(fake, GovernorRejection) and (fake.status_code, fake.code) == (429, "budget_hour")


def test_the_interactive_absolute_stop_is_a_429(tmp_path: Path) -> None:
    clock = FakeClock()
    c = caps(day_soft=D("30"), day_hard=D("50"), interactive_stop=D("75"))
    counter = spend(tmp_path, clock, owner="74.99")
    g = make_governor(tmp_path, counter, c, cat=cat(), env=trusted(c, clients=CLIENTS), clock=clock)
    stopped = g.process(as_client(body(PREMIUM, tokens=1000, tools=True), "owner", SECRET_A))
    assert isinstance(stopped, GovernorRejection)
    assert (stopped.status_code, stopped.code, stopped.body["error"]["cap"]) == (429, "budget_abs_stop", "75")


# --------------------------------------------------------------------------- G3: prompt budget
P = caps(prompt_tokens_max=60000)


def request(tokens: int, model: str = PREMIUM, **kw: object) -> Request:
    return Request(model=model, prompt_tokens=tokens, max_tokens=1, **kw)  # type: ignore[arg-type]


def test_prompt_at_the_limit_passes_and_one_token_over_is_trimmed_then_downgraded_then_rejected() -> None:
    clean = decide(request(60_000), Snapshot(), P, cat())
    assert (clean.action, clean.code, clean.actions, clean.trim_tokens, clean.target_model) == ("allow", "", (), 0, PREMIUM)
    at_limit_with_context = decide(request(60_000, trimmable_tokens=5), Snapshot(), P, cat())
    assert (at_limit_with_context.actions, at_limit_with_context.trim_tokens, at_limit_with_context.code) == ((), 0, "")
    lands_on_limit = decide(request(60_001, trimmable_tokens=1), Snapshot(), P, cat())            # trimmed to exactly 60,000
    assert (lands_on_limit.action, lands_on_limit.code, lands_on_limit.actions, lands_on_limit.target_model) ==         ("allow", "prompt_trim", ("trim_grounding",), PREMIUM)
    trimmed = decide(request(60_001, trimmable_tokens=2), Snapshot(), P, cat())
    assert (trimmed.action, trimmed.code, trimmed.actions, trimmed.trim_tokens, trimmed.target_model) == \
        ("allow", "prompt_trim", ("trim_grounding",), 2, PREMIUM)
    assert trimmed.reservation == D("1.199980")                                  # 59,999 tokens at $20 per million
    both = decide(request(60_002, trimmable_tokens=1), Snapshot(), P, cat())
    assert (both.action, both.code, both.actions, both.trim_tokens, both.target_model, both.target_arm) == \
        ("downgrade", "prompt_budget", ("trim_grounding", "downgrade_prompt_budget"), 1, CHEAP, "id-" + CHEAP)
    down = decide(request(60_001), Snapshot(), P, cat())
    assert (down.action, down.code, down.actions, down.target_model) == ("downgrade", "prompt_budget", ("downgrade_prompt_budget",), CHEAP)
    reject = decide(request(60_001, model=CHEAP), Snapshot(), P, cat())
    assert (reject.action, reject.code, reject.status, reject.cap, reject.reservation) == ("reject", "prompt_budget", 413, D("60000"), D("0"))


def test_a_large_context_lease_and_a_local_arm_are_not_trimmed() -> None:
    leased = decide(request(150_000, lease_ok=True), Snapshot(), P, cat())
    assert (leased.action, leased.code, leased.actions, leased.target_model) == ("allow", "", (), PREMIUM)
    local = cat(MODEL_LIST + [entry("delta", 0, loci_local=True)])
    free = decide(request(150_000, model="delta"), Snapshot(), P, local)
    assert (free.action, free.code, free.actions, free.reservation) == ("allow", "", (), D("0"))


def grounded(tokens_user: int) -> dict:
    data = body(PREMIUM, tokens=tokens_user)
    data["messages"].insert(0, {"role": "system", "content": "MEMORY MATCH (3 results from notes) " + "g" * 277})
    return data


def test_governor_trims_the_injected_grounding_first_when_enforcing_and_only_logs_when_observing(tmp_path: Path) -> None:
    c = caps(prompt_tokens_max=60000)
    events: list = []
    enforcing = make_governor(tmp_path, RecordingStore(), c, cat=cat(), events=events)
    data = grounded(59_950)
    assert grounding_tokens(data) == 110
    assert enforcing.process(data) is None
    assert [m["role"] for m in data["messages"]] == ["user"] and data["model"] == PREMIUM
    assert data["metadata"]["loci_governor_trimmed"] == 1
    assert [(e["event"], e["code"], e["actions"], e["trim_tokens"]) for e in events] == [("adjust", "prompt_trim", ["trim_grounding"], 110)]
    watching_events: list = []
    watching = make_governor(tmp_path, RecordingStore(), c, cat=cat(), mode="observe", events=watching_events)
    untouched = grounded(59_950)
    assert watching.process(untouched) is None
    assert len(untouched["messages"]) == 2 and untouched["model"] == PREMIUM
    assert [(e["event"], e["code"], e["actions"]) for e in watching_events] == [("would_block", "prompt_trim", ["trim_grounding"])]


def test_trim_removes_only_the_injected_context() -> None:
    data = {"messages": [{"role": "system", "content": "MEMORY MATCH (1 results from x) a"}, {"role": "system", "content": "be brief"},
                         {"role": "user", "content": "MEMORY MATCH (this is the user's own text)"}]}
    assert trim_grounding(data) == 1
    assert [m["content"] for m in data["messages"]] == ["be brief", "MEMORY MATCH (this is the user's own text)"]
    assert trim_grounding(data) == 0


def test_a_lease_header_is_checked_by_the_validator_for_that_client(tmp_path: Path) -> None:
    seen: list = []

    def validator(client: str, token: str) -> bool:
        seen.append((client, token))
        return token == "lease-1"

    c = caps(prompt_tokens_max=60000)
    clients = {"svc_a": CLIENTS["svc_a"]}
    env = trusted(c, clients=clients)
    g = make_governor(tmp_path, RecordingStore(), c, cat=cat(), env=env, lease_validator=validator)
    ok = as_client(body(PREMIUM, tokens=70_000), "svc_a", SECRET_A, **{LEASE_HEADER: "lease-1"})
    assert g.process(ok) is None and ok["model"] == PREMIUM
    bad = as_client(body(PREMIUM, tokens=70_000), "svc_a", SECRET_A, **{LEASE_HEADER: "lease-2"})
    assert g.process(bad) is None and bad["model"] == CHEAP
    assert seen == [("svc_a", "lease-1"), ("svc_a", "lease-2")]
    no_validator = make_governor(tmp_path, RecordingStore(), c, cat=cat(), env=env)
    claimed = as_client(body(PREMIUM, tokens=70_000), "svc_a", SECRET_A, **{LEASE_HEADER: "lease-1"})
    assert no_validator.process(claimed) is None and claimed["model"] == CHEAP


# --------------------------------------------------------------------------- G4: high-cost share
def test_arm_rank_comes_from_prices_and_not_from_names() -> None:
    renamed = [entry("zulu", 0.00002), entry("yankee", 0.00001), entry("xray", 0.000002)]

    def by_arm(catalog: Catalog) -> dict:
        return {catalog.groups()[name][0].arm_id: rank for name, rank in catalog.price_ranks().items()}

    assert by_arm(cat()) == {"id-alpha": 2, "id-bravo": 1, "id-charlie": 0}
    assert by_arm(cat(renamed)) == {"id-zulu": 2, "id-yankee": 1, "id-xray": 0}
    for model_list, top, mid in ((MODEL_LIST, "alpha", "id-bravo"), (renamed, "zulu", "id-yankee")):
        out = decide(request(1000, model=top), snap(hc_calls_hour=20), caps(), cat(model_list))
        assert (out.action, out.code, out.target_arm, out.reservation, out.explore_allowed) == \
            ("downgrade", "high_cost_rate", mid, D("0.010000"), False)
    assert cat().is_high_cost("alpha") and not cat().is_high_cost("bravo") and not cat().is_high_cost("charlie")
    assert not cat([entry("only", 0.00002)]).is_high_cost("only")                  # a single rank is not "high"


def test_the_hourly_call_line_and_the_daily_share_line_move_a_high_cost_request_one_rank_down() -> None:
    c = caps(day_hard=D("50"), high_cost_share=D("0.30"), high_cost_calls_per_hour=20)
    below = decide(request(1000), snap(hc_calls_hour=19), c, cat())
    assert (below.action, below.code, below.target_model, below.high_cost) == ("allow", "", PREMIUM, True)
    at_line = decide(request(1000), snap(hc_calls_hour=20), c, cat())
    assert (at_line.action, at_line.code, at_line.target_model, at_line.high_cost) == ("downgrade", "high_cost_rate", MID, False)
    share_ok = decide(request(1000), snap(hc_day="14.98"), c, cat())          # 14.98 + 0.02 = 15.00, not over
    assert (share_ok.action, share_ok.target_model) == ("allow", PREMIUM)
    share_over = decide(request(1000), snap(hc_day="14.99"), c, cat())
    assert (share_over.action, share_over.code, share_over.target_model) == ("downgrade", "high_cost_share", MID)
    off_premium = decide(request(1000, model=MID), snap(hc_calls_hour=99, hc_day="99"), c, cat())
    assert (off_premium.action, off_premium.target_model) == ("allow", MID)


def test_with_no_adequate_lower_arm_the_request_stays_put_and_is_flagged_not_dropped() -> None:
    no_tools = [entry(PREMIUM, 0.00002), entry(MID, 0.00001, tools=False), entry(CHEAP, 0.000002, tools=False)]
    out = decide(request(1000, has_tools=True), snap(hc_calls_hour=50), caps(), cat(no_tools))
    assert (out.action, out.target_model, out.notes, out.high_cost) == ("allow", PREMIUM, ("high_cost_no_alternative",), True)
    only_cheapest_fits = [entry(PREMIUM, 0.00002), entry(MID, 0.00001, tools=False), entry(CHEAP, 0.000002)]
    skip = decide(request(1000, has_tools=True), snap(hc_calls_hour=50), caps(), cat(only_cheapest_fits))
    assert (skip.action, skip.target_model, skip.target_arm) == ("downgrade", CHEAP, "id-" + CHEAP)


def test_governor_books_high_cost_calls_for_the_share_counters(tmp_path: Path) -> None:
    store = RecordingStore()
    g = make_governor(tmp_path, store, caps(), cat=cat())
    g.process(body(PREMIUM, tokens=1000))
    g.process(body(CHEAP, tokens=1000))
    assert [hc for _, _, hc in store.reservations] == [True, False]


def test_counter_keeps_high_cost_spend_and_hourly_calls(tmp_path: Path) -> None:
    clock = FakeClock()
    counter = SpendCounter(str(tmp_path / "c.json"), clock=clock, tz=NY)
    counter.reserve("a", D("1"), clock(), hc=True)
    counter.reserve("b", D("1"), clock(), hc=False)
    counter.settle("a", D("0.75"), clock(), final=True)
    counter.settle("b", D("0.25"), clock(), final=True)
    shot = counter.snapshot(clock())
    assert (shot.hc_day, shot.hc_calls_hour, shot.day_total) == (D("0.75"), 1, D("1.00"))
    clock.advance(3600)
    assert counter.snapshot(clock()).hc_calls_hour == 0


# --------------------------------------------------------------------------- G7: concurrency
class RecordingSemaphore(asyncio.Semaphore):
    made: list = []

    def __init__(self, value: int) -> None:
        super().__init__(value)
        self.limit = value
        self.acquired = 0
        self.released = 0
        RecordingSemaphore.made.append(self)

    async def acquire(self) -> bool:
        result = await super().acquire()
        self.acquired += 1
        return result

    def release(self) -> None:
        self.released += 1
        super().release()


def test_default_limits_are_8_per_arm_2_per_card_and_12_global() -> None:
    assert (Limits().arm, Limits().card, Limits().global_) == (8, 2, 12)
    limiter = ConcurrencyLimiter()
    assert (limiter.limit_for("arm:x"), limiter.limit_for("card:g0"), limiter.limit_for("global"), limiter.wait_s) == (8, 2, 12, 30.0)


@pytest.mark.asyncio
async def test_the_thirteenth_concurrent_call_waits_then_times_out_and_a_release_lets_the_next_in() -> None:
    limiter = ConcurrencyLimiter(Limits(), wait_s=0.05)
    for i in range(12):
        await limiter.acquire(f"c{i}", ["global", f"arm:m{i}"])
    assert limiter.in_use("global") == 12
    with pytest.raises(ConcurrencyTimeout, match="global"):
        await limiter.acquire("c12", ["global", "arm:m12"])
    assert (limiter.in_use("global"), limiter.in_use("arm:m12"), limiter.timeouts) == (12, 0, 1)
    assert limiter.release("c0") is True and limiter.release("c0") is False
    await limiter.acquire("c12", ["global", "arm:m12"])                      # positive twin
    assert limiter.in_use("global") == 12


@pytest.mark.asyncio
async def test_a_failed_second_acquire_gives_back_the_first_and_cancellation_does_too() -> None:
    RecordingSemaphore.made = []
    limiter = ConcurrencyLimiter(Limits(global_=2, arm=1, card=1), wait_s=0.05, semaphore_factory=RecordingSemaphore)
    await limiter.acquire("first", ["global", "arm:x"])
    with pytest.raises(ConcurrencyTimeout, match="arm:x"):
        await limiter.acquire("second", ["global", "arm:x"])
    glob = next(s for s in RecordingSemaphore.made if s.limit == 2)
    assert (glob.acquired, glob.released, limiter.in_use("global"), limiter.in_use("arm:x")) == (2, 1, 1, 1)
    slow = ConcurrencyLimiter(Limits(global_=2, arm=1, card=1), wait_s=30.0)
    await slow.acquire("holder", ["global", "arm:x"])
    task = asyncio.create_task(slow.acquire("waiter", ["global", "arm:x"]))
    await asyncio.sleep(0.02)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (slow.in_use("global"), slow.in_use("arm:x")) == (1, 1)
    assert slow.release("holder") is True and (slow.in_use("global"), slow.in_use("arm:x")) == (0, 0)


@pytest.mark.asyncio
async def test_a_slot_nobody_released_is_reclaimed_after_its_ttl() -> None:
    clock = FakeClock()
    limiter = ConcurrencyLimiter(Limits(global_=1), wait_s=0.05, clock=clock, slot_ttl_s=1800.0)
    await limiter.acquire("lost", ["global"])
    clock.advance(1800)
    with pytest.raises(ConcurrencyTimeout, match="global"):
        await limiter.acquire("next", ["global"])
    clock.advance(0.5)
    await limiter.acquire("next", ["global"])
    assert (limiter.swept, limiter.in_use("global")) == (1, 1)


@pytest.mark.asyncio
async def test_ninth_call_to_one_cloud_alias_gets_503_retry_after_and_its_reservation_is_released(tmp_path: Path) -> None:
    store = RecordingStore()
    limiter = ConcurrencyLimiter(Limits(), wait_s=0.05)
    g = make_governor(tmp_path, store, caps(), cat=cat(), limiter=limiter)
    ids = []
    for _ in range(8):
        data = body(PREMIUM, tokens=100)
        assert await g.admit(data) is None
        ids.append(data["metadata"]["loci_governor_id"])
    ninth = body(PREMIUM, tokens=100)
    rejection = await g.admit(ninth)
    assert isinstance(rejection, GovernorRejection)
    assert (rejection.status_code, rejection.code, rejection.headers) == (503, "concurrency", {"Retry-After": "5"})
    assert store.releases == [ninth["metadata"]["loci_governor_id"]] and store.denials == ["concurrency"]
    assert limiter.in_use("arm:" + PREMIUM) == 8
    g.settle(ids[0], D("0.001"), final=True)
    assert await g.admit(body(PREMIUM, tokens=100)) is None
    assert await g.admit(body(MID, tokens=100)) is None                       # another alias has its own slots


@pytest.mark.asyncio
async def test_local_arms_are_limited_per_card(tmp_path: Path) -> None:
    local = cat(MODEL_LIST + [entry("lone", 0, loci_local=True, loci_card="gpu0"), entry("pair", 0, loci_local=True, loci_card="gpu0"),
                              entry("solo", 0, loci_local=True, loci_card="gpu1")])
    limiter = ConcurrencyLimiter(Limits(), wait_s=0.05)
    g = make_governor(tmp_path, RecordingStore(), caps(), cat=local, limiter=limiter)
    assert await g.admit(body("lone", tokens=10)) is None
    assert await g.admit(body("pair", tokens=10)) is None
    third = await g.admit(body("lone", tokens=10))
    assert isinstance(third, GovernorRejection) and third.code == "concurrency"
    assert await g.admit(body("solo", tokens=10)) is None                     # a different card
    assert limiter.in_use("card:gpu0") == 2 and limiter.in_use("card:gpu1") == 1


@pytest.mark.asyncio
async def test_the_slot_is_released_when_the_call_succeeds_fails_or_the_gate_errors(tmp_path: Path) -> None:
    limiter = ConcurrencyLimiter(Limits(global_=3), wait_s=0.05)
    store = RecordingStore()
    g = make_governor(tmp_path, store, caps(), cat=cat(), limiter=limiter)
    cb = LociGovernor(g)
    ok, failed = body(PREMIUM, tokens=100), body(PREMIUM, tokens=100)
    await cb.async_pre_call_hook(None, None, ok, "acompletion")
    await cb.async_pre_call_hook(None, None, failed, "acompletion")
    assert limiter.in_use("global") == 2
    kwargs = {"litellm_params": {"metadata": ok["metadata"]}, "standard_logging_object": {"response_cost": 0.001}}
    await cb.async_log_success_event(kwargs, None, None, None)
    assert limiter.in_use("global") == 1
    attempt_only = {"litellm_params": {"metadata": failed["metadata"]}, "standard_logging_object": {"response_cost": 0}}
    await cb.async_log_failure_event(attempt_only, None, None, None)
    assert limiter.in_use("global") == 1                                      # one attempt failed; the call is still going
    await cb.async_post_call_failure_hook(failed, RuntimeError("upstream"), None)
    assert (limiter.in_use("global"), limiter.in_use("arm:" + PREMIUM), len(store.releases)) == (0, 0, 1)
    broken_store = RecordingStore()
    broken = make_governor(tmp_path, broken_store, caps(), cat=cat(), limiter=limiter)
    reads: list = []

    def flaky_catalog() -> Catalog:
        reads.append(1)
        if len(reads) > 1:                    # fine for the spend decision, gone by the time the slot is chosen
            raise KeyError("catalog unavailable")
        return cat()

    broken._catalog = flaky_catalog
    data = body(PREMIUM, tokens=100)
    result = await broken.admit(data)
    assert isinstance(result, GovernorRejection) and (result.status_code, result.code) == (503, "budget_gate_error")
    assert broken_store.releases == [data["metadata"]["loci_governor_id"]]
    assert limiter.in_use("global") == 0


@pytest.mark.asyncio
async def test_observe_mode_records_a_full_limiter_without_taking_or_waiting_for_slots(tmp_path: Path) -> None:
    limiter = ConcurrencyLimiter(Limits(global_=1), wait_s=0.05)
    await limiter.acquire("held", ["global"])
    events: list = []
    g = make_governor(tmp_path, RecordingStore(), caps(), cat=cat(), limiter=limiter, mode="observe", events=events)
    assert await g.admit(body(PREMIUM, tokens=100)) is None
    assert [(e["event"], e["code"], e["mode"]) for e in events] == [("would_block", "concurrency", "observe")]
    assert limiter.in_use("global") == 1


# --------------------------------------------------------------------------- observe stays inert for the enforce rules
def test_observe_mode_leaves_oversize_and_high_cost_requests_alone(tmp_path: Path) -> None:
    events: list = []
    g = make_governor(tmp_path, RecordingStore(snap=snap(hc_calls_hour=99)), caps(prompt_tokens_max=60000), cat=cat(),
                      mode="", events=events)
    data = grounded(70_000)
    before = json.dumps(data, sort_keys=True)
    assert g.process(data) is None
    assert json.dumps({k: v for k, v in data.items() if k != "metadata"}, sort_keys=True) == before
    assert [(e["event"], e["code"]) for e in events] == [("would_block", "prompt_budget")]


# --------------------------------------------------------------------------- virtual keys
def test_virtual_keys_need_a_database_in_the_installed_proxy_so_caps_live_in_the_governor() -> None:
    spec = importlib.util.find_spec("litellm.proxy.auth.user_api_key_auth")
    assert spec is not None and spec.origin is not None
    source = Path(spec.origin).read_text(encoding="utf-8")
    after = source[source.index("## Check DB"):][:700]
    assert "prisma_client is None" in after
    assert 'message="No connected db."' in after
    assert "ProxyErrorTypes.no_db_connection" in after
    assert "code=400" in after
