"""Provider router in SHADOW: it logs which arm it WOULD pick and never changes the routed model.

Opt in with LOCI_ROUTER_SHADOW=1 (default off: the callback does nothing, reads nothing, records nothing).
Register in the LiteLLM config:
    litellm_settings:
      callbacks:
        - loci_litellm.colony_router.loci_colony_router

What it does per request (in memory, well under 3 ms without the Thompson shadow row):
  - Enumerates arms at runtime from the router's model list (name, provider, price and capabilities come
    from `model_info`; no provider or model name is hardcoded). Seat-billed providers, entries with
    `model_info.seat_billed: true` and entries with `model_info.loci_router_arm: false` are never arms;
    extend the provider list with LOCI_ROUTER_EXCLUDE_PROVIDERS (comma separated).
  - Builds the safe set: drops arms in an open failure quorum, arms of a provider whose spend today has
    reached the per-provider share cap (default 45% of the daily cap), and arms without tool support for
    a request that carries tools.
  - Epsilon-greedy over the posterior-mean argmax. The logged propensity of the chosen arm is exact:
    P(a) = (1 - eps) * 1[a = a*] + eps / |S|. Epsilon is 0 when the governor level is >= 1, the PAUSE file
    exists, the resource scalar is < 0.35, or the request carries tools, a cached prefix, a delegated
    write, or is not a first turn.
  - A Thompson-sampling row beside it, with a Monte-Carlo propensity from >= 200 draws. Thompson is only
    ever a shadow row: its closed form is not valid, so it never drives a choice.
  - A control slice: a stable hash of the session id puts exactly CONTROL_SHARE of sessions on the
    incumbent (the model the request names). Never per call. A request with no session id is control.
  - Rows go to a bounded recorder (2000, drop-oldest, counted). With LOCI_ROUTER_LEARN_DIR set they are
    appended to <dir>/shadow_rows.jsonl after the call completes. No raw prompt text is recorded.

The only "action" is the log row. See `stats()` for every degraded condition (counted, never silent).

Canary (inert by default). The learned rule routes a request only when ALL of these hold: the router is not rolled
back, LOCI_ROUTER_CANARY_SHARE is 0.1, 0.5 or 0.9 (default 0: never), a valid frozen incumbent snapshot is loaded
(LOCI_ROUTER_SNAPSHOT), the session falls in the learned slice (control is the first 10% of the session hash, the
learned slice the next `share`), the request carries no tools and is not a delegated write, no PAUSE file exists,
the governor level is below 2 and a spend view is available. Epsilon during the canary is 0.03. The frozen
snapshot pins the caps: a snapshot whose caps differ from the pinned envelope is rejected (SnapshotRejected).
A watchdog tick that judges the candidate DEGENERATE against the incumbent slice rolls the router back to the
incumbent; it never resumes by itself (a marker file keeps it rolled back across restarts).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import random
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from fractions import Fraction
from typing import Any, Callable, Mapping, Optional

from litellm.integrations.custom_logger import CustomLogger
from loci_litellm.capability_gate import needs_tools
from loci_litellm.channels import ChannelBoard
from loci_litellm.fileutil import append_line, env_int

logger = logging.getLogger("loci_litellm.colony_router")

MIN_ARM_N = 30
THOMPSON_DRAWS = 200
RECORDER_MAX = 2000
EPS_MAX = 0.05
RESOURCE_FLOOR_FOR_EXPLORATION = 0.35
PRIOR = 1.0
DISCOUNT = 0.995
MAX_CELLS = 2048
ARMS_TTL_S = 60.0
LAGRANGE_INTERVAL_S = 300.0
LAGRANGE_ETA = 0.5
DEFAULT_SALT = "loci-router-v1"
ROWS_FILE = "shadow_rows.jsonl"

# Providers that are billed per seat, not per token: never arms (extend with LOCI_ROUTER_EXCLUDE_PROVIDERS).
SEAT_BILLED_PROVIDERS = frozenset({"github_copilot", "copilot"})
_CHAT_MODES = frozenset({"chat", "completion", "responses"})


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _decimal_env(name: str, default: Decimal) -> Decimal:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = Decimal(raw)
    except InvalidOperation:
        logger.warning("loci colony_router: %s=%r is not a number, using %s", name, raw, default)
        return default
    return value if value.is_finite() else default


@dataclass(frozen=True)
class Caps:
    """The pinned envelope the router runs inside. Money is Decimal."""

    daily_hard_cap_usd: Decimal = Decimal("50")
    provider_share: Decimal = Decimal("0.45")
    eps_max: Decimal = Decimal("0.05")
    control_share: Decimal = Decimal("0.10")
    canary_eps: Decimal = Decimal("0.03")
    lambda_max: Decimal = Decimal("5.0")
    min_arm_n: int = MIN_ARM_N

    @classmethod
    def from_env(cls) -> "Caps":
        base = cls()
        return cls(
            daily_hard_cap_usd=_decimal_env("LOCI_ROUTER_DAILY_CAP_USD", base.daily_hard_cap_usd),
            provider_share=base.provider_share, eps_max=base.eps_max, control_share=base.control_share,
            canary_eps=base.canary_eps, lambda_max=base.lambda_max, min_arm_n=base.min_arm_n,
        )

    def as_dict(self) -> dict:
        """The caps as strings, the form a frozen snapshot stores."""
        return {
            "daily_hard_cap_usd": str(self.daily_hard_cap_usd), "provider_share": str(self.provider_share),
            "eps_max": str(self.eps_max), "control_share": str(self.control_share),
            "canary_eps": str(self.canary_eps), "lambda_max": str(self.lambda_max), "min_arm_n": str(self.min_arm_n),
        }

    def provider_limit_usd(self) -> Decimal:
        return self.daily_hard_cap_usd * self.provider_share


# --- arms ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Arm:
    arm_id: str
    group: str  # the model name a request uses; routing to this arm means requesting this name
    provider: str
    unit_cost: Decimal = Decimal(0)  # input + output price per token; 0 when unknown or local
    supports_tools: bool = True


@dataclass(frozen=True)
class ArmSet:
    arms: tuple
    excluded: tuple = ()  # ((name, reason), ...)


def _provider_of(entry: Mapping, info: Mapping, params: Mapping) -> str:
    for candidate in (info.get("provider"), params.get("custom_llm_provider")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip().lower()
    model = params.get("model")
    if isinstance(model, str) and "/" in model:
        return model.split("/", 1)[0].strip().lower() or "unknown"
    return "unknown"


def _price(info: Mapping) -> Decimal:
    total = Decimal(0)
    for key in ("input_cost_per_token", "output_cost_per_token"):
        value = info.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
            continue
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            continue
        if number.is_finite() and number > 0:
            total += number
    return total


def enumerate_arms(model_list: Any, exclude_providers: Optional[frozenset] = None) -> ArmSet:
    """Arms from a LiteLLM-style model list: gateway deployments only, chat models, no seat-billed providers."""
    blocked = set(SEAT_BILLED_PROVIDERS if exclude_providers is None else exclude_providers)
    blocked |= {p.strip().lower() for p in os.environ.get("LOCI_ROUTER_EXCLUDE_PROVIDERS", "").split(",") if p.strip()}
    arms: list[Arm] = []
    excluded: list[tuple] = []
    seen: dict[str, int] = {}
    for entry in model_list if isinstance(model_list, (list, tuple)) else ():
        if not isinstance(entry, Mapping):
            continue
        name = entry.get("model_name")
        info = entry.get("model_info") if isinstance(entry.get("model_info"), Mapping) else {}
        params = entry.get("litellm_params") if isinstance(entry.get("litellm_params"), Mapping) else {}
        if not isinstance(name, str) or not name:
            excluded.append(("(unnamed)", "no_model_name"))
            continue
        provider = _provider_of(entry, info, params)
        if provider in blocked or info.get("seat_billed") is True:
            excluded.append((name, "seat_billed"))
            continue
        if info.get("loci_router_arm") is False:
            excluded.append((name, "opted_out"))
            continue
        mode = info.get("mode")
        if isinstance(mode, str) and mode not in _CHAT_MODES:
            excluded.append((name, "not_chat"))
            continue
        base = str(info.get("id") or name)
        seen[base] = seen.get(base, 0) + 1
        arm_id = base if seen[base] == 1 else f"{base}#{seen[base]}"
        arms.append(Arm(arm_id, name, provider, _price(info), info.get("supports_function_calling") is not False))
    return ArmSet(tuple(arms), tuple(excluded))


def proxy_model_list() -> Optional[list]:
    """The running proxy router's deployments, or None when no proxy router is available."""
    try:
        from litellm.proxy.proxy_server import llm_router

        deployments = getattr(llm_router, "model_list", None)
        return list(deployments) if deployments is not None else None
    except Exception as exc:  # no proxy in this process, or it is not initialised yet
        logger.debug("loci colony_router: no proxy model list: %r", exc)
        return None


# --- request ------------------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteRequest:
    requested: str
    session_id: str = ""
    task_class: str = "unclassified"
    tier: str = "-"
    has_tools: bool = False
    delegated_write: bool = False
    cached_prefix: bool = False
    first_turn: bool = True
    caller: str = "anon"
    call_id: str = ""


def _headers(data: Mapping) -> Mapping:
    for holder in (data.get("proxy_server_request"), data.get("metadata")):
        if isinstance(holder, Mapping) and isinstance(holder.get("headers"), Mapping):
            return {str(k).lower(): v for k, v in holder["headers"].items()}
    return {}


def _has_cache_control(data: Mapping) -> bool:
    if data.get("cache_control_injection_points"):
        return True
    for message in data.get("messages") or []:
        if not isinstance(message, Mapping):
            continue
        if "cache_control" in message:
            return True
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(p, Mapping) and "cache_control" in p for p in content):
            return True
    return False


def _first_turn(data: Mapping) -> bool:
    return not any(
        isinstance(m, Mapping) and m.get("role") in ("assistant", "tool", "function") for m in data.get("messages") or []
    )


def request_from_data(data: Mapping) -> RouteRequest:
    """Describe a LiteLLM request body. Session id, task class and tier come from metadata or the
    `x-loci-session` / `x-loci-task-class` / `x-loci-tier` headers; a delegated read-write run is flagged by
    `metadata.delegated_write` or the header `x-loci-delegation: rw|write`."""
    metadata = data.get("metadata") if isinstance(data.get("metadata"), Mapping) else {}
    headers = _headers(data)

    def pick(key: str, header: str, default: str) -> str:
        value = metadata.get(key) or headers.get(header)
        return str(value) if value else default

    delegation = str(metadata.get("delegation") or headers.get("x-loci-delegation") or "").lower()
    return RouteRequest(
        requested=str(data.get("model") or ""),
        session_id=pick("session_id", "x-loci-session", ""),
        task_class=pick("task_class", "x-loci-task-class", "unclassified"),
        tier=pick("tier", "x-loci-tier", "-"),
        has_tools=needs_tools(dict(data)),
        delegated_write=bool(metadata.get("delegated_write")) or delegation in ("rw", "write"),
        cached_prefix=_has_cache_control(data) or bool(metadata.get("cached_prefix")),
        first_turn=_first_turn(data),
        caller=pick("caller", "x-loci-client", "anon"),
        call_id=str(data.get("litellm_call_id") or ""),
    )


# --- control slice ------------------------------------------------------------------------------

def session_bucket(session_id: str, salt: str = DEFAULT_SALT) -> float:
    """A stable number in [0, 1) for a session. The same session and salt always give the same value."""
    digest = hashlib.sha256(f"{salt}\x00{session_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2.0**64


def session_hash(session_id: str, salt: str = DEFAULT_SALT) -> str:
    return hashlib.sha256(f"{salt}\x00{session_id}".encode("utf-8")).hexdigest()[:12]


# --- learning state -----------------------------------------------------------------------------

class CellStore:
    """Discounted Beta posterior per cell (arm, task class, tier). A cell stays on its prior until it has
    `min_n` labelled outcomes. Bounded to `max_cells` (least recently used dropped, counted)."""

    def __init__(self, min_n: int = MIN_ARM_N, discount: float = DISCOUNT, max_cells: int = MAX_CELLS) -> None:
        self.min_n = min_n
        self.discount = discount
        self.max_cells = max_cells
        self.evicted = 0
        self._cells: "OrderedDict[tuple, list]" = OrderedDict()

    def observe(self, arm_id: str, task_class: str, tier: str, quality: float) -> None:
        """A labelled outcome with quality in [0, 1]."""
        if not (isinstance(quality, (int, float)) and 0.0 <= quality <= 1.0):
            raise ValueError(f"quality must be within [0, 1], got {quality!r}")
        key = (arm_id, task_class, tier)
        cell = self._cells.setdefault(key, [PRIOR, PRIOR, 0])
        self._cells.move_to_end(key)
        cell[0] = PRIOR + (cell[0] - PRIOR) * self.discount + quality
        cell[1] = PRIOR + (cell[1] - PRIOR) * self.discount + (1.0 - quality)
        cell[2] += 1
        while len(self._cells) > self.max_cells:
            self._cells.popitem(last=False)
            self.evicted += 1

    def n(self, arm_id: str, task_class: str, tier: str) -> int:
        return self._cells.get((arm_id, task_class, tier), (0, 0, 0))[2]

    def proven(self, arm_id: str, task_class: str, tier: str) -> bool:
        return self.n(arm_id, task_class, tier) >= self.min_n

    def posterior(self, arm_id: str, task_class: str, tier: str) -> tuple:
        """(alpha, beta): the prior until the cell has min_n labelled outcomes."""
        cell = self._cells.get((arm_id, task_class, tier))
        if cell is None or cell[2] < self.min_n:
            return (PRIOR, PRIOR)
        return (cell[0], cell[1])

    def mean(self, arm_id: str, task_class: str, tier: str) -> float:
        alpha, beta = self.posterior(arm_id, task_class, tier)
        return alpha / (alpha + beta)

    def __len__(self) -> int:
        return len(self._cells)

    def state_bytes(self) -> int:
        return len(json.dumps([[*k, *v] for k, v in self._cells.items()]).encode("utf-8"))


class LagrangeController:
    """Dual ascent on the cost constraint: lambda' = clip(lambda + eta * (observed / target - 1), 0, lambda_max),
    at most once per `interval_s`. Spending twice the target raises lambda by eta each step up to the clamp;
    spending exactly the target leaves it unchanged."""

    def __init__(
        self, lam: float = 0.0, eta: float = LAGRANGE_ETA, lambda_max: float = 5.0,
        interval_s: float = LAGRANGE_INTERVAL_S,
    ) -> None:
        self.lam = lam
        self.eta = eta
        self.lambda_max = lambda_max
        self.interval_s = interval_s
        self._last: Optional[float] = None

    def step(self, observed: Decimal, target: Decimal, now: float) -> float:
        if target <= 0:
            raise ValueError(f"target must be positive, got {target}")
        if self._last is not None and now - self._last < self.interval_s:
            return self.lam
        self._last = now
        error = float((Decimal(observed) - target) / target)
        self.lam = min(max(self.lam + self.eta * error, 0.0), self.lambda_max)
        return self.lam


# --- recorder -----------------------------------------------------------------------------------

class ShadowRecorder:
    """Bounded queue of rows: when full, the oldest row is dropped and counted. Optionally appends to a file."""

    def __init__(self, maxlen: int = RECORDER_MAX, path: Optional[str] = None, max_bytes: int = 0) -> None:
        self.maxlen = maxlen
        self.path = path
        self.max_bytes = max_bytes
        self.dropped = 0
        self.recorded = 0
        self.written = 0
        self.write_errors = 0
        self._queue: deque = deque()
        self._lock = threading.Lock()

    def record(self, row: dict) -> None:
        with self._lock:
            if len(self._queue) >= self.maxlen:
                self._queue.popleft()
                self.dropped += 1
            self._queue.append(row)
            self.recorded += 1

    def pending(self) -> int:
        with self._lock:
            return len(self._queue)

    def rows(self) -> list:
        with self._lock:
            return list(self._queue)

    def flush(self) -> int:
        """Append the queued rows to the file and clear them. A failed write keeps them queued (bounded)."""
        if not self.path:
            return 0
        with self._lock:
            batch = list(self._queue)
            self._queue.clear()
        if not batch:
            return 0
        done = 0
        try:
            for row in batch:
                append_line(self.path, json.dumps(row, separators=(",", ":"), default=str), self.max_bytes)
                done += 1
        except OSError as exc:
            self.write_errors += 1
            logger.warning("loci colony_router: could not append shadow rows: %r", exc)
        with self._lock:
            for row in reversed(batch[done:]):
                if len(self._queue) >= self.maxlen:
                    self.dropped += 1
                else:
                    self._queue.appendleft(row)
        self.written += done
        return done


# --- verdicts, pass rule, quality breaker, frozen snapshot, audit ---------------------------------

MIN_SLICE_N = 30
_ONE_PP = Fraction(1, 100)
_QUALITY_LB_Z = 1.645  # one-sided 95%
QUALITY_FLOOR_MARGIN = -0.08
BREAKER_MARGIN = 0.05
BREAKER_DISABLE_S = 86400.0
ALLOWED_CANARY_SHARES = (0.0, 0.1, 0.5, 0.9)
SNAPSHOT_VERSION = 1
MAX_SNAPSHOT_MAPPING = 4096
ROLLBACK_MARKER = "ROLLED_BACK"
AUDIT_FILE = "audit.jsonl"
# External gates the preregistered pass rule needs from the offline harness; a missing one fails closed.
REQUIRED_EXTERNAL_GATES = (
    "holm", "ece", "conformal_coverage", "stratum_floor", "baseline_beat", "negative_control", "judge_gate",
)


class Verdict(str, Enum):
    HEALTHY = "healthy"
    DEGENERATE = "degenerate"
    TOO_EARLY = "too_early"


@dataclass(frozen=True)
class SliceStats:
    """Counts for one traffic slice (candidate or incumbent) over the comparison window."""

    n: int
    completed: int
    refused_or_empty: int
    accepted: int
    cost_usd: Decimal
    escalations: int
    p95_latency_s: float
    cache_read_tokens: int
    prompt_tokens: int
    labelled: int
    quality_ok: int
    mean_reward: float
    cap_breaches: int = 0
    attributed_escalations: int = 0  # governor escalations caused by this slice


@dataclass(frozen=True)
class VerdictResult:
    verdict: Verdict
    reasons: tuple = ()


def _finite(stats: SliceStats) -> bool:
    return Decimal(stats.cost_usd).is_finite() and all(
        math.isfinite(x) for x in (stats.p95_latency_s, stats.mean_reward)
    )


def judge(candidate: SliceStats, incumbent: SliceStats, min_n: int = MIN_SLICE_N) -> VerdictResult:
    """Healthy, degenerate or too early. Degenerate means the candidate got cheaper or scored better by
    failing: non-finite numbers, completion more than 1pp under the incumbent, refusal/empty more than 1pp
    over it (cheap-by-refusing when it also costs less per call), escalation drift beyond 2 standard errors,
    or reward up while quality is down by more than 1pp. A DEGENERATE verdict is final for the rollout."""
    if not (_finite(candidate) and _finite(incumbent)):
        return VerdictResult(Verdict.DEGENERATE, ("nonfinite",))
    if candidate.n < min_n or incumbent.n < min_n:
        return VerdictResult(Verdict.TOO_EARLY, ("burn_in",))
    reasons: list[str] = []
    if Fraction(candidate.completed, candidate.n) < Fraction(incumbent.completed, incumbent.n) - _ONE_PP:
        reasons.append("completion")
    if Fraction(candidate.refused_or_empty, candidate.n) > Fraction(incumbent.refused_or_empty, incumbent.n) + _ONE_PP:
        reasons.append("refusal")
        if Decimal(candidate.cost_usd) * incumbent.n < Decimal(incumbent.cost_usd) * candidate.n:
            reasons.append("cheap_by_refusing")
    pooled = (candidate.escalations + incumbent.escalations) / (candidate.n + incumbent.n)
    se = math.sqrt(pooled * (1.0 - pooled) * (1.0 / candidate.n + 1.0 / incumbent.n))
    if se > 0.0 and (candidate.escalations / candidate.n - incumbent.escalations / incumbent.n) / se > 2.0:
        reasons.append("escalation_drift")
    if candidate.labelled > 0 and incumbent.labelled > 0 and candidate.mean_reward > incumbent.mean_reward:
        if Fraction(candidate.quality_ok, candidate.labelled) < Fraction(incumbent.quality_ok, incumbent.labelled) - _ONE_PP:
            reasons.append("reward_up_quality_down")
    return VerdictResult(Verdict.DEGENERATE if reasons else Verdict.HEALTHY, tuple(reasons))


@dataclass(frozen=True)
class PassResult:
    passed: bool
    failures: tuple = ()


def evaluate_pass(
    candidate: SliceStats, incumbent: SliceStats, external_gates: Optional[Mapping[str, bool]] = None,
    min_n: int = MIN_SLICE_N,
) -> PassResult:
    """The preregistered pass rule for the learned router against the live incumbent slice: cost per accepted
    answer down at least 10%; one-sided 95% lower bound of the quality difference >= -0.08; completion no more
    than 1pp lower; refusal/empty no more than 1pp higher; p95 latency <= 1.25x; cache-read share not lower by
    more than 3pp; no cap breach and no governor escalation caused by the candidate; and every external gate
    (Holm, calibration, stratum floor, baseline, negative control, judge gate) true. A DEGENERATE or TOO_EARLY
    verdict fails the rule outright."""
    verdict = judge(candidate, incumbent, min_n)
    if verdict.verdict is Verdict.DEGENERATE:
        return PassResult(False, ("degenerate", *verdict.reasons))
    if verdict.verdict is Verdict.TOO_EARLY:
        return PassResult(False, ("too_early",))
    failures: list[str] = []
    if candidate.accepted <= 0 or incumbent.accepted <= 0:
        failures.append("no_accepted")
    elif Decimal(candidate.cost_usd) * incumbent.accepted > Decimal(incumbent.cost_usd) * candidate.accepted * Decimal("0.9"):
        failures.append("cost_per_accepted")
    if candidate.labelled <= 0 or incumbent.labelled <= 0:
        failures.append("no_labels")
    else:
        q_c, q_i = candidate.quality_ok / candidate.labelled, incumbent.quality_ok / incumbent.labelled
        se = math.sqrt(q_c * (1 - q_c) / candidate.labelled + q_i * (1 - q_i) / incumbent.labelled)
        if (q_c - q_i) - _QUALITY_LB_Z * se < QUALITY_FLOOR_MARGIN:
            failures.append("quality_lower_bound")
    if Fraction(candidate.completed, candidate.n) < Fraction(incumbent.completed, incumbent.n) - _ONE_PP:
        failures.append("completion")
    if Fraction(candidate.refused_or_empty, candidate.n) > Fraction(incumbent.refused_or_empty, incumbent.n) + _ONE_PP:
        failures.append("refusal")
    if candidate.p95_latency_s > incumbent.p95_latency_s * 1.25:
        failures.append("p95_latency")
    share_c = Fraction(candidate.cache_read_tokens, candidate.prompt_tokens) if candidate.prompt_tokens else Fraction(0)
    share_i = Fraction(incumbent.cache_read_tokens, incumbent.prompt_tokens) if incumbent.prompt_tokens else Fraction(0)
    if share_c < share_i - Fraction(3, 100):
        failures.append("cache_read_share")
    if candidate.cap_breaches:
        failures.append("cap_breach")
    if candidate.attributed_escalations:
        failures.append("governor_escalation")
    gates = external_gates or {}
    failures.extend(f"external:{name}" for name in REQUIRED_EXTERNAL_GATES if gates.get(name) is not True)
    return PassResult(not failures, tuple(failures))


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower end of the Wilson score interval for a proportion."""
    if n <= 0:
        return 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / denom


class AuditLog:
    """Bounded in-memory audit trail, optionally mirrored to a JSON-lines file. A failed write is counted."""

    def __init__(self, path: Optional[str] = None, maxlen: int = 1000) -> None:
        self.path = path
        self.write_errors = 0
        self.entries: deque = deque(maxlen=maxlen)

    def add(self, kind: str, ts: float, **fields: Any) -> dict:
        entry = {"ts": ts, "kind": kind, **fields}
        self.entries.append(entry)
        if self.path:
            try:
                append_line(self.path, json.dumps(entry, separators=(",", ":"), default=str))
            except OSError as exc:
                self.write_errors += 1
                logger.warning("loci colony_router: could not write an audit entry: %r", exc)
        return entry


def cell_key(req: RouteRequest) -> str:
    return f"{req.task_class}|{req.tier}"


class QualityBreaker:
    """Per-cell Wilson breaker on sampled quality results. When, with at least `min_n` results on each side,
    the lower bound of the downgraded arms falls more than `margin` under the baseline's, the downgrade is
    disabled for the cell for `disable_s` and an audit entry is written."""

    def __init__(
        self, audit: AuditLog, min_n: int = MIN_SLICE_N, margin: float = BREAKER_MARGIN,
        disable_s: float = BREAKER_DISABLE_S, max_cells: int = 1024,
    ) -> None:
        self.audit = audit
        self.min_n = min_n
        self.margin = margin
        self.disable_s = disable_s
        self.max_cells = max_cells
        self._counts: "OrderedDict[str, list]" = OrderedDict()
        self._until: dict[str, float] = {}

    def observe(self, cell: str, downgraded: bool, ok: bool) -> None:
        counts = self._counts.setdefault(cell, [0, 0, 0, 0])
        self._counts.move_to_end(cell)
        offset = 0 if downgraded else 2
        counts[offset] += bool(ok)
        counts[offset + 1] += 1
        while len(self._counts) > self.max_cells:
            self._counts.popitem(last=False)

    def check(self, cell: str, now: float) -> bool:
        """Trip the breaker if the evidence says so. True only on the call that trips it."""
        if now < self._until.get(cell, float("-inf")):
            return False
        counts = self._counts.get(cell)
        if counts is None or counts[1] < self.min_n or counts[3] < self.min_n:
            return False
        lb_down = wilson_lower_bound(counts[0], counts[1])
        lb_base = wilson_lower_bound(counts[2], counts[3])
        if lb_down >= lb_base - self.margin:
            return False
        self._until[cell] = now + self.disable_s
        self.audit.add(
            "quality_breaker_trip", now, cell=cell, lb_downgraded=lb_down, lb_baseline=lb_base,
            n_downgraded=counts[1], n_baseline=counts[3], disabled_until=now + self.disable_s,
        )
        return True

    def downgrade_allowed(self, cell: str, now: float) -> bool:
        until = self._until.get(cell)
        if until is None:
            return True
        if now < until:
            return False
        del self._until[cell]
        self._counts.pop(cell, None)  # a fresh sample after the pause
        return True


class SnapshotRejected(ValueError):
    """A frozen incumbent snapshot that is malformed, edited, or does not match the pinned caps."""


def _snapshot_digest(body: Mapping) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def freeze_snapshot(caps: Caps, mapping: Mapping[str, str], version: int = SNAPSHOT_VERSION) -> dict:
    """The JSON-able snapshot of an incumbent: its caps, its cell -> arm mapping ("*" = default) and a digest."""
    body = {"version": version, "caps": caps.as_dict(), "mapping": dict(mapping)}
    return {**body, "sha256": _snapshot_digest(body)}


@dataclass(frozen=True)
class IncumbentSnapshot:
    version: int
    caps: Mapping
    mapping: Mapping
    sha256: str

    def arm_for(self, cell: str) -> Optional[str]:
        return self.mapping.get(cell, self.mapping.get("*"))

    @classmethod
    def load(cls, source: Any, pinned: Caps) -> "IncumbentSnapshot":
        """Load a snapshot from a mapping or a JSON file path. Any cap that differs from `pinned` (wider or
        tighter), a missing or unknown cap, a digest mismatch or a malformed body raises SnapshotRejected."""
        data = source
        if isinstance(source, (str, os.PathLike)):
            try:
                with open(source, encoding="utf-8") as handle:
                    data = json.load(handle)
            except (OSError, ValueError) as exc:
                raise SnapshotRejected(f"snapshot unreadable: {exc!r}") from exc
        if not isinstance(data, Mapping):
            raise SnapshotRejected("snapshot is not a JSON object")
        caps = data.get("caps")
        if not isinstance(caps, Mapping):
            raise SnapshotRejected("snapshot cap table is missing")
        wanted = pinned.as_dict()
        for key in caps:
            if key not in wanted:
                raise SnapshotRejected(f"snapshot cap {key!r} is not a known cap")
        for key, pinned_value in wanted.items():
            if key not in caps:
                raise SnapshotRejected(f"snapshot cap {key!r} is missing")
            value = caps[key]
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise SnapshotRejected(f"snapshot cap {key!r} must be a decimal string, got {value!r}")
            try:
                same = Decimal(str(value)) == Decimal(pinned_value)
            except InvalidOperation:
                raise SnapshotRejected(f"snapshot cap {key!r} is not a number: {value!r}") from None
            if not same:
                raise SnapshotRejected(f"snapshot cap {key!r} differs from the pinned envelope: {value} != {pinned_value}")
        if data.get("version") != SNAPSHOT_VERSION:
            raise SnapshotRejected(f"unsupported snapshot version {data.get('version')!r}")
        mapping = data.get("mapping")
        if not isinstance(mapping, Mapping) or len(mapping) > MAX_SNAPSHOT_MAPPING or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in mapping.items()
        ):
            raise SnapshotRejected("snapshot mapping must be a bounded string-to-string object")
        body = {"version": data["version"], "caps": dict(caps), "mapping": dict(mapping)}
        digest = _snapshot_digest(body)
        if data.get("sha256") != digest:
            raise SnapshotRejected("snapshot sha256 does not match its contents (edited after freezing)")
        return cls(int(data["version"]), dict(caps), dict(mapping), digest)


# --- decision -----------------------------------------------------------------------------------

def propensity(eps: float, n: int, greedy: bool) -> float:
    """Probability that epsilon-greedy over n arms picks a given arm: (1 - eps) * 1[greedy] + eps / n."""
    return (1.0 - eps) * (1.0 if greedy else 0.0) + eps / n


@dataclass
class Decision:
    decision_id: str
    ts: float
    requested: str
    session_hash: str
    control: bool
    control_reason: str
    task_class: str
    tier: str
    n_arms: int
    safe_arms: list
    excluded: dict
    incumbent_arm: Optional[str]
    greedy_arm: Optional[str]
    chosen_arm: Optional[str]
    chosen_group: Optional[str]
    epsilon: float
    epsilon_reason: str
    explore: bool
    propensity: Optional[float]
    lam: float
    would_change: bool
    reason: str
    share_cap: str
    flags: dict
    pre: dict
    thompson: Optional[dict] = None
    probe_arms: list = field(default_factory=list)
    decision_us: int = 0
    acted: bool = False
    slice: str = "holdout"
    route_reason: str = ""
    model: Optional[str] = None

    def to_row(self) -> dict:
        row = dict(self.__dict__)
        row["kind"] = "decision"
        return row


class ColonyRouter:
    """The shadow decision engine. Deterministic given `clock` and the two seeded generators."""

    def __init__(
        self,
        arms_fn: Callable[[], Optional[list]] = proxy_model_list,
        *,
        caps: Caps = Caps(),
        epsilon: float = EPS_MAX,
        board: Optional[ChannelBoard] = None,
        cells: Optional[CellStore] = None,
        lagrange: Optional[LagrangeController] = None,
        recorder: Optional[ShadowRecorder] = None,
        spend_fn: Optional[Callable[[], Optional[Mapping[str, Decimal]]]] = None,
        level_fn: Callable[[], int] = lambda: 0,
        resource_fn: Callable[[], float] = lambda: 1.0,
        pause_fn: Callable[[], bool] = lambda: False,
        clock: Callable[[], float] = time.time,
        rng: Optional[random.Random] = None,
        thompson_rng: Optional[random.Random] = None,
        thompson_rate: float = 0.1,
        thompson_draws: int = THOMPSON_DRAWS,
        salt: str = DEFAULT_SALT,
        quality_floor: float = 0.5,
        canary_share: float = 0.0,
        snapshot: Optional[IncumbentSnapshot] = None,
        audit: Optional[AuditLog] = None,
        breaker: Optional[QualityBreaker] = None,
        rollback_marker: Optional[str] = None,
    ) -> None:
        if canary_share not in ALLOWED_CANARY_SHARES:
            raise ValueError(f"canary_share must be one of {ALLOWED_CANARY_SHARES}, got {canary_share!r}")
        self.caps = caps
        self.epsilon = min(max(float(epsilon), 0.0), float(caps.eps_max))
        self.arms_fn = arms_fn
        self.board = board or ChannelBoard()
        self.cells = cells or CellStore(caps.min_arm_n)
        self.lagrange = lagrange or LagrangeController(lambda_max=float(caps.lambda_max))
        self.recorder = recorder or ShadowRecorder()
        self.spend_fn = spend_fn
        self.level_fn = level_fn
        self.resource_fn = resource_fn
        self.pause_fn = pause_fn
        self.clock = clock
        self.rng = rng or random.Random()
        self.thompson_rng = thompson_rng or random.Random()
        self.thompson_rate = thompson_rate
        self.thompson_draws = max(int(thompson_draws), THOMPSON_DRAWS)
        self.salt = salt
        self.quality_floor = quality_floor
        self.canary_share = canary_share
        self.snapshot = snapshot
        self.audit = audit or AuditLog()
        self.breaker = breaker or QualityBreaker(self.audit)
        self.rollback_marker = rollback_marker
        self.state = "active"
        self.rollback_reason = ""
        self._control_share = float(caps.control_share)
        if rollback_marker and os.path.exists(rollback_marker):
            self.state, self.rollback_reason = "rolled_back", "marker"
        self._arm_set = ArmSet(())
        self._arms_at: Optional[float] = None
        self._pending: "OrderedDict[str, tuple]" = OrderedDict()
        self._counters = {
            "decisions": 0, "errors": 0, "no_arms": 0, "arms_unavailable": 0, "no_safe_arm": 0,
            "no_session": 0, "spend_unavailable": 0, "control": 0, "explored": 0, "would_change": 0,
            "outcomes": 0, "unmatched_outcomes": 0, "acted": 0, "rollbacks": 0, "marker_write_errors": 0,
            "snapshot_rejected": 0, "invalid_canary_share": 0, "watchdog_ticks": 0, "watchdog_no_data": 0,
            "watchdog_errors": 0,
        }

    # -- arms and safe set --
    def arm_set(self, now: float) -> ArmSet:
        if self._arms_at is None or now - self._arms_at >= ARMS_TTL_S or not self._arm_set.arms:
            try:
                model_list = self.arms_fn()
            except Exception as exc:
                logger.warning("loci colony_router: arms provider failed: %r", exc)
                model_list = None
            if model_list is None:
                self._counters["arms_unavailable"] += 1
            else:
                self._arm_set = enumerate_arms(model_list)
            self._arms_at = now
        return self._arm_set

    def _safe_set(self, arms: tuple, req: RouteRequest, now: float, incumbent_any: Optional[Arm] = None) -> tuple:
        spend = None
        limit = self.caps.provider_limit_usd()
        share_state = "unchecked"
        if self.spend_fn is not None:
            try:
                spend = self.spend_fn()
            except Exception as exc:
                logger.warning("loci colony_router: spend view failed: %r", exc)
        if spend is None:
            self._counters["spend_unavailable"] += 1
            share_state = "unavailable"
        else:
            share_state = "applied"
        safe: list[Arm] = []
        excluded: dict[str, str] = {}
        for arm in sorted(arms, key=lambda a: a.arm_id):
            self.board.register(arm.arm_id, now)
            if self.board.unhealthy(arm.arm_id, now):
                excluded[arm.arm_id] = "unhealthy"
            elif spend is not None and Decimal(spend.get(arm.provider, 0)) >= limit:
                excluded[arm.arm_id] = "provider_share_cap"
            elif req.has_tools and not arm.supports_tools:
                excluded[arm.arm_id] = "no_tool_support"
            else:
                safe.append(arm)
        if incumbent_any is not None and not self.breaker.downgrade_allowed(cell_key(req), now):
            for arm in [a for a in safe if a.unit_cost < incumbent_any.unit_cost]:
                safe.remove(arm)
                excluded[arm.arm_id] = "breaker"
        return safe, excluded, share_state

    def _incumbent_of(self, req: RouteRequest, pool) -> Optional[Arm]:
        """The incumbent's arm: the frozen snapshot's choice for the cell when it has one, else the arm
        serving the model the request names."""
        mapped = self.snapshot.arm_for(cell_key(req)) if self.snapshot else None
        found = next((a for a in pool if a.arm_id == mapped), None) if mapped else None
        return found or next((a for a in pool if a.group == req.requested), None)

    # -- epsilon --
    def epsilon_for(self, req: RouteRequest, n_safe: int, control: bool, base: Optional[float] = None) -> tuple:
        """(epsilon, reason). Reason is "" when exploration is allowed. `base` replaces the shadow epsilon
        (the canary passes its own, smaller one)."""
        if self.pause_fn():
            return 0.0, "pause"
        if int(self.level_fn()) >= 1:
            return 0.0, "governor_level"
        if float(self.resource_fn()) < RESOURCE_FLOOR_FOR_EXPLORATION:
            return 0.0, "resource"
        for flag, reason in (
            (req.has_tools, "tools"), (req.cached_prefix, "cached_prefix"),
            (req.delegated_write, "delegated_write"), (not req.first_turn, "not_first_turn"),
            (control, "control"), (n_safe < 2, "single_arm"),
        ):
            if flag:
                return 0.0, reason
        return (self.epsilon if base is None else base), ""

    # -- greedy --
    def _greedy(self, safe: list, req: RouteRequest, incumbent: Optional[Arm]) -> Arm:
        cell = (req.task_class, req.tier)
        proven = [a for a in safe if self.cells.proven(a.arm_id, *cell)]
        candidates = list(proven)
        if incumbent is not None and incumbent not in candidates:
            candidates.append(incumbent)
        if not candidates:
            return min(safe, key=lambda a: (a.unit_cost, a.arm_id))
        top_cost = max((a.unit_cost for a in candidates), default=Decimal(0))

        def score(arm: Arm) -> float:
            mean = self.cells.mean(arm.arm_id, *cell)
            if self.cells.proven(arm.arm_id, *cell) and mean < self.quality_floor:
                return -1.0
            norm = float(arm.unit_cost / top_cost) if top_cost > 0 else 0.0
            return mean - self.lagrange.lam * norm

        return min(candidates, key=lambda a: (-score(a), a is not incumbent, a.arm_id))

    def _thompson(self, safe: list, req: RouteRequest) -> dict:
        """One Thompson pick and its Monte-Carlo propensity: the share of `thompson_draws` independent
        posterior draws in which that arm has the highest sampled quality. The pick is the first draw's
        winner, so its count is at least 1."""
        cell = (req.task_class, req.tier)
        params = [(a.arm_id, *self.cells.posterior(a.arm_id, *cell)) for a in safe]
        picks = []
        for _ in range(self.thompson_draws):
            draws = [(self.thompson_rng.betavariate(alpha, beta), arm_id) for arm_id, alpha, beta in params]
            picks.append(max(draws, key=lambda d: d[0])[1])
        chosen = picks[0]
        return {"arm": chosen, "propensity": picks.count(chosen) / self.thompson_draws, "draws": self.thompson_draws}

    # -- decide --
    def decide(self, req: RouteRequest, now: Optional[float] = None) -> Decision:
        started = time.perf_counter()
        now = self.clock() if now is None else now
        arm_set = self.arm_set(now)
        slice_name, control_reason = self._slice(req)
        control = slice_name == "control"
        safe, excluded, share_state = self._safe_set(arm_set.arms, req, now, self._incumbent_of(req, arm_set.arms))
        incumbent = self._incumbent_of(req, safe)
        acting, route_reason = self._may_act(req, slice_name, share_state)
        flags = {
            "tools": req.has_tools, "cached_prefix": req.cached_prefix,
            "delegated_write": req.delegated_write, "first_turn": req.first_turn,
        }
        base = dict(
            decision_id=uuid.uuid4().hex[:16], ts=now, requested=req.requested,
            session_hash=session_hash(req.session_id, self.salt) if req.session_id else "",
            control=control, control_reason=control_reason, task_class=req.task_class, tier=req.tier,
            n_arms=len(arm_set.arms), safe_arms=[a.arm_id for a in safe], excluded=excluded,
            incumbent_arm=incumbent.arm_id if incumbent else None, lam=self.lagrange.lam,
            share_cap=share_state, flags=flags, slice=slice_name,
        )
        if not arm_set.arms or not safe:
            reason = "no_arms" if not arm_set.arms else "no_safe_arm"
            self._counters["no_arms" if not arm_set.arms else "no_safe_arm"] += 1
            decision = Decision(
                greedy_arm=None, chosen_arm=None, chosen_group=None, epsilon=0.0, epsilon_reason=reason,
                explore=False, propensity=None, would_change=False, reason=reason, pre={},
                route_reason=reason if acting else route_reason, **base,
            )
            return self._finish(decision, req, started)
        greedy = self._greedy(safe, req, incumbent)
        eps, eps_reason = self.epsilon_for(req, len(safe), control, float(self.caps.canary_eps) if acting else None)
        explore = False
        if control:
            chosen = incumbent or greedy
            prob = 1.0
        else:
            explore = eps > 0.0 and self.rng.random() < eps
            chosen = safe[self.rng.randrange(len(safe))] if explore else greedy
            prob = propensity(eps, len(safe), chosen is greedy) if eps > 0.0 else 1.0
        decision = Decision(
            greedy_arm=greedy.arm_id, chosen_arm=chosen.arm_id, chosen_group=chosen.group, epsilon=eps,
            epsilon_reason=eps_reason, explore=explore, propensity=prob,
            would_change=chosen.group != req.requested, reason="", acted=acting,
            route_reason="" if acting else route_reason, model=chosen.group if acting else None,
            pre={a.arm_id: self.board.values(a.arm_id, now) for a in {greedy, chosen, incumbent or greedy}}, **base,
        )
        if not control and self.thompson_rate > 0.0 and self.thompson_rng.random() < self.thompson_rate:
            decision.thompson = self._thompson(safe, req)
        decision.probe_arms = [a.arm_id for a in arm_set.arms if self.board.probe_due(a.arm_id, now)]
        return self._finish(decision, req, started)

    def _slice(self, req: RouteRequest) -> tuple:
        """("control" | "learned" | "holdout", reason). By session, never by call: the first control_share of
        the session hash is control, the next canary_share the learned slice, the rest is untouched."""
        if not req.session_id:
            self._counters["no_session"] += 1
            return "control", "no_session"
        bucket = session_bucket(req.session_id, self.salt)
        if bucket < self._control_share:
            return "control", "slice"
        if bucket < self._control_share + self.canary_share:
            return "learned", ""
        return "holdout", ""

    def _may_act(self, req: RouteRequest, slice_name: str, share_state: str) -> tuple:
        """Whether the learned rule routes this request, and if not, why. Every condition must hold."""
        if self.canary_share <= 0.0:
            return False, "canary_off"
        if self.state != "active":
            return False, "rolled_back"
        if self.snapshot is None:
            return False, "no_snapshot"
        if slice_name != "learned":
            return False, f"slice_{slice_name}"
        if req.has_tools:
            return False, "ineligible_tools"
        if req.delegated_write:
            return False, "ineligible_delegated_write"
        if self.pause_fn():
            return False, "pause"
        if int(self.level_fn()) >= 2:
            return False, "governor_level"
        if share_state != "applied":
            return False, "spend_unavailable"
        return True, ""

    def rollback(self, reason: str, now: Optional[float] = None) -> bool:
        """Swap back to the incumbent: the learned rule stops routing. Final: there is no resume. Returns
        False when already rolled back."""
        if self.state == "rolled_back":
            return False
        now = self.clock() if now is None else now
        self.state, self.rollback_reason = "rolled_back", reason
        self._counters["rollbacks"] += 1
        self.audit.add(
            "rollback", now, reason=reason, snapshot_sha=self.snapshot.sha256 if self.snapshot else None,
        )
        if self.rollback_marker:
            try:
                os.makedirs(os.path.dirname(self.rollback_marker) or ".", exist_ok=True)
                with open(self.rollback_marker, "w", encoding="utf-8") as handle:
                    json.dump({"ts": now, "reason": reason}, handle)
            except OSError as exc:
                self._counters["marker_write_errors"] += 1
                logger.error("loci colony_router: could not write the rollback marker: %r", exc)
        return True

    def _finish(self, decision: Decision, req: RouteRequest, started: float) -> Decision:
        decision.decision_us = int((time.perf_counter() - started) * 1_000_000)
        self._counters["decisions"] += 1
        self._counters["control"] += decision.control
        self._counters["explored"] += decision.explore
        self._counters["would_change"] += decision.would_change
        self._counters["acted"] += decision.acted
        self.recorder.record(decision.to_row())
        if req.call_id:
            self._pending[req.call_id] = (decision.decision_id, decision.pre)
            while len(self._pending) > RECORDER_MAX:
                self._pending.popitem(last=False)
        return decision

    # -- outcomes --
    def observe_result(
        self, arm_id: str, ok: bool, *, call_id: str = "", caller: str = "anon", status: Optional[int] = None,
        exc: Optional[BaseException] = None, latency_s: Optional[float] = None, now: Optional[float] = None,
    ) -> dict:
        """Feed a finished call back into the channels and record the post-action features."""
        now = self.clock() if now is None else now
        transient = False
        if ok:
            self.board.observe_success(arm_id, now)
        else:
            transient = self.board.observe_failure(arm_id, now, caller, exc=exc, status=status)
        pending = self._pending.pop(call_id, None) if call_id else None
        self._counters["outcomes"] += 1
        if pending is None:
            self._counters["unmatched_outcomes"] += 1
        row = {
            "kind": "outcome", "ts": now, "decision_id": pending[0] if pending else None, "arm": arm_id,
            "ok": bool(ok), "transient_failure": transient, "status": status, "latency_s": latency_s,
            "pre": pending[1].get(arm_id) if pending else None, "post": self.board.values(arm_id, now),
        }
        self.recorder.record(row)
        return row

    def stats(self) -> dict:
        now = self.clock()
        return {
            **self._counters,
            "arms": len(self._arm_set.arms), "arms_excluded": len(self._arm_set.excluded),
            "recorder_pending": self.recorder.pending(), "recorder_dropped": self.recorder.dropped,
            "recorder_recorded": self.recorder.recorded, "recorder_written": self.recorder.written,
            "recorder_write_errors": self.recorder.write_errors,
            "state": self.state, "rollback_reason": self.rollback_reason, "canary_share": self.canary_share,
            "snapshot_sha": self.snapshot.sha256 if self.snapshot else None, "audit_entries": len(self.audit.entries),
            "audit_write_errors": self.audit.write_errors,
            "cells": len(self.cells), "cells_evicted": self.cells.evicted, "arms_evicted": self.board.evicted,
            "state_bytes": self.board.state_bytes(now) + self.cells.state_bytes(),
        }


# --- watchdog -----------------------------------------------------------------------------------

class Watchdog:
    """Judges the learned slice against the incumbent slice once per tick (5 minutes) and rolls the router
    back on a DEGENERATE verdict. `stats_fn` returns (candidate, incumbent) SliceStats, or None when there is
    no data; no data is counted, never read as healthy or as degenerate."""

    def __init__(
        self, router: ColonyRouter, stats_fn: Callable[[], Optional[tuple]],
        interval_s: float = 300.0, min_n: int = MIN_SLICE_N,
    ) -> None:
        self.router = router
        self.stats_fn = stats_fn
        self.interval_s = interval_s
        self.min_n = min_n
        self.last: Optional[VerdictResult] = None
        self._last_tick: Optional[float] = None

    def tick(self, now: Optional[float] = None) -> Optional[VerdictResult]:
        now = self.router.clock() if now is None else now
        self._last_tick = now
        counters = self.router._counters
        counters["watchdog_ticks"] += 1
        try:
            pair = self.stats_fn()
        except Exception as exc:
            counters["watchdog_errors"] += 1
            logger.warning("loci colony_router: watchdog could not read slice stats: %r", exc)
            return None
        if pair is None:
            counters["watchdog_no_data"] += 1
            return None
        self.last = judge(pair[0], pair[1], self.min_n)
        if self.last.verdict is Verdict.DEGENERATE:
            self.router.rollback("degenerate:" + ",".join(self.last.reasons), now)
        return self.last

    def maybe_tick(self, now: Optional[float] = None) -> Optional[VerdictResult]:
        now = self.router.clock() if now is None else now
        if self._last_tick is not None and now - self._last_tick < self.interval_s:
            return None
        return self.tick(now)


# --- LiteLLM callback ---------------------------------------------------------------------------

def shadow_enabled() -> bool:
    return _truthy("LOCI_ROUTER_SHADOW")


def _router_from_env() -> ColonyRouter:
    pause = os.environ.get("LOCI_ROUTER_PAUSE_FILE", "").strip()
    learn_dir = os.environ.get("LOCI_ROUTER_LEARN_DIR", "").strip()
    recorder = ShadowRecorder(
        path=os.path.join(learn_dir, ROWS_FILE) if learn_dir else None,
        max_bytes=env_int("LOCI_ROUTER_ROWS_MAX_BYTES", 20_000_000),
    )
    try:
        epsilon = float(os.environ.get("LOCI_ROUTER_EPSILON", EPS_MAX))
    except ValueError:
        epsilon = EPS_MAX
    caps = Caps.from_env()
    rejected = invalid = 0
    try:
        share = float(os.environ.get("LOCI_ROUTER_CANARY_SHARE", "0") or 0)
    except ValueError:
        share = -1.0
    if share not in ALLOWED_CANARY_SHARES:
        logger.error("loci colony_router: LOCI_ROUTER_CANARY_SHARE must be one of %s; canary stays off", ALLOWED_CANARY_SHARES)
        share, invalid = 0.0, 1
    snapshot = None
    snapshot_path = os.environ.get("LOCI_ROUTER_SNAPSHOT", "").strip()
    if snapshot_path:
        try:
            snapshot = IncumbentSnapshot.load(snapshot_path, caps)
        except SnapshotRejected as exc:
            logger.error("loci colony_router: snapshot rejected, canary stays off: %s", exc)
            rejected = 1
    audit = AuditLog(os.path.join(learn_dir, AUDIT_FILE) if learn_dir else None)
    router = ColonyRouter(
        caps=caps, epsilon=epsilon, recorder=recorder, canary_share=share, snapshot=snapshot, audit=audit,
        pause_fn=(lambda: os.path.exists(pause)) if pause else (lambda: False),
        rollback_marker=os.path.join(learn_dir, ROLLBACK_MARKER) if learn_dir else None,
    )
    router._counters["snapshot_rejected"] += rejected
    router._counters["invalid_canary_share"] += invalid
    return router


class LociColonyRouter(CustomLogger):
    """Shadow router callback: decides, records, and returns None. Only when the router reports `acted` (the
    canary: see the module docstring) does it set `data["model"]`; otherwise it never writes to the request."""

    def __init__(self, router: Optional[ColonyRouter] = None, watchdog: Optional[Watchdog] = None) -> None:
        super().__init__()
        self._router = router
        self.watchdog = watchdog
        self._lock = threading.Lock()

    @property
    def router(self) -> ColonyRouter:
        with self._lock:
            if self._router is None:
                self._router = _router_from_env()
            return self._router

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> Any:
        if not shadow_enabled() or not isinstance(data, dict):
            return None
        router = self.router
        try:
            decision = router.decide(request_from_data(data))
            if decision.acted and decision.model and decision.model != data.get("model"):
                data["model"] = decision.model
                metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
                metadata["loci_router"] = {"decision_id": decision.decision_id, "slice": decision.slice}
                data["metadata"] = metadata
        except Exception as exc:
            router._counters["errors"] += 1
            logger.warning("loci colony_router: shadow decision failed, request unaffected: %r", exc)
        return None

    def _arm_of(self, kwargs: dict) -> Optional[str]:
        params = kwargs.get("litellm_params") if isinstance(kwargs.get("litellm_params"), dict) else {}
        metadata = params.get("metadata") if isinstance(params.get("metadata"), dict) else {}
        info = params.get("model_info") if isinstance(params.get("model_info"), dict) else {}
        arms = self.router.arm_set(self.router.clock()).arms
        ids = {a.arm_id for a in arms}
        if info.get("id") in ids:
            return info["id"]
        group = metadata.get("model_group") or kwargs.get("model")
        return next((a.arm_id for a in arms if a.group == group), None)

    async def _outcome(self, kwargs: dict, ok: bool, start_time: Any, end_time: Any) -> None:
        if not shadow_enabled():
            return
        router = self.router
        try:
            arm_id = self._arm_of(kwargs)
            if arm_id is not None:
                exc = kwargs.get("original_exception") or kwargs.get("exception")
                try:
                    latency = float((end_time - start_time).total_seconds())
                except Exception:
                    latency = None
                router.observe_result(
                    arm_id, ok, call_id=str(kwargs.get("litellm_call_id") or ""),
                    exc=exc if isinstance(exc, BaseException) else None, latency_s=latency,
                )
            if self.watchdog is not None:
                self.watchdog.maybe_tick()
            if router.recorder.path:
                await asyncio.to_thread(router.recorder.flush)
        except Exception as exc:
            router._counters["errors"] += 1
            logger.warning("loci colony_router: could not record an outcome: %r", exc)

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        await self._outcome(kwargs, True, start_time, end_time)

    async def async_log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        await self._outcome(kwargs, False, start_time, end_time)


loci_colony_router = LociColonyRouter()
