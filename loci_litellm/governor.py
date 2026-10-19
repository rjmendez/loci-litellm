"""Governor: a spend guard for the LiteLLM proxy. Register `loci_litellm.governor.loci_governor` in the config.

What it does, per request, before the call:
  G8  a STOP file (checked at most once a second) answers 503 `stopped`; a PAUSE file only switches off learning
      and exploration (cheap routing is untouched). Health calls and ZERO-tier calls always pass.
  G1  a worst-case reservation (Decimal) is taken against a local-day cap, a rolling-24h cap, and each client's
      cap. The reservation is the most expensive deployment over the alias's whole retry/fallback chain, times
      (1 + retries); the actual cost replaces it when the call settles.
  G2  an hourly trigger downgrades; a higher hourly line rejects non-interactive callers.
Interactive callers are downgraded, never rejected, until the absolute stop (default 1.5 x the hard cap).
A tool-carrying request over a hard cap is rejected (HTTP 429, `reason` and the cap in the body); a tool-free one
is routed to the cheapest adequate arm if that fits under the cap, and rejected if it does not.

Switches (all default to doing nothing):
  LOCI_GOVERNOR=observe|enforce   anything other than `enforce` is observe. Observe never blocks, downgrades or
                                  edits a request; it records the decision it WOULD have taken (`would_block`).
                                  `enforce` only takes effect when the owner envelope is present, pinned and
                                  sets `[governor] enforce = true`; otherwise it stays observe and
                                  stats()["enforce_refused_by_envelope"] counts the refusals.
  LOCI_GOVERNOR_DIR               state directory (default ~/.loci/governor): counter.json, STOP, PAUSE,
                                  envelope.toml, envelope.pin, audit/. Nothing is read or written until the
                                  first request.
  LOCI_GOVERNOR_LEDGER            cost ledger used to seed the counter (default: the cost ledger's own path).
  LOCI_GOVERNOR_OWNER_KEY_FILE    file holding the owner's HMAC key for signed `envelope_pin` audit entries.

Failure policy: the spend path (G1), the concurrency path and the STOP/envelope path fail CLOSED: an exception
in them answers 503 `budget_gate_error` (never a pass-through), and an unreadable spend counter falls back to
a built-in provisional cap. Only optimisation steps fail open, each counted; more than five fail-open events in
60 s turn the Governor fail-closed (`governor_degraded`) until they age out. Degraded operation is always
visible in stats(). Observe mode records all of this as would-block events and lets the request through.
"""
from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, tzinfo
from decimal import ROUND_UP, Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union
from zoneinfo import ZoneInfo

from litellm.integrations.custom_logger import CustomLogger
from loci_litellm.audit_chain import AuditChain, AuditUnavailable
from loci_litellm.capability_gate import needs_tools
from loci_litellm.envelope import Caps, EnvelopeState, EnvelopeWatcher

logger = logging.getLogger("loci_litellm.governor")

try:  # the proxy always has fastapi; the offline tools may not
    from fastapi import HTTPException as _RejectionBase
except ImportError:  # pragma: no cover
    _RejectionBase = Exception  # type: ignore[assignment,misc]

MODE_OBSERVE = "observe"
MODE_ENFORCE = "enforce"
ANONYMOUS = "anonymous"
MTOK = Decimal(1_000_000)
MICRO = Decimal("0.000001")
BUCKET_S = 900                      # counter bucket width: 15 minutes
BUCKETS_24H = 96
BUCKETS_HOUR = 4
UNKNOWN_IN_PER_MTOK = Decimal("15")  # price assumed for a deployment whose model_info has none
UNKNOWN_OUT_PER_MTOK = Decimal("75")
DEFAULT_RETRIES = 2
INFLIGHT_TTL_S = 900.0
MAX_CLIENTS_IN_COUNTER = 16

_EXEMPT_CALL_TYPES = frozenset({"health", "health_check", "model_list", "list_models"})


def parse_mode(value: Optional[str]) -> str:
    """`enforce` (any case, trimmed) is enforce; anything else, including unset, is observe."""
    return MODE_ENFORCE if (value or "").strip().lower() == MODE_ENFORCE else MODE_OBSERVE


# ----------------------------------------------------------------------------- rejection
class GovernorRejection(_RejectionBase):  # type: ignore[valid-type,misc]
    """Raised from the pre-call hook; the proxy turns it into an HTTP response."""

    def __init__(self, status_code: int, code: str, message: str, cap: Optional[Decimal] = None,
                 headers: Optional[dict] = None) -> None:
        error: dict[str, Any] = {"type": "governor_rejection", "reason": code, "message": message}
        if cap is not None:
            error["cap"] = format(cap, "f")
        self.code = code
        self.cap = cap
        self.body = {"error": error}
        if _RejectionBase is Exception:
            super().__init__(message)
            self.status_code = status_code
            self.detail = self.body
            self.headers = headers
        else:
            super().__init__(status_code=status_code, detail=self.body, headers=headers)


# ----------------------------------------------------------------------------- time
def tz_from_name(name: str) -> Optional[tzinfo]:
    """None means the machine's local zone (DST-aware); otherwise an IANA zone."""
    if not name or name == "local":
        return None
    return ZoneInfo(name)


def local_day(ts: float, tz: Optional[tzinfo]) -> str:
    """The calendar day (YYYY-MM-DD) of an epoch time in `tz`, or in the machine's zone when tz is None."""
    moment = datetime.fromtimestamp(ts, tz) if tz is not None else datetime.fromtimestamp(ts)
    return moment.date().isoformat()


# ----------------------------------------------------------------------------- request sizing
def _text_len(content: Any) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                total += len(str(part.get("text") or "")) + (1500 if part.get("type") in ("image_url", "input_image", "image") else 0)
            elif isinstance(part, str):
                total += len(part)
        return total
    return 0


def estimate_prompt_tokens(data: dict) -> int:
    """A deliberately high estimate: one token per three characters plus a per-message overhead."""
    chars = 0
    for message in data.get("messages") or []:
        if isinstance(message, dict):
            chars += _text_len(message.get("content")) + 16
            if message.get("tool_calls"):
                chars += len(json.dumps(message["tool_calls"], default=str))
    for key in ("tools", "functions"):
        if data.get(key):
            chars += len(json.dumps(data[key], default=str))
    for key in ("input", "prompt"):
        value = data.get(key)
        if isinstance(value, (str, list)):
            chars += _text_len(value) if isinstance(value, str) else len(json.dumps(value, default=str))
    return (chars + 2) // 3


def sanitize_max_tokens(raw: Any, ceiling: int) -> int:
    """The completion size to reserve for: the request's max_tokens when it is a sane positive number, and
    `ceiling` when it is missing, malformed, non-positive or larger than the ceiling."""
    if isinstance(raw, bool) or raw is None:
        return ceiling
    value: Any = raw
    if isinstance(raw, str):
        try:
            value = int(raw.strip())
        except ValueError:
            return ceiling
    if isinstance(value, float):
        if not math.isfinite(value):
            return ceiling
        value = int(value)
    if not isinstance(value, int) or value < 1:
        return ceiling
    return min(value, ceiling)


# ----------------------------------------------------------------------------- arms and prices
@dataclass(frozen=True)
class Arm:
    arm_id: str
    model_name: str
    in_per_mtok: Optional[Decimal] = None     # None = model_info has no price: the unknown-price default applies
    out_per_mtok: Optional[Decimal] = None
    supports_tools: bool = False
    max_input_tokens: Optional[int] = None
    local: bool = False

    @property
    def price_in(self) -> Decimal:
        return UNKNOWN_IN_PER_MTOK if self.in_per_mtok is None else self.in_per_mtok

    @property
    def price_out(self) -> Decimal:
        return UNKNOWN_OUT_PER_MTOK if self.out_per_mtok is None else self.out_per_mtok

    @property
    def priced(self) -> bool:
        return self.in_per_mtok is not None and self.out_per_mtok is not None


UNKNOWN_ARM = Arm(arm_id="(unknown)", model_name="(unknown)")


def _per_mtok(value: Any) -> Optional[Decimal]:
    """A LiteLLM per-token price as dollars per million tokens, exactly."""
    if value is None or isinstance(value, bool):
        return None
    try:
        price = Decimal(str(value)) * MTOK
    except InvalidOperation:
        return None
    return price if price.is_finite() and price >= 0 else None


class Catalog:
    """The arms the proxy can route to, with fallback chains. Built from LiteLLM `model_list` entries so that
    prices come from model_info and nothing here knows a model or host name."""

    def __init__(self, arms: Sequence[Arm] = (), fallbacks: Optional[Mapping[str, Sequence[str]]] = None,
                 num_retries: int = DEFAULT_RETRIES, p95_completion: Optional[Mapping[str, int]] = None) -> None:
        self.arms = tuple(arms)
        self.fallbacks = {k: tuple(v) for k, v in (fallbacks or {}).items()}
        self.num_retries = num_retries
        self.p95_completion = dict(p95_completion or {})
        self._groups: dict[str, list[Arm]] = {}
        for arm in self.arms:
            self._groups.setdefault(arm.model_name, []).append(arm)

    @classmethod
    def from_model_list(cls, model_list: Iterable[Mapping[str, Any]],
                        fallbacks: Optional[Mapping[str, Sequence[str]]] = None,
                        num_retries: int = DEFAULT_RETRIES,
                        p95_completion: Optional[Mapping[str, int]] = None) -> "Catalog":
        from loci_litellm.cost_ledger import _is_local_deployment

        arms: list[Arm] = []
        for entry in model_list:
            if not isinstance(entry, Mapping) or not entry.get("model_name"):
                continue
            info = entry.get("model_info") if isinstance(entry.get("model_info"), Mapping) else {}
            params = entry.get("litellm_params") if isinstance(entry.get("litellm_params"), Mapping) else {}
            name = str(entry["model_name"])
            local = info.get("loci_local")
            if not isinstance(local, bool):
                local = _is_local_deployment(str(params.get("custom_llm_provider") or ""), str(params.get("model") or ""),
                                             str(params.get("api_base") or ""))
            ctx = info.get("max_input_tokens")
            arms.append(Arm(
                arm_id=str(info.get("id") or name), model_name=name,
                in_per_mtok=_per_mtok(info.get("input_cost_per_token", params.get("input_cost_per_token"))),
                out_per_mtok=_per_mtok(info.get("output_cost_per_token", params.get("output_cost_per_token"))),
                supports_tools=info.get("supports_function_calling") is True,
                max_input_tokens=ctx if isinstance(ctx, int) and not isinstance(ctx, bool) else None,
                local=bool(local)))
        return cls(arms, fallbacks, num_retries, p95_completion)

    def groups(self) -> dict[str, list[Arm]]:
        return self._groups

    def chain(self, model_name: str) -> list[Arm]:
        """Every deployment reachable from the alias: its own, then each fallback alias, transitively."""
        seen: list[str] = []
        queue = [model_name]
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            seen.append(name)
            queue.extend(self.fallbacks.get(name, ()))
        out: list[Arm] = []
        for name in seen:
            out.extend(self._groups.get(name, ()))
        return out

    def reservation(self, chain: Sequence[Arm], prompt_tokens: int, completion_tokens: int,
                    retries: Optional[int] = None) -> Decimal:
        """Worst case over the chain: the dearest arm's cost for (prompt, completion), times 1 + retries attempts.
        An empty chain is priced at the unknown-price default. Rounded up to a millionth of a dollar."""
        attempts = 1 + max(0, self.num_retries if retries is None else retries)
        arms = list(chain) or [UNKNOWN_ARM]
        worst = Decimal(0)
        for arm in arms:
            completion = completion_tokens
            cap = self.p95_completion.get(arm.arm_id)
            if cap is not None:
                completion = min(completion, max(0, cap))
            cost = (Decimal(prompt_tokens) * arm.price_in + Decimal(completion) * arm.price_out) / MTOK
            worst = max(worst, cost)
        return (worst * attempts).quantize(MICRO, rounding=ROUND_UP)


# ----------------------------------------------------------------------------- decision
@dataclass(frozen=True)
class Request:
    model: str
    prompt_tokens: int = 0
    max_tokens: Any = None
    has_tools: bool = False
    client: str = ANONYMOUS
    interactive: bool = False
    rw: bool = False
    client_cap: Optional[Decimal] = None


@dataclass(frozen=True)
class Snapshot:
    """Spend so far, with the reservations of calls in flight already included."""
    day_total: Decimal = Decimal(0)
    rolling_24h: Decimal = Decimal(0)
    hour: Decimal = Decimal(0)
    client_day: Decimal = Decimal(0)
    degraded: bool = False
    hc_day: Decimal = Decimal(0)
    hc_calls_hour: int = 0


@dataclass(frozen=True)
class Decision:
    action: str                       # allow | downgrade | reject
    code: str = ""                    # why, when action is not a clean allow
    status: int = 200
    cap: Optional[Decimal] = None
    reservation: Decimal = Decimal(0)
    target_model: str = ""
    target_arm: str = ""
    notes: tuple[str, ...] = ()
    explore_allowed: bool = True

    @property
    def would_block(self) -> bool:
        return self.action != "allow"


@dataclass(frozen=True)
class Candidate:
    model_name: str
    arm_id: str
    reservation: Decimal


def _group_adequate(arms: Sequence[Arm], req: Request, prompt_tokens: int) -> bool:
    if req.has_tools and not all(a.supports_tools for a in arms):
        return False
    return all(a.max_input_tokens is None or a.max_input_tokens >= prompt_tokens for a in arms)


def cheapest_adequate(catalog: Catalog, req: Request, caps: Caps, *, exclude: Sequence[str] = ()) -> Optional[Candidate]:
    """The alias with the lowest worst-case reservation that can serve the request (tools, context)."""
    completion = sanitize_max_tokens(req.max_tokens, caps.max_completion_ceiling)
    best: Optional[Candidate] = None
    for name, arms in catalog.groups().items():
        if name in exclude or not _group_adequate(arms, req, req.prompt_tokens):
            continue
        cost = catalog.reservation(catalog.chain(name), req.prompt_tokens, completion)
        if best is None or cost < best.reservation:
            cheapest_arm = min(arms, key=lambda a: a.price_in + a.price_out)
            best = Candidate(name, cheapest_arm.arm_id, cost)
    return best


def decide(req: Request, snap: Snapshot, caps: Caps, catalog: Catalog) -> Decision:
    """The pure spend decision. No clock, no files, no environment: everything it needs is an argument."""
    completion = sanitize_max_tokens(req.max_tokens, caps.max_completion_ceiling)
    chain = catalog.chain(req.model)
    reservation = catalog.reservation(chain, req.prompt_tokens, completion)
    notes: list[str] = []
    hard_day = caps.day_hard
    if snap.degraded:
        hard_day = min(hard_day, caps.provisional_day)
        notes += ["ledger_degraded", "provisional_cap"]
    soft_day = min(caps.day_soft, hard_day)
    own_arm = chain[0].arm_id if chain else UNKNOWN_ARM.arm_id

    def hard_violation(amount: Decimal) -> Optional[tuple[str, Decimal]]:
        if snap.day_total + amount > hard_day:
            return "budget_day", hard_day
        if snap.rolling_24h + amount > caps.rolling_24h:
            return "budget_rolling", caps.rolling_24h
        if not req.interactive and snap.hour + amount > caps.hour_reject:
            return "budget_hour", caps.hour_reject
        if req.client_cap is not None and snap.client_day + amount > req.client_cap:
            return "budget_client", req.client_cap
        return None

    def soft_violation(amount: Decimal) -> Optional[tuple[str, Decimal]]:
        if snap.day_total + amount > soft_day:
            return "budget_soft", soft_day
        if snap.hour + amount > caps.hour_downgrade:
            return "budget_hour_downgrade", caps.hour_downgrade
        return None

    def allow(extra: Sequence[str] = (), explore: bool = True) -> Decision:
        return Decision("allow", "", 200, None, reservation, req.model, own_arm, tuple(notes) + tuple(extra), explore)

    def downgrade(code: str, cap: Decimal, cand: Candidate) -> Decision:
        return Decision("downgrade", code, 200, cap, cand.reservation, cand.model_name, cand.arm_id, tuple(notes), False)

    def reject(code: str, cap: Decimal, status: int = 429) -> Decision:
        return Decision("reject", code, status, cap, reservation, req.model, own_arm, tuple(notes), False)

    if req.interactive and snap.day_total + reservation > caps.interactive_stop:
        return reject("budget_abs_stop", caps.interactive_stop)

    cand = cheapest_adequate(catalog, req, caps)
    cheaper = cand if cand is not None and cand.model_name != req.model and cand.reservation < reservation else None

    violation = hard_violation(reservation)
    if violation is not None:
        code, cap = violation
        if req.interactive:
            return downgrade(code, cap, cheaper) if cheaper else allow(["no_cheaper_arm"], explore=False)
        if req.has_tools:
            return reject(code, cap)
        if cheaper is not None and hard_violation(cheaper.reservation) is None:
            return downgrade(code, cap, cheaper)
        return reject(code, cap)
    soft = soft_violation(reservation)
    if soft is not None:
        code, cap = soft
        return downgrade(code, cap, cheaper) if cheaper else allow(["no_cheaper_arm"], explore=False)
    return allow()


# ----------------------------------------------------------------------------- spend counter
def _dec(value: Any) -> Decimal:
    amount = Decimal(str(value))
    if not amount.is_finite():
        raise InvalidOperation("not finite")
    return amount


def _fmt(value: Decimal) -> str:
    return format(value.quantize(MICRO, rounding=ROUND_UP), "f")


class SpendCounter:
    """The Governor's own spend counter. A small JSON file (written through a temp file and a rename, under 4 KiB)
    holds the local-day total, per-client day totals and 15-minute buckets for the rolling windows. It is seeded
    from the cost ledger and its rotated `.1` when absent or corrupt, and at start the larger of counter and
    ledger wins per bucket, so a rotated ledger can never lower it. If neither the counter nor the ledger can be
    read the counter runs in memory only and `degraded` is set (the caller then applies a provisional cap)."""

    def __init__(self, path: str, ledger_file: Optional[str] = None, *, clock: Callable[[], float] = time.time,
                 tz: Optional[tzinfo] = None, inflight_ttl_s: float = INFLIGHT_TTL_S) -> None:
        self._path = Path(path)
        self._ledger = ledger_file
        self._clock = clock
        self._tz = tz
        self._ttl = inflight_ttl_s
        self._lock = threading.RLock()
        self._loaded = False
        self.degraded = False
        self._day = ""
        self._day_total = Decimal(0)
        self._clients: dict[str, Decimal] = {}
        self._hc_day = Decimal(0)
        self._buckets: dict[int, Decimal] = {}
        self._hc_calls: dict[int, int] = {}
        self._inflight: dict[str, tuple[Decimal, str, float]] = {}
        self.denials: dict[str, int] = {}
        self.reservations_taken = 0
        self.save_failures = 0
        self.ledger_bad_rows = 0
        self.seeded_from_ledger = 0
        self.settled_at_reservation = 0

    # -- loading
    def _read_counter(self) -> Optional[dict]:
        """The parsed counter file; None when absent; ValueError when it exists but is unusable."""
        try:
            text = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ValueError(f"counter unreadable: {exc!r}") from exc
        try:
            doc = json.loads(text)
            if not isinstance(doc, dict) or doc.get("v") != 1:
                raise ValueError("bad counter schema")
            return {
                "day": str(doc["day"]), "day_total": _dec(doc["day_total"]),
                "clients": {str(k): _dec(v) for k, v in doc.get("clients", {}).items()},
                "hc_day": _dec(doc.get("hc_day", "0")),
                "buckets": {int(k): _dec(v) for k, v in doc.get("buckets", {}).items()},
                "hc_calls": {int(k): int(v) for k, v in doc.get("hc_calls", {}).items()},
            }
        except (ValueError, KeyError, TypeError, AttributeError, InvalidOperation) as exc:
            raise ValueError(f"counter corrupt: {exc!r}") from exc

    def _read_ledger(self, now: float, today: str) -> Optional[tuple[dict[int, Decimal], Decimal]]:
        """(buckets, today's total) from the ledger and its `.1`; None when it exists but cannot be read."""
        if not self._ledger:
            return None
        buckets: dict[int, Decimal] = {}
        day_total = Decimal(0)
        oldest = now - (BUCKETS_24H + 1) * BUCKET_S
        for candidate in (self._ledger + ".1", self._ledger):
            try:
                handle = open(candidate, "r", encoding="utf-8")
            except FileNotFoundError:
                continue
            except OSError:
                return None
            try:
                with handle:
                    for line in handle:
                        try:
                            row = json.loads(line)
                            ts, cost = float(row["ts"]), _dec(row.get("cost_usd") or 0)
                            if cost < 0:
                                raise ValueError("negative cost")
                        except (ValueError, KeyError, TypeError, InvalidOperation):
                            self.ledger_bad_rows += 1
                            continue
                        if local_day(ts, self._tz) == today:
                            day_total += cost
                        if ts >= oldest:
                            key = int(ts // BUCKET_S)
                            buckets[key] = buckets.get(key, Decimal(0)) + cost
            except (OSError, UnicodeDecodeError):
                return None
        return buckets, day_total

    def _ensure(self) -> None:
        if self._loaded:
            return
        now = float(self._clock())
        today = local_day(now, self._tz)
        counter: Optional[dict] = None
        counter_bad = False
        try:
            counter = self._read_counter()
        except ValueError:
            counter_bad = True
        ledger = self._read_ledger(now, today)
        if counter is None and ledger is None and (counter_bad or self._ledger):
            # nothing readable to start from: an absent ledger counts as readable (zero rows), so this means
            # an unreadable ledger, or a damaged counter with no ledger configured
            self.degraded = True
        if counter is not None:
            if counter["day"] == today:
                self._day_total = counter["day_total"]
                self._clients = counter["clients"]
                self._hc_day = counter["hc_day"]
            self._buckets = counter["buckets"]
            self._hc_calls = counter["hc_calls"]
        self._day = today
        if ledger is not None:
            lbuckets, lday = ledger
            for key, value in lbuckets.items():
                self._buckets[key] = max(self._buckets.get(key, Decimal(0)), value)
            self._day_total = max(self._day_total, lday)
            if counter is None:
                self.seeded_from_ledger += 1
        self._loaded = True

    # -- bookkeeping
    def _roll_day(self, now: float) -> None:
        today = local_day(now, self._tz)
        if today != self._day:
            self._day, self._day_total, self._clients, self._hc_day = today, Decimal(0), {}, Decimal(0)
        oldest = int(now // BUCKET_S) - BUCKETS_24H
        for key in [k for k in self._buckets if k < oldest]:
            del self._buckets[key]
        for key in [k for k in self._hc_calls if k < oldest]:
            del self._hc_calls[key]

    def _sweep(self, now: float) -> None:
        for call_id in [c for c, (_, _, ts) in self._inflight.items() if now - ts > self._ttl]:
            del self._inflight[call_id]

    def snapshot(self, now: float, client: str = ANONYMOUS) -> Snapshot:
        with self._lock:
            self._ensure()
            self._roll_day(now)
            self._sweep(now)
            bucket = int(now // BUCKET_S)
            inflight = sum((amount for amount, _, _ in self._inflight.values()), Decimal(0))
            mine = sum((amount for amount, who, _ in self._inflight.values() if who == client), Decimal(0))
            rolling = sum((v for k, v in self._buckets.items() if bucket - BUCKETS_24H + 1 <= k <= bucket), Decimal(0))
            hour = sum((v for k, v in self._buckets.items() if bucket - BUCKETS_HOUR + 1 <= k <= bucket), Decimal(0))
            hc_calls = sum(n for k, n in self._hc_calls.items() if bucket - BUCKETS_HOUR + 1 <= k <= bucket)
            return Snapshot(day_total=self._day_total + inflight, rolling_24h=rolling + inflight, hour=hour + inflight,
                            client_day=self._clients.get(client, Decimal(0)) + mine, degraded=self.degraded,
                            hc_day=self._hc_day, hc_calls_hour=hc_calls)

    def reserve(self, call_id: str, amount: Decimal, now: float, client: str = ANONYMOUS, hc: bool = False) -> None:
        with self._lock:
            self._ensure()
            self._inflight[call_id] = (amount, client, now)
            self.reservations_taken += 1
            if hc:
                key = int(now // BUCKET_S)
                self._hc_calls[key] = self._hc_calls.get(key, 0) + 1

    def reserved(self, call_id: str) -> Optional[Decimal]:
        with self._lock:
            held = self._inflight.get(call_id)
            return None if held is None else held[0]

    def settle(self, call_id: str, actual: Optional[Decimal], now: float, client: str = ANONYMOUS,
               final: bool = True, hc: bool = False) -> Decimal:
        """Add one attempt's actual cost. `actual` None (the provider reported none) books the reservation
        instead. `final` also drops the reservation. Returns the amount booked."""
        with self._lock:
            self._ensure()
            held = self._inflight.get(call_id)
            booked = actual
            if booked is None:
                booked = held[0] if held is not None else Decimal(0)
                self.settled_at_reservation += 1
            if held is not None:
                client = held[1]
                if final:
                    del self._inflight[call_id]
            self._roll_day(now)
            self._day_total += booked
            self._clients[client] = self._clients.get(client, Decimal(0)) + booked
            if hc:
                self._hc_day += booked
            key = int(now // BUCKET_S)
            self._buckets[key] = self._buckets.get(key, Decimal(0)) + booked
            self._save()
            return booked

    def release(self, call_id: str) -> None:
        with self._lock:
            self._inflight.pop(call_id, None)

    def record_denial(self, code: str) -> None:
        with self._lock:
            self.denials[code] = self.denials.get(code, 0) + 1

    def _save(self) -> None:
        clients = dict(sorted(self._clients.items(), key=lambda kv: kv[1], reverse=True)[:MAX_CLIENTS_IN_COUNTER])
        doc = {"v": 1, "day": self._day, "day_total": _fmt(self._day_total),
               "clients": {k: _fmt(v) for k, v in clients.items()}, "hc_day": _fmt(self._hc_day),
               "buckets": {str(k): _fmt(v) for k, v in sorted(self._buckets.items())},
               "hc_calls": {str(k): n for k, n in sorted(self._hc_calls.items())}}
        tmp = self._path.with_name(self._path.name + ".tmp")
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
            os.replace(tmp, self._path)
        except OSError as exc:
            self.save_failures += 1
            logger.warning("loci governor: counter not saved (%d failures): %r", self.save_failures, exc)

    def stats(self) -> dict:
        with self._lock:
            return {"degraded": self.degraded, "inflight": len(self._inflight),
                    "reservations_taken": self.reservations_taken, "denials": dict(self.denials),
                    "save_failures": self.save_failures, "ledger_bad_rows": self.ledger_bad_rows,
                    "seeded_from_ledger": self.seeded_from_ledger,
                    "settled_at_reservation": self.settled_at_reservation}


# ----------------------------------------------------------------------------- STOP / PAUSE, breaker
class StopFiles:
    """STOP and PAUSE marker files, each looked at no more than once per `cache_s` (default 1 s). A marker whose
    state cannot be determined (a stat error other than 'not there') counts as present: fail closed."""

    def __init__(self, stop_path: str, pause_path: str, *, clock: Callable[[], float] = time.time,
                 cache_s: float = 1.0) -> None:
        self._paths = {"stop": Path(stop_path), "pause": Path(pause_path)}
        self._clock = clock
        self._cache_s = cache_s
        self._seen: dict[str, tuple[float, bool]] = {}
        self.stat_errors = 0

    def _present(self, which: str) -> bool:
        now = self._clock()
        cached = self._seen.get(which)
        if cached is not None and now - cached[0] < self._cache_s:
            return cached[1]
        try:
            self._paths[which].stat()
            present = True
        except FileNotFoundError:
            present = False
        except OSError:
            self.stat_errors += 1
            present = True
        self._seen[which] = (now, present)
        return present

    def stopped(self) -> bool:
        return self._present("stop")

    def paused(self) -> bool:
        return self._present("pause")


class FailOpenBreaker:
    """Counts fail-open events; more than `limit` inside `window_s` makes the Governor fail closed."""

    def __init__(self, clock: Callable[[], float] = time.time, window_s: float = 60.0, limit: int = 5) -> None:
        self._clock = clock
        self._window = window_s
        self._limit = limit
        self._events: list[float] = []
        self.total = 0

    def _trim(self) -> None:
        cutoff = self._clock() - self._window
        self._events = [t for t in self._events if t > cutoff]

    def record(self) -> None:
        self._trim()
        self._events.append(self._clock())
        self.total += 1

    def tripped(self) -> bool:
        self._trim()
        return len(self._events) > self._limit


# ----------------------------------------------------------------------------- the governor
Optimiser = Callable[[dict, "Context"], None]


@dataclass
class Context:
    """What an optimisation step may look at while the Governor processes one request."""
    mode: str
    client: str
    notes: list = field(default_factory=list)


def _metadata(data: dict) -> dict:
    metadata = data.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        data["metadata"] = metadata
    return metadata


def _headers(data: dict) -> dict:
    request = data.get("proxy_server_request")
    headers = request.get("headers") if isinstance(request, dict) else None
    return {str(k).lower(): v for k, v in headers.items()} if isinstance(headers, dict) else {}


class Governor:
    def __init__(self, *, store: Any, envelope: Callable[[], EnvelopeState], catalog: Callable[[], Catalog],
                 stop: StopFiles, audit: Optional[AuditChain] = None,
                 mode: Union[str, Callable[[], str]] = MODE_OBSERVE, clock: Callable[[], float] = time.time,
                 optimisers: Sequence[Optimiser] = (), breaker: Optional[FailOpenBreaker] = None,
                 event_sink: Optional[Callable[[dict], None]] = None) -> None:
        self._store = store
        self._envelope = envelope
        self._catalog = catalog
        self._stop = stop
        self._audit = audit
        self._mode = mode
        self._clock = clock
        self._optimisers = list(optimisers)
        self._breaker = breaker or FailOpenBreaker(clock)
        self._sink = event_sink
        self._counts: dict[str, int] = {}
        self._enforcing = False     # sticky: once enforcing, a later envelope fault keeps enforcing on the defaults

    # -- small helpers
    def _count(self, name: str, n: int = 1) -> None:
        self._counts[name] = self._counts.get(name, 0) + n

    def requested_mode(self) -> str:
        return parse_mode(self._mode() if callable(self._mode) else self._mode)

    def effective_mode(self, env: EnvelopeState) -> str:
        """Enforce needs the request (env var) AND a trusted, pinned envelope that says enforce = true. Once this
        process has enforced, an envelope that later fails its check (tampered, missing, unpinned) does not
        switch enforcement off: it keeps enforcing on the built-in default caps, with the alarm in the notes."""
        if self.requested_mode() != MODE_ENFORCE:
            self._enforcing = False
            return MODE_OBSERVE
        if env.trusted and env.enforce:
            self._enforcing = True
            return MODE_ENFORCE
        if self._enforcing:
            self._count("enforce_on_defaults")
            return MODE_ENFORCE
        self._count("enforce_refused_by_envelope")
        return MODE_OBSERVE

    def paused(self) -> bool:
        return self._stop.paused()

    def learning_allowed(self) -> bool:
        """PAUSE and STOP both halt learning, background agents and exploration; neither touches cheap routing."""
        return not (self._stop.paused() or self._stop.stopped())

    def stats(self) -> dict:
        out: dict[str, Any] = dict(self._counts)
        out["fail_open_events"] = self._breaker.total
        out["fail_closed_breaker"] = self._breaker.tripped()
        out["stop_stat_errors"] = getattr(self._stop, "stat_errors", 0)
        if hasattr(self._store, "stats"):
            out["store"] = self._store.stats()
        return out

    def _emit(self, event: dict) -> bool:
        """Log and audit one event. Returns False when an audit write the caller depends on failed."""
        logger.info("loci governor %s", json.dumps(event, sort_keys=True, default=str))
        if self._sink is not None:
            try:
                self._sink(event)
            except Exception as exc:   # an observer must not break the request path
                self._count("event_sink_errors")
                logger.warning("loci governor: event sink failed: %r", exc)
        if self._audit is None:
            return True
        try:
            self._audit.append(str(event.get("event", "event")), event)
            return True
        except AuditUnavailable as exc:
            self._count("audit_failures")
            logger.warning("loci governor: audit unavailable: %r", exc)
            return False

    # -- request side
    def _build_request(self, data: dict, env: EnvelopeState, catalog: Catalog) -> Request:
        metadata = _metadata(data)
        return Request(model=str(data.get("model") or ""), prompt_tokens=estimate_prompt_tokens(data),
                       max_tokens=data.get("max_tokens", data.get("max_completion_tokens")),
                       has_tools=needs_tools(data), client=ANONYMOUS, interactive=False,
                       rw=metadata.get("loci_rw") is True)

    def _is_exempt(self, data: dict, call_type: Any) -> bool:
        if str(call_type or "") in _EXEMPT_CALL_TYPES:
            return True
        metadata = data.get("metadata")
        return isinstance(metadata, dict) and str(metadata.get("loci_tier") or "").upper() == "ZERO"

    def process(self, data: dict, call_type: Any = None) -> Optional[GovernorRejection]:
        """Decide for one request. Edits `data` only when enforcing (a downgrade) and for bookkeeping metadata.
        Returns the rejection to raise, or None to let the request go on."""
        if not isinstance(data, dict):
            return None
        self._count("calls")
        if self._is_exempt(data, call_type):
            self._count("exempt")
            return None
        now = float(self._clock())
        env = self._envelope()
        mode = self.effective_mode(env)
        call_id = uuid.uuid4().hex
        req: Optional[Request] = None
        decision: Decision
        try:
            if self._stop.stopped():
                decision = Decision("reject", "stopped", 503, notes=("stop_file",), explore_allowed=False)
            elif self._breaker.tripped():
                decision = Decision("reject", "governor_degraded", 503, notes=("fail_open_breaker",), explore_allowed=False)
            else:
                ctx = Context(mode=mode, client=ANONYMOUS)
                for optimiser in self._optimisers:
                    try:
                        optimiser(data, ctx)
                    except Exception as exc:     # optimisation only: fail open, counted
                        self._breaker.record()
                        self._count("fail_open_events")
                        logger.warning("loci governor: optimiser failed open: %r", exc)
                if self._breaker.tripped():
                    decision = Decision("reject", "governor_degraded", 503, notes=("fail_open_breaker",), explore_allowed=False)
                else:
                    catalog = self._catalog()
                    req = self._build_request(data, env, catalog)
                    snap = self._store.snapshot(now, req.client)
                    decision = decide(req, snap, env.caps, catalog)
                    if env.level != "L0":
                        decision = replace(decision, notes=decision.notes + (f"envelope_{env.alarm}",))
        except Exception as exc:                  # G1/G7/G8 fail closed
            self._count("fail_closed_events")
            logger.warning("loci governor: spend gate error: %r", exc)
            decision = Decision("reject", "budget_gate_error", 503, notes=(type(exc).__name__,), explore_allowed=False)
        return self._apply(data, decision, req, mode, call_id, now)

    def _apply(self, data: dict, decision: Decision, req: Optional[Request], mode: str, call_id: str,
               now: float) -> Optional[GovernorRejection]:
        metadata = _metadata(data)
        rw = bool(req and req.rw)
        client = req.client if req else ANONYMOUS
        if decision.action != "allow":
            event = {"event": "would_block" if mode == MODE_OBSERVE else "block" if decision.action == "reject" else "downgrade",
                     "code": decision.code, "action": decision.action, "status": decision.status,
                     "cap": None if decision.cap is None else format(decision.cap, "f"),
                     "reservation": format(decision.reservation, "f"), "model": data.get("model"),
                     "target": decision.target_model, "client": client, "mode": mode, "notes": list(decision.notes)}
            self._count("would_block" if mode == MODE_OBSERVE else "enforced_" + decision.action)
            self._store.record_denial(decision.code)
            audited = self._emit(event)
        else:
            audited = True
            if rw:
                audited = self._emit({"event": "call", "client": client, "model": data.get("model"),
                                      "reservation": format(decision.reservation, "f"), "mode": mode})
        if rw and not audited:
            self._count("audit_unavailable")
            if mode == MODE_ENFORCE:
                return GovernorRejection(503, "audit_unavailable", "the audit trail is unavailable for this client",
                                         headers={"Retry-After": "30"})
        metadata["loci_governor_mode"] = mode
        if mode == MODE_ENFORCE:
            if decision.action == "reject":
                return GovernorRejection(decision.status, decision.code, _message(decision), decision.cap,
                                         headers={"Retry-After": "30"} if decision.status == 503 else None)
            if decision.action == "downgrade":
                metadata["loci_governor_downgraded_from"] = data.get("model")
                data["model"] = decision.target_model
            metadata["loci_explore_allowed"] = decision.explore_allowed
        if self._stop.paused():
            metadata["loci_paused"] = True
        # Observe mode lets a would-block request through, but still books its spend (reservation, then actual).
        metadata["loci_governor_id"] = call_id
        try:
            self._store.reserve(call_id, decision.reservation, now, client)
        except Exception as exc:
            self._count("fail_closed_events")
            logger.warning("loci governor: reservation failed: %r", exc)
            if mode == MODE_ENFORCE:
                return GovernorRejection(503, "budget_gate_error", "the spend gate could not reserve",
                                         headers={"Retry-After": "30"})
        return None

    # -- settle side
    def settle(self, call_id: str, actual: Optional[Decimal], final: bool) -> None:
        try:
            self._store.settle(call_id, actual, float(self._clock()), ANONYMOUS, final)
        except Exception as exc:
            self._count("settle_errors")
            logger.warning("loci governor: could not settle %s: %r", call_id, exc)

    def release(self, call_id: str) -> None:
        try:
            self._store.release(call_id)
        except Exception as exc:
            self._count("settle_errors")
            logger.warning("loci governor: could not release %s: %r", call_id, exc)


def _message(decision: Decision) -> str:
    cap = "" if decision.cap is None else f" (cap {format(decision.cap, 'f')})"
    return f"request refused by the spend governor: {decision.code}{cap}"


def _call_id_of(kwargs: Mapping[str, Any]) -> str:
    for source in (kwargs.get("litellm_params"), (kwargs.get("standard_logging_object") or {})):
        metadata = source.get("metadata") if isinstance(source, Mapping) else None
        if isinstance(metadata, Mapping) and metadata.get("loci_governor_id"):
            return str(metadata["loci_governor_id"])
    return ""


def _cost_of(kwargs: Mapping[str, Any]) -> Optional[Decimal]:
    slo = kwargs.get("standard_logging_object")
    raw = slo.get("response_cost") if isinstance(slo, Mapping) else None
    if raw is None:
        raw = kwargs.get("response_cost")
    if raw is None:
        return None
    try:
        amount = Decimal(str(raw))
    except InvalidOperation:
        return None
    return amount if amount.is_finite() and amount >= 0 else None


class LociGovernor(CustomLogger):
    """The LiteLLM callback. Builds its parts from the environment on first use."""

    def __init__(self, governor: Optional[Governor] = None) -> None:
        super().__init__()
        self._governor = governor
        self._build_lock = threading.Lock()

    @property
    def governor(self) -> Governor:
        if self._governor is None:
            with self._build_lock:
                if self._governor is None:
                    self._governor = build_governor_from_env()
        return self._governor

    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> Any:
        rejection = self.governor.process(data, call_type)
        if rejection is not None:
            raise rejection
        return None

    async def async_log_success_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        call_id = _call_id_of(kwargs)
        if call_id:
            self.governor.settle(call_id, _cost_of(kwargs), final=True)

    async def async_log_failure_event(self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any) -> None:
        call_id = _call_id_of(kwargs)
        if call_id:
            self.governor.settle(call_id, _cost_of(kwargs) or Decimal(0), final=False)

    async def async_post_call_failure_hook(self, request_data: dict, original_exception: Exception,
                                           user_api_key_dict: Any, traceback_str: Optional[str] = None) -> Any:
        metadata = request_data.get("metadata") if isinstance(request_data, dict) else None
        call_id = metadata.get("loci_governor_id") if isinstance(metadata, dict) else None
        if call_id:
            self.governor.release(str(call_id))
        return None


def _state_dir() -> Path:
    return Path(os.environ.get("LOCI_GOVERNOR_DIR") or (Path.home() / ".loci" / "governor"))


def _litellm_catalog() -> Catalog:
    """The live router's model list (a fresh read every minute is not needed: the router holds it in memory)."""
    import sys

    proxy_server = sys.modules.get("litellm.proxy.proxy_server")      # not loaded outside the proxy: no router
    router = getattr(proxy_server, "llm_router", None)
    if router is None:
        return Catalog()
    settings = getattr(router, "fallbacks", None)
    fallbacks: dict[str, list[str]] = {}
    for item in settings or []:
        if isinstance(item, dict):
            for key, value in item.items():
                fallbacks[str(key)] = [str(v) for v in value]
    return Catalog.from_model_list(router.model_list, fallbacks, int(getattr(router, "num_retries", DEFAULT_RETRIES) or 0))


def build_governor_from_env(clock: Callable[[], float] = time.time) -> Governor:
    from loci_litellm.backend_resolver import ledger_path

    base = _state_dir()
    audit = AuditChain(str(base / "audit"))
    key_file = os.environ.get("LOCI_GOVERNOR_OWNER_KEY_FILE")
    owner_key: Optional[bytes] = None
    if key_file:
        try:
            owner_key = Path(key_file).read_bytes().strip() or None
        except OSError:
            owner_key = None
    watcher = EnvelopeWatcher(str(base / "envelope.toml"), str(base / "envelope.pin"), audit, owner_key, clock=clock)
    ledger = os.environ.get("LOCI_GOVERNOR_LEDGER") or ledger_path()
    holder: dict[str, Any] = {}

    def envelope() -> EnvelopeState:
        state = watcher.state()
        if "store" not in holder or holder.get("tz") != state.day_timezone:
            try:
                tz = tz_from_name(state.day_timezone)
            except Exception:
                tz = None
            holder["tz"] = state.day_timezone
            holder["store"] = SpendCounter(str(base / "counter.json"), ledger, clock=clock, tz=tz)
        return state

    class _LazyStore:
        def __getattr__(self, name: str) -> Any:
            envelope()
            return getattr(holder["store"], name)

    return Governor(store=_LazyStore(), envelope=envelope, catalog=_litellm_catalog,
                    stop=StopFiles(str(base / "STOP"), str(base / "PAUSE"), clock=clock), audit=audit,
                    mode=lambda: os.environ.get("LOCI_GOVERNOR", ""), clock=clock)


loci_governor = LociGovernor()
