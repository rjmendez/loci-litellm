"""Channels: bounded, lazily decaying health signals per routing arm (trail, alarm, resource).

A channel is a number that decays exponentially toward a rest value with a fixed half-life and moves
only by bounded steps:

    x' = clamp(rest + (x - rest) * 0.5 ** (dt / half_life) + clip(deposit, -0.18, +0.18), floor, cap)

Bounds: trail <= 0.92, alarm <= 0.88, resource >= 0.08. Decay is computed lazily when a channel is
read or deposited into; a clock that runs backwards never amplifies a value (dt is clipped at 0).

Only transient provider failures (408, 429, 5xx, timeouts, connection errors) deposit alarm. Permanent
errors (400, 401, 403, 404 ...) say nothing about provider health and deposit nothing.

An arm is only reported unhealthy on a quorum: at least 3 transient failures from at least 2 distinct
callers inside 120 s. One noisy caller cannot take a provider out of the safe set. A starved arm gets
at most one probe per interval.

Everything here is in memory, bounded (arms, failures per arm) and takes `now` from the caller, so it
is deterministic under test. Nothing in this module routes a request.
"""
from __future__ import annotations

import json
import math
from collections import OrderedDict, deque
from typing import Any, Optional

from loci_litellm.circuit_monitor import _is_transient

STEP_CLAMP = 0.18
TRAIL_CAP = 0.92
ALARM_CAP = 0.88
RESOURCE_FLOOR = 0.08

# Half-lives in seconds. Proposals: they are frozen by the preregistration, not tuned online.
HALF_LIFE_S = {"alarm": 180.0, "latency": 720.0, "cost": 3600.0, "quality": 43200.0}

QUORUM_FAILURES = 3
QUORUM_CALLERS = 2
QUORUM_WINDOW_S = 120.0
PROBE_AFTER_S = 120.0

ALARM_GAIN = 0.3  # a transient failure asks for this much; the step clamp bounds it to 0.18
TRAIL_GAIN = 0.1  # a success asks for this much

MAX_ARMS = 256
_MAX_FAILURES_KEPT = 64


def decay_factor(dt: float, half_life_s: float) -> float:
    """Fraction of a deviation that remains after `dt` seconds: exactly 0.5 at one half-life.

    A negative `dt` (clock stepped back) is treated as 0, so a value never grows by being read."""
    if not (half_life_s > 0.0) or not math.isfinite(half_life_s):
        raise ValueError(f"half_life_s must be a positive finite number, got {half_life_s!r}")
    return 0.5 ** (max(dt, 0.0) / half_life_s)


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


class Channel:
    """One decaying value with a floor, a cap and a rest value it relaxes toward."""

    __slots__ = ("floor", "cap", "half_life_s", "rest", "_value", "_updated")

    def __init__(
        self, floor: float, cap: float, half_life_s: float, rest: float = 0.0,
        value: Optional[float] = None, now: float = 0.0,
    ) -> None:
        if floor > cap:
            raise ValueError(f"floor {floor!r} is above cap {cap!r}")
        decay_factor(0.0, half_life_s)  # validates the half-life once
        self.floor = floor
        self.cap = cap
        self.half_life_s = half_life_s
        self.rest = rest
        self._value = clamp(rest if value is None else value, floor, cap)
        self._updated = now

    def value_at(self, now: float) -> float:
        """The decayed value at `now`. Does not change the channel."""
        return self.rest + (self._value - self.rest) * decay_factor(now - self._updated, self.half_life_s)

    def deposit(self, amount: float, now: float) -> float:
        """Decay to `now`, add `amount` limited to +-STEP_CLAMP, clamp to the bounds, return the new value."""
        if not math.isfinite(amount):
            raise ValueError(f"deposit must be finite, got {amount!r}")
        step = clamp(amount, -STEP_CLAMP, STEP_CLAMP)
        self._value = clamp(self.value_at(now) + step, self.floor, self.cap)
        self._updated = max(self._updated, now)
        return self._value


def is_transient(exc: Optional[BaseException] = None, status: Optional[int] = None) -> bool:
    """Is this failure evidence about provider health? 408, 429 and 5xx are; other 4xx are not.
    Without a status code the exception is classified like the circuit monitor does."""
    if status is None and exc is not None:
        status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if status in (408, 429) or status >= 500:
            return True
        if 400 <= status < 500:
            return False
    return exc is not None and _is_transient(exc)


class ArmChannels:
    """Trail (what has worked), alarm (what has been failing) and resource for one arm."""

    __slots__ = ("trail", "alarm", "resource")

    def __init__(self, now: float = 0.0) -> None:
        self.trail = Channel(0.0, TRAIL_CAP, HALF_LIFE_S["quality"], rest=0.0, now=now)
        self.alarm = Channel(0.0, ALARM_CAP, HALF_LIFE_S["alarm"], rest=0.0, now=now)
        self.resource = Channel(RESOURCE_FLOOR, 1.0, HALF_LIFE_S["cost"], rest=1.0, now=now)

    def values(self, now: float) -> dict:
        return {
            "trail": round(self.trail.value_at(now), 6),
            "alarm": round(self.alarm.value_at(now), 6),
            "resource": round(self.resource.value_at(now), 6),
        }


class ChannelBoard:
    """Channels, failure quorum and starvation probes for every arm seen. Bounded to MAX_ARMS arms
    (least recently touched arm is dropped first and counted in `evicted`)."""

    def __init__(self, max_arms: int = MAX_ARMS) -> None:
        self.max_arms = max_arms
        self.evicted = 0
        self._arms: "OrderedDict[str, ArmChannels]" = OrderedDict()
        self._failures: dict[str, deque] = {}
        self._last_traffic: dict[str, float] = {}
        self._last_probe: dict[str, float] = {}
        self._unhealthy_until: dict[str, float] = {}

    def _arm(self, arm: str, now: float) -> ArmChannels:
        channels = self._arms.get(arm)
        if channels is None:
            channels = self._arms[arm] = ArmChannels(now)
            self._last_traffic.setdefault(arm, now)
            while len(self._arms) > self.max_arms:
                old, _ = self._arms.popitem(last=False)
                for table in (self._failures, self._last_traffic, self._last_probe, self._unhealthy_until):
                    table.pop(old, None)
                self.evicted += 1
        else:
            self._arms.move_to_end(arm)
        return channels

    def register(self, arm: str, now: float) -> None:
        """Make the arm known (starts its starvation clock) without recording an outcome."""
        self._arm(arm, now)

    def observe_success(self, arm: str, now: float) -> None:
        channels = self._arm(arm, now)
        channels.trail.deposit(TRAIL_GAIN, now)
        self._last_traffic[arm] = now

    def observe_failure(
        self, arm: str, now: float, caller: str = "anon",
        exc: Optional[BaseException] = None, status: Optional[int] = None,
    ) -> bool:
        """Record a failed call. Returns True when it was transient (and so deposited alarm)."""
        channels = self._arm(arm, now)
        self._last_traffic[arm] = now
        if not is_transient(exc, status):
            return False
        channels.alarm.deposit(ALARM_GAIN, now)
        log = self._failures.setdefault(arm, deque(maxlen=_MAX_FAILURES_KEPT))
        log.append((now, caller))
        if self._quorum(arm, now):
            self._unhealthy_until[arm] = now + HALF_LIFE_S["alarm"]
        return True

    def _quorum(self, arm: str, now: float) -> bool:
        recent = [(t, c) for t, c in self._failures.get(arm, ()) if 0.0 <= now - t <= QUORUM_WINDOW_S]
        return len(recent) >= QUORUM_FAILURES and len({c for _, c in recent}) >= QUORUM_CALLERS

    def unhealthy(self, arm: str, now: float) -> bool:
        """True while a failure quorum is open, and for one alarm half-life after it last formed."""
        return self._quorum(arm, now) or now < self._unhealthy_until.get(arm, float("-inf"))

    def probe_due(self, arm: str, now: float) -> bool:
        """True exactly once per PROBE_AFTER_S of silence (no traffic and no earlier probe)."""
        self._arm(arm, now)
        quiet_since = max(self._last_traffic.get(arm, now), self._last_probe.get(arm, float("-inf")))
        if now - quiet_since >= PROBE_AFTER_S:
            self._last_probe[arm] = now
            return True
        return False

    def values(self, arm: str, now: float) -> dict:
        channels = self._arms.get(arm)
        return (channels or ArmChannels(now)).values(now)

    def snapshot(self, now: float) -> dict[str, Any]:
        return {arm: ch.values(now) for arm, ch in self._arms.items()}

    def state_bytes(self, now: float) -> int:
        """Size of the JSON form of the state, to keep it well under the 1 MiB budget."""
        return len(json.dumps(self.snapshot(now)).encode("utf-8"))
