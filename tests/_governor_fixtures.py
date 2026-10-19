"""Shared fakes and builders for the Governor tests. Recording fakes only: nothing here raises on purpose."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional

from loci_litellm.envelope import DEFAULT_CAPS, Caps, EnvelopeState
from loci_litellm.governor import (
    Arm,
    Catalog,
    Governor,
    Snapshot,
    StopFiles,
)

D = Decimal
BASE_TS = 1_760_000_000.0     # 2025-10-09 08:53:20 UTC


class FakeClock:
    def __init__(self, start: float = BASE_TS) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class RecordingStore:
    """Stands in for the spend counter: serves a canned snapshot and records every call made to it."""
    snap: Snapshot = field(default_factory=Snapshot)
    reservations: list = field(default_factory=list)
    denials: list = field(default_factory=list)
    settlements: list = field(default_factory=list)
    releases: list = field(default_factory=list)
    snapshot_clients: list = field(default_factory=list)

    def snapshot(self, now: float, client: str = "anonymous") -> Snapshot:
        self.snapshot_clients.append(client)
        return self.snap

    def reserve(self, call_id: str, amount: Decimal, now: float, client: str = "anonymous", hc: bool = False) -> None:
        self.reservations.append((amount, client, hc))

    def settle(self, call_id: str, actual: Optional[Decimal], now: float, client: str = "anonymous",
               final: bool = True, hc: bool = False) -> Decimal:
        self.settlements.append((call_id, actual, final))
        return actual or Decimal(0)

    def release(self, call_id: str) -> None:
        self.releases.append(call_id)

    def record_denial(self, code: str) -> None:
        self.denials.append(code)


def caps(**overrides: Any) -> Caps:
    """Defaults with generous other lines, so a test only trips the line it sets."""
    base = replace(DEFAULT_CAPS, day_soft=D("1000"), day_hard=D("1000"), rolling_24h=D("1000"),
                   hour_downgrade=D("1000"), hour_reject=D("1000"), interactive_stop=D("1000"))
    return replace(base, **overrides)


def trusted(c: Caps, enforce: bool = True, clients: Optional[dict] = None) -> EnvelopeState:
    return EnvelopeState(caps=c, enforce=enforce, day_timezone="local", clients=clients or {}, level="L0",
                         alarm="", sha256="x", source="envelope")


# One expensive arm and one cheap arm, both tool-capable; prices are dollars per million tokens.
BIG = Arm("arm-big", "big", D("20"), D("0"), supports_tools=True, max_input_tokens=200_000)
SMALL = Arm("arm-small", "small", D("2"), D("0"), supports_tools=True, max_input_tokens=200_000)
TEXTONLY = Arm("arm-text", "text", D("1"), D("0"), supports_tools=False, max_input_tokens=200_000)


def catalog(*arms: Arm, retries: int = 0, **kwargs: Any) -> Catalog:
    return Catalog(arms or (BIG, SMALL), num_retries=retries, **kwargs)


def make_governor(tmp_path: Path, store: Any, c: Caps, *, mode: str = "enforce", env: Optional[EnvelopeState] = None,
                  cat: Optional[Catalog] = None, clock: Optional[FakeClock] = None, audit: Any = None,
                  optimisers: tuple = (), events: Optional[list] = None, stop: Optional[StopFiles] = None,
                  **kwargs: Any) -> Governor:
    clock = clock or FakeClock()
    state = env or trusted(c)
    return Governor(store=store, envelope=lambda: state, catalog=lambda: cat or catalog(), clock=clock,
                    stop=stop or StopFiles(str(tmp_path / "STOP"), str(tmp_path / "PAUSE"), clock=clock),
                    audit=audit, mode=mode, optimisers=optimisers,
                    event_sink=None if events is None else events.append, **kwargs)


def body(model: str = "big", tokens: int = 1000, tools: bool = False, **extra: Any) -> dict:
    """A request whose estimated prompt is exactly `tokens` tokens (the estimator counts 3 characters a token)."""
    tool_defs = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}] if tools else []
    chars = 3 * tokens - 2 - 16 - (len(json.dumps(tool_defs)) if tools else 0)
    data: dict[str, Any] = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "x" * max(0, chars)}]}
    if tools:
        data["tools"] = tool_defs
    data.update(extra)
    return data
