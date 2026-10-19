"""The owner envelope: spending caps, the enforce switch and the client table, read from a file the
callbacks never write.

The envelope is a TOML (or JSON, by suffix) file. Its sha256 is compared with a pin: the first token of the
pin file, or a signed `envelope_pin` entry in the audit chain (an HMAC-SHA256 of the digest under an owner
key). Anything other than a valid, pinned, well-formed file fails CLOSED to the built-in defaults, level L3
and an alarm string: `envelope_missing`, `envelope_unpinned`, `caps_hash_mismatch` or `envelope_invalid`.
A one-byte change is a mismatch. Which accounts may write the files is an operating-system matter; this
module only detects.

    [governor]
    enforce = false          # the only way to let LOCI_GOVERNOR=enforce take effect
    day_timezone = "local"   # "local" (the machine's zone, DST-aware) or an IANA name

    [caps]                   # dollars; anything omitted keeps the built-in default
    day_soft = 30
    day_hard = 50
    rolling_24h = 60
    hour_downgrade = 10
    hour_reject = 25
    interactive_stop = 75
    anonymous_day = 10
    high_cost_share = "0.30"
    high_cost_calls_per_hour = 20
    prompt_tokens_max = 60000

    [clients.scripts]
    secret_sha256 = "<hex sha256 of the client's header secret>"
    cap_day = 20
    interactive = false
    rw = false
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
import tomllib
from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from loci_litellm.audit_chain import AuditChain, AuditUnavailable

LEVEL_OK = "L0"
LEVEL_FAIL_CLOSED = "L3"

ALARM_MISSING = "envelope_missing"
ALARM_UNPINNED = "envelope_unpinned"
ALARM_MISMATCH = "caps_hash_mismatch"
ALARM_INVALID = "envelope_invalid"


class EnvelopeInvalid(ValueError):
    """The envelope file parsed but a value is missing its type, negative or inconsistent."""


@dataclass(frozen=True)
class Caps:
    day_soft: Decimal = Decimal("30")
    day_hard: Decimal = Decimal("50")
    rolling_24h: Decimal = Decimal("60")
    hour_downgrade: Decimal = Decimal("10")
    hour_reject: Decimal = Decimal("25")
    interactive_stop: Decimal = Decimal("75")
    anonymous_day: Decimal = Decimal("10")
    provisional_day: Decimal = Decimal("25")       # applies when the spend counter is unreadable
    high_cost_share: Decimal = Decimal("0.30")
    high_cost_calls_per_hour: int = 20
    prompt_tokens_max: int = 60000
    max_completion_ceiling: int = 32768            # max_tokens that is missing or malformed reserves this many


DEFAULT_CAPS = Caps()
_MONEY_FIELDS = ("day_soft", "day_hard", "rolling_24h", "hour_downgrade", "hour_reject", "interactive_stop",
                 "anonymous_day", "provisional_day", "high_cost_share")
_INT_FIELDS = ("high_cost_calls_per_hour", "prompt_tokens_max", "max_completion_ceiling")


@dataclass(frozen=True)
class ClientEntry:
    name: str
    secret_sha256: str
    cap_day: Optional[Decimal] = None
    interactive: bool = False
    rw: bool = False


@dataclass(frozen=True)
class EnvelopeState:
    caps: Caps
    enforce: bool
    day_timezone: str
    clients: Mapping[str, ClientEntry] = field(default_factory=dict)
    level: str = LEVEL_FAIL_CLOSED
    alarm: str = ALARM_MISSING
    sha256: str = ""
    source: str = "defaults"          # "envelope" when the file was accepted

    @property
    def trusted(self) -> bool:
        return self.source == "envelope" and self.level == LEVEL_OK


def default_state(alarm: str, sha256: str = "") -> EnvelopeState:
    """Built-in defaults, never enforcing, at the fail-closed level."""
    return EnvelopeState(caps=DEFAULT_CAPS, enforce=False, day_timezone="local", clients={},
                         level=LEVEL_FAIL_CLOSED, alarm=alarm, sha256=sha256, source="defaults")


def _money(value: Any, name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, str, float)):
        raise EnvelopeInvalid(f"{name} must be a number")
    try:
        amount = Decimal(str(value))
    except InvalidOperation as exc:
        raise EnvelopeInvalid(f"{name} is not a decimal number") from exc
    if not amount.is_finite() or amount < 0:
        raise EnvelopeInvalid(f"{name} must be a finite number >= 0")
    return amount


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EnvelopeInvalid(f"{name} must be an integer >= 0")
    return value


def parse_envelope(raw: bytes, suffix: str = ".toml") -> tuple[Caps, bool, str, dict[str, ClientEntry]]:
    try:
        text = raw.decode("utf-8")
        doc = json.loads(text) if suffix.lower() == ".json" else tomllib.loads(text)
    except (UnicodeDecodeError, ValueError) as exc:     # tomllib.TOMLDecodeError is a ValueError
        raise EnvelopeInvalid(f"envelope does not parse: {exc}") from exc
    if not isinstance(doc, dict):
        raise EnvelopeInvalid("envelope must be a table")
    caps_in = doc.get("caps", {})
    gov = doc.get("governor", {})
    if not isinstance(caps_in, dict) or not isinstance(gov, dict):
        raise EnvelopeInvalid("[caps] and [governor] must be tables")
    values: dict[str, Any] = {}
    for key, value in caps_in.items():
        if key in _MONEY_FIELDS:
            values[key] = _money(value, key)
        elif key in _INT_FIELDS:
            values[key] = _count(value, key)
        else:
            raise EnvelopeInvalid(f"unknown cap {key!r}")
    caps = replace(DEFAULT_CAPS, **values)
    if not (caps.day_soft <= caps.day_hard <= caps.interactive_stop):
        raise EnvelopeInvalid("need day_soft <= day_hard <= interactive_stop")
    if not caps.hour_downgrade <= caps.hour_reject:
        raise EnvelopeInvalid("need hour_downgrade <= hour_reject")
    if caps.day_hard > caps.rolling_24h:
        raise EnvelopeInvalid("need day_hard <= rolling_24h")
    enforce = gov.get("enforce", False)
    if not isinstance(enforce, bool):
        raise EnvelopeInvalid("governor.enforce must be true or false")
    tz = gov.get("day_timezone", "local")
    if not isinstance(tz, str) or not tz:
        raise EnvelopeInvalid("governor.day_timezone must be a string")
    clients: dict[str, ClientEntry] = {}
    clients_in = doc.get("clients", {})
    if not isinstance(clients_in, dict):
        raise EnvelopeInvalid("[clients] must be a table")
    for name, entry in clients_in.items():
        if name == "anonymous":
            raise EnvelopeInvalid("client name 'anonymous' is reserved for unidentified callers")
        if not isinstance(entry, dict) or not isinstance(entry.get("secret_sha256"), str) or len(entry["secret_sha256"]) != 64:
            raise EnvelopeInvalid(f"client {name!r} needs a 64-hex secret_sha256")
        cap = entry.get("cap_day")
        flags = (entry.get("interactive", False), entry.get("rw", False))
        if not all(isinstance(flag, bool) for flag in flags):
            raise EnvelopeInvalid(f"client {name!r}: interactive and rw must be booleans")
        clients[name] = ClientEntry(name=name, secret_sha256=entry["secret_sha256"].lower(),
                                    cap_day=None if cap is None else _money(cap, f"clients.{name}.cap_day"),
                                    interactive=flags[0], rw=flags[1])
    return caps, enforce, tz, clients


def sign_pin(owner_key: bytes, digest: str) -> str:
    return hmac.new(owner_key, digest.encode("ascii"), hashlib.sha256).hexdigest()


def record_pin(audit: AuditChain, owner_key: bytes, digest: str) -> dict:
    """The owner's act of accepting an envelope: a signed entry in the audit chain."""
    return audit.append("envelope_pin", {"sha256": digest, "sig": sign_pin(owner_key, digest)})


def file_digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EnvelopeWatcher:
    """Re-reads and re-checks the envelope at most every `recheck_s` (default 60 s), with an injected clock."""

    def __init__(self, envelope_path: str, pin_path: str, audit: Optional[AuditChain] = None,
                 owner_key: Optional[bytes] = None, *, clock: Callable[[], float] = time.time,
                 recheck_s: float = 60.0) -> None:
        self._path = Path(envelope_path)
        self._pin = Path(pin_path)
        self._audit = audit
        self._key = owner_key
        self._clock = clock
        self._recheck_s = recheck_s
        self._state: Optional[EnvelopeState] = None
        self._checked_at = 0.0
        self.checks = 0
        self.alarms: dict[str, int] = {}

    def state(self) -> EnvelopeState:
        now = self._clock()
        if self._state is None or now - self._checked_at >= self._recheck_s:
            self._state = self._evaluate()
            self._checked_at = now
            self.checks += 1
            if self._state.alarm:
                self.alarms[self._state.alarm] = self.alarms.get(self._state.alarm, 0) + 1
        return self._state

    def _pinned_digest(self) -> Optional[str]:
        try:
            tokens = self._pin.read_text(encoding="utf-8").split()
        except OSError:
            return None
        return tokens[0].lower() if tokens else None

    def _signed_accept(self, digest: str) -> bool:
        if self._audit is None or self._key is None:
            return False
        try:
            if not self._audit.verify().ok:
                return False
            expected = sign_pin(self._key, digest)
            for entry in self._audit.entries():
                data = entry.get("data")
                if entry.get("kind") == "envelope_pin" and isinstance(data, dict) and data.get("sha256") == digest \
                        and isinstance(data.get("sig"), str) and hmac.compare_digest(data["sig"], expected):
                    return True
        except AuditUnavailable:
            return False
        return False

    def _evaluate(self) -> EnvelopeState:
        try:
            raw = self._path.read_bytes()
        except OSError:
            return default_state(ALARM_MISSING)
        digest = hashlib.sha256(raw).hexdigest()
        pinned = self._pinned_digest()
        if pinned is None and not self._signed_accept(digest):
            return default_state(ALARM_UNPINNED, digest)
        if pinned != digest and not self._signed_accept(digest):
            return default_state(ALARM_MISMATCH, digest)
        try:
            caps, enforce, tz, clients = parse_envelope(raw, self._path.suffix)
        except EnvelopeInvalid:
            return default_state(ALARM_INVALID, digest)
        return EnvelopeState(caps=caps, enforce=enforce, day_timezone=tz, clients=clients, level=LEVEL_OK,
                             alarm="", sha256=digest, source="envelope")
