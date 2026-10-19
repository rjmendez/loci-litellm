"""Hash of a request's cacheable prefix, and a bounded per-prefix dwell tracker. stdlib only, no LiteLLM.

A provider's prompt cache keys on the start of the prompt. Text a client or a hook adds in front of or
inside that start (reminder tags, a retrieved-memory block, a grounding status notice) changes from call
to call and breaks the match, so the hash is taken after removing those blocks. Two requests whose hashes
are equal share a cacheable prefix; the dwell tracker uses that to keep them on one arm.

What is hashed (nothing is stored; the result is 16 hex characters of a sha256 over length-framed segments):
the tool definitions, every instruction-slot text (`system`, `instructions`, system and developer
messages), and the first conversational message that still has text after stripping. Runs of whitespace
count as one space, and only the first TEXT_CAP (16,384) characters of each text are used, which keeps
the hash under a millisecond for a very large prompt; texts that differ only after that point hash equal.

Which text is removed is configurable: `[cache] prefix_strip_markers = [...]` in the private config or
LOCI_PREFIX_STRIP_MARKERS ('||' separated); unset means DEFAULT_STRIP_MARKERS, which are what the grounding
hook injects (the MEMORY MATCH block and the LOCI_GROUNDING_STATUS notice) plus the reminder tags and the
client retrieval header seen in front-end requests. The client markers of `[grounding] client_rag_ok_markers`
(LOCI_CLIENT_RAG_OK_MARKERS) are always added. An entry is one of:
  - `open=>close`: every span from `open` to the next `close` is removed (to the end if never closed);
  - a plain marker: an instruction-slot part (a system or developer message, a block of a `system` or
    `instructions` list) that starts with it is dropped whole, because the grounding hook puts its block in a
    part of its own; anywhere else (user text, a string field such as `system`, `instructions` or `prompt`)
    the marker and everything up to the next blank line are removed, because there a client or the hook has
    prepended the block to text that must stay. A retrieved block with blank lines inside it, prepended to
    such text, can leave a remainder in the hash.

The dwell tracker (DwellTracker) is off unless LOCI_ARM_DWELL is set to 1: nothing in this package calls it,
and when it is off `choose` returns the first candidate and keeps no state.
"""
from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional, Sequence

logger = logging.getLogger("loci_litellm.prefix_hash")

DEFAULT_STRIP_MARKERS: tuple[str, ...] = (
    "<system-reminder>=></system-reminder>",
    "MEMORY MATCH (",              # the grounding hook's context block (grounding_hook._OWN_CONTEXT_PREFIX)
    "LOCI_GROUNDING_STATUS:",      # the grounding hook's failure notice
    "AUTOMATIC LOCI RAG",          # a client's own retrieval header
)
HASH_LEN = 16
TEXT_CAP = 16_384        # characters of each text (and of the tool definitions) that are hashed

_INSTRUCTION_KEYS = ("instructions", "system", "system_instruction", "systemInstruction")
_INSTRUCTION_ROLES = ("system", "developer")


# ---------------------------------------------------------------- marker configuration

def effective_markers() -> tuple[str, ...]:
    """The configured strip markers (or the defaults) plus the configured client retrieval markers."""
    from loci_litellm.backend_resolver import BackendResolver
    configured = BackendResolver.prefix_strip_markers()
    base = DEFAULT_STRIP_MARKERS if configured is None else configured
    extra = BackendResolver.client_rag_ok_markers()
    return tuple(dict.fromkeys(base + extra))


@functools.lru_cache(maxsize=32)
def _rules(markers: tuple[str, ...]) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...]]:
    pairs: list = []
    plain: list = []
    for entry in markers:
        if "=>" in entry:
            open_, close = entry.split("=>", 1)
            if open_ and close:
                pairs.append((open_, close))
        elif entry:
            plain.append(entry)
    return tuple(pairs), tuple(plain)


# ---------------------------------------------------------------- stripping

def _cut_spans(text: str, open_: str, close: str) -> str:
    out: list = []
    pos = 0
    while True:
        start = text.find(open_, pos)
        if start < 0:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:start])
        end = text.find(close, start + len(open_))
        if end < 0:
            return "".join(out)
        pos = end + len(close)


def _cut_paragraphs(text: str, marker: str) -> str:
    """Remove each occurrence of `marker` and the text after it up to the next blank line."""
    out: list = []
    pos = 0
    while True:
        start = text.find(marker, pos)
        if start < 0:
            out.append(text[pos:])
            return "".join(out)
        out.append(text[pos:start])
        end = text.find("\n\n", start + len(marker))
        if end < 0:
            return "".join(out)
        pos = end


def _strip(text: str, markers: tuple[str, ...], whole_part: bool) -> str:
    """`text` with the marker blocks removed and whitespace runs collapsed; "" when the part is dropped."""
    pairs, plain = _rules(markers)
    text = text[:TEXT_CAP]
    for open_, close in pairs:
        text = _cut_spans(text, open_, close)
    for marker in plain:
        if whole_part and text.lstrip().startswith(marker):
            return ""
        text = _cut_paragraphs(text, marker)
    return " ".join(text.split())


# ---------------------------------------------------------------- extraction

def _texts(value: Any, markers: tuple[str, ...], whole_part: bool) -> list[str]:
    """Stripped, non-empty texts of a content value; `whole_part` says a text that starts with a plain marker
    is dropped whole (an instruction-slot part) instead of cut at the next blank line."""
    if isinstance(value, str):
        cleaned = _strip(value, markers, whole_part)
        return [cleaned] if cleaned else []
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return _texts(value["text"], markers, whole_part)
        inner = value.get("parts", value.get("content"))
        if inner is not None:
            return _texts(inner, markers, whole_part)
        kind = value.get("type")
        return [f"[{kind}]"] if isinstance(kind, str) else []
    if isinstance(value, (list, tuple)):
        found: list = []
        for item in value:
            found += _texts(item, markers, whole_part)
        return found
    return []


def _segments(data: Mapping[str, Any], markers: tuple[str, ...]) -> list[tuple[str, str]]:
    segments: list = []
    tools = data.get("tools")
    if tools:
        try:
            segments.append(("tools", json.dumps(tools, sort_keys=True, separators=(",", ":"),
                                                 default=str)[:TEXT_CAP]))
        except (TypeError, ValueError):
            segments.append(("tools", repr(tools)[:TEXT_CAP]))
    for key in _INSTRUCTION_KEYS:
        value = data.get(key)
        for text in _texts(value, markers, not isinstance(value, str)):   # a list holds parts of their own
            segments.append(("instr", text))
    messages = data.get("messages")
    first: list = []
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if message.get("role") in _INSTRUCTION_ROLES:
            segments += [("instr", t) for t in _texts(content, markers, True)]
        elif not first:
            first = _texts(content, markers, False)
    if not first:
        for key in ("input", "prompt"):
            value = data.get(key)
            if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
                for item in value:
                    first = _texts(item.get("content"), markers, False)
                    if first:
                        break
            elif value:
                first = _texts(value, markers, False)
            if first:
                break
    segments += [("first", t) for t in first]
    return segments


def prefix_hash(data: Mapping[str, Any], markers: Optional[Sequence[str]] = None) -> str:
    """16 hex characters identifying the cacheable prefix of a request body (any API shape), or "" when the
    body has no text or tools to hash. `markers` overrides effective_markers()."""
    if not isinstance(data, Mapping):
        return ""
    chosen = tuple(markers) if markers is not None else effective_markers()
    segments = _segments(data, chosen)
    if not segments:
        return ""
    digest = hashlib.sha256()
    for tag, text in segments:
        raw = text.encode("utf-8", "replace")
        digest.update(f"{tag}:{len(raw)}:".encode("ascii"))
        digest.update(raw)
    return digest.hexdigest()[:HASH_LEN]


def request_of(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """The request body a LiteLLM callback's kwargs describe, in the shape prefix_hash reads."""
    optional = kwargs.get("optional_params")
    optional = optional if isinstance(optional, Mapping) else {}
    body: dict = {}
    for key in ("messages", "input", "prompt", "instructions", "system"):
        value = kwargs.get(key, optional.get(key))
        if value:
            body[key] = value
    tools = kwargs.get("tools", optional.get("tools"))
    if tools:
        body["tools"] = tools
    return body


# ---------------------------------------------------------------- dwell tracker

def _money(value: Any) -> Decimal:
    if isinstance(value, float):
        raise TypeError("money amounts are Decimal, int or str, never float")
    return Decimal(value) if isinstance(value, (int, str)) else value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("%s=%r is not a number; using %s", name, raw, default)
        return default


def dwell_enabled() -> bool:
    return os.environ.get("LOCI_ARM_DWELL", "").strip().lower() in ("1", "true", "yes", "on")


class DwellTracker:
    """Keeps requests that share a prefix hash on one arm for `dwell_s` seconds, so the arm's prompt cache is
    not thrown away by per-call balancing.

    `choose(key, candidates, ...)` takes the arms the router ranks best-first and returns the arm to use:
      - first sight of a key (or an entry older than `dwell_s`): the first candidate is pinned;
      - within the dwell: the pinned arm, whatever the router now prefers;
      - the pinned arm has an alarm above `alarm_threshold` (or `budget_breach` is true, or the arm is no
        longer a candidate): the request moves to the first candidate that is not alarmed, whatever the
        cache loss; with none, it stays;
      - after the dwell, a different best candidate replaces the pin only when `gain` (what moving is worth)
        is greater than `cache_loss` (what the pinned arm's warm cache is worth); otherwise the pin is kept.
    `gain` and `cache_loss` are Decimal, int or str, never float. The table holds at most `max_entries`
    keys (least recently used out); only the key, the arm name and a timestamp are kept."""

    def __init__(self, dwell_s: Optional[float] = None, max_entries: int = 10_000,
                 alarm_threshold: Optional[float] = None, enabled: Optional[bool] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.dwell_s = _env_float("LOCI_ARM_DWELL_SECONDS", 300.0) if dwell_s is None else float(dwell_s)
        self.alarm_threshold = (_env_float("LOCI_ARM_DWELL_ALARM", 0.6) if alarm_threshold is None
                                else float(alarm_threshold))
        self.enabled = dwell_enabled() if enabled is None else bool(enabled)
        self._max = int(max_entries)
        self._clock = clock
        self._table: "OrderedDict[str, tuple[str, float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._counts: dict = {}

    def _bump(self, name: str) -> None:
        self._counts[name] = self._counts.get(name, 0) + 1

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._table), **{k: v for k, v in sorted(self._counts.items())}}

    def _pin(self, key: str, arm: str, now: float) -> None:
        self._table[key] = (arm, now)
        self._table.move_to_end(key)
        while len(self._table) > self._max:
            self._table.popitem(last=False)
            self._bump("evicted")

    def choose(self, key: str, candidates: Sequence[str], alarms: Optional[Mapping[str, float]] = None,
               gain: Any = 0, cache_loss: Any = 0, budget_breach: bool = False) -> str:
        if not candidates:
            raise ValueError("choose needs at least one candidate arm")
        best = candidates[0]
        if not self.enabled:
            with self._lock:
                self._bump("passthrough")
            return best
        if not key:
            with self._lock:
                self._bump("unkeyed")
            return best
        alarms = alarms or {}
        gain_d, loss_d = _money(gain), _money(cache_loss)
        with self._lock:
            now = self._clock()
            entry = self._table.get(key)
            if entry is None:
                self._pin(key, best, now)
                self._bump("pinned")
                return best
            arm, since = entry
            self._table.move_to_end(key)
            alarmed = float(alarms.get(arm, 0.0)) > self.alarm_threshold
            if arm not in candidates or alarmed or budget_breach:
                reason = "unavailable" if arm not in candidates else ("budget" if budget_breach else "alarm")
                for other in candidates:
                    if other != arm and float(alarms.get(other, 0.0)) <= self.alarm_threshold:
                        self._pin(key, other, now)
                        self._bump("moved_" + reason)
                        return other
                self._bump("stuck_" + reason)
                return arm if arm in candidates else best
            if now - since < self.dwell_s:
                self._bump("held")
                return arm
            if best == arm or float(alarms.get(best, 0.0)) > self.alarm_threshold:
                self._table[key] = (arm, now)          # same arm again: a fresh dwell
                self._bump("renewed")
                return arm
            if gain_d > loss_d:
                self._pin(key, best, now)
                self._bump("moved_gain")
                return best
            self._table[key] = (arm, now)
            self._bump("refused_cost")
            return arm
