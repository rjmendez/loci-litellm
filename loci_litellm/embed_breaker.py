"""Circuit breaker for the embedding endpoint, shared by the grounding hook and the memory writer.

After `threshold` consecutive failures the breaker opens and `allow()` returns False for a cooldown, so
callers fail fast instead of each request waiting out a timeout. When the cooldown ends one probe is
allowed through: success closes the breaker, failure reopens it with a doubled cooldown (capped).

State is per process. Environment (read when a breaker is created):
  LOCI_EMBED_BREAKER_THRESHOLD       consecutive failures that open it (default 3)
  LOCI_EMBED_BREAKER_COOLDOWN_S      first cooldown in seconds (default 15)
  LOCI_EMBED_BREAKER_MAX_COOLDOWN_S  cooldown cap in seconds (default 300)
  LOCI_EMBED_BREAKER_DISABLED=1      never open
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable, Dict

logger = logging.getLogger("loci_litellm.embed_breaker")


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        logger.warning("loci embed breaker: %s is not a number; using %s", name, default)
        return float(default)


class EmbedBreaker:
    def __init__(self, key: str, threshold: int = 3, cooldown: float = 15.0, max_cooldown: float = 300.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.key = key
        self.threshold = max(1, int(threshold))
        self.base_cooldown = float(cooldown)
        self.max_cooldown = max(float(max_cooldown), float(cooldown))
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._open_until = 0.0        # 0 = closed
        self._cooldown = self.base_cooldown
        self._probing = False
        self.times_opened = 0

    def allow(self) -> bool:
        with self._lock:
            if not self._open_until:
                return True
            if self._clock() < self._open_until:
                return False
            if self._probing:          # one probe at a time
                return False
            self._probing = True
            return True

    def record_success(self) -> None:
        with self._lock:
            if self._open_until:
                logger.warning("loci embed breaker: closed again (%s answered)", self.key)
            self._failures = 0
            self._open_until = 0.0
            self._cooldown = self.base_cooldown
            self._probing = False

    def record_failure(self) -> None:
        with self._lock:
            self._probing = False
            self._failures += 1
            if self._open_until:
                self._cooldown = min(self._cooldown * 2, self.max_cooldown)
            elif self._failures < self.threshold:
                return
            self._open_until = self._clock() + self._cooldown
            self.times_opened += 1
            logger.warning(
                "loci embed breaker: OPEN for %.0fs after %d consecutive failures (%s); grounding and "
                "memory write-back skip embedding until it answers",
                self._cooldown, self._failures, self.key,
            )

    def snapshot(self) -> dict:
        with self._lock:
            remaining = max(0.0, self._open_until - self._clock()) if self._open_until else 0.0
            return {"key": self.key, "open": bool(self._open_until), "consecutive_failures": self._failures,
                    "open_for_s": round(remaining, 1), "times_opened": self.times_opened}


class _Disabled:
    key = ""
    times_opened = 0

    def allow(self) -> bool:
        return True

    def record_success(self) -> None:
        pass

    def record_failure(self) -> None:
        pass

    def snapshot(self) -> dict:
        return {"key": "", "open": False, "disabled": True}


_registry: Dict[str, EmbedBreaker] = {}
_registry_lock = threading.Lock()


def get(key: str):
    """The breaker for an embedding endpoint (one per distinct key, created on first use)."""
    if os.environ.get("LOCI_EMBED_BREAKER_DISABLED", "").strip().lower() in ("1", "true", "yes", "on"):
        return _Disabled()
    with _registry_lock:
        breaker = _registry.get(key)
        if breaker is None:
            breaker = EmbedBreaker(
                key,
                threshold=int(_number("LOCI_EMBED_BREAKER_THRESHOLD", 3)),
                cooldown=_number("LOCI_EMBED_BREAKER_COOLDOWN_S", 15.0),
                max_cooldown=_number("LOCI_EMBED_BREAKER_MAX_COOLDOWN_S", 300.0),
            )
            _registry[key] = breaker
        return breaker


def reset_all() -> None:
    """Forget every breaker (tests)."""
    with _registry_lock:
        _registry.clear()
