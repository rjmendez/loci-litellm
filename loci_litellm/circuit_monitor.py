"""LociCircuitMonitor: an advisory per-model-group circuit breaker for the LiteLLM proxy.

  - Observes async success/failure log events keyed by model group (else the requested model).
  - Counts consecutive transient failures (timeouts, connection errors, HTTP 408 / 5xx); a success
    resets the count. Permanent errors (auth, bad request, model not found) are logged at INFO, not counted.
  - At LOCI_CB_FAILURE_THRESHOLD (default 5) the circuit opens and a WARNING is logged; backoff is a
    fixed 1, 2, 4 ... 64 s sequence.
  - async_pre_call_hook logs a WARNING and sets metadata["loci_circuit_open"] = <group> when open.
  - Never blocks, redirects or reorders a request (the hook always returns None). The signal is the
    "loci_litellm.circuit_monitor" log; get_status() is in-process only.
  - State is per process and in memory. Calls LiteLLM logs through its sync path are not observed.

Register in litellm config.yaml:
    litellm_settings:
      callbacks:
        - loci_litellm.circuit_monitor.loci_circuit_monitor
"""

from __future__ import annotations

import logging
import os
import threading
import time
from enum import Enum
from typing import Any

from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("loci_litellm.circuit_monitor")

# ── Tunables ─────────────────────────────────────────────────────────────────
# consecutive transient failures before a circuit opens
_FAILURE_THRESHOLD = int(os.environ.get("LOCI_CB_FAILURE_THRESHOLD", "5"))
_BACKOFF_SEQUENCE = [1, 2, 4, 8, 16, 32, 64]  # seconds


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class GroupHealth:
    """Health counters and circuit state for one model group. Thread-safe."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.backoff_level = 0
        self.backoff_until = 0.0
        self.total_failures = 0
        self.total_successes = 0
        self.circuit_open_count = 0
        self.lock = threading.RLock()

    def record_failure(self) -> None:
        with self.lock:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()
            should_open = (
                self.state == CircuitState.HALF_OPEN
                or (self.state == CircuitState.CLOSED and self.failure_count >= _FAILURE_THRESHOLD)
            )
            if should_open:
                self._open_circuit()

    def record_success(self) -> None:
        with self.lock:
            self.success_count += 1
            self.total_successes += 1
            self.failure_count = 0
            if self.state != CircuitState.CLOSED:
                self.state = CircuitState.CLOSED
                self.backoff_level = 0
                self.backoff_until = 0.0
                logger.info("loci circuit_monitor: %s circuit closed (recovered)", self.name)

    def _open_circuit(self) -> None:
        self.state = CircuitState.OPEN
        self.circuit_open_count += 1
        delay = _BACKOFF_SEQUENCE[min(self.backoff_level, len(_BACKOFF_SEQUENCE) - 1)]
        self.backoff_level = min(self.backoff_level + 1, len(_BACKOFF_SEQUENCE) - 1)
        self.backoff_until = time.time() + delay
        logger.warning(
            "loci circuit_monitor: %s circuit OPENED after %d consecutive transient failures, backoff %ds",
            self.name, self.failure_count, delay,
        )

    def can_proceed(self) -> bool:
        """Whether the circuit is not currently open. Informational: callers do not block on it."""
        with self.lock:
            if self.state == CircuitState.CLOSED:
                return True
            if self.state == CircuitState.OPEN:
                now = time.time()
                if now >= self.backoff_until:
                    self.state = CircuitState.HALF_OPEN
                    logger.info("loci circuit_monitor: %s half_open (testing recovery)", self.name)
                    return True
                return False
            # HALF_OPEN: no single-probe slot; every request passes and a transient failure re-opens.
            return True

    def status(self) -> dict:
        with self.lock:
            return {
                "group": self.name,
                "state": self.state.value,
                "failures": self.failure_count,
                "successes": self.success_count,
                "total_failures": self.total_failures,
                "total_successes": self.total_successes,
                "circuit_open_count": self.circuit_open_count,
                "backoff_until": self.backoff_until,
            }


_TRANSIENT_CLASS_NAMES = (
    "serviceunavailableerror", "badgatewayerror", "internalservererror", "apiconnectionerror",
)


def _is_transient(exc: BaseException) -> bool:
    """Transient provider/transport failure? Checks HTTP status (408, >= 500), then class name, then message."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and not isinstance(status, bool):
        if status == 408 or status >= 500:
            return True
        if 400 <= status < 500:
            return False
    name = type(exc).__name__.lower()
    if "timeout" in name or "connection" in name or name in _TRANSIENT_CLASS_NAMES:
        return True
    msg = str(exc).lower()
    signals = (
        "timeout", "timed out", "temporar", "connection aborted", "connection reset",
        "connection refused", "service unavailable", "bad gateway", "gateway timeout",
    )
    return any(s in msg for s in signals)


class LociCircuitMonitor(CustomLogger):
    """Advisory per-model-group circuit breaker (see the module docstring).

    The hooks catch their own exceptions and log a WARNING; a swallowed error means the event was not counted.
    """

    def __init__(self) -> None:
        super().__init__()
        self._providers: dict[str, GroupHealth] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _circuit_key(payload: dict) -> str:
        litellm_params = payload.get("litellm_params") or {}
        for metadata in (litellm_params.get("metadata"), payload.get("metadata")):
            if not isinstance(metadata, dict):
                continue
            group = metadata.get("model_group")
            if isinstance(group, str) and group:
                return group
        model = payload.get("model")
        return model if isinstance(model, str) and model else "unknown"

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> Exception | str | dict | None:
        """Check circuit state before the call. Advisory only: does not block."""
        try:
            provider = self._circuit_key(data)
            if provider == "unknown":
                return None
            # Read only: entries come from events, never client-supplied model strings, so the table cannot grow unbounded.
            with self._lock:
                health = self._providers.get(provider)
            if health is not None and not health.can_proceed():
                logger.warning(
                    "loci circuit_monitor: %s circuit open; request proceeds unchanged (advisory only)",
                    provider,
                )
                metadata = data.get("metadata")
                if not isinstance(metadata, dict):
                    metadata = {}
                metadata["loci_circuit_open"] = provider
                data["metadata"] = metadata
        except Exception as exc:
            logger.warning("loci circuit_monitor: pre_call check failed, request unaffected: %r", exc)
        return None

    async def async_log_success_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Record a success for the model group."""
        try:
            provider = self._circuit_key(kwargs)
            if provider == "unknown":
                return
            with self._lock:
                health = self._providers.setdefault(provider, GroupHealth(provider))
            health.record_success()
        except Exception as exc:
            logger.warning("loci circuit_monitor: could not record a success event: %r", exc)

    async def async_log_failure_event(
        self,
        kwargs: dict,
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        """Record a failure for the model group, classified transient or permanent."""
        try:
            provider = self._circuit_key(kwargs)
            if provider == "unknown":
                return
            exc = kwargs.get("original_exception") or kwargs.get("exception")
            is_transient = isinstance(exc, BaseException) and _is_transient(exc)
            with self._lock:
                health = self._providers.setdefault(provider, GroupHealth(provider))
            if is_transient:
                health.record_failure()
                logger.debug(
                    "loci circuit_monitor: %s transient failure recorded (%d/%d consecutive)",
                    provider, health.failure_count, _FAILURE_THRESHOLD,
                )
            else:
                # config-type errors (auth, model not found) say nothing about provider health
                logger.info(
                    "loci circuit_monitor: %s failure not counted toward the circuit (%s)",
                    provider, type(exc).__name__ if exc is not None else "no exception on the event",
                )
        except Exception as err:
            logger.warning("loci circuit_monitor: could not record a failure event: %r", err)

    def get_status(self) -> dict:
        """Circuit status per model group seen in events. In-process only; no endpoint exposes it."""
        with self._lock:
            return {
                name: health.status()
                for name, health in self._providers.items()
            }


# ── Singleton instance ──────────────────────────────────────────────────────
loci_circuit_monitor = LociCircuitMonitor()
