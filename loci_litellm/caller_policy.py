"""Hard request policy for a verified caller (the Hermes agent): enforced in the proxy, not advisory.

Applies only when the caller proved its identity: `x-loci-client` + `x-loci-client-secret` matching the owner
envelope (see client_keys.identify). For that caller, a pre-call violation is a rejection, not a log line:

  model_not_allowed          `model` is not a deployment name the proxy serves (no invented model names). The
                             set is read from the live router, never hardcoded (LOCI_POLICY_EXTRA_MODELS adds).
  malformed_tool_call_history  an assistant `tool_calls[].function.arguments` that is not valid JSON object text.
                             Such a turn is replayed on every later call, so it is refused at the boundary.
  policy_error               the policy itself failed. Fails CLOSED for a verified caller (503), never passes.

Unverified callers (no header, wrong secret) are not constrained here: the governor already caps them, and they
are logged by client_keys as `client_unknown`. Enforcement that depends on the caller's own claim would not be
enforcement.

Every rejection is appended to `policy_violations.jsonl` in the governor state dir, so violations are auditable.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

from litellm.integrations.custom_logger import CustomLogger

from loci_litellm.client_keys import identify
from loci_litellm.fileutil import append_line

logger = logging.getLogger("loci_litellm.caller_policy")

try:  # the proxy always has fastapi; offline tools may not
    from fastapi import HTTPException as _RejectionBase
except ImportError:  # pragma: no cover
    _RejectionBase = Exception  # type: ignore[assignment,misc]

ENFORCED_CLIENTS = frozenset({"hermes"})
VIOLATIONS_FILE = "policy_violations.jsonl"


class PolicyRejection(_RejectionBase):  # type: ignore[valid-type,misc]
    """Raised from the pre-call hook; the proxy turns it into an HTTP response."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.code = code
        self.body = {"error": {"type": "policy_rejection", "reason": code, "message": message}}
        if _RejectionBase is Exception:
            super().__init__(message)
            self.status_code = status_code
            self.detail = self.body
        else:
            super().__init__(status_code=status_code, detail=self.body)


def _violations_path() -> Path:
    base = os.environ.get("LOCI_GOVERNOR_DIR") or str(Path.home() / ".loci" / "governor")
    return Path(base) / VIOLATIONS_FILE


def _record(code: str, client: str, detail: str) -> None:
    row = {"ts": round(time.time(), 3), "client": client, "reason": code, "detail": detail[:300]}
    try:
        append_line(str(_violations_path()), json.dumps(row, separators=(",", ":")))
    except OSError as exc:  # the rejection still happens; only the audit line is lost
        logger.warning("caller policy: could not write violation record: %r", exc)


def _headers(data: dict) -> dict:
    request = data.get("proxy_server_request")
    headers = request.get("headers") if isinstance(request, dict) else None
    return {str(k).lower(): v for k, v in headers.items()} if isinstance(headers, dict) else {}


def _deployments() -> Optional[list]:
    """The live router's deployments, or None when the router is not reachable (local import keeps it cheap)."""
    from loci_litellm.colony_router import proxy_model_list

    return proxy_model_list()


def _router_models() -> Optional[set]:
    """Deployment names the live router serves (plus LOCI_POLICY_EXTRA_MODELS), or None when unreachable."""
    deployments = _deployments()
    if deployments is None:
        return None
    names = {str(d.get("model_name")) for d in deployments if isinstance(d, dict) and d.get("model_name")}
    extra = {m.strip() for m in os.environ.get("LOCI_POLICY_EXTRA_MODELS", "").split(",") if m.strip()}
    return names | extra


def _tool_call_args_bad(messages: Any) -> Optional[str]:
    """First assistant tool call whose arguments are not a JSON object, as a description; None when all parse."""
    if not isinstance(messages, list):
        return None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if not isinstance(call, dict):
                return f"message {index}: tool_call is not an object"
            fn = call.get("function") if isinstance(call.get("function"), dict) else {}
            args = fn.get("arguments", "")
            if isinstance(args, dict):
                continue
            try:
                parsed = json.loads(args) if isinstance(args, str) else None
            except ValueError as exc:
                return f"message {index}: {fn.get('name', '?')} arguments not JSON ({exc.msg})"
            if not isinstance(parsed, dict):
                return f"message {index}: {fn.get('name', '?')} arguments not a JSON object"
    return None


class LociCallerPolicy(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict: Any, cache: Any, data: dict, call_type: Any) -> Any:
        if not isinstance(data, dict):
            return None
        try:
            from loci_litellm.governor import loci_governor  # the governor owns the envelope (trust source)

            env = loci_governor.governor._envelope()
            who = identify(_headers(data), env.clients, env.caps.anonymous_day)
            if not (who.known and who.name in ENFORCED_CLIENTS):
                return None
            self._enforce(data, who.name)
        except PolicyRejection:
            raise
        except Exception as exc:
            # a verified caller must not pass when the policy itself breaks
            logger.error("caller policy: failed closed for a verified caller: %r", exc)
            _record("policy_error", "hermes", repr(exc))
            raise PolicyRejection(503, "policy_error", "the request policy could not be evaluated") from exc
        return None

    def _enforce(self, data: dict, client: str) -> None:
        models = _router_models()
        if models is None:
            raise PolicyRejection(503, "policy_error", "the router model list is unavailable")
        model = str(data.get("model") or "")
        if model not in models:
            _record("model_not_allowed", client, model)
            raise PolicyRejection(400, "model_not_allowed",
                                  f"model {model!r} is not served by this proxy; use one of GET /v1/models")
        bad = _tool_call_args_bad(data.get("messages"))
        if bad:
            _record("malformed_tool_call_history", client, bad)
            raise PolicyRejection(400, "malformed_tool_call_history",
                                  f"a prior tool call has invalid arguments ({bad}); fix and resend")


loci_caller_policy = LociCallerPolicy()
