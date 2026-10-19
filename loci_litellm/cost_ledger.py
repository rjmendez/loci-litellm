"""Cost and reliability ledger: one metadata-only JSON line per call reported through LiteLLM's
async callbacks (no prompt or response text). One row per attempt: a retry or fallback is its own
row, so cost_report counts are per attempt. Register `loci_litellm.cost_ledger.loci_cost_ledger`
in the LiteLLM config; LOCI_COST_LEDGER_DISABLED=1 turns it off.

Row fields (complete set): ts, call_id, trace_id, group, deployment, model, host, local, ok,
call_type, stream, prompt_tokens, completion_tokens, cache_read_tokens, cache_write_tokens,
empty_response, empty_suspect_stream, cost_usd, latency_s, cache_hit, error_class, internal, grounding,
agent (first 60 chars of the User-Agent), key_alias, session_id, has_tools, task_class, prefix_hash,
max_tokens, client_id, config_hash, cache_regime, finish_reason. `agent`, `key_alias` and `client_id`
identify callers: the file is sensitive.
cache_*_tokens are provider-reported and included in prompt_tokens; empty_response means a
successful call with no answer text and no tool call (a streamed call that reported 0 completion tokens is
labelled empty_suspect_stream instead (a key written only when true): usage is often missing on streams, so it is unverified, not a
failure, and stays out of empty_response); latency_s is null when unknown.
Correlation fields (never any prompt or response text): session_id is the first 12 hex characters of the
sha256 of the caller's session id (empty when none), never the raw id; has_tools is true when the request
offered tools or functions; task_class, client_id and prefix_hash are labels a router or hook put in the
request metadata (`loci_task_class`, `loci_client_id`, `loci_prefix_hash`; empty when absent, truncated to
24, 32 and 12 characters; prefix_hash is computed from the request by loci_litellm.prefix_hash when the
metadata has none, unless LOCI_LEDGER_PREFIX_HASH=0); max_tokens is the requested output cap (null when unset); config_hash is
LOCI_CONFIG_HASH, else the first 12 hex characters of the sha256 of the file named by CONFIG_FILE_PATH
(empty when neither is usable); cache_regime is the operator label in LOCI_CACHE_REGIME (empty when
unset); finish_reason is the first choice's finish reason (empty when unknown or the call failed).
`local` is
an address-class heuristic on the api_base host (loopback, link-local, private, CGNAT, localhost, or
a host in LOCI_LOCAL_HOSTS, comma separated). Calls with no group and no api_base are "(unrouted)".
File: LOCI_COST_LEDGER_FILE, else `[cost] ledger_file` in the private config, else
~/.loci/litellm_cost_ledger.jsonl. Append-only; at LOCI_COST_LEDGER_MAX_BYTES (default 50 MB, 0 =
unbounded) it rotates to <file>.1. Write failures log a WARNING and never fail the request.
"""
from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from litellm.integrations.custom_logger import CustomLogger
from loci_litellm.backend_resolver import BackendResolver, ledger_path  # noqa: F401  (re-exported)
from loci_litellm.fileutil import append_line, env_int
from loci_litellm.prefix_hash import prefix_hash, request_of

logger = logging.getLogger("loci_litellm.cost_ledger")

_WRITE_LOCK = threading.Lock()

# Group for events with no model group and no api_base (rejected before routing).
UNROUTED = "(unrouted)"


def _disabled() -> bool:
    return os.environ.get("LOCI_COST_LEDGER_DISABLED", "").strip().lower() in ("1", "true", "yes", "on")


_CGNAT = ipaddress.ip_network("100.64.0.0/10")


def _local_hosts() -> set:
    raw = os.environ.get("LOCI_LOCAL_HOSTS", "")
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


def _local_providers() -> set:
    raw = os.environ.get("LOCI_LOCAL_PROVIDERS", "")
    return {p.strip().lower() for p in raw.split(",") if p.strip()}


def _is_local(api_base: str) -> bool:
    """Address-class heuristic: loopback, link-local, private, CGNAT (Tailscale) or a host named in
    LOCI_LOCAL_HOSTS. Any other hostname is treated as not local."""
    host = (urlparse(api_base or "").hostname or "").lower()
    if not host:
        return False
    if host == "localhost" or host in _local_hosts():
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local or ip in _CGNAT


_WARN_EVERY = 100
_warn_counts: dict = {}


def _warn_throttled(kind: str, message: str, *args: Any) -> None:
    """Log the first occurrence of `kind` and then every _WARN_EVERY-th, with the running count."""
    count = _warn_counts.get(kind, 0) + 1
    _warn_counts[kind] = count
    if count == 1 or count % _WARN_EVERY == 0:
        logger.warning(message + " (occurrence %d)", *args, count)


def _is_local_deployment(provider: str, model: str, api_base: str) -> bool:
    """A provider named in LOCI_LOCAL_PROVIDERS is local whatever its host name; anything else only when the
    host is a local address."""
    local = _local_providers()
    if local:
        prefix = (model or "").lower().split("/", 1)[0]
        if (provider or "").lower() in local or ("/" in (model or "") and prefix in local):
            return True
    return _is_local(api_base)


_COMPLETION_CALLS = ("completion", "text_completion", "acompletion", "atext_completion", "responses",
                     "aresponses", "anthropic_messages", "generate_content", "agenerate_content")


def _usage_of(response_obj: Any, slo: dict) -> Any:
    usage = getattr(response_obj, "usage", None)
    if usage is None and isinstance(response_obj, dict):
        usage = response_obj.get("usage")
    if usage is None:
        resp = slo.get("response")
        usage = resp.get("usage") if isinstance(resp, dict) else None
    return usage


def _field(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _cache_tokens(response_obj: Any, slo: dict) -> tuple[int, int]:
    """(cache_read, cache_write) prompt tokens as reported, 0 when absent. Reads OpenAI-shaped
    prompt_tokens_details first, then the Anthropic-shaped usage fields."""
    usage = _usage_of(response_obj, slo)
    if usage is None:
        return 0, 0
    details = _field(usage, "prompt_tokens_details")
    read = _field(details, "cached_tokens") if details is not None else None
    write = _field(details, "cache_write_tokens") if details is not None else None
    if write is None and details is not None:
        write = _field(details, "cache_creation_tokens")
    if read is None:
        read = _field(usage, "cache_read_input_tokens") or _field(usage, "_cache_read_input_tokens")
    if write is None:
        write = _field(usage, "cache_creation_input_tokens") or _field(usage, "_cache_creation_input_tokens")
    return int(_num(read)), int(_num(write))


def _text_state(call_type: str, response_obj: Any) -> str:
    """"text" when the response carries answer text or a tool call, "empty" when it has chat-completion
    choices that carry neither, "unknown" for non-text call types and shapes without choices."""
    if call_type not in _COMPLETION_CALLS or response_obj is None:
        return "unknown"
    choices = getattr(response_obj, "choices", None)
    if choices is None and isinstance(response_obj, dict):
        choices = response_obj.get("choices")
    if not choices:
        return "unknown"                  # not a chat-completion shape
    for choice in choices:
        message = _field(choice, "message") or _field(choice, "delta")
        text = _field(message, "content") if message is not None else None
        if text is None:
            text = _field(choice, "text")
        if isinstance(text, list):
            text = "".join(str(_field(part, "text") or "") for part in text)
        if isinstance(text, str) and text.strip():
            return "text"
        if message is not None and (_field(message, "tool_calls") or _field(message, "function_call")):
            return "text"
    return "empty"


def _first_finish_reason(response_obj: Any) -> str:
    choices = getattr(response_obj, "choices", None)
    if choices is None and isinstance(response_obj, dict):
        choices = response_obj.get("choices")
    if not choices or not isinstance(choices, (list, tuple)):
        return ""
    return str(_field(choices[0], "finish_reason") or "")[:_FINISH_LEN]


_SESSION_HASH_LEN = 12
_PREFIX_LEN = 12
_TASK_CLASS_LEN = 24
_CLIENT_ID_LEN = 32
_CONFIG_HASH_LEN = 12
_CACHE_REGIME_LEN = 24
_FINISH_LEN = 20


def _session_hash(session_id: Any) -> str:
    """Truncated sha256 of a session id: enough to group one session's calls, not enough to recover or
    replay the id."""
    raw = str(session_id or "").strip()
    if not raw:
        return ""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_SESSION_HASH_LEN]


_config_hash_cache: dict = {}


def _config_hash() -> str:
    """LOCI_CONFIG_HASH, else a hash of the proxy config file (re-read only when its size or mtime
    changes); empty, with a throttled warning, when the file is named but unreadable."""
    explicit = os.environ.get("LOCI_CONFIG_HASH", "").strip()
    if explicit:
        return explicit[:_CONFIG_HASH_LEN]
    path = os.environ.get("CONFIG_FILE_PATH", "").strip()
    if not path:
        return ""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
        if _config_hash_cache.get("key") != key:
            digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:_CONFIG_HASH_LEN]
            _config_hash_cache.update(key=key, value=digest)
        return str(_config_hash_cache["value"])
    except OSError as exc:
        _warn_throttled("config_hash", "loci cost ledger: could not hash the proxy config: %r", exc)
        return ""


def _meta_str(sources: tuple, names: tuple, limit: int) -> str:
    for source in sources:
        if isinstance(source, dict):
            for name in names:
                value = source.get(name)
                if value:
                    return str(value)[:limit]
    return ""


def _request_prefix_hash(kwargs: dict) -> str:
    """Hash of the request's cacheable prefix (see prefix_hash), computed here when no hook supplied one.
    LOCI_LEDGER_PREFIX_HASH=0 turns the computation off; a failure leaves the field empty and is counted in
    the throttled warning."""
    if os.environ.get("LOCI_LEDGER_PREFIX_HASH", "1").strip().lower() in ("0", "false", "no", "off"):
        return ""
    try:
        return prefix_hash(request_of(kwargs))[:_PREFIX_LEN]
    except Exception as exc:
        _warn_throttled("prefix_hash", "loci cost ledger: could not hash the request prefix: %r", exc)
        return ""


def _max_tokens(slo: dict, kwargs: dict) -> Optional[int]:
    for source in (slo.get("model_parameters"), kwargs.get("optional_params")):
        if isinstance(source, dict):
            for name in ("max_completion_tokens", "max_tokens"):
                value = source.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
                    return int(value)
    return None


def _has_tools(slo: dict, kwargs: dict) -> bool:
    for source in (kwargs.get("optional_params"), slo.get("model_parameters"), kwargs):
        if isinstance(source, dict) and (source.get("tools") or source.get("functions")):
            return True
    return False


def _num(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


class LociCostLedger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def build_row(kwargs: dict[str, Any], ok: bool, start_time: Any, end_time: Any,
                  response_obj: Any = None) -> Optional[dict[str, Any]]:
        slo: Any = kwargs.get("standard_logging_object")
        slo = slo if isinstance(slo, dict) else {}
        params: Any = kwargs.get("litellm_params")
        params = params if isinstance(params, dict) else {}
        meta_slo: Any = slo.get("metadata")
        meta_slo = meta_slo if isinstance(meta_slo, dict) else {}
        meta_params: Any = params.get("metadata")
        meta_params = meta_params if isinstance(meta_params, dict) else {}
        requester: Any = meta_slo.get("requester_metadata")
        api_base = str(slo.get("api_base") or params.get("api_base") or kwargs.get("api_base") or "")
        group = slo.get("model_group") or meta_params.get("model_group") or kwargs.get("model") or ""
        latency: Optional[float]
        try:
            latency = float(slo.get("response_time") or (end_time - start_time).total_seconds())
        except Exception:
            latency = None
        error = slo.get("error_information") if isinstance(slo.get("error_information"), dict) else {}
        call_type = str(slo.get("call_type") or kwargs.get("call_type") or "")
        cache_read, cache_write = _cache_tokens(response_obj, slo)
        streamed = bool(slo.get("stream") or kwargs.get("stream"))
        completion_tokens = int(_num(slo.get("completion_tokens")))
        state = _text_state(call_type, response_obj)
        # A stream that reports no completion tokens and shows no text cannot be told from a stream whose
        # usage was simply not reported: label it suspect instead of counting a failed answer.
        suspect = (bool(ok) and streamed and call_type in _COMPLETION_CALLS
                   and completion_tokens == 0 and state != "text")
        sources = (meta_params, meta_slo, requester)
        row: dict[str, Any] = {
            "ts": round(time.time(), 3),
            "call_id": str(slo.get("litellm_call_id") or kwargs.get("litellm_call_id") or ""),
            "trace_id": str(slo.get("trace_id") or ""),
            "group": str(group) or UNROUTED,
            "deployment": str(slo.get("model_id") or meta_params.get("model_id") or ""),
            "model": str(slo.get("model") or kwargs.get("model") or ""),
            "host": urlparse(api_base).hostname or "",
            "local": _is_local_deployment(str(slo.get("custom_llm_provider") or ""),
                                          str(slo.get("model") or kwargs.get("model") or ""), api_base),
            "ok": bool(ok),
            "call_type": call_type,
            "stream": streamed,
            "prompt_tokens": int(_num(slo.get("prompt_tokens"))),
            "completion_tokens": completion_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
            "empty_response": bool(ok) and state == "empty" and not suspect,
            "cost_usd": _num(slo.get("response_cost") or kwargs.get("response_cost")),
            "latency_s": None if latency is None else round(latency, 3),
            "cache_hit": bool(slo.get("cache_hit")),
            "error_class": str(error.get("error_class") or error.get("error_code") or "") if not ok else "",
            "internal": BackendResolver.is_internal_call(kwargs.get("metadata"), meta_params, requester),
            "grounding": str(meta_params.get("loci_grounding_status") or ""),
            "agent": str(slo.get("user_agent") or "")[:60],
            "key_alias": str(meta_slo.get("user_api_key_alias") or ""),
            "session_id": _session_hash(slo.get("session_id") or _meta_str(sources, ("session_id",), 256)),
            "has_tools": _has_tools(slo, kwargs),
            "task_class": _meta_str(sources, ("loci_task_class", "task_class"), _TASK_CLASS_LEN),
            "prefix_hash": (_meta_str(sources, ("loci_prefix_hash",), _PREFIX_LEN)
                            or _request_prefix_hash(kwargs)),
            "max_tokens": _max_tokens(slo, kwargs),
            "client_id": _meta_str(sources, ("loci_client_id", "client_id"), _CLIENT_ID_LEN),
            "config_hash": _config_hash(),
            "cache_regime": os.environ.get("LOCI_CACHE_REGIME", "").strip()[:_CACHE_REGIME_LEN],
            "finish_reason": _first_finish_reason(response_obj) if ok else "",
        }
        if suspect:
            row["empty_suspect_stream"] = True      # written only when true: keeps the common row small
        return row

    @staticmethod
    def _append(row: dict[str, Any]) -> None:
        line = json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
        with _WRITE_LOCK:
            append_line(ledger_path(), line, env_int("LOCI_COST_LEDGER_MAX_BYTES", 50_000_000))

    async def _record(self, kwargs: dict[str, Any], ok: bool, start_time: Any, end_time: Any,
                      response_obj: Any = None) -> None:
        if _disabled():
            return
        try:
            row = self.build_row(kwargs, ok, start_time, end_time, response_obj)
            await asyncio.to_thread(self._append, row)
        except Exception as exc:
            _warn_throttled("write", "loci cost ledger: could not record a call: %r", exc)

    async def async_log_success_event(self, kwargs: dict[str, Any], response_obj: Any,
                                      start_time: Any, end_time: Any) -> None:
        await self._record(kwargs, True, start_time, end_time, response_obj)

    async def async_log_failure_event(self, kwargs: dict[str, Any], response_obj: Any,
                                      start_time: Any, end_time: Any) -> None:
        await self._record(kwargs, False, start_time, end_time)


loci_cost_ledger = LociCostLedger()
