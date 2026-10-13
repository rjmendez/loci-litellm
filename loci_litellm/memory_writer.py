from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any, Callable, Optional

import httpx
from litellm.integrations.custom_logger import CustomLogger
from loci_litellm import embed_breaker as _embed_breaker
from loci_litellm.backend_resolver import BackendResolver, build_embed_request, parse_embedding
from loci_litellm.redact import redact_secrets

logger = logging.getLogger("loci_litellm.memory_writer")


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


_warned_env: set[str] = set()  # mutable-ok: once-per-variable warning memo


def _env_number(name: str, default: Any, cast: Callable[[str], Any] = float) -> Any:
    """Read a numeric env var; on a bad value use the default and say so once (WARNING)."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return cast(raw.strip())
    except (TypeError, ValueError):
        if name not in _warned_env:
            _warned_env.add(name)
            logger.warning(
                "loci memory writer: %s=%r is not a valid %s; using default %r",
                name, raw, cast.__name__, default,
            )
        return default


# Cap on distinct failure reasons tracked, so odd error text cannot grow the dict.
_MAX_FAILURE_KEYS = 64


class LociMemoryWriter(CustomLogger):
    """Writes a (prompt preview, answer preview) record to Qdrant for each completed completion.

    - Fail-open: a failure never reaches the request. It is logged at WARNING (first occurrence
      of each reason, then every 100th) and counted in ``failure_counts``.
    - The upsert uses ``wait=false``: HTTP 200 means accepted, not applied or searchable.
    - Stored text is the prompt (first 300 chars) and answer (first LOCI_MEM_MAX_CHARS chars),
      with secrets redacted unless LOCI_MEM_REDACT_DISABLED is set.
    - ``investigation_id`` is the process-wide LOCI_INVESTIGATION_ID, stamped on every record.
    - Streams are recorded from litellm's success event; one litellm cannot assemble is skipped.
    """

    def __init__(
        self,
        _http_client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    ) -> None:
        super().__init__()
        self._resolver = BackendResolver
        self._http_client_factory: Callable[[], httpx.AsyncClient] = (
            _http_client_factory or (lambda: httpx.AsyncClient())
        )
        self.failure_counts: dict[str, int] = {}  # mutable-ok: operator-visible failure counter, bounded by _MAX_FAILURE_KEYS

    def _note_failure(self, reason: str) -> None:
        """Count a fail-open event and log it at WARNING (first of each reason, then every 100th)."""
        if reason not in self.failure_counts and len(self.failure_counts) >= _MAX_FAILURE_KEYS:
            reason = "other"
        count = self.failure_counts.get(reason, 0) + 1
        self.failure_counts[reason] = count
        if count == 1 or count % 100 == 0:
            logger.warning(
                "loci memory writer: record NOT written (fail-open): %s [occurrence %d]", reason, count
            )

    def _client_for_timeout(self, timeout: float) -> httpx.AsyncClient:
        client = self._http_client_factory()
        if hasattr(client, "timeout"):
            try:
                client.timeout = timeout  # pyright: ignore[reportAttributeAccessIssue]
            except Exception as exc:
                self._note_failure(
                    f"could not apply the configured timeout to the HTTP client ({type(exc).__name__}); client default applies"
                )
        return client

    @staticmethod
    def _text_of(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            bits = [p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") in ("text", "input_text")
                    and isinstance(p.get("text"), str)]
            return " ".join(b.strip() for b in bits if b.strip()).strip()
        return ""

    @staticmethod
    def _extract_user_message(kwargs: dict[str, Any]) -> str:
        messages: Any = kwargs.get("messages") or []
        if not messages:
            prompt: Any = kwargs.get("prompt")
            if isinstance(prompt, list) and prompt:
                prompt = prompt[-1]
            if isinstance(prompt, str) and prompt.strip():
                return prompt.strip()
            items: Any = kwargs.get("input")
            if isinstance(items, str):
                return items.strip()
            if isinstance(items, list):
                for item in reversed(items):
                    if isinstance(item, dict) and (item.get("role") or "user") == "user":
                        text = LociMemoryWriter._text_of(item.get("content"))
                        if text:
                            return text
            return ""
        if not isinstance(messages, list):
            return ""

        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "user":
                continue
            content: Any = message.get("content") or ""
            if isinstance(content, str):
                return content.strip()
            if isinstance(content, list):
                bits: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text: Any = part.get("text") or ""
                    if isinstance(text, str) and text.strip():
                        bits.append(text.strip())
                return " ".join(bits).strip()
        return ""

    def _extract_response_text(self, response_obj: Any) -> str:
        try:
            choices: Any = getattr(response_obj, "choices", None) or []
            if not choices:
                output: Any = getattr(response_obj, "output", None)
                if output is None and isinstance(response_obj, dict):
                    output = response_obj.get("output")
                bits: list[str] = []
                for item in output if isinstance(output, list) else []:
                    content: Any = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
                    for part in content if isinstance(content, list) else []:
                        text: Any = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
                        if isinstance(text, str) and text.strip():
                            bits.append(text.strip())
                return " ".join(bits).strip()
            completion_text: Any = getattr(choices[0], "text", None)
            if isinstance(completion_text, str) and completion_text.strip():
                return completion_text.strip()
            message: Any = choices[0].message
            content: Any = getattr(message, "content", "") or ""
            if isinstance(content, str):
                stripped = content.strip()
                if stripped:
                    return stripped
            if isinstance(content, list):
                bits: list[str] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    text: Any = part.get("text") or ""
                    if isinstance(text, str) and text.strip():
                        bits.append(text.strip())
                joined = " ".join(bits).strip()
                if joined:
                    return joined
            # Reasoning-only responses (empty content) are not returned: not an answer.
        except Exception as exc:
            self._note_failure(f"could not extract response text ({type(exc).__name__})")
            return ""
        return ""

    @staticmethod
    def _is_internal_call(kwargs: dict[str, Any]) -> bool:
        """True when the client set the metadata.loci_internal_call opt-out (honored unless
        LOCI_HONOR_INTERNAL_CALL=0). The flag is client-supplied, so any caller can suppress write-back."""
        params: Any = kwargs.get("litellm_params")
        params = params if isinstance(params, dict) else {}
        server_request: Any = params.get("proxy_server_request")
        body: Any = server_request.get("body") if isinstance(server_request, dict) else None
        sources = [kwargs.get("metadata"), params.get("metadata"),
                   body.get("metadata") if isinstance(body, dict) else None]
        return BackendResolver.is_internal_call(*sources)

    @staticmethod
    def _point_id_for(prompt: str, response: str, model: str = "") -> str:
        """Hash of model, prompt and response: only an identical triple overwrites an earlier record."""
        h = hashlib.sha256()
        for part in (model, prompt, response):
            h.update(part.encode("utf-8", "replace"))
            h.update(b"\x00")
        return h.hexdigest()[:32]

    async def _embed_text(self, text: str) -> Optional[list[float]]:
        request = build_embed_request(self._resolver, text, "document")
        if request is None:
            self._note_failure("embedding endpoint not configured (set LOCI_EMBED_API_BASE and LOCI_EMBED_MODEL)")
            return None
        url, headers, body = request
        timeout: float = _env_number("LOCI_MEM_EMBED_TIMEOUT", 3.0)
        breaker = _embed_breaker.get(url)
        if not breaker.allow():
            self._note_failure("embedder circuit open (embedding skipped)")
            return None
        try:
            async with self._client_for_timeout(timeout) as client:
                response = await client.post(url, json=body, headers=headers)
                response.raise_for_status()
                data: Any = response.json()
            vector = parse_embedding(data)
        except Exception as exc:
            breaker.record_failure()
            self._note_failure(f"embedding request failed ({type(exc).__name__})")
            return None
        breaker.record_success()
        return vector

    async def _upsert_point(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        qdrant_url: str = self._resolver.qdrant_url()
        if not qdrant_url:
            self._note_failure("no Qdrant URL resolved (set LOCI_QDRANT_URL)")
            return

        # wait=false: acceptance only; the write is unconfirmed.
        url: str = f"{qdrant_url.rstrip('/')}/collections/{collection}/points?wait=false"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key: str = self._resolver.qdrant_api_key()
        if api_key:
            headers["api-key"] = api_key

        body: dict[str, Any] = {
            "points": [
                {
                    "id": point_id,
                    "vector": {os.environ.get("LOCI_MEMORY_VECTOR_NAME", "dense"): vector},
                    "payload": payload,
                },
            ]
        }
        timeout: float = _env_number("LOCI_MEM_QDRANT_TIMEOUT", 2.0)
        async with self._client_for_timeout(timeout) as client:
            response = await client.put(url, json=body, headers=headers)
            response.raise_for_status()

    async def _write_qa_point(self, kwargs: dict[str, Any], prompt: str, response: str) -> None:
        min_prompt_len: int = _env_number("LOCI_MEM_MIN_PROMPT_LEN", 20, int)
        if len(prompt) < min_prompt_len or len(response) < min_prompt_len:
            return
        if not _env_flag("LOCI_MEM_REDACT_DISABLED"):
            prompt, response = redact_secrets(prompt), redact_secrets(response)

        # The vector embeds the prompt only; recall queries are also prompts.
        vector: Optional[list[float]] = await self._embed_text(prompt)
        if vector is None:
            return

        collection: str = self._resolver.memory_write_collection()
        prompt_text: str = prompt[:300]
        response_limit: int = _env_number("LOCI_MEM_MAX_CHARS", 500, int)
        response_preview: str = response[:response_limit]

        payload: dict[str, Any] = {
            "text": f"Q: {prompt_text}\nA: {response_preview}",
            "prompt": prompt_text,
            "response_preview": response_preview,
            "model": kwargs.get("model") or "",
            "created_at_ts": time.time(),
            "source": "litellm",
            # Unverified model output, so not "observed"/"medium".
            "record_type": "inferred",
            "confidence": "low",
            "provenance": "model_output",
            "pheromone": 0.0,
            # Process-global: the same id for every request.
            "investigation_id": os.environ.get("LOCI_INVESTIGATION_ID", ""),
        }
        point_id: str = self._point_id_for(prompt, response, str(kwargs.get("model") or ""))
        await self._upsert_point(collection, point_id, vector, payload)

    async def async_log_success_event(
        self,
        kwargs: dict[str, Any],
        response_obj: Any,
        start_time: float,
        end_time: float,
    ) -> None:
        try:
            if _env_flag("LOCI_MEMORY_WRITER_DISABLED") or self._is_internal_call(kwargs):
                return

            prompt: str = self._extract_user_message(kwargs)
            response: str = self._extract_response_text(response_obj)
            if not prompt or not response:
                return

            await self._write_qa_point(kwargs, prompt, response)
        except Exception as exc:
            self._note_failure(f"write-back failed ({type(exc).__name__}: {str(exc)[:120]})")
            return


loci_memory_writer = LociMemoryWriter()
