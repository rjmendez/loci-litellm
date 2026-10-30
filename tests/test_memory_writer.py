from __future__ import annotations

from types import SimpleNamespace

import pytest

from loci_litellm.memory_writer import LociMemoryWriter


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> object:
        return self._payload


class FakeAsyncClient:
    def __init__(
        self,
        *,
        embed_payload: object | None = None,
        qdrant_payload: object | None = None,
        embed_exception: Exception | None = None,
        qdrant_exception: Exception | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self._embed_payload = embed_payload or {"data": [{"embedding": [0.1] * 768}]}
        self._qdrant_payload = qdrant_payload or {"result": []}
        self._embed_exception = embed_exception
        self._qdrant_exception = qdrant_exception

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"method": "POST", "url": url, "kwargs": kwargs})
        if "/embeddings" in url:
            if self._embed_exception is not None:
                raise self._embed_exception
            return FakeResponse(self._embed_payload)
        raise AssertionError(f"Unexpected POST URL: {url}")

    async def put(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append({"method": "PUT", "url": url, "kwargs": kwargs})
        if "/points?wait=false" in url:
            if self._qdrant_exception is not None:
                raise self._qdrant_exception
            return FakeResponse(self._qdrant_payload)
        raise AssertionError(f"Unexpected PUT URL: {url}")


def _response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
    )


def _reasoning_only_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=None,
                    reasoning_content=text,
                    provider_specific_fields={"reasoning": text},
                )
            )
        ],
    )


@pytest.mark.asyncio
async def test_stores_qa_pair_to_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    monkeypatch.delenv("LOCI_MEMORY_WRITER_DISABLED", raising=False)
    monkeypatch.delenv("LOCI_INVESTIGATION_ID", raising=False)

    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    prompt = "Please explain the Loci write-back flow and how Qdrant upserts memory records."
    response = (
        "The writer embeds the question and answer, then upserts the result into the "
        "loci_memory collection with a deterministic point ID."
    )

    await writer.async_log_success_event(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
        _response(response),
        0.0,
        0.0,
    )

    qdrant_calls = [call for call in client.calls if call["method"] == "PUT"]
    assert len(qdrant_calls) == 1
    point = qdrant_calls[0]["kwargs"]["json"]["points"][0]
    payload = point["payload"]
    assert point["vector"]["dense"] == [0.1] * 768
    assert prompt[:300] in payload["text"]
    assert response[:500] in payload["text"]


@pytest.mark.asyncio
async def test_deterministic_point_id() -> None:
    writer = LociMemoryWriter()
    prompt = "Ask the memory writer to store this exact request for comparison."
    response = "This is a deterministic payload used to validate point IDs."

    point_id = writer._point_id_for(prompt, response)
    assert point_id == writer._point_id_for(prompt, response)


@pytest.mark.asyncio
async def test_skips_short_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)

    await writer.async_log_success_event(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "short"}]},
        _response("This answer is long enough to pass if the prompt were long enough."),
        0.0,
        0.0,
    )

    assert client.calls == []


@pytest.mark.asyncio
async def test_fail_open_on_embed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    client = FakeAsyncClient(embed_exception=ConnectionError("embed failed"))
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    prompt = "This is long enough to trigger the embedding request but it should fail open."
    response = "The embed request fails, and the writer should swallow the error without raising."

    await writer.async_log_success_event(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
        _response(response),
        0.0,
        0.0,
    )

    assert all(call["method"] != "PUT" for call in client.calls)


@pytest.mark.asyncio
async def test_fail_open_on_qdrant_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    client = FakeAsyncClient(qdrant_exception=ConnectionError("qdrant failed"))
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    prompt = "This prompt is long enough for the embedding call, but the qdrant upsert fails."
    response = "The writer should fail-open and avoid raising on the Qdrant write failure."

    await writer.async_log_success_event(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
        _response(response),
        0.0,
        0.0,
    )

    assert any(call["method"] == "PUT" for call in client.calls)


@pytest.mark.asyncio
async def test_disabled_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_MEMORY_WRITER_DISABLED", "1")
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)

    await writer.async_log_success_event(
        {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "This prompt is long enough to write."}],
        },
        _response("This response is long enough to store and should be skipped when disabled."),
        0.0,
        0.0,
    )

    assert client.calls == []


@pytest.mark.asyncio
async def test_investigation_id_in_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    monkeypatch.setenv("LOCI_INVESTIGATION_ID", "inv-42")
    monkeypatch.delenv("LOCI_MEMORY_WRITER_DISABLED", raising=False)

    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    prompt = "This prompt is intentionally long enough to write into the investigation collection."
    response = "The resulting payload should include the investigation_id field for attribution."

    await writer.async_log_success_event(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
        _response(response),
        0.0,
        0.0,
    )

    qdrant_calls = [call for call in client.calls if call["method"] == "PUT"]
    assert qdrant_calls
    payload = qdrant_calls[0]["kwargs"]["json"]["points"][0]["payload"]
    assert payload["investigation_id"] == "inv-42"


@pytest.mark.asyncio
async def test_reasoning_only_response_is_not_stored_as_an_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    prompt = "This prompt is long enough to verify reasoning-only provider responses are skipped."
    reasoning = "The provider returned reasoning text without a content field."

    await writer.async_log_success_event(
        {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": prompt}]},
        _reasoning_only_response(reasoning),
        0.0,
        0.0,
    )

    assert client.calls == []


# ── tenant tagging (LOCI_GROUNDING_TENANT_SCOPE) ───────────────────────────────

def _env_for_write(monkeypatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    monkeypatch.delenv("LOCI_MEMORY_WRITER_DISABLED", raising=False)


_PROMPT = "Explain the Loci write-back flow and how Qdrant upserts a memory record for this request."
_RESP = "The writer embeds the question and upserts the result into the loci_memory collection with an id."


def _kw(client_header: str | None):
    kw = {"model": "m", "messages": [{"role": "user", "content": _PROMPT}]}
    if client_header is not None:
        kw["litellm_params"] = {"proxy_server_request": {"headers": {"x-loci-client": client_header}}}
    return kw


async def _written_payload(monkeypatch, scope: str | None, client_header: str | None) -> dict:
    _env_for_write(monkeypatch)
    if scope is None:
        monkeypatch.delenv("LOCI_GROUNDING_TENANT_SCOPE", raising=False)
    else:
        monkeypatch.setenv("LOCI_GROUNDING_TENANT_SCOPE", scope)
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    await writer.async_log_success_event(_kw(client_header), _response(_RESP), 0.0, 0.0)
    put = [c for c in client.calls if c["method"] == "PUT"]
    assert len(put) == 1
    return put[0]["kwargs"]["json"]["points"][0]["payload"]


@pytest.mark.asyncio
async def test_write_tags_tenant_when_scoping_on_and_client_named(monkeypatch) -> None:
    payload = await _written_payload(monkeypatch, scope="1", client_header="Hermes")
    assert payload["tenant"] == "hermes"


@pytest.mark.asyncio
async def test_write_omits_tenant_when_anonymous(monkeypatch) -> None:
    # scoping on but no client header -> no tenant field (is_empty = shared/legacy on retrieval)
    payload = await _written_payload(monkeypatch, scope="1", client_header=None)
    assert "tenant" not in payload


@pytest.mark.asyncio
async def test_write_omits_tenant_when_scoping_off(monkeypatch) -> None:
    # positive twin: flag off -> no tenant field even when the caller named itself
    payload = await _written_payload(monkeypatch, scope=None, client_header="hermes")
    assert "tenant" not in payload


# ── internal-call opt-out gating (LOCI_INTERNAL_CALL_REQUIRES_CLIENT) ───────────

def _ic_kwargs(marker: bool, client: str | None):
    lp: dict = {"metadata": {"loci_internal_call": marker}}
    if client is not None:
        lp["proxy_server_request"] = {"headers": {"x-loci-client": client}}
    return {"litellm_params": lp}


def test_internal_marker_is_honored_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LOCI_INTERNAL_CALL_REQUIRES_CLIENT", raising=False)
    assert LociMemoryWriter._is_internal_call(_ic_kwargs(True, None)) is True


def test_internal_marker_ignored_without_client_when_required(monkeypatch) -> None:
    monkeypatch.setenv("LOCI_INTERNAL_CALL_REQUIRES_CLIENT", "1")
    assert LociMemoryWriter._is_internal_call(_ic_kwargs(True, None)) is False


def test_internal_marker_honored_with_client_when_required(monkeypatch) -> None:
    # positive twin: the same marker IS honored once the caller identifies itself
    monkeypatch.setenv("LOCI_INTERNAL_CALL_REQUIRES_CLIENT", "1")
    assert LociMemoryWriter._is_internal_call(_ic_kwargs(True, "hermes")) is True


def test_no_marker_is_never_internal(monkeypatch) -> None:
    monkeypatch.setenv("LOCI_INTERNAL_CALL_REQUIRES_CLIENT", "1")
    assert LociMemoryWriter._is_internal_call(_ic_kwargs(False, "hermes")) is False
