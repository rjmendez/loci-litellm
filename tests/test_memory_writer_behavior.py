from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from litellm.integrations.custom_logger import CustomLogger

from loci_litellm.memory_writer import LociMemoryWriter
from tests.test_memory_writer import FakeAsyncClient

PROMPT = "Please explain the Loci write-back flow and how Qdrant upserts memory records."
ANSWER = "The writer embeds the question and answer, then upserts them into the memory collection."


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    monkeypatch.delenv("LOCI_MEMORY_WRITER_DISABLED", raising=False)
    monkeypatch.delenv("LOCI_INVESTIGATION_ID", raising=False)
    monkeypatch.delenv("LOCI_MEM_MAX_CHARS", raising=False)


def _resp(text: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def _kw(prompt: str = PROMPT, model: str = "m1") -> dict:
    return {"model": model, "messages": [{"role": "user", "content": prompt}]}


async def _run(kwargs=None, response=None, client=None):
    client = client or FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    await writer.async_log_success_event(kwargs or _kw(), response or _resp(ANSWER), 0.0, 0.0)
    return writer, client


def _puts(client):
    return [c for c in client.calls if c["method"] == "PUT"]


# --- failures are logged and counted ---

@pytest.mark.asyncio
async def test_embed_failure_warns_counts_and_attempted_embedding(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="loci_litellm.memory_writer")
    writer, client = await _run(client=FakeAsyncClient(embed_exception=ConnectionError("down")))

    posts = [c for c in client.calls if c["method"] == "POST"]
    assert len(posts) == 1 and "/embeddings" in posts[0]["url"]
    assert _puts(client) == []
    assert any(r.levelno == logging.WARNING and "embedding request failed" in r.getMessage()
               for r in caplog.records)
    assert sum(writer.failure_counts.values()) >= 1


@pytest.mark.asyncio
async def test_qdrant_failure_warns_and_counts(caplog) -> None:
    caplog.set_level(logging.WARNING, logger="loci_litellm.memory_writer")
    writer, client = await _run(client=FakeAsyncClient(qdrant_exception=ConnectionError("qdrant down")))

    assert len(_puts(client)) == 1
    assert any(r.levelno == logging.WARNING and "write-back failed" in r.getMessage()
               for r in caplog.records)
    assert sum(writer.failure_counts.values()) == 1


@pytest.mark.asyncio
async def test_missing_qdrant_url_is_reported(monkeypatch, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="loci_litellm.memory_writer")
    from loci_litellm.backend_resolver import BackendResolver
    monkeypatch.setattr(BackendResolver, "qdrant_url", staticmethod(lambda: ""))
    writer, client = await _run()

    assert _puts(client) == []
    assert any("no Qdrant URL" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_repeated_failures_log_once_but_count_every_time(monkeypatch, caplog) -> None:
    monkeypatch.setenv("LOCI_EMBED_BREAKER_DISABLED", "1")   # this test is about the counting, not the breaker
    caplog.set_level(logging.WARNING, logger="loci_litellm.memory_writer")
    client = FakeAsyncClient(embed_exception=ConnectionError("down"))
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    for _ in range(3):
        await writer.async_log_success_event(_kw(), _resp(ANSWER), 0.0, 0.0)

    assert list(writer.failure_counts.values()) == [3]
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 1


# --- bad numeric env ---

@pytest.mark.asyncio
async def test_bad_numeric_env_falls_back_with_warning(monkeypatch, caplog) -> None:
    caplog.set_level(logging.WARNING, logger="loci_litellm.memory_writer")
    import loci_litellm.memory_writer as mw
    mw._warned_env.discard("LOCI_MEM_MAX_CHARS")
    monkeypatch.setenv("LOCI_MEM_MAX_CHARS", "lots")
    _, client = await _run()

    assert len(_puts(client)) == 1
    assert any("LOCI_MEM_MAX_CHARS" in r.getMessage() and r.levelno == logging.WARNING
               for r in caplog.records)


# --- provenance ---

@pytest.mark.asyncio
async def test_payload_does_not_claim_observed_or_medium_confidence() -> None:
    _, client = await _run()
    payload = _puts(client)[0]["kwargs"]["json"]["points"][0]["payload"]

    assert payload["record_type"] == "inferred"
    assert payload["confidence"] == "low"
    assert payload["provenance"] == "model_output"
    assert payload["source"] == "litellm"
    assert "importance" not in payload


# --- point id ---

def test_point_id_distinguishes_long_shared_prefixes_and_models() -> None:
    f = LociMemoryWriter._point_id_for
    base_p = "x" * 250
    assert f(base_p + "A", "resp") != f(base_p + "B", "resp")
    assert f("p", "r" * 150 + "1") != f("p", "r" * 150 + "2")
    assert f("p", "r", "m1") != f("p", "r", "m2")
    assert f("ab", "c") != f("a", "bc")
    assert f("p", "r", "m") == f("p", "r", "m")


@pytest.mark.asyncio
async def test_put_id_equals_computed_id_and_embeds_prompt_only() -> None:
    _, client = await _run()
    point = _puts(client)[0]["kwargs"]["json"]["points"][0]
    assert point["id"] == LociMemoryWriter._point_id_for(PROMPT, ANSWER, "m1")
    embed = [c for c in client.calls if c["method"] == "POST"][0]
    assert embed["kwargs"]["json"]["input"] == PROMPT


# --- truncation ---

@pytest.mark.asyncio
async def test_prompt_and_answer_are_truncated_to_limits() -> None:
    _, client = await _run(_kw("p" * 1000), _resp("a" * 2000))
    payload = _puts(client)[0]["kwargs"]["json"]["points"][0]["payload"]
    assert len(payload["prompt"]) == 300
    assert len(payload["response_preview"]) == 500


# --- reasoning fields ---

@pytest.mark.asyncio
async def test_reasoning_fields_are_not_used_as_the_answer() -> None:
    msg = SimpleNamespace(content="", reasoning_content="step by step thinking that is long enough",
                          provider_specific_fields={"reasoning": "more thinking that is long enough"})
    _, client = await _run(response=SimpleNamespace(choices=[SimpleNamespace(message=msg)]))
    assert client.calls == []


# --- streaming ---

def test_no_stream_buffering_and_no_stream_override() -> None:
    w = LociMemoryWriter()
    assert not hasattr(w, "_stream_buffers")
    assert LociMemoryWriter.async_log_stream_event is CustomLogger.async_log_stream_event


@pytest.mark.asyncio
async def test_a_streamed_call_is_written_once_from_the_assembled_response() -> None:
    kwargs = _kw()
    kwargs["stream"] = True
    writer, client = await _run(kwargs, _resp(ANSWER))
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="par"), finish_reason="stop")])
    await writer.async_log_stream_event(kwargs, chunk, 0.0, 0.0)
    assert len(_puts(client)) == 1


# --- internal-call marker ---

def test_internal_call_docstring_names_the_client_flag() -> None:
    assert "client" in LociMemoryWriter._is_internal_call.__doc__.lower()
