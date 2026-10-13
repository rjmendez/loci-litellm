"""Requests the memory system makes for itself carry `metadata.loci_internal_call`; they are neither grounded nor
written back, so Loci's judges and classifiers do not pollute memory or retrieve their own output."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from loci_litellm.backend_resolver import BackendResolver
from loci_litellm.grounding_hook import LociGroundingHook
from loci_litellm.memory_writer import LociMemoryWriter
from tests.test_memory_writer import FakeAsyncClient

PROMPT = "Please explain the Loci write-back flow and how Qdrant upserts memory records."
ANSWER = "The writer embeds the question and answer, then upserts them with a deterministic point ID."


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_HONOR_INTERNAL_CALL", raising=False)
    monkeypatch.delenv("LOCI_MEMORY_WRITER_DISABLED", raising=False)
    monkeypatch.delenv("LOCI_GROUNDING_DISABLED", raising=False)
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")


def test_marker_detection() -> None:
    assert BackendResolver.is_internal_call({"loci_internal_call": True})
    assert not BackendResolver.is_internal_call({"loci_internal_call": "yes"}, None, {})
    assert not BackendResolver.is_internal_call(None)


def test_the_marker_can_be_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_HONOR_INTERNAL_CALL", "0")
    assert not BackendResolver.is_internal_call({"loci_internal_call": True})


@pytest.mark.asyncio
async def test_an_internal_call_is_not_grounded() -> None:
    hook = LociGroundingHook()
    data = {"model": "m", "messages": [{"role": "user", "content": PROMPT}],
            "metadata": {"loci_internal_call": True}}

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert result is None
    assert data["metadata"]["loci_grounding_status"] == "skipped"
    assert data["metadata"]["loci_grounding_reason"] == "internal_call"
    assert [m["role"] for m in data["messages"]] == ["user"]


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["metadata", "litellm_params", "proxy_body"])
async def test_an_internal_call_is_not_written_back(where: str) -> None:
    marker = {"loci_internal_call": True}
    kwargs = {"model": "m", "messages": [{"role": "user", "content": PROMPT}]}
    if where == "metadata":
        kwargs["metadata"] = marker
    elif where == "litellm_params":
        kwargs["litellm_params"] = {"metadata": marker}
    else:
        kwargs["litellm_params"] = {"proxy_server_request": {"body": {"metadata": marker}}}
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=ANSWER))])

    await writer.async_log_success_event(kwargs, response, 0.0, 0.0)

    assert client.calls == []


@pytest.mark.asyncio
async def test_ordinary_traffic_is_written_back() -> None:
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=ANSWER))])

    await writer.async_log_success_event({"model": "m", "messages": [{"role": "user", "content": PROMPT}],
                                          "litellm_params": {"metadata": {"other": 1}}}, response, 0.0, 0.0)

    assert any(c["method"] == "PUT" for c in client.calls)
