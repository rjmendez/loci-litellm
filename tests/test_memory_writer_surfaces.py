from __future__ import annotations

from types import SimpleNamespace

import pytest

from loci_litellm.memory_writer import LociMemoryWriter
from tests.test_memory_writer import FakeAsyncClient

PROMPT = "Please explain the Loci write-back flow and how Qdrant upserts memory records."
ANSWER = "The writer embeds the question and answer, then upserts them with a deterministic point ID."


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    monkeypatch.delenv("LOCI_MEMORY_WRITER_DISABLED", raising=False)
    monkeypatch.delenv("LOCI_INVESTIGATION_ID", raising=False)


async def _written(kwargs: dict, response: object) -> list[dict]:
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)

    await writer.async_log_success_event(kwargs, response, 0.0, 0.0)

    puts = [c for c in client.calls if c["method"] == "PUT"]
    return [c["kwargs"]["json"]["points"][0]["payload"] for c in puts]


def _responses_output(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])],
    )


@pytest.mark.asyncio
async def test_a_responses_api_call_is_written_back() -> None:
    kwargs = {"model": "test-model", "input": [{"role": "user", "content": [{"type": "input_text", "text": PROMPT}]}]}

    (payload,) = await _written(kwargs, _responses_output(ANSWER))

    assert payload["prompt"] == PROMPT
    assert payload["response_preview"] == ANSWER
    assert payload["source"] == "litellm"


@pytest.mark.asyncio
async def test_a_responses_string_input_with_a_dict_shaped_output_is_written_back() -> None:
    response = {"output": [{"type": "message", "content": [{"type": "output_text", "text": ANSWER}]}]}

    (payload,) = await _written({"model": "test-model", "input": PROMPT}, response)

    assert (payload["prompt"], payload["response_preview"]) == (PROMPT, ANSWER)


@pytest.mark.asyncio
async def test_the_latest_user_item_of_a_responses_input_list_is_the_prompt() -> None:
    kwargs = {"model": "test-model", "input": [
        {"role": "user", "content": "An older question that is long enough to be stored on its own."},
        {"role": "assistant", "content": [{"type": "output_text", "text": "older answer"}]},
        {"role": "user", "content": [{"type": "input_text", "text": PROMPT}]},
    ]}

    (payload,) = await _written(kwargs, _responses_output(ANSWER))

    assert payload["prompt"] == PROMPT


@pytest.mark.asyncio
async def test_a_text_completion_is_written_back() -> None:
    response = SimpleNamespace(choices=[SimpleNamespace(text=ANSWER)])

    (payload,) = await _written({"model": "test-model", "prompt": PROMPT}, response)

    assert (payload["prompt"], payload["response_preview"]) == (PROMPT, ANSWER)


@pytest.mark.asyncio
async def test_a_responses_call_with_no_output_text_writes_nothing() -> None:
    response = SimpleNamespace(output=[SimpleNamespace(type="reasoning", content=[])])

    assert await _written({"model": "test-model", "input": PROMPT}, response) == []


@pytest.mark.asyncio
async def test_a_short_responses_prompt_is_skipped() -> None:
    assert await _written({"model": "test-model", "input": "short"}, _responses_output(ANSWER)) == []


@pytest.mark.asyncio
async def test_chat_messages_win_over_other_fields() -> None:
    kwargs = {"model": "test-model", "messages": [{"role": "user", "content": PROMPT}], "input": "ignored input field text"}
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=ANSWER))])

    (payload,) = await _written(kwargs, response)

    assert (payload["prompt"], payload["response_preview"]) == (PROMPT, ANSWER)
