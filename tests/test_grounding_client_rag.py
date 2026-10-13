"""Detection of existing client-side or earlier-hop grounding, skip reasons and the response header."""

from __future__ import annotations

import pytest

from loci_litellm.grounding_hook import GroundingOutcome, LociGroundingHook

GROUNDED = GroundingOutcome(
    status="grounded",
    context='MEMORY MATCH (1 results from 1 Qdrant collection) for "q":\n[a|0.9] fact',
)
CLIENT_OK = "[CLIENT-RAG: OK]\nCollections: notes\nfacts\n[/CLIENT-RAG]"
CLIENT_DOWN = "[CLIENT-RAG: UNAVAILABLE]\nMandatory retrieval was attempted but failed\n[/CLIENT-RAG]"
QUESTION = "What did we decide about the relay?"


class _Resolver:
    def __init__(self, client_trusted: bool = True, markers: tuple = ("[CLIENT-RAG: OK]",)) -> None:
        self._client_trusted = client_trusted
        self._markers = markers

    @staticmethod
    def is_disabled() -> bool:
        return False

    def client_rag_ok_markers(self) -> tuple:
        return self._markers

    def client_rag_trusted(self) -> bool:
        return self._client_trusted


class _RecordingHook(LociGroundingHook):
    def __init__(self, client_trusted: bool = True, markers: tuple = ("[CLIENT-RAG: OK]",)) -> None:
        super().__init__(resolver=_Resolver(client_trusted, markers))
        self.intents: list[str] = []

    async def _ground(self, intent: str) -> GroundingOutcome:
        self.intents.append(intent)
        return GROUNDED


def _request(*messages: dict) -> dict:
    return {"model": "auto-router", "messages": list(messages)}


@pytest.mark.asyncio
async def test_a_client_block_that_succeeded_means_no_proxy_side_injection() -> None:
    hook = _RecordingHook()
    data = _request({"role": "user", "content": f"{QUESTION}\n{CLIENT_OK}"})

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert hook.intents == []
    assert [m["role"] for m in result["messages"]] == ["user"]
    assert result["metadata"]["loci_grounding_status"] == "client_grounded"


@pytest.mark.asyncio
async def test_the_same_question_without_a_client_block_is_grounded_by_the_proxy() -> None:
    hook = _RecordingHook()
    data = _request({"role": "user", "content": QUESTION})

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert hook.intents == [QUESTION]
    assert [m["role"] for m in result["messages"]] == ["system", "user"]
    assert result["messages"][0]["content"] == GROUNDED.context
    assert result["metadata"]["loci_grounding_status"] == "grounded"


@pytest.mark.asyncio
async def test_a_client_block_that_says_unavailable_is_the_case_the_proxy_covers() -> None:
    hook = _RecordingHook()
    data = _request({"role": "user", "content": f"{QUESTION}\n{CLIENT_DOWN}"})

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert len(hook.intents) == 1
    assert result["messages"][0]["role"] == "system"
    assert result["metadata"]["loci_grounding_status"] == "grounded"


@pytest.mark.asyncio
async def test_a_memory_match_system_message_from_an_earlier_hop_is_not_duplicated() -> None:
    hook = _RecordingHook()
    data = _request({"role": "system", "content": GROUNDED.context}, {"role": "user", "content": QUESTION})

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert hook.intents == []
    assert [m["role"] for m in result["messages"]] == ["system", "user"]
    assert result["metadata"]["loci_grounding_status"] == "already_grounded"


@pytest.mark.asyncio
async def test_memory_match_text_in_a_user_message_does_not_count_as_grounding() -> None:
    hook = _RecordingHook()
    data = _request({"role": "user", "content": f"{GROUNDED.context}\n{QUESTION}"})

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert len(hook.intents) == 1
    assert result["metadata"]["loci_grounding_status"] == "grounded"


@pytest.mark.asyncio
async def test_distrusting_client_retrieval_grounds_even_when_the_client_block_is_ok() -> None:
    hook = _RecordingHook(client_trusted=False)
    data = _request({"role": "user", "content": f"{QUESTION}\n{CLIENT_OK}"})

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert len(hook.intents) == 1
    assert result["metadata"]["loci_grounding_status"] == "grounded"


@pytest.mark.asyncio
async def test_a_client_block_inside_multimodal_content_parts_is_recognised() -> None:
    hook = _RecordingHook()
    data = _request({"role": "user", "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "text", "text": CLIENT_OK},
        {"type": "image_url", "image_url": {"url": "http://x.invalid/i.png"}},
    ]})

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert hook.intents == []
    assert result["metadata"]["loci_grounding_status"] == "client_grounded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call_type", "content", "reason"),
    [
        ("aembedding", QUESTION, "call_type"),
        ("acompletion", "   ", "no_user_message"),
        ("acompletion", "/clear", "navigation_command"),
    ],
)
async def test_requests_that_are_not_grounded_say_why(call_type: str, content: str, reason: str) -> None:
    hook = _RecordingHook()
    data = _request({"role": "user", "content": content})

    result = await hook.async_pre_call_hook(None, None, data, call_type)

    assert result is None
    assert hook.intents == []
    assert data["metadata"] == {"loci_grounding_status": "skipped", "loci_grounding_reason": reason}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({"loci_grounding_status": "grounded"}, {"x-loci-grounding": "grounded"}),
        ({"loci_grounding_status": "miss", "loci_grounding_reason": "no_matching_context"},
         {"x-loci-grounding": "miss", "x-loci-grounding-reason": "no_matching_context"}),
        ({"loci_grounding_status": "error", "loci_grounding_reason": "bad\r\nSet-Cookie: x"},
         {"x-loci-grounding": "error"}),
        ({}, None),
    ],
)
async def test_the_response_header_reports_the_grounding_status(metadata: dict, expected: dict | None) -> None:
    hook = _RecordingHook()

    headers = await hook.async_post_call_response_headers_hook({"metadata": metadata}, None, object())

    assert headers == expected
