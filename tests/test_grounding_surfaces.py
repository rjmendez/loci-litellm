"""Grounding injection and detection on the completions, Responses, Anthropic and chat request shapes."""

from __future__ import annotations

import pytest

from loci_litellm.grounding_hook import GroundingOutcome, LociGroundingHook

CONTEXT = 'MEMORY MATCH (1 results from 1 Qdrant collection) for "q":\n[a|0.9] fact'
CLIENT_OK = "[CLIENT-RAG: OK]\nCollections: notes\nfacts\n[/CLIENT-RAG]"
CLIENT_DOWN = "[CLIENT-RAG: UNAVAILABLE]\nretrieval failed\n[/CLIENT-RAG]"
QUESTION = "What did we decide about the relay?"


class _Resolver:
    @staticmethod
    def is_disabled() -> bool:
        return False

    @staticmethod
    def client_rag_ok_markers() -> tuple:
        return ("[CLIENT-RAG: OK]",)

    @staticmethod
    def client_rag_trusted() -> bool:
        return True


class _Hook(LociGroundingHook):
    def __init__(self, outcome: GroundingOutcome | None = None) -> None:
        super().__init__(resolver=_Resolver())
        self._outcome = outcome or GroundingOutcome(status="grounded", context=CONTEXT)
        self.intents: list[str] = []

    async def _ground(self, intent: str) -> GroundingOutcome:
        self.intents.append(intent)
        return self._outcome


# ── /v1/completions ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_text_completion_prompt_gets_the_context_prepended() -> None:
    hook = _Hook()
    data = {"model": "m", "prompt": QUESTION}

    result = await hook.async_pre_call_hook(None, None, data, "atext_completion")

    assert hook.intents == [QUESTION]
    assert result["prompt"] == f"{CONTEXT}\n\n{QUESTION}"
    assert "messages" not in result
    assert result["metadata"]["loci_grounding_status"] == "grounded"


@pytest.mark.asyncio
async def test_a_batch_of_text_completion_prompts_each_get_the_context() -> None:
    hook = _Hook()
    data = {"model": "m", "prompt": [QUESTION, "second prompt"]}

    result = await hook.async_pre_call_hook(None, None, data, "atext_completion")

    assert hook.intents == ["second prompt"]
    assert result["prompt"] == [f"{CONTEXT}\n\n{QUESTION}", f"{CONTEXT}\n\nsecond prompt"]


@pytest.mark.asyncio
async def test_text_completion_with_a_client_ok_block_is_left_alone() -> None:
    hook = _Hook()
    data = {"model": "m", "prompt": f"{QUESTION}\n{CLIENT_OK}"}

    result = await hook.async_pre_call_hook(None, None, data, "atext_completion")

    assert hook.intents == []
    assert result["prompt"] == f"{QUESTION}\n{CLIENT_OK}"
    assert result["metadata"]["loci_grounding_status"] == "client_grounded"


# ── /v1/responses ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_responses_string_input_gets_the_context_as_instructions() -> None:
    hook = _Hook()
    data = {"model": "m", "input": QUESTION}

    result = await hook.async_pre_call_hook(None, None, data, "aresponses")

    assert hook.intents == [QUESTION]
    assert result["instructions"] == CONTEXT
    assert result["input"] == QUESTION
    assert "messages" not in result


@pytest.mark.asyncio
async def test_responses_context_goes_ahead_of_existing_instructions_and_list_input_is_read() -> None:
    hook = _Hook()
    data = {
        "model": "m",
        "instructions": "You are terse.",
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "older question"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "older answer"}]},
            {"role": "user", "content": [{"type": "input_text", "text": QUESTION}]},
        ],
    }

    result = await hook.async_pre_call_hook(None, None, data, "aresponses")

    assert hook.intents == [QUESTION]
    assert result["instructions"] == f"{CONTEXT}\n\nYou are terse."
    assert len(result["input"]) == 3
    assert "messages" not in result


@pytest.mark.asyncio
async def test_responses_instructions_carrying_our_own_block_are_not_grounded_twice() -> None:
    hook = _Hook()
    data = {"model": "m", "instructions": CONTEXT, "input": QUESTION}

    result = await hook.async_pre_call_hook(None, None, data, "aresponses")

    assert hook.intents == []
    assert result["instructions"] == CONTEXT
    assert result["metadata"]["loci_grounding_status"] == "already_grounded"


@pytest.mark.asyncio
async def test_responses_failure_notice_lands_in_instructions() -> None:
    hook = _Hook(GroundingOutcome(status="miss", reason="no_matching_context"))
    data = {"model": "m", "input": QUESTION}

    result = await hook.async_pre_call_hook(None, None, data, "aresponses")

    assert result["instructions"].startswith("LOCI_GROUNDING_STATUS: miss.")
    assert result["metadata"]["loci_grounding_reason"] == "no_matching_context"
    assert "messages" not in result


# ── /v1/messages (Anthropic) ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_anthropic_string_system_gets_the_context_ahead_of_it() -> None:
    hook = _Hook()
    data = {"model": "m", "system": "You are terse.", "messages": [{"role": "user", "content": QUESTION}]}

    result = await hook.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert hook.intents == [QUESTION]
    assert result["system"] == f"{CONTEXT}\n\nYou are terse."
    assert [m["role"] for m in result["messages"]] == ["user"]


@pytest.mark.asyncio
async def test_anthropic_block_list_system_gets_a_leading_text_block() -> None:
    hook = _Hook()
    original = [{"type": "text", "text": "You are terse.", "cache_control": {"type": "ephemeral"}}]
    data = {"model": "m", "system": list(original), "messages": [{"role": "user", "content": QUESTION}]}

    result = await hook.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert result["system"] == [{"type": "text", "text": CONTEXT}, *original]


@pytest.mark.asyncio
async def test_anthropic_without_a_system_field_gets_one_and_no_system_message() -> None:
    hook = _Hook()
    data = {"model": "m", "messages": [{"role": "user", "content": [{"type": "text", "text": QUESTION}]}]}

    result = await hook.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert result["system"] == CONTEXT
    assert [m["role"] for m in result["messages"]] == ["user"]


@pytest.mark.asyncio
async def test_anthropic_client_unavailable_block_is_grounded_here() -> None:
    hook = _Hook()
    data = {"model": "m", "messages": [{"role": "user", "content": f"{QUESTION}\n{CLIENT_DOWN}"}]}

    result = await hook.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert len(hook.intents) == 1
    assert result["system"] == CONTEXT
    assert result["metadata"]["loci_grounding_status"] == "grounded"


@pytest.mark.asyncio
async def test_anthropic_system_carrying_our_block_is_not_grounded_twice() -> None:
    hook = _Hook()
    data = {"model": "m", "system": [{"type": "text", "text": CONTEXT}],
            "messages": [{"role": "user", "content": QUESTION}]}

    result = await hook.async_pre_call_hook(None, None, data, "anthropic_messages")

    assert hook.intents == []
    assert result["metadata"]["loci_grounding_status"] == "already_grounded"


# ── chat shape ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_chat_still_gets_a_leading_system_message() -> None:
    hook = _Hook()
    data = {"model": "m", "messages": [{"role": "user", "content": QUESTION}]}

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert result["messages"] == [{"role": "system", "content": CONTEXT}, {"role": "user", "content": QUESTION}]
    assert "system" not in result and "instructions" not in result
