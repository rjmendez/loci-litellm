"""Client RAG marker configuration (TOML, env var) and the trust switch."""

from __future__ import annotations

from pathlib import Path

import pytest

from loci_litellm import backend_resolver
from loci_litellm.backend_resolver import BackendResolver
from loci_litellm.grounding_hook import GroundingOutcome, LociGroundingHook

QUESTION = "What did we decide about the relay?"
BLOCK = "[CLIENT-RAG: OK]\nnotes\n[/CLIENT-RAG]"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.delenv("LOCI_CLIENT_RAG_OK_MARKERS", raising=False)
    monkeypatch.delenv("LOCI_TRUST_CLIENT_RAG", raising=False)
    config = tmp_path / "backends.toml"
    monkeypatch.setattr(backend_resolver, "_CONFIG_PATH", str(config))
    backend_resolver._config.cache_clear()
    yield config
    backend_resolver._config.cache_clear()


def test_with_nothing_configured_there_are_no_markers() -> None:
    assert BackendResolver.client_rag_ok_markers() == ()


def test_markers_come_from_the_private_toml(_isolated: Path) -> None:
    _isolated.write_text('[grounding]\nclient_rag_ok_markers = ["[A: OK]", "  [B: OK]  ", ""]\n')
    backend_resolver._config.cache_clear()

    assert BackendResolver.client_rag_ok_markers() == ("[A: OK]", "[B: OK]")


def test_the_env_var_wins_over_the_toml_and_splits_on_double_bar(
    _isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _isolated.write_text('[grounding]\nclient_rag_ok_markers = ["[FROM-TOML]"]\n')
    backend_resolver._config.cache_clear()
    monkeypatch.setenv("LOCI_CLIENT_RAG_OK_MARKERS", "[X: OK]||[Y: OK]")

    assert BackendResolver.client_rag_ok_markers() == ("[X: OK]", "[Y: OK]")


class _Hook(LociGroundingHook):
    def __init__(self) -> None:
        super().__init__(resolver=BackendResolver)
        self.intents: list[str] = []

    async def _ground(self, intent: str) -> GroundingOutcome:
        self.intents.append(intent)
        return GroundingOutcome(status="grounded", context='MEMORY MATCH (1 results from 1 Qdrant collection) for "q":\n[a|0.9] f')


@pytest.mark.asyncio
async def test_a_client_block_is_ignored_until_its_marker_is_configured(
    _isolated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hook = _Hook()
    data = {"model": "m", "messages": [{"role": "user", "content": f"{QUESTION}\n{BLOCK}"}]}

    unconfigured = await hook.async_pre_call_hook(None, None, dict(data, messages=list(data["messages"])), "acompletion")

    assert len(hook.intents) == 1
    assert unconfigured["metadata"]["loci_grounding_status"] == "grounded"

    monkeypatch.setenv("LOCI_CLIENT_RAG_OK_MARKERS", "[CLIENT-RAG: OK]")
    configured = await hook.async_pre_call_hook(None, None, dict(data, messages=list(data["messages"])), "acompletion")

    assert len(hook.intents) == 1
    assert configured["metadata"]["loci_grounding_status"] == "client_grounded"


@pytest.mark.asyncio
async def test_trust_off_ignores_configured_markers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_CLIENT_RAG_OK_MARKERS", "[CLIENT-RAG: OK]")
    monkeypatch.setenv("LOCI_TRUST_CLIENT_RAG", "0")
    hook = _Hook()
    data = {"model": "m", "messages": [{"role": "user", "content": f"{QUESTION}\n{BLOCK}"}]}

    result = await hook.async_pre_call_hook(None, None, data, "acompletion")

    assert len(hook.intents) == 1
    assert result["metadata"]["loci_grounding_status"] == "grounded"
