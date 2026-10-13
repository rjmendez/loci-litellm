"""Capability gate: tags requests that need a capability so the router can skip unsuitable deployments."""
from __future__ import annotations

import pytest

from loci_litellm.capability_gate import LociCapabilityGate, needs_images, needs_tools

TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}
IMAGE_MESSAGE = {"role": "user", "content": [{"type": "text", "text": "what is this"},
                                             {"type": "image_url", "image_url": {"url": "https://x.test/a.png"}}]}


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("LOCI_CAPABILITY_GATE_DISABLED", "LOCI_GATE_TOOLS_EXCLUDE_TAGS", "LOCI_GATE_IMAGES_EXCLUDE_TAGS"):
        monkeypatch.delenv(name, raising=False)


async def _run(data: dict) -> dict:
    await LociCapabilityGate().async_pre_call_hook(None, None, data, "acompletion")
    return data


def test_tool_definitions_and_tool_turns_are_detected() -> None:
    assert needs_tools({"tools": [TOOL], "messages": []})
    assert needs_tools({"functions": [{"name": "f"}]})
    assert needs_tools({"messages": [{"role": "tool", "content": "x"}]})
    assert needs_tools({"messages": [{"role": "assistant", "tool_calls": [{"id": "1"}]}]})
    assert not needs_tools({"messages": [{"role": "user", "content": "hi"}]})
    assert not needs_tools({"tools": [], "messages": []})


def test_image_parts_are_detected() -> None:
    assert needs_images({"messages": [IMAGE_MESSAGE]})
    assert not needs_images({"messages": [{"role": "user", "content": "hi"}]})
    assert not needs_images({"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]})


@pytest.mark.asyncio
async def test_nothing_is_tagged_until_tags_are_configured() -> None:
    data = await _run({"tools": [TOOL], "messages": [IMAGE_MESSAGE]})
    assert "metadata" not in data


@pytest.mark.asyncio
async def test_a_tool_request_gets_the_configured_exclusion_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_GATE_TOOLS_EXCLUDE_TAGS", "local, small")
    data = await _run({"model": "g", "tools": [TOOL], "messages": [{"role": "user", "content": "hi"}]})
    assert data["metadata"]["tags"] == ["!local", "!small"]
    assert data["metadata"]["loci_capability_gate"] == "tools"


@pytest.mark.asyncio
async def test_each_capability_uses_its_own_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_GATE_TOOLS_EXCLUDE_TAGS", "no-tools")
    monkeypatch.setenv("LOCI_GATE_IMAGES_EXCLUDE_TAGS", "text-only")
    data = await _run({"messages": [IMAGE_MESSAGE]})
    assert data["metadata"]["tags"] == ["!text-only"]
    both = await _run({"tools": [TOOL], "messages": [IMAGE_MESSAGE]})
    assert both["metadata"]["tags"] == ["!no-tools", "!text-only"]
    assert both["metadata"]["loci_capability_gate"] == "tools,images"


@pytest.mark.asyncio
async def test_existing_tags_are_kept_and_not_duplicated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_GATE_TOOLS_EXCLUDE_TAGS", "local")
    data = await _run({"tools": [TOOL], "metadata": {"tags": ["team-a", "!local"]}})
    assert data["metadata"]["tags"] == ["team-a", "!local"]


@pytest.mark.asyncio
async def test_a_plain_request_is_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_GATE_TOOLS_EXCLUDE_TAGS", "local")
    data = await _run({"model": "g", "messages": [{"role": "user", "content": "hi"}]})
    assert "metadata" not in data


@pytest.mark.asyncio
async def test_the_off_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_GATE_TOOLS_EXCLUDE_TAGS", "local")
    monkeypatch.setenv("LOCI_CAPABILITY_GATE_DISABLED", "1")
    assert "metadata" not in await _run({"tools": [TOOL]})
