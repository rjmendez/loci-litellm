from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from loci_litellm import backend_resolver
from loci_litellm.backend_resolver import BackendResolver
from loci_litellm.memory_writer import LociMemoryWriter
from tests.test_memory_writer import FakeAsyncClient

PROMPT = "Please explain the Loci write-back flow and how Qdrant upserts memory records."
ANSWER = "The writer embeds the question and answer, then upserts them with a deterministic point ID."


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.delenv("LOCI_MEMORY_WRITE_COLLECTION", raising=False)
    monkeypatch.delenv("LOCI_MEMORY_WRITER_DISABLED", raising=False)
    monkeypatch.delenv("LOCI_INVESTIGATION_ID", raising=False)
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.local/v1")
    monkeypatch.setenv("LOCI_QDRANT_URL", "http://qdrant.local")
    monkeypatch.setenv("LOCI_MEM_MIN_PROMPT_LEN", "20")
    config = tmp_path / "backends.toml"
    monkeypatch.setattr(backend_resolver, "_CONFIG_PATH", str(config))
    backend_resolver._config.cache_clear()
    yield config
    backend_resolver._config.cache_clear()


def test_the_default_write_collection_is_generic() -> None:
    assert BackendResolver.memory_write_collection() == "loci_memory"


def test_the_private_config_names_the_write_collection(_isolated: Path) -> None:
    _isolated.write_text('[memory]\nwrite_collection = "private_store"\n')
    backend_resolver._config.cache_clear()

    assert BackendResolver.memory_write_collection() == "private_store"


def test_the_env_var_wins_over_the_private_config(_isolated: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _isolated.write_text('[memory]\nwrite_collection = "private_store"\n')
    backend_resolver._config.cache_clear()
    monkeypatch.setenv("LOCI_MEMORY_WRITE_COLLECTION", "from_env")

    assert BackendResolver.memory_write_collection() == "from_env"


@pytest.mark.asyncio
async def test_the_memory_writer_upserts_into_the_configured_collection(_isolated: Path) -> None:
    _isolated.write_text('[memory]\nwrite_collection = "private_store"\n')
    backend_resolver._config.cache_clear()
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=ANSWER))])

    await writer.async_log_success_event({"model": "m", "messages": [{"role": "user", "content": PROMPT}]}, response, 0.0, 0.0)

    puts = [c for c in client.calls if c["method"] == "PUT"]
    assert [c["url"] for c in puts] == ["http://qdrant.local/collections/private_store/points?wait=false"]
