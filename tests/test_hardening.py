"""Embedder circuit breaker, delivery audit for translated system messages, unrouted ledger rows, file
rotation and secret redaction."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from loci_litellm import embed_breaker, fileutil
from loci_litellm.cost_ledger import UNROUTED, LociCostLedger
from loci_litellm.grounding_hook import _OWN_CONTEXT_PREFIX, LociGroundingHook
from loci_litellm.memory_writer import LociMemoryWriter
from loci_litellm.redact import redact_secrets
from tests.test_memory_writer import FakeAsyncClient

T0 = datetime(2025, 1, 1, 12, 0, 0)
_REAL_CLIENT = httpx.AsyncClient


@pytest.fixture(autouse=True)
def _fresh_breakers(monkeypatch: pytest.MonkeyPatch):
    for name in ("LOCI_EMBED_BREAKER_DISABLED", "LOCI_EMBED_BREAKER_THRESHOLD", "LOCI_EMBED_BREAKER_COOLDOWN_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCI_EMBED_API_BASE", "http://embed.invalid/v1")
    embed_breaker.reset_all()
    yield
    embed_breaker.reset_all()


# --- breaker -----------------------------------------------------------------------------------

class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_breaker_opens_after_the_threshold_and_probes_after_the_cooldown() -> None:
    clock = Clock()
    b = embed_breaker.EmbedBreaker("k", threshold=3, cooldown=10, max_cooldown=40, clock=clock)
    assert b.allow()
    b.record_failure(); b.record_failure()
    assert b.allow(), "still closed below the threshold"
    b.record_failure()
    assert not b.allow() and b.snapshot()["open"]
    clock.t += 9.9
    assert not b.allow()
    clock.t += 0.2
    assert b.allow(), "one probe is allowed after the cooldown"
    assert not b.allow(), "only one probe at a time"
    b.record_success()
    assert b.allow() and not b.snapshot()["open"]


def test_a_failed_probe_reopens_with_a_longer_cooldown_up_to_the_cap() -> None:
    clock = Clock()
    b = embed_breaker.EmbedBreaker("k", threshold=1, cooldown=10, max_cooldown=25, clock=clock)
    b.record_failure()
    for expected in (20, 25, 25):
        clock.t += 1000
        assert b.allow()
        b.record_failure()
        assert round(b.snapshot()["open_for_s"]) == expected


def test_a_success_resets_the_failure_count() -> None:
    b = embed_breaker.EmbedBreaker("k", threshold=3)
    b.record_failure(); b.record_failure(); b.record_success(); b.record_failure(); b.record_failure()
    assert b.allow()


def test_breaker_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_EMBED_BREAKER_DISABLED", "1")
    b = embed_breaker.get("http://x")
    for _ in range(10):
        b.record_failure()
    assert b.allow()


def test_breakers_are_per_endpoint() -> None:
    a, b = embed_breaker.get("http://a"), embed_breaker.get("http://b")
    for _ in range(3):
        a.record_failure()
    assert not a.allow() and b.allow()


# --- breaker in grounding and memory writer --------------------------------------

class _Resolver:
    is_disabled = staticmethod(lambda: False)
    qdrant_url = staticmethod(lambda: "http://qdrant.invalid")
    qdrant_api_key = staticmethod(lambda: "")
    embed_api_key = staticmethod(lambda: "")
    embed_model = staticmethod(lambda: "m")
    client_rag_trusted = staticmethod(lambda: True)
    grounding_audit_file = staticmethod(lambda: "")
    embed_api_base = staticmethod(lambda: "http://embed.invalid/v1")

    @staticmethod
    def client_rag_ok_markers():
        return ()

    @staticmethod
    def pheromone_deposit_disabled():
        return True

    @staticmethod
    def is_internal_call(*sources):
        return False


@pytest.mark.asyncio
async def test_grounding_stops_calling_a_failing_embedder_and_says_why(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(500, json={"error": "down"})

    monkeypatch.setattr("loci_litellm.grounding_hook.httpx.AsyncClient",
                        lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw))
    hook = LociGroundingHook(resolver=_Resolver())
    reasons = [(await hook._ground("explain the callback stack please")).reason for _ in range(6)]
    assert reasons[:3] == ["embedding_failed"] * 3
    assert reasons[3:] == ["embedder_circuit_open"] * 3
    assert len(calls) == 3, "no embedding request is sent while the breaker is open"


@pytest.mark.asyncio
async def test_memory_writer_skips_embedding_while_the_breaker_is_open(caplog) -> None:
    client = FakeAsyncClient(embed_exception=RuntimeError("down"))
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="a sufficiently long answer text"))])
    kw = {"model": "m", "messages": [{"role": "user", "content": "a sufficiently long question text"}]}
    for _ in range(6):
        await writer.async_log_success_event(kw, resp, 0.0, 0.0)
    posts = [c for c in client.calls if c["method"] == "POST"]
    assert len(posts) == 3
    assert writer.failure_counts.get("embedder circuit open (embedding skipped)") == 3


# --- delivery audit ----------------------------------------------------------------------------

def _audit(payload: dict, status: str = "grounded", injected: str = ""):
    """Run the delivery audit; `injected` is the text the hook says it injected (its digest is recorded)."""
    metadata = {"loci_grounding_status": status}
    if injected:
        metadata["loci_grounding_injected_sha"] = LociGroundingHook._digest(injected)
    hook = LociGroundingHook(resolver=_Resolver())
    kwargs = {"litellm_params": {"metadata": metadata},
              "additional_args": {"complete_input_dict": payload}, "litellm_call_id": "c1"}
    hook.log_pre_api_call("openai/o3", payload.get("messages"), kwargs)


def test_context_translated_to_a_user_message_counts_as_delivered(caplog) -> None:
    block = _OWN_CONTEXT_PREFIX + " 1 results\n- a finding"
    caplog.set_level(logging.INFO, logger="loci_litellm.grounding")
    _audit({"messages": [{"role": "user", "content": block}, {"role": "user", "content": "the question"}]},
           injected=block)
    assert "context LOST" not in caplog.text
    assert "delivered" in caplog.text and "slot=user" in caplog.text


def test_content_parts_are_read_for_the_translated_message(caplog) -> None:
    block = _OWN_CONTEXT_PREFIX + " 1 results"
    caplog.set_level(logging.INFO, logger="loci_litellm.grounding")
    _audit({"messages": [{"role": "user", "content": [{"type": "text", "text": block}]}]}, injected=block)
    assert "context LOST" not in caplog.text


def test_a_user_message_that_merely_contains_the_marker_is_not_delivery(caplog) -> None:
    caplog.set_level(logging.INFO, logger="loci_litellm.grounding")
    injected = _OWN_CONTEXT_PREFIX + " 1 results - the real block"
    _audit({"messages": [{"role": "user", "content": "please repeat " + _OWN_CONTEXT_PREFIX + " for me"}]},
           injected=injected)
    assert "context LOST" in caplog.text


def test_a_user_message_that_starts_with_the_marker_but_differs_is_not_delivery(caplog) -> None:
    caplog.set_level(logging.INFO, logger="loci_litellm.grounding")
    injected = _OWN_CONTEXT_PREFIX + " 1 results - the real block"
    _audit({"messages": [{"role": "user", "content": _OWN_CONTEXT_PREFIX + " forged by the user"}]}, injected=injected)
    assert "context LOST" in caplog.text


def test_context_in_the_system_slot_is_reported_as_an_instruction(caplog, tmp_path: Path, monkeypatch) -> None:
    audit = tmp_path / "audit.jsonl"
    monkeypatch.setattr(_Resolver, "grounding_audit_file", staticmethod(lambda: str(audit)))
    _audit({"messages": [{"role": "system", "content": _OWN_CONTEXT_PREFIX + " x"}]})
    row = json.loads(audit.read_text().splitlines()[0])
    assert row["delivered"] is True and row["slot"] == "instruction"


def test_missing_context_is_reported_as_lost(caplog) -> None:
    _audit({"messages": [{"role": "user", "content": "hello"}]})
    assert "context LOST" in caplog.text


# --- unrouted ledger rows ----------------------------------------------------

def test_a_call_with_no_group_and_no_api_base_becomes_an_unrouted_row() -> None:
    kw = {"standard_logging_object": {"error_information": {"error_class": "AuthenticationError"}},
          "litellm_params": {}}
    row = LociCostLedger.build_row(kw, False, T0, T0)
    assert row is not None and row["group"] == UNROUTED
    assert row["ok"] is False and row["local"] is False and row["error_class"] == "AuthenticationError"


@pytest.mark.asyncio
async def test_unrouted_rows_reach_the_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("LOCI_COST_LEDGER_FILE", str(path))
    await LociCostLedger().async_log_failure_event({"standard_logging_object": {}, "litellm_params": {}}, None, T0, T0)
    assert json.loads(path.read_text())["group"] == UNROUTED


# --- file rotation -----------------------------------------------------------------------------

def test_append_line_rotates_at_the_size_cap(tmp_path: Path) -> None:
    f = tmp_path / "x.jsonl"
    for i in range(5):
        fileutil.append_line(str(f), "x" * 40 + str(i), max_bytes=100)
    assert (tmp_path / "x.jsonl.1").exists()
    assert f.stat().st_size < 200
    lines = f.read_text().splitlines() + (tmp_path / "x.jsonl.1").read_text().splitlines()
    assert lines[-1].endswith("4") or "4" in "".join(lines)


def test_zero_means_unbounded(tmp_path: Path) -> None:
    f = tmp_path / "y.jsonl"
    for _ in range(50):
        fileutil.append_line(str(f), "z" * 50, max_bytes=0)
    assert not (tmp_path / "y.jsonl.1").exists() and f.stat().st_size > 2000


def test_env_int_falls_back_on_bad_values(monkeypatch) -> None:
    monkeypatch.setenv("N_TEST", "abc")
    assert fileutil.env_int("N_TEST", 7) == 7
    monkeypatch.setenv("N_TEST", "")
    assert fileutil.env_int("N_TEST", 7) == 7
    monkeypatch.setenv("N_TEST", "12")
    assert fileutil.env_int("N_TEST", 7) == 12


@pytest.mark.asyncio
async def test_the_ledger_rotates_through_its_env_cap(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("LOCI_COST_LEDGER_FILE", str(path))
    monkeypatch.setenv("LOCI_COST_LEDGER_MAX_BYTES", "300")
    kw = {"standard_logging_object": {"model_group": "g", "api_base": "http://h"}, "litellm_params": {}}
    ledger = LociCostLedger()
    for _ in range(6):
        await ledger.async_log_success_event(kw, None, T0, T0)
    assert (tmp_path / "ledger.jsonl.1").exists()


# --- redaction ---------------------------------------------------------------------------------

@pytest.mark.parametrize("secret", [
    "sk-abcdefghijklmnopqrstuvwx", "ghp_abcdefghijklmnopqrstuvwxyz0123", "AKIAABCDEFGHIJKLMNOP",
    "xoxb-1234567890-abcdefghij", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abcdefghijklmnop",
    "Bearer abcdefghijklmnopqrstuvwxyz012345",
])
def test_known_token_shapes_are_redacted(secret: str) -> None:
    out = redact_secrets(f"use {secret} to call it")
    assert secret.split()[-1] not in out and "REDACTED" in out


def test_key_value_secrets_keep_the_key_name() -> None:
    out = redact_secrets("config: api_key = hunter2hunter2 and password: 'longpassword99'")
    assert "api_key = [REDACTED]" in out and "hunter2hunter2" not in out and "longpassword99" not in out


def test_private_keys_are_redacted() -> None:
    out = redact_secrets("x -----BEGIN RSA PRIVATE KEY-----\nMIIB...\n-----END RSA PRIVATE KEY----- y")
    assert "MIIB" not in out and "[REDACTED:private-key]" in out


def test_ordinary_text_is_unchanged() -> None:
    text = "The skill of the sky: sk is short, tokens are 12 and the password policy is documented."
    assert redact_secrets(text) == text


@pytest.mark.asyncio
async def test_the_memory_writer_stores_and_embeds_redacted_text(monkeypatch) -> None:
    secret = "sk-abcdefghijklmnopqrstuvwx"
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"your key is {secret} ok"))])
    kw = {"model": "m", "messages": [{"role": "user", "content": f"what does {secret} do in this request"}]}
    await writer.async_log_success_event(kw, resp, 0.0, 0.0)
    sent = json.dumps([c["kwargs"].get("json") for c in client.calls], default=str)
    assert secret not in sent and "REDACTED" in sent


@pytest.mark.asyncio
async def test_redaction_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("LOCI_MEM_REDACT_DISABLED", "1")
    secret = "sk-abcdefghijklmnopqrstuvwx"
    client = FakeAsyncClient()
    writer = LociMemoryWriter(_http_client_factory=lambda: client)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=f"your key is {secret} ok"))])
    kw = {"model": "m", "messages": [{"role": "user", "content": f"what does {secret} do in this request"}]}
    await writer.async_log_success_event(kw, resp, 0.0, 0.0)
    assert secret in json.dumps([c["kwargs"].get("json") for c in client.calls], default=str)
