"""Ledger enrichment: correlation fields, the empty-stream reclassification, and old-row tolerance."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from loci_litellm import cost_ledger, cost_report
from loci_litellm.cost_ledger import LociCostLedger

T0 = datetime(2025, 1, 1, 12, 0, 0)

OLD_FIELDS = {
    "ts", "call_id", "trace_id", "group", "deployment", "model", "host", "local", "ok", "call_type", "stream",
    "prompt_tokens", "completion_tokens", "cache_read_tokens", "cache_write_tokens", "empty_response",
    "cost_usd", "latency_s", "cache_hit", "error_class", "internal", "grounding", "agent", "key_alias",
}
NEW_FIELDS = {
    "session_id", "has_tools", "task_class", "prefix_hash", "max_tokens", "client_id",
    "config_hash", "cache_regime", "finish_reason",
}
SESSION_RAW = "raw-session-abc"
SESSION_HASH = "61fed90e2ba1"          # sha256("raw-session-abc") first 12 hex
OTHER_SESSION_HASH = "5f711b23ee4f"    # sha256("other-session") first 12 hex


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCI_COST_LEDGER_FILE", str(tmp_path / "ledger.jsonl"))
    for name in ("LOCI_COST_LEDGER_DISABLED", "LOCI_CONFIG_HASH", "CONFIG_FILE_PATH", "LOCI_CACHE_REGIME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cost_ledger, "_warn_counts", {})
    monkeypatch.setattr(cost_ledger, "_config_hash_cache", {})


def _kwargs(*, stream=True, completion_tokens=12, session=SESSION_RAW, tools=True, max_tokens=4096,
            meta=None, **slo_extra) -> dict:
    slo = {"litellm_call_id": "c1", "model_group": "medium-tier", "model_id": "dep", "model": "m",
           "api_base": "https://gw.example.test/v1", "call_type": "acompletion", "response_cost": 0.01,
           "prompt_tokens": 5000, "completion_tokens": completion_tokens, "response_time": 1.0,
           "stream": stream, "session_id": session, "metadata": {},
           "model_parameters": {"max_tokens": max_tokens} if max_tokens else {}}
    slo.update(slo_extra)
    optional = {"tools": [{"type": "function", "function": {"name": "f"}}]} if tools else {}
    params_meta = {"loci_task_class": "chat", "loci_client_id": "client-a",
                   "loci_prefix_hash": "0123456789abcdef", "loci_grounding_status": "grounded"}
    params_meta.update(meta or {})
    return {"standard_logging_object": slo, "litellm_params": {"metadata": params_meta},
            "optional_params": optional, "stream": stream,
            "messages": [{"role": "user", "content": "SECRET-PROMPT"}]}


def _resp(content="SECRET-REPLY", finish="stop", tool_calls=None):
    return NS(choices=[NS(message=NS(content=content, tool_calls=tool_calls), finish_reason=finish)], usage=None)


def _row(kwargs=None, response=None, ok=True) -> dict:
    return LociCostLedger.build_row(kwargs or _kwargs(), ok, T0, T0 + timedelta(seconds=1),
                                    _resp() if response is None else response)


# ---------------------------------------------------------------- the new keys

def test_a_streamed_call_row_carries_exactly_the_new_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_CACHE_REGIME", "regime-b")
    monkeypatch.setenv("LOCI_CONFIG_HASH", "abcdef0123456789")
    row = _row()
    assert set(row) == OLD_FIELDS | NEW_FIELDS
    assert {k: row[k] for k in NEW_FIELDS | {"stream"}} == {
        "stream": True,
        "session_id": SESSION_HASH,
        "has_tools": True,
        "task_class": "chat",
        "prefix_hash": "0123456789ab",
        "max_tokens": 4096,
        "client_id": "client-a",
        "config_hash": "abcdef012345",
        "cache_regime": "regime-b",
        "finish_reason": "stop",
    }


def test_a_row_with_no_extra_metadata_has_the_empty_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_LEDGER_PREFIX_HASH", "0")       # the request has messages; do not compute a hash
    kw = _kwargs(session="", tools=False, max_tokens=0, meta={"loci_task_class": "", "loci_client_id": "",
                                                              "loci_prefix_hash": ""})
    row = _row(kw, response=NS(choices=[NS(message=NS(content="x", tool_calls=None))], usage=None))
    assert {k: row[k] for k in NEW_FIELDS} == {
        "session_id": "", "has_tools": False, "task_class": "", "prefix_hash": "", "max_tokens": None,
        "client_id": "", "config_hash": "", "cache_regime": "", "finish_reason": "",
    }
    assert "empty_suspect_stream" not in row


def test_the_session_id_is_hashed_and_never_stored_raw() -> None:
    line = json.dumps(_row(), separators=(",", ":"), sort_keys=True)
    assert SESSION_RAW not in line
    assert '"session_id":"%s"' % SESSION_HASH in line
    # twin: a different session gives a different, equally short hash
    assert _row(_kwargs(session="other-session"))["session_id"] == OTHER_SESSION_HASH


def test_a_session_id_in_request_metadata_is_hashed_too() -> None:
    row = _row(_kwargs(session="", meta={"session_id": SESSION_RAW}))
    assert row["session_id"] == SESSION_HASH


def test_no_prompt_or_response_text_reaches_the_serialised_row() -> None:
    line = json.dumps(_row(), separators=(",", ":"), sort_keys=True)
    assert "SECRET-PROMPT" not in line and "SECRET-REPLY" not in line
    # twin: the same row does carry the metadata fields, so the absence above is not an empty row
    assert '"task_class":"chat"' in line and '"finish_reason":"stop"' in line


def test_the_row_grows_by_at_most_200_bytes() -> None:
    row = _row()
    old = {k: v for k, v in row.items() if k not in NEW_FIELDS}
    size = lambda r: len(json.dumps(r, separators=(",", ":"), sort_keys=True))   # noqa: E731
    grown = size(row) - size(old)
    assert grown == 193
    assert grown <= 200
    # the suspect flag is written only when true, and is the one case that goes past the 200 byte budget
    suspect = _row(_kwargs(stream=True, completion_tokens=0), response=_resp(content=""))
    assert "empty_suspect_stream" not in row
    assert set(suspect) == set(row) | {"empty_suspect_stream"} and suspect["empty_suspect_stream"] is True


def test_tools_and_max_tokens_are_read_from_the_request() -> None:
    assert _row(_kwargs(tools=False))["has_tools"] is False
    assert _row(_kwargs(tools=True))["has_tools"] is True
    assert _row(_kwargs(max_tokens=0))["max_tokens"] is None
    kw = _kwargs(max_tokens=0)
    kw["optional_params"] = {"max_completion_tokens": 900, "max_tokens": 100}
    assert _row(kw)["max_tokens"] == 900


def test_a_failed_call_has_no_finish_reason() -> None:
    assert _row(ok=False)["finish_reason"] == ""
    assert _row(ok=True)["finish_reason"] == "stop"


def test_the_config_hash_follows_the_config_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_bytes(b"model_list: []\n")           # bytes: no newline translation on Windows
    monkeypatch.setenv("CONFIG_FILE_PATH", str(cfg))
    first = _row()["config_hash"]
    assert first == "383e57317bf3"                       # sha256 of the file bytes, first 12 hex
    assert _row()["config_hash"] == first
    cfg.write_bytes(b"model_list: [a]\n")
    assert _row()["config_hash"] == "5f4e05d9124d"


def test_an_unreadable_config_gives_an_empty_hash_and_a_warning(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
                                                                caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setenv("CONFIG_FILE_PATH", str(tmp_path / "missing.yaml"))
    with caplog.at_level(logging.WARNING, logger="loci_litellm.cost_ledger"):
        assert _row()["config_hash"] == ""
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "could not hash the proxy config" in caplog.text


# ---------------------------------------------------------------- empty stream vs empty response

def _flags(row: dict) -> tuple:
    """(empty_suspect_stream, empty_response); the suspect key is written only when it is true."""
    return row.get("empty_suspect_stream", False), row["empty_response"]


def test_a_stream_with_no_completion_tokens_and_no_text_is_suspect_not_empty() -> None:
    row = _row(_kwargs(stream=True, completion_tokens=0), response=_resp(content=""))
    assert _flags(row) == (True, False)


def test_the_same_empty_call_without_streaming_is_an_empty_response() -> None:
    row = _row(_kwargs(stream=False, completion_tokens=0), response=_resp(content=""))
    assert _flags(row) == (False, True)


def test_a_stream_that_reported_tokens_but_showed_no_text_stays_an_empty_response() -> None:
    row = _row(_kwargs(stream=True, completion_tokens=7), response=_resp(content=""))
    assert _flags(row) == (False, True)


def test_a_stream_with_text_is_neither_even_when_usage_was_not_reported() -> None:
    row = _row(_kwargs(stream=True, completion_tokens=0), response=_resp(content="an answer"))
    assert _flags(row) == (False, False)


def test_a_stream_with_a_tool_call_is_neither() -> None:
    resp = _resp(content="", tool_calls=[{"id": "t"}])
    row = _row(_kwargs(stream=True, completion_tokens=0), response=resp)
    assert _flags(row) == (False, False)


def test_a_stream_whose_response_shape_is_unseen_and_reports_no_tokens_is_suspect() -> None:
    kw = _kwargs(stream=True, completion_tokens=0)
    row = LociCostLedger.build_row(kw, True, T0, T0, None)
    assert _flags(row) == (True, False)


def test_a_failed_or_non_text_stream_is_never_suspect() -> None:
    failed = _row(_kwargs(stream=True, completion_tokens=0), response=_resp(content=""), ok=False)
    assert _flags(failed) == (False, False)
    emb = _row(_kwargs(stream=True, completion_tokens=0, call_type="aembedding"), response=_resp(content=""))
    assert _flags(emb) == (False, False)


# ---------------------------------------------------------------- readers

def _old_row(**kw) -> dict:
    base = {"ts": time.time(), "group": "medium-tier", "deployment": "dep", "local": True, "ok": True,
            "prompt_tokens": 1000, "completion_tokens": 10, "cost_usd": 0.0, "latency_s": 1.0, "internal": False,
            "cache_read_tokens": 400, "cache_write_tokens": 0, "empty_response": False}
    base.update(kw)
    return base


def test_old_rows_without_the_new_keys_total_exactly_as_before() -> None:
    rows = [_old_row(), _old_row(empty_response=True), _old_row(ok=False, error_class="Timeout")]
    s = cost_report.summarise(rows, {})["medium-tier"]
    assert (s["calls"], s["local_calls"], s["local_success"]) == (3, 3, 0.3333)
    assert (s["empty_responses"], s["empty_suspect_streams"], s["prompt_tokens"]) == (1, 0, 3000)
    assert (s["cache_read_tokens"], s["cache_read_share"]) == (1200, 0.4)


def test_suspect_streams_are_counted_apart_and_do_not_lower_success() -> None:
    rows = [_old_row(), _old_row(), _old_row(empty_suspect_stream=True, stream=True, completion_tokens=0),
            _old_row(empty_response=True)]
    s = cost_report.summarise(rows, {})["medium-tier"]
    assert (s["empty_suspect_streams"], s["empty_responses"]) == (1, 1)
    assert s["local_success"] == 0.75                # 3 of 4: only the real empty response is a failure


def test_a_ledger_file_mixing_old_and_new_rows_reads_back(tmp_path: Path) -> None:
    new = json.loads(json.dumps(_row()))
    path = tmp_path / "mixed.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [_old_row(), new]) + "\n", encoding="utf-8")
    stats: dict = {}
    assert [r.get("session_id") for r in cost_report.read_rows(str(path), 0.0, stats)] == [None, SESSION_HASH]
    assert (stats["read"], stats["skipped"]) == (2, 0)


# ---------------------------------------------------------------- write path

@pytest.mark.asyncio
async def test_a_write_failure_logs_a_warning_and_the_request_is_unaffected(
        monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    attempts: list = []

    def failing_append(path: str, line: str, max_bytes: int = 0) -> None:
        attempts.append((path, json.loads(line), max_bytes))
        raise OSError("disk full")

    monkeypatch.setattr(cost_ledger, "append_line", failing_append)
    kw = _kwargs()
    before = json.dumps(kw, sort_keys=True)
    with caplog.at_level(logging.WARNING, logger="loci_litellm.cost_ledger"):
        result = await LociCostLedger().async_log_success_event(kw, _resp(), T0, T0 + timedelta(seconds=1))
    assert result is None
    assert json.dumps(kw, sort_keys=True) == before            # the request object was not touched
    assert len(attempts) == 1 and attempts[0][1]["session_id"] == SESSION_HASH
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "could not record a call" in caplog.text and "disk full" in caplog.text


@pytest.mark.asyncio
async def test_a_successful_write_lands_one_line(tmp_path: Path) -> None:
    await LociCostLedger().async_log_success_event(_kwargs(), _resp(), T0, T0 + timedelta(seconds=1))
    lines = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["session_id"] == SESSION_HASH
