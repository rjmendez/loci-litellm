"""The ledger's prefix_hash field: supplied by a hook, else computed from the request, never any text."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from loci_litellm import cost_ledger
from loci_litellm.cost_ledger import LociCostLedger

T0 = datetime(2025, 1, 1, 12, 0, 0)
HOOK_HASH = "0123456789ab"
COMPUTED = "21096ee9ff29"       # first 12 hex of sha256("first:13:SECRET-PROMPT")


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOCI_COST_LEDGER_FILE", str(tmp_path / "ledger.jsonl"))
    monkeypatch.delenv("LOCI_LEDGER_PREFIX_HASH", raising=False)
    monkeypatch.setattr(cost_ledger, "_warn_counts", {})


def _row(hook_hash: str) -> dict:
    slo = {"litellm_call_id": "c1", "model_group": "g", "model": "m", "api_base": "https://gw.example.test/v1",
           "call_type": "acompletion", "prompt_tokens": 10, "completion_tokens": 2, "metadata": {}}
    params = {"metadata": {"loci_prefix_hash": hook_hash}}
    kwargs = {"standard_logging_object": slo, "litellm_params": params,
              "messages": [{"role": "user", "content": "SECRET-PROMPT"}]}
    response = NS(choices=[NS(message=NS(content="ok", tool_calls=None), finish_reason="stop")], usage=None)
    return LociCostLedger.build_row(kwargs, True, T0, T0 + timedelta(seconds=1), response)


def test_without_a_hook_supplied_hash_the_row_hashes_the_request_prefix() -> None:
    row = _row("")
    assert row["prefix_hash"] == COMPUTED
    assert "SECRET-PROMPT" not in json.dumps(row)


def test_a_hash_the_hook_put_in_the_metadata_wins_over_the_computed_one() -> None:
    assert _row(HOOK_HASH)["prefix_hash"] == HOOK_HASH


def test_the_computed_prefix_hash_can_be_switched_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_LEDGER_PREFIX_HASH", "0")
    assert _row("")["prefix_hash"] == ""
    monkeypatch.setenv("LOCI_LEDGER_PREFIX_HASH", "1")
    assert _row("")["prefix_hash"] == COMPUTED


def test_a_failing_prefix_hash_leaves_the_field_empty_and_warns(monkeypatch: pytest.MonkeyPatch,
                                                                caplog: pytest.LogCaptureFixture) -> None:
    calls: list = []

    def broken(body):
        calls.append(body)
        raise RuntimeError("hash exploded")

    monkeypatch.setattr(cost_ledger, "prefix_hash", broken)
    with caplog.at_level(logging.WARNING, logger="loci_litellm.cost_ledger"):
        row = _row("")
    assert row["prefix_hash"] == "" and row["call_id"] == "c1"        # the rest of the row is intact
    assert calls == [{"messages": [{"role": "user", "content": "SECRET-PROMPT"}]}]
    assert "could not hash the request prefix" in caplog.text and "hash exploded" in caplog.text
