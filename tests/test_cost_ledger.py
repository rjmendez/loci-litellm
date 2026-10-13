from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from loci_litellm import cost_report
from loci_litellm.cost_ledger import LociCostLedger, _is_local, ledger_path

T0 = datetime(2025, 1, 1, 12, 0, 0)


def _kwargs(**slo):
    base = {"litellm_call_id": "c1", "model_group": "medium-tier", "model_id": "dep-1", "model": "resident",
            "api_base": "http://127.0.0.1:11435", "call_type": "acompletion", "response_cost": 0.0,
            "prompt_tokens": 100, "completion_tokens": 20, "response_time": 1.5,
            "metadata": {"user_api_key_alias": "k1"}}
    base.update(slo)
    return {"standard_logging_object": base, "litellm_params": {"metadata": {"loci_grounding_status": "grounded"}}}


@pytest.fixture(autouse=True)
def _ledger(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("LOCI_COST_LEDGER_FILE", str(path))
    monkeypatch.delenv("LOCI_COST_LEDGER_DISABLED", raising=False)
    monkeypatch.delenv("LOCI_HONOR_INTERNAL_CALL", raising=False)
    return path


def test_hosts_are_classified_local_or_cloud() -> None:
    assert _is_local("http://127.0.0.1:11435") and _is_local("http://localhost:11434")
    assert _is_local("http://192.168.1.5:11434") and _is_local("http://[::1]:11434")
    assert not _is_local("https://gateway.ai.cloudflare.com/v1/x") and not _is_local("")


def test_the_default_path_is_under_the_loci_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LOCI_COST_LEDGER_FILE")
    assert ledger_path().endswith("litellm_cost_ledger.jsonl")


def test_a_row_has_metadata_only() -> None:
    row = LociCostLedger.build_row(_kwargs(), True, T0, T0 + timedelta(seconds=2))
    assert row["group"] == "medium-tier" and row["local"] is True and row["ok"] is True
    assert row["prompt_tokens"] == 100 and row["latency_s"] == 1.5 and row["grounding"] == "grounded"
    assert row["key_alias"] == "k1" and row["internal"] is False
    assert not {"messages", "response", "prompt", "content"} & set(row)


def test_internal_calls_are_marked() -> None:
    kw = _kwargs(metadata={"requester_metadata": {"loci_internal_call": True}})
    assert LociCostLedger.build_row(kw, True, T0, T0)["internal"] is True


@pytest.mark.asyncio
async def test_success_and_failure_append_lines(_ledger: Path) -> None:
    ledger = LociCostLedger()
    await ledger.async_log_success_event(_kwargs(), None, T0, T0 + timedelta(seconds=1))
    failing = _kwargs(error_information={"error_class": "Timeout"}, api_base="https://cloud.example/v1")
    await ledger.async_log_failure_event(failing, None, T0, T0 + timedelta(seconds=3))

    rows = [json.loads(line) for line in _ledger.read_text().splitlines()]
    assert [(r["ok"], r["local"], r["error_class"]) for r in rows] == [(True, True, ""), (False, False, "Timeout")]


@pytest.mark.asyncio
async def test_it_can_be_disabled(_ledger: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_COST_LEDGER_DISABLED", "1")
    await LociCostLedger().async_log_success_event(_kwargs(), None, T0, T0)
    assert not _ledger.exists()


@pytest.mark.asyncio
async def test_a_write_failure_is_logged_and_does_not_raise(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setenv("LOCI_COST_LEDGER_FILE", str(tmp_path))  # a directory: open() fails
    await LociCostLedger().async_log_success_event(_kwargs(), None, T0, T0)
    assert "could not record" in caplog.text


def _row(group, local, ok=True, lat=1.0, p=100, c=20, cost=0.0, deployment="d", error=""):
    return {"ts": time.time(), "group": group, "local": local, "ok": ok, "latency_s": lat, "prompt_tokens": p,
            "completion_tokens": c, "cost_usd": cost, "deployment": deployment, "error_class": error}


def test_report_prices_local_tokens_at_the_cheapest_cloud_deployment(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        "model_list:\n"
        "- model_name: medium-tier\n  litellm_params: {model: ollama/resident}\n"
        "- model_name: medium-tier\n  litellm_params: {model: a}\n  model_info: {input_cost_per_token: 0.000002, output_cost_per_token: 0.00001}\n"
        "- model_name: medium-tier\n  litellm_params: {model: b}\n  model_info: {input_cost_per_token: 0.000001, output_cost_per_token: 0.000005}\n")
    prices = cost_report.cloud_prices(str(config))
    assert prices == {"medium-tier": (0.000001, 0.000005)}

    rows = [_row("medium-tier", True, p=1000, c=100) for _ in range(60)] + [_row("medium-tier", False, cost=0.01)]
    summary = cost_report.summarise(rows, prices)["medium-tier"]

    assert summary["local_savings_usd"] == pytest.approx(60 * (1000 * 0.000001 + 100 * 0.000005))
    assert summary["cloud_spend_usd"] == pytest.approx(0.01)
    assert summary["meets_slo"] is True and summary["recommendation"] == "local meets the SLO"


def test_report_withholds_the_recommendation_when_local_is_flaky_or_thin() -> None:
    flaky = [_row("g", True, ok=(i % 5 != 0), error="Timeout") for i in range(100)]
    assert cost_report.summarise(flaky, {})["g"]["meets_slo"] is False
    thin = [_row("g", True) for _ in range(5)]
    assert "only 5 local calls" in cost_report.summarise(thin, {})["g"]["recommendation"]


def test_a_listed_local_provider_is_local_even_behind_a_private_name(monkeypatch) -> None:
    kwargs = _kwargs(api_base="https://llm.lab.example.net:4443", model="ollama/resident")
    assert LociCostLedger.build_row(kwargs, True, T0, T0)["local"] is False
    monkeypatch.setenv("LOCI_LOCAL_PROVIDERS", "ollama, vllm")
    assert LociCostLedger.build_row(kwargs, True, T0, T0)["local"] is True
    cloud = LociCostLedger.build_row(_kwargs(api_base="https://gateway.example/v1", model="openai/anthropic/x"), True, T0, T0)
    assert cloud["local"] is False
