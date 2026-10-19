"""Ledger telemetry: cache tokens, empty responses, success rate, and a standalone report."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import yaml

from loci_litellm import cost_report
from loci_litellm.cost_ledger import LociCostLedger

PKG_ROOT = Path(__file__).resolve().parents[1]
T0 = datetime(2025, 1, 1, 12, 0, 0)

FIELDS = {
    "ts", "call_id", "trace_id", "group", "deployment", "model", "host", "local", "ok", "call_type", "stream",
    "prompt_tokens", "completion_tokens", "cache_read_tokens", "cache_write_tokens", "empty_response",
    "cost_usd", "latency_s", "cache_hit", "error_class", "internal", "grounding", "agent", "key_alias",
    # added by the ledger-enrichment change (see tests/test_cost_ledger_fields.py for their values)
    "session_id", "has_tools", "task_class", "prefix_hash", "max_tokens", "client_id",
    "config_hash", "cache_regime", "finish_reason",
}


def _kwargs(call_type: str = "acompletion") -> dict:
    slo = {"litellm_call_id": "c1", "model_group": "medium-tier", "model_id": "dep", "model": "m",
           "api_base": "https://gw.example.test/v1", "call_type": call_type, "response_cost": 0.01,
           "prompt_tokens": 50000, "completion_tokens": 20, "response_time": 1.0, "metadata": {}}
    return {"standard_logging_object": slo, "litellm_params": {"metadata": {}}}


def _resp(content="hello", tool_calls=None, reasoning=None, usage=None):
    message = NS(content=content, tool_calls=tool_calls, reasoning_content=reasoning)
    return NS(choices=[NS(message=message)], usage=usage)


def _row(response, call_type="acompletion", ok=True):
    return LociCostLedger.build_row(_kwargs(call_type), ok, T0, T0 + timedelta(seconds=1), response)


def test_the_row_has_exactly_the_documented_fields() -> None:
    assert set(_row(_resp())) == FIELDS


def test_openai_shaped_cache_tokens_are_recorded() -> None:
    usage = NS(prompt_tokens_details=NS(cached_tokens=40000, cache_write_tokens=500))
    row = _row(_resp(usage=usage))
    assert (row["cache_read_tokens"], row["cache_write_tokens"]) == (40000, 500)


def test_anthropic_shaped_cache_tokens_are_recorded() -> None:
    usage = {"cache_read_input_tokens": 1234, "cache_creation_input_tokens": 77}
    row = _row({"choices": [{"message": {"content": "hi"}}], "usage": usage})
    assert (row["cache_read_tokens"], row["cache_write_tokens"]) == (1234, 77)


def test_no_usage_means_zero_not_an_error() -> None:
    row = _row(_resp(usage=None))
    assert (row["cache_read_tokens"], row["cache_write_tokens"]) == (0, 0)
    assert _row(None)["cache_read_tokens"] == 0


@pytest.mark.parametrize("response,expected", [
    (_resp(content=""), True),                                              # nothing at all
    (_resp(content=None, reasoning="thinking only"), True),                 # reasoning ate the budget
    (_resp(content="   \n"), True),
    (_resp(content="an answer"), False),
    (_resp(content="", tool_calls=[{"id": "t"}]), False),                   # a tool call is an answer
    (_resp(content=[{"text": "part"}]), False),
    (None, False),                                                          # unknown shape: do not guess
    (NS(data=[1, 2]), False),
])
def test_empty_response_detection(response, expected) -> None:
    assert _row(response)["empty_response"] is expected


def test_embeddings_and_failures_are_never_empty_responses() -> None:
    assert _row(_resp(content=""), call_type="aembedding")["empty_response"] is False
    assert _row(_resp(content=""), ok=False)["empty_response"] is False


def _ledger(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "ledger.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _r(**kw) -> dict:
    base = {"ts": time.time(), "group": "medium-tier", "deployment": "local-dep", "local": True, "ok": True,
            "prompt_tokens": 1000, "completion_tokens": 10, "cost_usd": 0.0, "latency_s": 1.0, "internal": False,
            "cache_read_tokens": 0, "cache_write_tokens": 0, "empty_response": False}
    base.update(kw)
    return base


def test_an_empty_answer_is_not_counted_as_a_local_success() -> None:
    rows = [_r() for _ in range(8)] + [_r(empty_response=True) for _ in range(2)]
    s = cost_report.summarise(rows, {"medium-tier": (1e-6, 5e-6)})["medium-tier"]
    assert s["local_calls"] == 10 and s["local_empty_responses"] == 2
    assert s["local_success"] == 0.8


def test_cache_share_is_reported_and_old_rows_read_as_zero() -> None:
    rows = [_r(group="g", local=False, prompt_tokens=1000, cache_read_tokens=900),
            {"ts": time.time(), "group": "g", "deployment": "d", "local": False, "ok": True,
             "prompt_tokens": 1000, "completion_tokens": 5}]            # a row from before the fields existed
    s = cost_report.summarise(rows, {})["g"]
    assert s["cache_read_tokens"] == 900 and s["prompt_tokens"] == 2000
    assert s["cache_read_share"] == 0.45


def test_the_report_runs_without_litellm_importable(tmp_path: Path) -> None:
    """`python -m loci_litellm.cost_report` is an offline tool: it must not need the proxy stack."""
    ledger = _ledger(tmp_path, [_r(), _r(group="cloud-g", local=False, deployment="c", cost_usd=0.5,
                                         cache_read_tokens=500)])
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model_list": [{"model_name": "medium-tier", "model_info": {
        "input_cost_per_token": 1e-6, "output_cost_per_token": 5e-6}}]}), encoding="utf-8")
    code = (
        "import sys, runpy\n"
        "sys.modules['litellm'] = None          # any `import litellm` now raises ImportError\n"
        f"sys.path.insert(0, {str(PKG_ROOT)!r})\n"
        f"sys.argv = ['cost_report', '--config', {str(config)!r}, '--ledger', {str(ledger)!r}, '--since-hours', '48']\n"
        "runpy.run_module('loci_litellm.cost_report', run_name='__main__')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr[-600:]
    assert "cache%" in done.stdout and "empty" in done.stdout and "medium-tier" in done.stdout


def test_importing_the_package_does_not_load_the_callbacks() -> None:
    code = ("import sys; sys.modules['litellm'] = None; sys.path.insert(0, %r)\n"
            "import loci_litellm\n"
            "assert loci_litellm.__version__ != 'unknown'\n"
            "assert 'loci_litellm.grounding_hook' not in sys.modules\n"
            "print('ok')\n") % str(PKG_ROOT)
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-600:]


def test_lazy_attributes_resolve() -> None:
    import loci_litellm

    assert loci_litellm.BackendResolver.__name__ == "BackendResolver"
    assert loci_litellm.loci_grounding_hook is not None
    with pytest.raises(AttributeError):
        loci_litellm.not_a_thing  # noqa: B018
