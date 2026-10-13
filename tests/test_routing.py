"""Circuit monitor, cost ledger, cost report and backend resolver."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import litellm
import pytest

from loci_litellm import backend_resolver, cost_report
from loci_litellm.backend_resolver import BackendResolver
from loci_litellm.cost_ledger import LociCostLedger, _is_local
from loci_litellm import cost_ledger
from loci_litellm.circuit_monitor import _FAILURE_THRESHOLD, LociCircuitMonitor, _is_transient

T0 = datetime(2025, 1, 1, 12, 0, 0)


# --- circuit monitor ---

def _ev(group: str, exc=None, **extra) -> dict:
    d: dict = {"model": "x", "litellm_params": {"metadata": {"model_group": group}}}
    if exc is not None:
        d["exception"] = exc
    d.update(extra)
    return d


def _svc(cls, **kw):
    return cls(message="x", llm_provider="p", model="m", **kw)


@pytest.mark.parametrize("exc, expected", [
    (_svc(litellm.ServiceUnavailableError), True),
    (_svc(litellm.BadGatewayError), True),
    (_svc(litellm.InternalServerError), True),
    (_svc(litellm.APIConnectionError), True),
    (litellm.Timeout(message="x", model="m", llm_provider="p"), True),
    (_svc(litellm.AuthenticationError), False),
    (_svc(litellm.NotFoundError), False),
    (_svc(litellm.BadRequestError), False),
    (ValueError("model not found"), False),
])
def test_real_litellm_exception_classes_are_classified(exc, expected) -> None:
    assert _is_transient(exc) is expected


@pytest.mark.asyncio
async def test_provider_5xx_opens_the_circuit() -> None:
    router = LociCircuitMonitor()
    for _ in range(_FAILURE_THRESHOLD):
        await router.async_log_failure_event(_ev("g", _svc(litellm.ServiceUnavailableError)), None, 0, 0)
    assert router.get_status()["g"]["state"] == "open"


@pytest.mark.asyncio
async def test_pre_call_never_redirects_and_does_not_create_entries() -> None:
    router = LociCircuitMonitor()
    for name in ("random-1", "random-2", "random-3"):
        data = {"model": name}
        assert await router.async_pre_call_hook(None, None, data, "acompletion") is None
    assert router.get_status() == {}

    for _ in range(_FAILURE_THRESHOLD):
        await router.async_log_failure_event(_ev("g", TimeoutError("t")), None, 0, 0)
    data = {"model": "g", "metadata": None}
    assert await router.async_pre_call_hook(None, None, data, "acompletion") is None
    assert data["metadata"]["loci_circuit_open"] == "g"


@pytest.mark.asyncio
async def test_half_open_lets_every_request_through() -> None:
    router = LociCircuitMonitor()
    for _ in range(_FAILURE_THRESHOLD):
        await router.async_log_failure_event(_ev("g", TimeoutError("t")), None, 0, 0)
    health = router._providers["g"]
    health.backoff_until = 0.0
    assert health.can_proceed() is True and health.state.value == "half_open"
    assert health.can_proceed() is True and health.can_proceed() is True


@pytest.mark.asyncio
async def test_permanent_errors_neither_count_nor_reset_the_consecutive_count() -> None:
    router = LociCircuitMonitor()
    for _ in range(_FAILURE_THRESHOLD - 1):
        await router.async_log_failure_event(_ev("g", TimeoutError("t")), None, 0, 0)
    for _ in range(20):
        await router.async_log_failure_event(_ev("g", ValueError("bad")), None, 0, 0)
    assert router.get_status()["g"]["state"] == "closed"
    await router.async_log_failure_event(_ev("g", TimeoutError("t")), None, 0, 0)
    assert router.get_status()["g"]["state"] == "open"


@pytest.mark.asyncio
async def test_malformed_events_do_not_raise_and_are_logged(caplog) -> None:
    router = LociCircuitMonitor()
    bad = {"litellm_params": "not-a-dict"}
    with caplog.at_level(logging.WARNING, logger="loci_litellm.circuit_monitor"):
        await router.async_log_failure_event(bad, None, 0, 0)
        await router.async_log_success_event(bad, None, 0, 0)
        assert await router.async_pre_call_hook(None, None, dict(bad), "x") is None
    assert caplog.text.count("loci circuit_monitor:") >= 3


def test_recovery_window_tunable_is_absent() -> None:
    from loci_litellm import circuit_monitor
    assert not hasattr(circuit_monitor, "_RECOVERY_WINDOW_S")


@pytest.mark.asyncio
async def test_non_counted_failures_are_logged_at_info_with_class(caplog) -> None:
    router = LociCircuitMonitor()
    with caplog.at_level(logging.INFO, logger="loci_litellm.circuit_monitor"):
        await router.async_log_failure_event(_ev("g", _svc(litellm.AuthenticationError)), None, 0, 0)
    assert "AuthenticationError" in caplog.text


# --- cost ledger ---

@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "ledger.jsonl"
    monkeypatch.setenv("LOCI_COST_LEDGER_FILE", str(path))
    monkeypatch.delenv("LOCI_COST_LEDGER_DISABLED", raising=False)
    monkeypatch.delenv("LOCI_HONOR_INTERNAL_CALL", raising=False)
    monkeypatch.delenv("LOCI_LOCAL_HOSTS", raising=False)
    getattr(cost_ledger, "_warn_counts", {}).clear()
    return path


def _kw(**slo) -> dict:
    base = {"litellm_call_id": "c1", "model_group": "g", "model_id": "d", "model": "m",
            "api_base": "http://127.0.0.1:1", "response_time": 1.0}
    base.update(slo)
    return {"standard_logging_object": base, "litellm_params": {"metadata": {}}}


def test_tailscale_and_named_hosts_are_local(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _is_local("http://100.101.102.103:11434")
    assert not _is_local("http://gpu-box:11434")
    monkeypatch.setenv("LOCI_LOCAL_HOSTS", "GPU-Box, other")
    assert _is_local("http://gpu-box:11434") and _is_local("http://other/v1")
    assert not _is_local("http://100.128.0.1")  # outside 100.64.0.0/10 and public


def test_row_fields_are_an_exact_allowlist() -> None:
    row = LociCostLedger.build_row(_kw(), True, T0, T0)
    assert set(row) == {
        "ts", "call_id", "trace_id", "group", "deployment", "model", "host", "local", "ok", "call_type",
        "stream", "prompt_tokens", "completion_tokens", "cache_read_tokens", "cache_write_tokens", "empty_response",
        "cost_usd", "latency_s", "cache_hit", "error_class", "internal", "grounding", "agent", "key_alias"}


def test_unknown_latency_is_null_not_zero() -> None:
    kw = _kw(response_time=None)
    row = LociCostLedger.build_row(kw, True, None, None)
    assert row["latency_s"] is None


@pytest.mark.asyncio
async def test_unattributable_calls_are_recorded_as_unrouted(_env: Path) -> None:
    kw = {"standard_logging_object": {}, "litellm_params": {}}
    await LociCostLedger().async_log_failure_event(kw, None, T0, T0)
    assert json.loads(_env.read_text())["group"] == "(unrouted)"


@pytest.mark.asyncio
async def test_write_failure_warnings_are_throttled(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setenv("LOCI_COST_LEDGER_FILE", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="loci_litellm.cost_ledger"):
        for _ in range(10):
            await LociCostLedger().async_log_success_event(_kw(), None, T0, T0)
    assert caplog.text.count("could not record") == 1


def test_internal_marker_is_read_from_each_source(monkeypatch: pytest.MonkeyPatch) -> None:
    marker = {"loci_internal_call": True}
    top = _kw(); top["metadata"] = dict(marker)
    assert LociCostLedger.build_row(top, True, T0, T0)["internal"] is True
    params = _kw(); params["litellm_params"]["metadata"] = dict(marker)
    assert LociCostLedger.build_row(params, True, T0, T0)["internal"] is True
    monkeypatch.setenv("LOCI_HONOR_INTERNAL_CALL", "0")
    assert LociCostLedger.build_row(params, True, T0, T0)["internal"] is False


# --- cost report ---

def _r(group="g", local=True, ok=True, lat=1.0, internal=False, **kw) -> dict:
    row = {"ts": time.time(), "group": group, "local": local, "ok": ok, "latency_s": lat, "prompt_tokens": 10,
           "completion_tokens": 10, "cost_usd": 0.0, "deployment": "d", "error_class": "", "internal": internal}
    row.update(kw)
    return row


def test_unpriced_group_reports_null_savings_not_zero() -> None:
    s = cost_report.summarise([_r()], {})["g"]
    assert s["local_savings_usd"] is None


def test_verdict_names_the_failing_criterion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_LOCAL_SLO_MIN_CALLS", "1")
    slow = cost_report.summarise([_r(lat=99.0) for _ in range(5)], {})["g"]
    assert slow["meets_slo"] is False and "p95" in slow["recommendation"]
    flaky = cost_report.summarise([_r(ok=False) for _ in range(5)], {})["g"]
    assert "success" in flaky["recommendation"]
    assert "put it first" not in flaky["recommendation"] and "keep cloud first" not in flaky["recommendation"]
    good = cost_report.summarise([_r() for _ in range(5)], {})["g"]
    assert good["meets_slo"] is True


def test_p95_boundary_is_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOCI_LOCAL_SLO_MIN_CALLS", "1")
    monkeypatch.setenv("LOCI_LOCAL_SLO_P95_S", "20")
    assert cost_report.summarise([_r(lat=20.0)], {})["g"]["meets_slo"] is True
    assert cost_report.summarise([_r(lat=20.01)], {})["g"]["meets_slo"] is False


def test_internal_rows_can_be_excluded_and_null_latency_is_skipped() -> None:
    rows = [_r(internal=True, ok=False), _r(lat=None), _r(lat=2.0)]
    assert cost_report.summarise(rows, {})["g"]["local_calls"] == 3
    out = cost_report.summarise(rows, {}, include_internal=False)["g"]
    assert out["local_calls"] == 2 and out["local_p95_s"] == 2.0


def test_malformed_ledger_lines_are_counted_not_fatal(tmp_path: Path) -> None:
    f = tmp_path / "l.jsonl"
    f.write_text("\n".join([json.dumps(_r()), "{not json", json.dumps([1]), json.dumps({"ts": "abc"}),
                            json.dumps({"group": "no-ts"}), json.dumps(_r(ts=1.0))]) + "\n", encoding="utf-8")
    stats: dict = {}
    rows = list(cost_report.read_rows(str(f), time.time() - 3600, stats))
    assert len(rows) == 1 and stats["skipped"] == 4 and stats["read"] == 2


def test_bad_slo_env_falls_back_with_warning(monkeypatch, capsys) -> None:
    monkeypatch.setenv("LOCI_LOCAL_SLO_SUCCESS", "high")
    cost_report.summarise([_r()], {})
    assert "LOCI_LOCAL_SLO_SUCCESS" in capsys.readouterr().err


def test_main_json_and_missing_ledger(tmp_path: Path, capsys, _env: Path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text("model_list:\n- model_name: g\n  model_info: {input_cost_per_token: 1, output_cost_per_token: 1}\n")
    assert cost_report.main(["--config", str(config)]) == 2
    _env.write_text(json.dumps(_r(internal=True)) + "\n{bad\n", encoding="utf-8")
    assert cost_report.main(["--config", str(config), "--json", "--exclude-internal"]) == 0
    cap = capsys.readouterr()
    assert json.loads(cap.out) == {}
    assert "skipped 1" in cap.err
    assert cost_report.main(["--config", str(config)]) == 0
    assert "internal calls included" in capsys.readouterr().out


# --- backend resolver ---

@pytest.fixture
def cfg_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    def _use(path: Path) -> None:
        monkeypatch.setattr(backend_resolver, "_CONFIG_PATH", str(path))
        backend_resolver._config.cache_clear()
        getattr(backend_resolver, "_warned_sections", set()).clear()
    yield _use
    backend_resolver._config.cache_clear()
    getattr(backend_resolver, "_warned_sections", set()).clear()


def test_broken_config_warns_and_exposes_the_reason(cfg_file, tmp_path: Path, caplog) -> None:
    p = tmp_path / "b.toml"
    p.write_text("this is = = not toml", encoding="utf-8")
    cfg_file(p)
    with caplog.at_level(logging.WARNING, logger="loci_litellm.backends"):
        assert BackendResolver.qdrant() == ("", "")
    assert "EMPTY config" in caplog.text and str(p) in backend_resolver.config_load_error()


def test_non_utf8_config_is_read_as_utf8_not_locale(cfg_file, tmp_path: Path) -> None:
    p = tmp_path / "b.toml"
    p.write_bytes('[memory]\nwrite_collection = "café"\n'.encode("utf-8"))
    cfg_file(p)
    assert BackendResolver.memory_write_collection() == "café"
    assert backend_resolver.config_load_error() == ""


def test_explicit_loci_config_pointing_nowhere_is_loud(cfg_file, tmp_path: Path, monkeypatch, caplog) -> None:
    monkeypatch.setenv("LOCI_CONFIG", str(tmp_path / "missing.toml"))
    cfg_file(tmp_path / "missing.toml")
    with caplog.at_level(logging.WARNING, logger="loci_litellm.backends"):
        backend_resolver._config()
    assert "missing file" in backend_resolver.config_load_error() and "EMPTY config" in caplog.text


def test_non_table_section_does_not_raise(cfg_file, tmp_path: Path, caplog) -> None:
    p = tmp_path / "b.toml"
    p.write_text('ollama = "x"\n', encoding="utf-8")
    cfg_file(p)
    with caplog.at_level(logging.WARNING, logger="loci_litellm.backends"):
        assert backend_resolver._cfg("ollama", "url", "dflt") == "dflt"
    assert "not a table" in caplog.text


def test_unused_resolver_surface_is_absent() -> None:
    for name in ("ollama_gen_url", "ollama_gen_model", "vllm_url", "vllm_model", "rerank_model"):
        assert not hasattr(BackendResolver, name)
    assert not hasattr(backend_resolver, "_LOCAL_VLLM")
