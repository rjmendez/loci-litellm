from __future__ import annotations

import pytest

import time

from loci_litellm.circuit_monitor import _FAILURE_THRESHOLD, CircuitState, LociCircuitMonitor, GroupHealth


def _event(group: str, deployment_model: str, exception: Exception | None = None) -> dict:
    payload: dict = {
        "model": deployment_model,
        "litellm_params": {"metadata": {"model_group": group}},
    }
    if exception is not None:
        payload["exception"] = exception
    return payload


async def _fail(router: LociCircuitMonitor, group: str, deployment_model: str, times: int, exc: Exception) -> None:
    for _ in range(times):
        await router.async_log_failure_event(_event(group, deployment_model, exc), None, 0.0, 0.0)


async def _pre_call_metadata(router: LociCircuitMonitor, requested_model: str) -> dict:
    data: dict = {"model": requested_model, "messages": [{"role": "user", "content": "hi"}]}
    await router.async_pre_call_hook(None, None, data, "acompletion")
    return data.get("metadata") or {}


@pytest.mark.asyncio
async def test_ignores_failures_with_no_model_identity() -> None:
    router = LociCircuitMonitor()

    await router.async_log_failure_event({"exception": TimeoutError("timed out")}, None, 0.0, 0.0)

    assert router.get_status() == {}


@pytest.mark.asyncio
async def test_failures_recorded_by_events_open_the_circuit_seen_by_pre_call() -> None:
    router = LociCircuitMonitor()

    await _fail(router, "simple-tier", "workers-ai/@cf/model", _FAILURE_THRESHOLD, TimeoutError("timed out"))

    assert (await _pre_call_metadata(router, "simple-tier"))["loci_circuit_open"] == "simple-tier"


@pytest.mark.asyncio
async def test_circuit_is_tracked_per_model_group() -> None:
    router = LociCircuitMonitor()

    await _fail(router, "simple-tier", "workers-ai/@cf/model", _FAILURE_THRESHOLD, TimeoutError("timed out"))

    assert "loci_circuit_open" not in await _pre_call_metadata(router, "complex-tier")


@pytest.mark.asyncio
async def test_deployments_of_one_group_share_a_circuit_regardless_of_provider_prefix() -> None:
    router = LociCircuitMonitor()
    deployments = ("anthropic/claude-sonnet-5", "workers-ai/@cf/model", "resident")

    for index in range(_FAILURE_THRESHOLD):
        await _fail(router, "complex-tier", deployments[index % len(deployments)], 1, TimeoutError("timed out"))

    assert list(router.get_status()) == ["complex-tier"]
    assert (await _pre_call_metadata(router, "complex-tier"))["loci_circuit_open"] == "complex-tier"


@pytest.mark.asyncio
async def test_success_from_a_fallback_deployment_closes_the_group_circuit() -> None:
    router = LociCircuitMonitor()
    await _fail(router, "simple-tier", "resident", _FAILURE_THRESHOLD, TimeoutError("timed out"))

    await router.async_log_success_event(_event("simple-tier", "workers-ai/@cf/model"), object(), 0.0, 0.0)

    assert "loci_circuit_open" not in await _pre_call_metadata(router, "simple-tier")
    assert router.get_status()["simple-tier"]["state"] == "closed"


@pytest.mark.asyncio
async def test_permanent_errors_do_not_count_toward_the_circuit() -> None:
    router = LociCircuitMonitor()

    await _fail(router, "medium-tier", "anthropic/claude-haiku-5.5", _FAILURE_THRESHOLD * 2, ValueError("model not found"))

    assert "loci_circuit_open" not in await _pre_call_metadata(router, "medium-tier")


def _open(health: GroupHealth) -> None:
    for _ in range(_FAILURE_THRESHOLD):
        health.record_failure()


def test_failures_while_open_do_not_reopen_or_extend_the_backoff() -> None:
    health = GroupHealth("simple-tier")
    _open(health)
    first_deadline = health.status()["backoff_until"]

    for _ in range(_FAILURE_THRESHOLD * 2):
        health.record_failure()

    status = health.status()
    assert status["circuit_open_count"] == 1
    assert status["backoff_until"] == first_deadline


def test_backoff_grows_each_time_a_half_open_probe_fails() -> None:
    health = GroupHealth("simple-tier")
    _open(health)
    delays = []
    for _ in range(3):
        delays.append(health.status()["backoff_until"] - time.time())
        health.backoff_until = 0.0
        assert health.can_proceed() is True
        assert health.state == CircuitState.HALF_OPEN
        health.record_failure()

    assert delays[1] >= delays[0] * 1.5
    assert delays[2] >= delays[1] * 1.5
    assert health.status()["circuit_open_count"] == 4


def test_recovery_resets_the_backoff_sequence() -> None:
    health = GroupHealth("simple-tier")
    _open(health)
    health.backoff_until = 0.0
    health.can_proceed()
    health.record_failure()
    health.record_success()

    _open(health)

    assert 0.0 < health.status()["backoff_until"] - time.time() <= 1.0
