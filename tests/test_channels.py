"""Channels: decay, step clamp, bounds, transient-only alarm, failure quorum, starvation probe."""
from __future__ import annotations

import json
import random

import pytest

from loci_litellm import channels
from loci_litellm.channels import (
    ALARM_CAP, RESOURCE_FLOOR, STEP_CLAMP, TRAIL_CAP, ArmChannels, Channel, ChannelBoard, decay_factor, is_transient,
)


class _Status(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"http {status_code}")
        self.status_code = status_code


# --- decay ---

def test_decay_is_exactly_one_half_at_one_half_life() -> None:
    assert abs(decay_factor(180.0, 180.0) - 0.5) < 1e-9
    assert abs(decay_factor(360.0, 180.0) - 0.25) < 1e-9
    assert decay_factor(0.0, 180.0) == 1.0


def test_channel_value_halves_after_one_half_life() -> None:
    channel = Channel(0.0, 0.88, 180.0, value=0.8, now=1000.0)

    assert abs(channel.value_at(1180.0) - 0.4) < 1e-9
    assert channel.value_at(1000.0) == 0.8


def test_a_clock_stepping_backwards_never_amplifies_a_value() -> None:
    channel = Channel(0.0, 0.88, 60.0, value=0.4, now=100.0)

    assert channel.value_at(40.0) == 0.4
    assert decay_factor(-500.0, 60.0) == 1.0


def test_decay_relaxes_toward_the_rest_value_not_toward_zero() -> None:
    channel = Channel(0.08, 1.0, 100.0, rest=1.0, value=0.2, now=0.0)

    assert abs(channel.value_at(100.0) - 0.6) < 1e-9


@pytest.mark.parametrize("bad", [0.0, -5.0, float("inf"), float("nan")])
def test_a_non_positive_or_non_finite_half_life_is_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="half_life_s must be a positive finite number"):
        decay_factor(1.0, bad)


# --- step clamp and bounds ---

def test_a_large_deposit_is_limited_to_the_step_clamp() -> None:
    channel = Channel(0.0, TRAIL_CAP, 1000.0)

    assert channel.deposit(1.0, 0.0) == STEP_CLAMP == 0.18


def test_a_small_deposit_is_taken_as_given() -> None:
    channel = Channel(0.0, TRAIL_CAP, 1000.0)

    assert channel.deposit(0.05, 0.0) == 0.05


def test_a_large_negative_deposit_is_limited_to_the_step_clamp() -> None:
    channel = Channel(RESOURCE_FLOOR, 1.0, 1000.0, rest=1.0)

    assert abs(channel.deposit(-1.0, 0.0) - 0.82) < 1e-9


def test_a_non_finite_deposit_is_rejected_and_changes_nothing() -> None:
    channel = Channel(0.0, 0.9, 100.0, value=0.3)

    with pytest.raises(ValueError, match="deposit must be finite"):
        channel.deposit(float("nan"), 1.0)
    assert channel.value_at(0.0) == 0.3


def test_bounds_hold_over_1000_seeded_updates_and_are_reached() -> None:
    rng = random.Random(20261018)
    arm = ArmChannels()
    now = 0.0
    trail, alarm, resource = [], [], []
    for _ in range(1000):
        now += rng.uniform(0.0, 5.0)
        trail.append(arm.trail.deposit(rng.uniform(-0.1, 1.0), now))
        alarm.append(arm.alarm.deposit(rng.uniform(-0.1, 1.0), now))
        resource.append(arm.resource.deposit(rng.uniform(-1.0, 0.1), now))

    assert max(trail) == TRAIL_CAP == 0.92
    assert max(alarm) == ALARM_CAP == 0.88
    assert min(resource) == RESOURCE_FLOOR == 0.08
    assert min(trail) >= 0.0 and min(alarm) >= 0.0
    assert max(resource) <= 1.0


# --- transient classification ---

@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_transient_statuses_are_transient(status: int) -> None:
    assert is_transient(status=status) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_statuses_are_not_transient(status: int) -> None:
    assert is_transient(status=status) is False


def test_status_is_read_from_the_exception_and_falls_back_to_its_kind() -> None:
    assert is_transient(_Status(503)) is True
    assert is_transient(_Status(401)) is False
    assert is_transient(TimeoutError("timed out")) is True
    assert is_transient(ValueError("model not found")) is False
    assert is_transient() is False


def test_a_503_deposits_alarm() -> None:
    board = ChannelBoard()

    deposited = board.observe_failure("arm-a", 10.0, "caller-1", status=503)

    assert deposited is True
    assert board.values("arm-a", 10.0) == {"trail": 0.0, "alarm": 0.18, "resource": 1.0}


@pytest.mark.parametrize("status", [400, 401])
def test_a_permanent_error_deposits_nothing(status: int) -> None:
    board = ChannelBoard()

    deposited = board.observe_failure("arm-a", 10.0, "caller-1", status=status)

    assert deposited is False
    assert board.values("arm-a", 10.0) == {"trail": 0.0, "alarm": 0.0, "resource": 1.0}


def test_a_permanent_error_never_forms_a_quorum() -> None:
    board = ChannelBoard()
    for index, caller in enumerate(["a", "b", "c", "d"]):
        board.observe_failure("arm-a", float(index), caller, status=401)

    assert board.unhealthy("arm-a", 4.0) is False


def test_a_success_deposits_trail_and_no_alarm() -> None:
    board = ChannelBoard()

    board.observe_success("arm-a", 5.0)

    assert board.values("arm-a", 5.0) == {"trail": 0.1, "alarm": 0.0, "resource": 1.0}


# --- quorum ---

def test_two_failures_from_one_caller_do_not_trip_quorum() -> None:
    board = ChannelBoard()
    board.observe_failure("arm-a", 0.0, "caller-1", status=503)
    board.observe_failure("arm-a", 1.0, "caller-1", status=503)

    assert board.unhealthy("arm-a", 2.0) is False


def test_three_failures_from_one_caller_do_not_trip_quorum() -> None:
    board = ChannelBoard()
    for t in (0.0, 1.0, 2.0):
        board.observe_failure("arm-a", t, "caller-1", status=503)

    assert board.unhealthy("arm-a", 3.0) is False


def test_three_failures_from_two_callers_trip_quorum() -> None:
    board = ChannelBoard()
    board.observe_failure("arm-a", 0.0, "caller-1", status=503)
    board.observe_failure("arm-a", 1.0, "caller-2", status=503)
    assert board.unhealthy("arm-a", 1.0) is False
    board.observe_failure("arm-a", 2.0, "caller-1", status=503)

    assert board.unhealthy("arm-a", 2.0) is True


def test_failures_older_than_the_window_do_not_count() -> None:
    board = ChannelBoard()
    board.observe_failure("arm-a", 0.0, "caller-1", status=503)
    board.observe_failure("arm-a", 1.0, "caller-2", status=503)
    board.observe_failure("arm-a", 121.5, "caller-1", status=503)

    assert board.unhealthy("arm-a", 121.5) is False


def test_an_open_quorum_lingers_for_one_alarm_half_life() -> None:
    board = ChannelBoard()
    for t, caller in ((0.0, "a"), (5.0, "b"), (10.0, "a")):
        board.observe_failure("arm-a", t, caller, status=503)

    assert board.unhealthy("arm-a", 189.99) is True
    assert board.unhealthy("arm-a", 190.0) is False


def test_quorum_is_per_arm() -> None:
    board = ChannelBoard()
    for t, caller in ((0.0, "a"), (1.0, "b"), (2.0, "a")):
        board.observe_failure("arm-a", t, caller, status=503)

    assert board.unhealthy("arm-a", 3.0) is True
    assert board.unhealthy("arm-b", 3.0) is False


def test_a_failure_exactly_120_seconds_old_still_counts_toward_quorum() -> None:
    board = ChannelBoard()
    board.observe_failure("arm-a", 0.0, "caller-1", status=503)
    board.observe_failure("arm-a", 60.0, "caller-2", status=503)
    board.observe_failure("arm-a", 120.0, "caller-1", status=503)

    assert board.unhealthy("arm-a", 120.0) is True


def test_alarm_is_not_fed_by_a_permanent_error_even_when_a_transient_one_precedes_it() -> None:
    board = ChannelBoard()
    board.observe_failure("arm-a", 0.0, "caller-1", status=503)
    board.observe_failure("arm-a", 0.0, "caller-1", status=401)
    board.observe_failure("arm-a", 0.0, "caller-1", status=400)

    assert board.values("arm-a", 0.0)["alarm"] == 0.18


# --- starvation probe ---

def test_starvation_probe_emits_exactly_one_probe_at_120_seconds() -> None:
    board = ChannelBoard()
    board.register("arm-a", 0.0)

    probes = [t for t in range(0, 200) if board.probe_due("arm-a", float(t))]

    assert probes == [120]


def test_probes_repeat_once_per_interval_and_traffic_resets_the_clock() -> None:
    board = ChannelBoard()
    board.register("arm-a", 0.0)
    assert [t for t in range(0, 250) if board.probe_due("arm-a", float(t))] == [120, 240]

    other = ChannelBoard()
    other.register("arm-b", 0.0)
    other.observe_success("arm-b", 100.0)
    assert [t for t in range(0, 230) if other.probe_due("arm-b", float(t))] == [220]


# --- bounds on memory ---

def test_the_board_evicts_the_least_recently_touched_arm_and_counts_it() -> None:
    board = ChannelBoard(max_arms=3)
    for index in range(5):
        board.observe_success(f"arm-{index}", float(index))

    assert sorted(board.snapshot(10.0)) == ["arm-2", "arm-3", "arm-4"]
    assert board.evicted == 2


def test_state_for_the_maximum_number_of_arms_is_far_below_one_mebibyte() -> None:
    board = ChannelBoard()
    for index in range(channels.MAX_ARMS + 10):
        board.observe_failure(f"gateway-deployment-{index:04d}", float(index), "caller", status=503)

    size = board.state_bytes(1000.0)

    assert size == len(json.dumps(board.snapshot(1000.0)).encode("utf-8"))
    assert len(board.snapshot(1000.0)) == channels.MAX_ARMS
    assert size < 1024 * 1024 // 4
