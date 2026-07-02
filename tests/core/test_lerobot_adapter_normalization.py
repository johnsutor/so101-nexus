"""Tests for LeRobot sim follower normalization helpers."""

from __future__ import annotations

import math
import types
from typing import Any

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.motors import MotorCalibration

from so101_nexus.config import SO101_JOINT_NAMES


def _calibration(*, drive_mode: int = 0) -> dict[str, MotorCalibration]:
    return {
        name: MotorCalibration(
            id=i,
            drive_mode=drive_mode,
            homing_offset=0,
            range_min=1000,
            range_max=3000,
        )
        for i, name in enumerate(SO101_JOINT_NAMES, start=1)
    }


def test_build_so101_motors_uses_lerobot_modes() -> None:
    from lerobot.motors import MotorNormMode

    from so101_nexus.lerobot_adapter.normalization import build_so101_motors

    motors = build_so101_motors(use_degrees=True)

    for name in SO101_JOINT_NAMES[:-1]:
        assert motors[name].norm_mode is MotorNormMode.DEGREES
    assert motors["gripper"].norm_mode is MotorNormMode.RANGE_0_100


def test_normalize_ticks_matches_upstream_feetech_private_method() -> None:
    from lerobot.motors.feetech import FeetechMotorsBus

    from so101_nexus.lerobot_adapter.normalization import (
        build_so101_motors,
        normalize_ticks,
    )

    motors = build_so101_motors(use_degrees=False)
    calibration = _calibration(drive_mode=1)
    ticks = {name: 1200 + i * 150 for i, name in enumerate(SO101_JOINT_NAMES)}
    bus = FeetechMotorsBus(port="/dev/null", motors=motors, calibration=calibration)
    expected_by_id = bus._normalize({motors[name].id: tick for name, tick in ticks.items()})
    expected = {name: expected_by_id[motors[name].id] for name in ticks}

    assert normalize_ticks(ticks, motors=motors, calibration=calibration) == expected


def test_unnormalize_values_matches_upstream_feetech_private_method() -> None:
    from lerobot.motors.feetech import FeetechMotorsBus

    from so101_nexus.lerobot_adapter.normalization import (
        build_so101_motors,
        unnormalize_values,
    )

    motors = build_so101_motors(use_degrees=False)
    calibration = _calibration(drive_mode=1)
    values = {
        "shoulder_pan": -40.0,
        "shoulder_lift": -20.0,
        "elbow_flex": 0.0,
        "wrist_flex": 20.0,
        "wrist_roll": 40.0,
        "gripper": 75.0,
    }
    bus = FeetechMotorsBus(port="/dev/null", motors=motors, calibration=calibration)
    expected_by_id = bus._unnormalize({motors[name].id: val for name, val in values.items()})
    expected = {name: expected_by_id[motors[name].id] for name in values}

    assert unnormalize_values(values, motors=motors, calibration=calibration) == expected


def test_sim_rad_tick_roundtrip_uses_calibration_midpoint() -> None:
    from so101_nexus.lerobot_adapter.normalization import (
        motor_ticks_to_sim_rad,
        sim_rad_to_motor_ticks,
    )

    gripper_limits = (-0.2, 1.2)
    calibration = _calibration()
    qpos = np.array([0.0, math.radians(45), -0.3, 0.1, -0.2, 0.5])

    ticks = sim_rad_to_motor_ticks(
        qpos,
        calibration=calibration,
        gripper_limits_rad=gripper_limits,
    )
    roundtrip = motor_ticks_to_sim_rad(
        ticks,
        calibration=calibration,
        gripper_limits_rad=gripper_limits,
    )

    assert ticks["shoulder_pan"] == 2000
    np.testing.assert_allclose(roundtrip[:5], qpos[:5], atol=1 / 650)
    assert roundtrip[5] == pytest.approx(qpos[5], abs=1 / 1000)


def test_gripper_conversion_uses_caller_provided_limits() -> None:
    from so101_nexus.lerobot_adapter.normalization import (
        motor_ticks_to_sim_rad,
        sim_rad_to_motor_ticks,
    )

    calibration = _calibration()
    lower_upper = (0.25, 0.75)
    qpos = np.zeros(6)
    qpos[-1] = lower_upper[1]

    ticks = sim_rad_to_motor_ticks(
        qpos,
        calibration=calibration,
        gripper_limits_rad=lower_upper,
    )
    roundtrip = motor_ticks_to_sim_rad(
        ticks,
        calibration=calibration,
        gripper_limits_rad=lower_upper,
    )

    assert ticks["gripper"] == 3000
    assert roundtrip[-1] == pytest.approx(lower_upper[1])


def test_leader_action_to_sim_qpos_round_trip() -> None:
    from so101_nexus.lerobot_adapter.normalization import (
        build_so101_motors,
        leader_action_to_sim_qpos,
        normalize_ticks,
        sim_rad_to_motor_ticks,
    )

    motors = build_so101_motors(use_degrees=True)
    calibration = _calibration()
    gripper_limits_rad = (-0.2, 1.2)
    original_qpos = np.array([0.1, -0.2, 0.3, 0.0, -0.4, 0.6], dtype=np.float64)
    ticks = sim_rad_to_motor_ticks(
        original_qpos,
        calibration=calibration,
        gripper_limits_rad=gripper_limits_rad,
    )
    normalized = normalize_ticks(ticks, motors=motors, calibration=calibration)
    action = {f"{name}.pos": value for name, value in normalized.items()}

    recovered = leader_action_to_sim_qpos(
        action,
        motors=motors,
        calibration=calibration,
        gripper_limits_rad=gripper_limits_rad,
    )

    np.testing.assert_allclose(recovered, original_qpos, atol=1e-2)


def test_body_conversion_rejects_equal_calibration_ranges() -> None:
    from so101_nexus.lerobot_adapter.normalization import (
        motor_ticks_to_sim_rad,
        sim_rad_to_motor_ticks,
    )

    calibration = _calibration()
    calibration["shoulder_pan"] = MotorCalibration(
        id=1,
        drive_mode=0,
        homing_offset=0,
        range_min=2000,
        range_max=2000,
    )
    ticks = dict.fromkeys(SO101_JOINT_NAMES, 2000)

    with pytest.raises(ValueError, match="shoulder_pan"):
        sim_rad_to_motor_ticks(
            np.zeros(6),
            calibration=calibration,
            gripper_limits_rad=(-0.2, 1.2),
        )
    with pytest.raises(ValueError, match="shoulder_pan"):
        motor_ticks_to_sim_rad(
            ticks,
            calibration=calibration,
            gripper_limits_rad=(-0.2, 1.2),
        )


class _FakeEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self._ctrl_low = np.array([-1.0, -1.1, -1.2, -1.3, -1.4, 0.25])
        self._ctrl_high = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 0.75])
        self.qpos = np.arange(6, dtype=np.float64)

    def _get_current_qpos(self) -> np.ndarray:
        return self.qpos.copy()


def test_read_sim_qpos_reads_unwrapped_current_qpos() -> None:
    from so101_nexus.lerobot_adapter.normalization import read_sim_qpos

    env = _FakeEnv()

    np.testing.assert_array_equal(read_sim_qpos(env), env.qpos)


def test_read_gripper_limits_rad_reads_env_control_range() -> None:
    from so101_nexus.lerobot_adapter.normalization import read_gripper_limits_rad

    assert read_gripper_limits_rad(_FakeEnv()) == (0.25, 0.75)


def test_action_for_env_clips_to_control_range() -> None:
    from so101_nexus.lerobot_adapter.normalization import action_for_env

    action = action_for_env(_FakeEnv(), np.array([-2.0, 0.0, 2.0, 0.0, 0.0, 1.0]))

    np.testing.assert_allclose(action, np.array([-1.0, 0.0, 1.2, 0.0, 0.0, 0.75]))


def test_read_sim_qpos_supports_tensor_shape() -> None:
    torch = pytest.importorskip("torch")

    from so101_nexus.lerobot_adapter.normalization import read_sim_qpos

    class _Env:
        def _get_current_qpos(self) -> Any:
            return torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])

    np.testing.assert_array_equal(read_sim_qpos(_Env()), np.arange(6, dtype=np.float32))


def test_dataset_row_to_sim_qpos_inverts_recorder_pipeline() -> None:
    from so101_nexus import SO101_GRIPPER_LIMITS_RAD, dataset_row_to_sim_qpos
    from so101_nexus.lerobot_adapter.normalization import (
        MOTOR_NAMES,
        build_so101_motors,
        normalize_ticks,
        sim_rad_to_motor_ticks,
    )
    from so101_nexus.lerobot_adapter.synthetic_calibration import (
        build_synthetic_calibration,
    )

    calibration = build_synthetic_calibration()
    motors = build_so101_motors(use_degrees=True)
    qpos = np.array([0.3, -0.4, 0.5, -0.2, 0.1, 0.113], dtype=np.float64)

    ticks = sim_rad_to_motor_ticks(
        qpos, calibration=calibration, gripper_limits_rad=SO101_GRIPPER_LIMITS_RAD
    )
    values = normalize_ticks(ticks, motors=motors, calibration=calibration)
    row = np.array([values[name] for name in MOTOR_NAMES])

    decoded = dataset_row_to_sim_qpos(row)
    np.testing.assert_allclose(decoded, qpos, atol=2e-3)


def test_read_privileged_state_none_without_compute_method() -> None:
    from so101_nexus.lerobot_adapter.normalization import read_privileged_state

    # No _compute_obs_components: env is not a component-based sim env.
    env = types.SimpleNamespace(unwrapped=types.SimpleNamespace())
    assert read_privileged_state(env) is None


def test_read_privileged_state_none_when_observations_falsy() -> None:
    from so101_nexus.lerobot_adapter.normalization import read_privileged_state

    # Callable present but config.observations is empty -> skip the channel.
    unwrapped = types.SimpleNamespace(
        _compute_obs_components=lambda: np.zeros(3, dtype=np.float32),
        config=types.SimpleNamespace(observations=[]),
    )
    assert read_privileged_state(types.SimpleNamespace(unwrapped=unwrapped)) is None


def test_read_privileged_state_none_for_camera_only_observations() -> None:
    from so101_nexus.lerobot_adapter.normalization import read_privileged_state

    # All components report size == 0 (cameras): no privileged scalar vector.
    cameras = [types.SimpleNamespace(size=0), types.SimpleNamespace(size=0)]
    unwrapped = types.SimpleNamespace(
        _compute_obs_components=lambda: np.empty(0, dtype=np.float32),
        config=types.SimpleNamespace(observations=cameras),
    )
    assert read_privileged_state(types.SimpleNamespace(unwrapped=unwrapped)) is None


def test_read_privileged_state_returns_float32_vector_via_unwrap() -> None:
    from so101_nexus.lerobot_adapter.normalization import read_privileged_state

    vector = np.arange(4, dtype=np.float64)
    unwrapped = types.SimpleNamespace(
        _compute_obs_components=lambda: vector,
        config=types.SimpleNamespace(observations=[types.SimpleNamespace(size=4)]),
    )
    # Outer env only exposes `unwrapped`; the method lives on the inner env,
    # so a returned vector proves the unwrap path is exercised.
    wrapped = types.SimpleNamespace(unwrapped=unwrapped)

    result = read_privileged_state(wrapped)

    assert result is not None
    assert result.dtype == np.float32
    assert result.shape == (4,)
    np.testing.assert_array_equal(result, np.arange(4, dtype=np.float32))
