"""Normalization helpers for the LeRobot simulator follower adapter."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

from so101_nexus_core.config import SO101_JOINT_NAMES

if TYPE_CHECKING:
    from collections.abc import Mapping

MOTOR_NAMES = SO101_JOINT_NAMES
BODY_MOTOR_NAMES = SO101_JOINT_NAMES[:-1]
GRIPPER_NAME = "gripper"
MOTOR_MODEL = "sts3215"
MOTOR_RESOLUTION = 4096
TICKS_PER_RADIAN = (MOTOR_RESOLUTION - 1) / (2 * math.pi)
GripperLimitsRad = tuple[float, float]


def build_so101_motors(*, use_degrees: bool) -> dict[str, Motor]:
    """Return SO101 motors configured with LeRobot's norm modes."""
    body_mode = MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    return {
        "shoulder_pan": Motor(1, MOTOR_MODEL, body_mode),
        "shoulder_lift": Motor(2, MOTOR_MODEL, body_mode),
        "elbow_flex": Motor(3, MOTOR_MODEL, body_mode),
        "wrist_flex": Motor(4, MOTOR_MODEL, body_mode),
        "wrist_roll": Motor(5, MOTOR_MODEL, body_mode),
        "gripper": Motor(6, MOTOR_MODEL, MotorNormMode.RANGE_0_100),
    }


def _bus_for(
    *,
    motors: Mapping[str, Motor],
    calibration: dict[str, MotorCalibration],
) -> FeetechMotorsBus:
    return FeetechMotorsBus(port="/dev/null", motors=dict(motors), calibration=calibration)


def normalize_ticks(
    ticks: dict[str, int],
    *,
    motors: Mapping[str, Motor],
    calibration: dict[str, MotorCalibration],
) -> dict[str, float]:
    """Normalize name-keyed raw motor ticks using LeRobot's Feetech math."""
    bus = _bus_for(motors=motors, calibration=calibration)
    id_values = {motors[name].id: int(value) for name, value in ticks.items()}
    normalized = bus._normalize(id_values)
    return {name: float(normalized[motors[name].id]) for name in ticks}


def unnormalize_values(
    values: dict[str, float],
    *,
    motors: Mapping[str, Motor],
    calibration: dict[str, MotorCalibration],
) -> dict[str, int]:
    """Unnormalize name-keyed LeRobot values using LeRobot's Feetech math."""
    bus = _bus_for(motors=motors, calibration=calibration)
    id_values = {motors[name].id: float(value) for name, value in values.items()}
    unnormalized = bus._unnormalize(id_values)
    return {name: int(unnormalized[motors[name].id]) for name in values}


def _validate_gripper_limits(gripper_limits_rad: GripperLimitsRad) -> GripperLimitsRad:
    lower, upper = (float(gripper_limits_rad[0]), float(gripper_limits_rad[1]))
    if upper == lower:
        raise ValueError("gripper_limits_rad lower and upper bounds must differ")
    return lower, upper


def sim_rad_to_motor_ticks(
    qpos_rad: np.ndarray,
    *,
    calibration: dict[str, MotorCalibration],
    gripper_limits_rad: GripperLimitsRad,
) -> dict[str, int]:
    """Convert simulator joint radians to calibrated raw motor ticks."""
    qpos = np.asarray(qpos_rad, dtype=np.float64)
    if qpos.shape != (len(MOTOR_NAMES),):
        raise ValueError(f"qpos_rad shape {qpos.shape} != expected ({len(MOTOR_NAMES)},)")

    ticks: dict[str, int] = {}
    for index, name in enumerate(BODY_MOTOR_NAMES):
        cal = calibration[name]
        mid = (cal.range_min + cal.range_max) / 2
        sign = -1 if cal.drive_mode else 1
        ticks[name] = round(sign * qpos[index] * TICKS_PER_RADIAN + mid)

    lower, upper = _validate_gripper_limits(gripper_limits_rad)
    cal = calibration[GRIPPER_NAME]
    frac = (qpos[-1] - lower) / (upper - lower)
    if cal.drive_mode:
        frac = 1 - frac
    ticks[GRIPPER_NAME] = round(cal.range_min + frac * (cal.range_max - cal.range_min))
    return ticks


def motor_ticks_to_sim_rad(
    ticks: dict[str, int],
    *,
    calibration: dict[str, MotorCalibration],
    gripper_limits_rad: GripperLimitsRad,
) -> np.ndarray:
    """Convert calibrated raw motor ticks to simulator joint radians."""
    qpos = np.zeros(len(MOTOR_NAMES), dtype=np.float64)
    for index, name in enumerate(BODY_MOTOR_NAMES):
        cal = calibration[name]
        mid = (cal.range_min + cal.range_max) / 2
        sign = -1 if cal.drive_mode else 1
        qpos[index] = sign * (int(ticks[name]) - mid) / TICKS_PER_RADIAN

    lower, upper = _validate_gripper_limits(gripper_limits_rad)
    cal = calibration[GRIPPER_NAME]
    frac = (int(ticks[GRIPPER_NAME]) - cal.range_min) / (cal.range_max - cal.range_min)
    if cal.drive_mode:
        frac = 1 - frac
    qpos[-1] = lower + frac * (upper - lower)
    return qpos


def _unwrap_env(env: object) -> object:
    return getattr(env, "unwrapped", env)


def _as_qpos_vector(value: Any) -> np.ndarray:
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape != (len(MOTOR_NAMES),):
        raise ValueError(f"sim qpos shape {arr.shape} != expected ({len(MOTOR_NAMES)},)")
    return arr.astype(np.float64, copy=False)


def read_sim_qpos(env: object) -> np.ndarray:
    """Read current simulator joint positions in SO101 joint order."""
    unwrapped = _unwrap_env(env)
    get_current_qpos = getattr(unwrapped, "_get_current_qpos", None)
    if callable(get_current_qpos):
        return _as_qpos_vector(get_current_qpos())

    agent = getattr(unwrapped, "agent", None)
    robot = getattr(agent, "robot", None)
    get_qpos = getattr(robot, "get_qpos", None)
    if callable(get_qpos):
        return _as_qpos_vector(get_qpos())

    raise TypeError(
        "Simulator env must expose _get_current_qpos() or agent.robot.get_qpos()."
    )


def _control_bounds(env: object) -> tuple[np.ndarray, np.ndarray] | None:
    unwrapped = _unwrap_env(env)
    if hasattr(unwrapped, "_ctrl_low") and hasattr(unwrapped, "_ctrl_high"):
        low = np.asarray(unwrapped._ctrl_low, dtype=np.float64)
        high = np.asarray(unwrapped._ctrl_high, dtype=np.float64)
        return low, high

    action_space = getattr(unwrapped, "action_space", None)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is not None and high is not None:
        return np.asarray(low, dtype=np.float64), np.asarray(high, dtype=np.float64)

    return None


def read_gripper_limits_rad(env: object) -> GripperLimitsRad:
    """Read gripper simulator limits from the active env control range."""
    bounds = _control_bounds(env)
    if bounds is None:
        raise TypeError("Simulator env does not expose gripper control limits.")
    low, high = bounds
    gripper_index = MOTOR_NAMES.index(GRIPPER_NAME)
    return float(low[gripper_index]), float(high[gripper_index])


def clip_qpos_to_env_ctrlrange(env: object, qpos_rad: np.ndarray) -> np.ndarray:
    """Clip qpos to env actuator control bounds when the env exposes them."""
    qpos = np.asarray(qpos_rad, dtype=np.float64)
    bounds = _control_bounds(env)
    if bounds is None:
        return qpos.copy()
    low, high = bounds
    return np.clip(qpos, low, high)


def action_for_env(env: object, qpos_rad: np.ndarray) -> np.ndarray:
    """Return the absolute-position action vector for the active simulator env."""
    return clip_qpos_to_env_ctrlrange(env, qpos_rad)
