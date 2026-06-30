"""Pure unit conversions between LeRobot dataset rows and simulator radians.

These helpers turn recorded ``action`` / ``observation.state`` six-vectors into
simulator joint radians (and back) without a calibration file. They depend only
on NumPy and the joint-name config, so they import cleanly without the
``teleop`` extra (LeRobot is not required). Like the reward functions, they
accept Python floats, NumPy arrays, and torch tensors by duck typing; ``torch``
is never imported here and tensor support relies on operator overloading, so the
same call decodes a single row or a batched ``(..., 6)`` policy tensor.
"""

from __future__ import annotations

import math

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES

GripperLimitsRad = tuple[float, float]
_DEG2RAD = math.pi / 180.0
_RAD2DEG = 180.0 / math.pi
_GRIPPER_INDEX = len(SO101_JOINT_NAMES) - 1
SO101_GRIPPER_LIMITS_RAD: GripperLimitsRad = (math.radians(-10.0), math.radians(100.0))
"""Default SO101 gripper jaw travel in radians (-10 deg .. +100 deg).

Matches the gripper actuator control range across the vendored SO101 MuJoCo
models. Pass the env gripper control bounds (``env.action_space`` low/high at the
gripper index, or ``read_gripper_limits_rad(env)`` from the adapter) for the
exact runtime limits of a specific environment.
"""


def _validate_gripper_limits(gripper_limits_rad: GripperLimitsRad) -> GripperLimitsRad:
    lower, upper = (float(gripper_limits_rad[0]), float(gripper_limits_rad[1]))
    if upper == lower:
        raise ValueError("gripper_limits_rad lower and upper bounds must differ")
    return lower, upper


def dataset_row_to_sim_qpos(
    row,
    *,
    gripper_limits_rad: GripperLimitsRad = SO101_GRIPPER_LIMITS_RAD,
):
    """Decode a normalized LeRobot dataset row into simulator joint radians.

    Recorded ``action`` and ``observation.state`` six-vectors mix units: body
    joints ``shoulder_pan..wrist_roll`` use ``MotorNormMode.DEGREES`` while the
    gripper uses ``MotorNormMode.RANGE_0_100`` (percent of jaw travel, not
    degrees). This inverts that per-motor map, avoiding the silent corruption of
    decoding the whole vector with ``np.deg2rad``.

    Parameters
    ----------
    row:
        NumPy array, torch tensor, or sequence of shape ``(..., 6)`` of recorded
        values in SO101 joint order.
    gripper_limits_rad:
        Simulator gripper ``(low, high)`` control range in radians. Defaults to
        the SO101 gripper travel; pass the env gripper bounds for exactness.

    Returns
    -------
    Same array type as ``row`` with body joints converted via ``deg2rad`` and the
    gripper mapped linearly from ``[0, 100]`` onto ``[low, high]``.

    Notes
    -----
    Assumes the library recording convention (synthetic calibration,
    ``drive_mode=0``). This is a pure unit transform; clamping to the actuator
    range stays the caller's job.
    """
    values = row if hasattr(row, "shape") else np.asarray(row, dtype=np.float64)
    if values.shape[-1] != len(SO101_JOINT_NAMES):
        raise ValueError(
            f"dataset row last dim {values.shape[-1]} != expected {len(SO101_JOINT_NAMES)}"
        )
    lower, upper = _validate_gripper_limits(gripper_limits_rad)
    qpos = values * _DEG2RAD
    qpos[..., _GRIPPER_INDEX] = lower + values[..., _GRIPPER_INDEX] / 100.0 * (upper - lower)
    return qpos


def sim_qpos_to_dataset_row(
    qpos,
    *,
    gripper_limits_rad: GripperLimitsRad = SO101_GRIPPER_LIMITS_RAD,
):
    """Encode simulator joint radians into a normalized LeRobot dataset row.

    Inverse of :func:`dataset_row_to_sim_qpos`: body joints become degrees and
    the gripper its ``RANGE_0_100`` percent. Useful for replaying a simulator
    trajectory or policy output as LeRobot dataset rows.

    Parameters
    ----------
    qpos:
        NumPy array, torch tensor, or sequence of shape ``(..., 6)`` of simulator
        joint radians in SO101 joint order.
    gripper_limits_rad:
        Simulator gripper ``(low, high)`` control range in radians.

    Returns
    -------
    Same array type as ``qpos`` with body joints in degrees and the gripper as a
    ``[0, 100]`` percent of ``[low, high]``.
    """
    values = qpos if hasattr(qpos, "shape") else np.asarray(qpos, dtype=np.float64)
    if values.shape[-1] != len(SO101_JOINT_NAMES):
        raise ValueError(
            f"sim qpos last dim {values.shape[-1]} != expected {len(SO101_JOINT_NAMES)}"
        )
    lower, upper = _validate_gripper_limits(gripper_limits_rad)
    row = values * _RAD2DEG
    row[..., _GRIPPER_INDEX] = (values[..., _GRIPPER_INDEX] - lower) / (upper - lower) * 100.0
    return row
