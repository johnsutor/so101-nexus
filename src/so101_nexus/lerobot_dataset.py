"""Pure conversions between LeRobot dataset rows and simulator state.

These helpers turn recorded ``action`` / ``observation.state`` six-vectors into
simulator joint radians (and back) without a calibration file, and re-lay a
recorded ``observation.environment_state`` matrix onto a newer observation
component layout. They depend only on NumPy and the joint-name config, so they
import cleanly without the ``teleop`` extra (LeRobot is not required). Like the
reward functions, the unit conversions accept Python floats, NumPy arrays, and
torch tensors by duck typing; ``torch`` is never imported here and tensor
support relies on operator overloading, so the same call decodes a single row
or a batched ``(..., 6)`` policy tensor.
"""

from __future__ import annotations

import math
from collections import Counter
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.observations import privileged_state_feature_names

if TYPE_CHECKING:
    from collections.abc import Sequence

    from so101_nexus.observations import Observation

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


def relabel_environment_state(
    env_state: np.ndarray,
    recorded_names: Sequence[str],
    observations: Sequence[Observation],
    *,
    dt: float,
    episode_index: np.ndarray | None = None,
) -> np.ndarray:
    """Re-lay a recorded ``observation.environment_state`` matrix onto a new layout.

    A dataset recorded before a component was added to a task's default
    ``observations`` carries fewer columns than the env now emits, so its rows
    cannot be fed to a policy trained against the current env. This maps the
    recorded columns onto the target layout by name and reconstructs
    ``JointVelocities`` offline as a backward finite difference of the recorded
    ``JointPositions`` columns, which is the same relabeling a real SO-101
    control loop performs on consecutive servo position readings. The result is
    an interval average, not the instantaneous ``qvel`` a live env reports, so
    relabeled velocities are unbiased but noisier than recorded ones.

    Parameters
    ----------
    env_state:
        Recorded state matrix of shape ``(n_frames, len(recorded_names))``.
    recorded_names:
        Per-column names as declared by the dataset's
        ``observation.environment_state`` feature.
    observations:
        Target observation components (an ``EnvironmentConfig.observations``
        list); their concatenation order defines the output layout.
    dt:
        Simulated seconds between consecutive frames, i.e. the env's
        ``control_dt`` (physics timestep times substeps), NOT the dataset's
        ``1 / fps``. The teleop recorder sleeps to pace the operator but
        advances the simulation exactly one step per recorded frame, so using
        the wall-clock frame period rescales every velocity by
        ``control_dt * fps``.
    episode_index:
        Per-frame episode id of shape ``(n_frames,)``. Finite differences never
        cross an episode boundary; the first frame of each episode gets zero
        velocity, matching the zeroed ``qvel`` both backends write on reset.
        ``None`` treats every frame as one episode. Rows must be grouped by
        episode and chronological within each episode (LeRobot's own row order,
        i.e. sorted by ``index``); a non-contiguous episode id is rejected, but
        rows shuffled *within* one episode cannot be detected and would produce
        differences between temporally unrelated frames.

    Returns
    -------
    numpy.ndarray
        C-contiguous ``float32`` array of shape ``(n_frames, target_dim)``.

    Raises
    ------
    ValueError
        If a target column is neither recorded nor reconstructible, if
        ``recorded_names`` or the target layout repeats a name, or if
        ``episode_index`` has the wrong length or a non-contiguous episode.
    """
    state = np.asarray(env_state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != len(recorded_names):
        raise ValueError(
            f"env_state shape {state.shape} does not match {len(recorded_names)} recorded names"
        )
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    target_names = privileged_state_feature_names(observations)
    _reject_duplicate_names(recorded_names, "recorded_names")
    _reject_duplicate_names(target_names, "target observation layout")
    column = {name: i for i, name in enumerate(recorded_names)}
    missing = [name for name in target_names if name not in column]

    n_joints = len(SO101_JOINT_NAMES)
    velocity_names = [f"joint_velocities_{i}" for i in range(n_joints)]
    reconstructed: dict[str, np.ndarray] = {}
    if missing:
        unreconstructible = [name for name in missing if name not in velocity_names]
        if unreconstructible:
            raise ValueError(
                f"recorded environment_state cannot supply target columns {unreconstructible}; "
                "re-record the dataset against the current observation layout"
            )
        position_names = [f"joint_positions_{i}" for i in range(n_joints)]
        if any(name not in column for name in position_names):
            raise ValueError(
                "joint velocities require recorded joint_positions_0.."
                f"{n_joints - 1}, which the dataset does not declare"
            )
        qpos = state[:, [column[name] for name in position_names]]
        qvel = np.zeros_like(qpos)
        for start, stop in _episode_blocks(episode_index, len(state)):
            qvel[start + 1 : stop] = (qpos[start + 1 : stop] - qpos[start : stop - 1]) / dt
        reconstructed = dict(zip(velocity_names, qvel.T, strict=True))

    out = np.empty((len(state), len(target_names)), dtype=np.float32)
    for i, name in enumerate(target_names):
        out[:, i] = state[:, column[name]] if name in column else reconstructed[name]
    return out


def _reject_duplicate_names(names: Sequence[str], label: str) -> None:
    """Raise if ``names`` repeats an entry, which would alias two state columns."""
    duplicates = sorted({name for name, n in Counter(names).items() if n > 1})
    if duplicates:
        raise ValueError(
            f"{label} repeats {duplicates}; observation column names must be unique "
            "or columns would silently alias"
        )


def _episode_blocks(episode_index: np.ndarray | None, n_frames: int) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` row ranges per episode, rejecting unusable input."""
    if episode_index is None:
        return [(0, n_frames)]
    ids = np.asarray(episode_index)
    if ids.shape != (n_frames,):
        raise ValueError(f"episode_index shape {ids.shape} does not match {n_frames} frames")
    blocks = _contiguous_blocks(ids)
    if len({int(ids[start]) for start, _ in blocks}) != len(blocks):
        raise ValueError(
            "episode_index has a non-contiguous episode; rows must be grouped by episode "
            "and chronological within each (sort the dataset by `index` first)"
        )
    return blocks


def _contiguous_blocks(episode_index: np.ndarray) -> list[tuple[int, int]]:
    """Return ``(start, stop)`` row ranges for each run of equal episode ids."""
    if episode_index.size == 0:
        return []
    edges = np.flatnonzero(episode_index[1:] != episode_index[:-1]) + 1
    bounds = [0, *edges.tolist(), episode_index.size]
    return list(pairwise(bounds))
