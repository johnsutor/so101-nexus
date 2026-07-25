"""Tests for the lerobot-free dataset row to sim qpos conversions."""

from __future__ import annotations

import math
import subprocess
import sys

import numpy as np
import pytest

from so101_nexus import (
    SO101_GRIPPER_LIMITS_RAD,
    EndEffectorPose,
    GraspState,
    JointPositions,
    JointVelocities,
    dataset_row_to_sim_qpos,
    privileged_state_feature_names,
    relabel_environment_state,
    sim_qpos_to_dataset_row,
)


def test_dataset_row_gripper_decodes_as_range_0_100_not_degrees() -> None:
    low, high = SO101_GRIPPER_LIMITS_RAD
    row = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 15.0])
    decoded = dataset_row_to_sim_qpos(row)

    assert decoded[5] == pytest.approx(low + (15.0 / 100.0) * (high - low))
    # The naive deg2rad-the-whole-vector decode is the bug; it is far off here.
    assert abs(decoded[5] - math.radians(15.0)) > 0.1


def test_sim_qpos_to_dataset_row_round_trips_batched() -> None:
    rng = np.random.default_rng(0)
    qpos = rng.uniform(-0.5, 0.5, size=(4, 6))
    low, high = SO101_GRIPPER_LIMITS_RAD
    qpos[:, 5] = rng.uniform(low, high, size=4)

    row = sim_qpos_to_dataset_row(qpos)
    np.testing.assert_allclose(dataset_row_to_sim_qpos(row), qpos, atol=1e-12)
    np.testing.assert_allclose(row[:, :5], np.rad2deg(qpos[:, :5]), atol=1e-12)
    assert np.all((row[:, 5] >= 0.0) & (row[:, 5] <= 100.0))


def test_dataset_row_to_sim_qpos_respects_custom_gripper_limits() -> None:
    row = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0])
    decoded = dataset_row_to_sim_qpos(row, gripper_limits_rad=(0.0, 2.0))
    assert decoded[5] == pytest.approx(1.0)


def test_dataset_row_to_sim_qpos_supports_torch_tensor() -> None:
    torch = pytest.importorskip("torch")

    row_np = np.array([[10.0, -20.0, 30.0, -5.0, 1.0, 15.0], [0.0, 0.0, 0.0, 0.0, 0.0, 100.0]])
    decoded_t = dataset_row_to_sim_qpos(torch.tensor(row_np))

    assert isinstance(decoded_t, torch.Tensor)
    np.testing.assert_allclose(decoded_t.numpy(), dataset_row_to_sim_qpos(row_np), atol=1e-12)


def test_dataset_row_to_sim_qpos_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="6"):
        dataset_row_to_sim_qpos(np.zeros(5))


def test_decode_helpers_import_without_lerobot() -> None:
    """The decode path must work without the teleop extra (LeRobot)."""
    code = (
        "import sys; sys.modules['lerobot'] = None;"  # make `import lerobot` raise
        "import numpy as np;"
        "from so101_nexus import dataset_row_to_sim_qpos, SO101_GRIPPER_LIMITS_RAD;"
        "low, high = SO101_GRIPPER_LIMITS_RAD;"
        "row = np.zeros(6); row[5] = 50.0;"
        "q = dataset_row_to_sim_qpos(row);"
        "assert q.shape == (6,);"
        "assert abs(q[5] - (low + 0.5 * (high - low))) < 1e-9;"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")


def _legacy_recording(n_frames: int, seed: int = 0):
    """Build a pre-JointVelocities recording: JointPositions + GraspState."""
    legacy = [JointPositions(), GraspState()]
    names = privileged_state_feature_names(legacy)
    rng = np.random.default_rng(seed)
    state = rng.uniform(-1.0, 1.0, size=(n_frames, len(names))).astype(np.float32)
    return state, names


def test_relabel_reconstructs_joint_velocities_by_finite_difference() -> None:
    state, names = _legacy_recording(8)
    target = [JointPositions(), JointVelocities(), GraspState()]
    dt = 1.0 / 30.0

    out = relabel_environment_state(state, names, target, dt=dt)

    assert out.shape == (8, 13)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, :6], state[:, :6], rtol=0, atol=0)
    np.testing.assert_allclose(out[:, 12], state[:, 6], rtol=0, atol=0)
    # First frame has no predecessor: reset zeroes qvel in both backends.
    np.testing.assert_array_equal(out[0, 6:12], np.zeros(6, dtype=np.float32))
    np.testing.assert_allclose(
        out[1:, 6:12], (state[1:, :6] - state[:-1, :6]) / dt, rtol=1e-5, atol=1e-5
    )


def test_relabel_does_not_difference_across_episode_boundaries() -> None:
    state, names = _legacy_recording(6, seed=1)
    episode_index = np.array([0, 0, 0, 1, 1, 1])

    out = relabel_environment_state(
        state,
        names,
        [JointPositions(), JointVelocities()],
        dt=0.05,
        episode_index=episode_index,
    )

    np.testing.assert_array_equal(out[0, 6:12], np.zeros(6, dtype=np.float32))
    np.testing.assert_array_equal(out[3, 6:12], np.zeros(6, dtype=np.float32))
    np.testing.assert_allclose(
        out[4, 6:12], (state[4, :6] - state[3, :6]) / 0.05, rtol=1e-5, atol=1e-5
    )


def test_relabel_reorders_recorded_columns_by_name() -> None:
    recorded = [GraspState(), JointPositions()]
    names = privileged_state_feature_names(recorded)
    state = np.arange(7, dtype=np.float32)[None, :]

    out = relabel_environment_state(state, names, [JointPositions(), GraspState()], dt=0.1)

    np.testing.assert_array_equal(out[0], np.array([1, 2, 3, 4, 5, 6, 0], dtype=np.float32))


def test_relabel_rejects_target_columns_it_cannot_reconstruct() -> None:
    state, names = _legacy_recording(3)

    with pytest.raises(ValueError, match="end_effector_pose_0"):
        relabel_environment_state(state, names, [JointPositions(), EndEffectorPose()], dt=0.1)


def test_relabel_requires_recorded_joint_positions_for_velocities() -> None:
    recorded = [GraspState()]
    names = privileged_state_feature_names(recorded)
    state = np.zeros((3, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="joint_positions"):
        relabel_environment_state(state, names, [GraspState(), JointVelocities()], dt=0.1)


@pytest.mark.parametrize("dt", [0.0, -0.1])
def test_relabel_rejects_non_positive_dt(dt: float) -> None:
    state, names = _legacy_recording(3)

    with pytest.raises(ValueError, match="dt must be > 0"):
        relabel_environment_state(state, names, [JointPositions(), JointVelocities()], dt=dt)


def test_relabel_rejects_state_width_mismatch() -> None:
    state, names = _legacy_recording(3)

    with pytest.raises(ValueError, match="does not match"):
        relabel_environment_state(state[:, :-1], names, [JointPositions()], dt=0.1)


def test_relabel_rejects_episode_index_length_mismatch() -> None:
    """A short episode_index used to leave the uncovered tail at zero velocity."""
    state, names = _legacy_recording(4)
    target = [JointPositions(), JointVelocities()]

    with pytest.raises(ValueError, match="episode_index shape"):
        relabel_environment_state(state, names, target, dt=0.02, episode_index=np.array([0, 0]))


@pytest.mark.parametrize(
    "episode_index",
    [
        pytest.param(np.array([0, 1, 0, 1]), id="interleaved"),
        pytest.param(np.array([1, 1, 0, 1]), id="episode_resumes"),
    ],
)
def test_relabel_rejects_non_contiguous_episodes(episode_index: np.ndarray) -> None:
    """Unsorted rows used to silently difference temporally unrelated frames."""
    state, names = _legacy_recording(4)
    target = [JointPositions(), JointVelocities()]

    with pytest.raises(ValueError, match="non-contiguous episode"):
        relabel_environment_state(state, names, target, dt=0.02, episode_index=episode_index)


def test_relabel_rejects_duplicate_recorded_names() -> None:
    """Duplicate names would alias two columns onto one source index."""
    state = np.zeros((2, 12), dtype=np.float32)
    names = [*privileged_state_feature_names([JointPositions()])] * 2

    with pytest.raises(ValueError, match="repeats"):
        relabel_environment_state(state, names, [JointPositions()], dt=0.02)


def test_relabel_rejects_duplicate_target_components() -> None:
    state, names = _legacy_recording(3)

    with pytest.raises(ValueError, match="repeats"):
        relabel_environment_state(state, names, [JointPositions(), JointPositions()], dt=0.02)


def test_relabel_output_is_c_contiguous_on_both_paths() -> None:
    """Layout must not depend on whether any column needed reconstruction."""
    state, names = _legacy_recording(4)
    episode_index = np.zeros(4, dtype=np.int64)

    reconstructed = relabel_environment_state(
        state, names, [JointPositions(), JointVelocities()], dt=0.02, episode_index=episode_index
    )
    passthrough = relabel_environment_state(
        state, names, [JointPositions(), GraspState()], dt=0.02, episode_index=episode_index
    )

    assert reconstructed.flags["C_CONTIGUOUS"]
    assert passthrough.flags["C_CONTIGUOUS"]
    # Both paths must return an independent copy, never a view on the input.
    passthrough[0, 0] = 12345.0
    assert state[0, 0] != 12345.0
