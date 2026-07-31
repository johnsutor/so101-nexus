"""Warp tests for the gaze observations and the object-velocity observation.

The gaze predicate is shared geometry (``so101_nexus.gaze``) fed from two
independently written camera plumbings, so the cases here pin the batched
plumbing (per-world camera pose, per-world FOV) and its agreement with the
MuJoCo backend at matched simulator state.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.warp


def _pick_env(num_envs=4, observations=None, seed=0):
    from so101_nexus.config import PickConfig
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    config = PickConfig(observations=observations)
    return WarpPickLiftVectorEnv(num_envs=num_envs, config=config, device="cpu", seed=seed)


def _expected_gaze_state(env, half_fov_rad, cam_id):
    """Recompute the predicate from the batched camera arrays, in NumPy."""
    cam_pos = env._cam_xpos[:, cam_id, :].numpy()
    axis = -env._cam_xmat[:, cam_id, :, 2].numpy()
    to_obj = env._gaze_target_pos().numpy() - cam_pos
    to_obj = to_obj / np.linalg.norm(to_obj, axis=1, keepdims=True)
    cos = np.clip((axis * to_obj).sum(1), -1.0, 1.0)
    return (np.arccos(cos) <= half_fov_rad).astype(np.float32)


def test_gaze_state_matches_the_batched_camera_geometry():
    import torch

    from so101_nexus.observations import GazeState, JointPositions

    env = _pick_env(observations=[JointPositions(), GazeState()])
    obs, _ = env.reset(seed=0)
    expected = _expected_gaze_state(env, env._static_half_fov_rad, env._wrist_cam_id)
    torch.testing.assert_close(obs[:, 6], torch.from_numpy(expected))


def test_gaze_reads_the_wrist_camera_in_a_multi_camera_scene():
    """An overhead camera puts a second camera in the model, so a wrong camera id
    stops being indistinguishable from the right one."""
    import mujoco
    import torch

    from so101_nexus.observations import GazeState, JointPositions, OverheadCamera

    env = _pick_env(
        observations=[JointPositions(), GazeState(), OverheadCamera(width=32, height=24)]
    )
    assert env.mjm.ncam == 2
    wrist_id = mujoco.mj_name2id(env.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    overhead_id = mujoco.mj_name2id(env.mjm, mujoco.mjtObj.mjOBJ_CAMERA, "overhead_cam")
    assert env._wrist_cam_id == wrist_id != overhead_id

    obs, _ = env.reset(seed=0)
    got = obs["state"][:, 6]
    torch.testing.assert_close(
        got, torch.from_numpy(_expected_gaze_state(env, env._static_half_fov_rad, wrist_id))
    )
    # The overhead camera looks straight down at the whole workspace, so keying on
    # it would report every object in view; the two must not coincide here.
    from_overhead = _expected_gaze_state(env, env._static_half_fov_rad, overhead_id)
    assert not np.array_equal(got.numpy(), from_overhead)


def test_gaze_state_is_per_world_not_broadcast():
    """Each world sees its own object, so the predicate must vary across worlds."""
    import torch

    from so101_nexus.observations import GazeState, JointPositions
    from so101_nexus.testing import component_slice

    env = _pick_env(num_envs=6, observations=[JointPositions(), GazeState()])
    env.reset(seed=0)
    gaze = component_slice(env, GazeState)
    cam_pos = env._cam_xpos[:, env._wrist_cam_id, :]
    axis = -env._cam_xmat[:, env._wrist_cam_id, :, 2]
    # Half the worlds get the object dead ahead, the rest get it behind the camera.
    ahead = torch.arange(6) % 2 == 0
    sign = torch.where(ahead, 1.0, -1.0).unsqueeze(1)
    targets = cam_pos + axis * sign * 0.15
    env._gaze_target_pos = lambda: targets
    got = env._compute_state_vector()[:, gaze][:, 0]
    torch.testing.assert_close(got, ahead.to(torch.float32))


def test_gaze_state_follows_per_world_fov_randomization():
    """Wrist-camera DR gives each world its own FOV; the boundary must move with it."""
    import torch

    from so101_nexus.observations import GazeState, JointPositions, WristCamera

    env = _pick_env(
        num_envs=4,
        observations=[JointPositions(), GazeState(), WristCamera(width=32, height=24)],
    )
    env.reset(seed=0)
    cam_pos = env._cam_xpos[:, env._wrist_cam_id, :]
    axis = -env._cam_xmat[:, env._wrist_cam_id, :, 2]
    perp = torch.stack([axis[:, 1], -axis[:, 0], torch.zeros(4)], dim=1)
    perp = perp / perp.norm(dim=1, keepdim=True)
    bearing = torch.deg2rad(env._cam_fovy[:, env._wrist_cam_id]) * 0.5
    # Two worlds just inside their own half-FOV, two just outside it.
    inside = torch.tensor([True, False, True, False])
    angle = torch.where(inside, bearing * 0.9, bearing * 1.1).unsqueeze(1)
    targets = cam_pos + (torch.cos(angle) * axis + torch.sin(angle) * perp) * 0.15
    env._gaze_target_pos = lambda: targets
    torch.testing.assert_close(env._is_looking_at(), inside.to(torch.float32))


def test_lookat_gaze_state_equals_success():
    import torch

    from so101_nexus.config import LookAtConfig
    from so101_nexus.observations import GazeState, JointPositions
    from so101_nexus.warp.look_at_env import WarpLookAtVectorEnv

    env = WarpLookAtVectorEnv(
        num_envs=4,
        config=LookAtConfig(observations=[JointPositions(), GazeState()]),
        device="cpu",
        seed=0,
    )
    env.reset(seed=0)
    obs, _, _, _, info = env.step(torch.zeros((4, 6)))
    torch.testing.assert_close(obs[:, 6], info["success"].to(torch.float32))


def test_object_velocity_reports_the_target_free_joint_velocity():
    import torch

    from so101_nexus.observations import JointPositions, ObjectVelocity
    from so101_nexus.testing import component_slice

    env = _pick_env(num_envs=3, observations=[JointPositions(), ObjectVelocity()])
    env.reset(seed=0)
    cols = env._target_dadr[:, None] + torch.arange(6)
    env.qvel[env._world_rows[:, None], cols] = torch.tensor([[0.3, -0.2, 0.1, 1.0, -1.5, 0.5]] * 3)
    got = env._compute_state_vector()[:, component_slice(env, ObjectVelocity)]
    torch.testing.assert_close(got, torch.tensor([[0.3, -0.2, 0.1, 1.0, -1.5, 0.5]] * 3))


def test_stack_cube_object_velocity_tracks_cube_a():
    import torch

    from so101_nexus.config import StackCubeConfig
    from so101_nexus.observations import JointPositions, ObjectVelocity
    from so101_nexus.testing import component_slice
    from so101_nexus.warp.stack_cube import WarpStackCubeVectorEnv

    env = WarpStackCubeVectorEnv(
        num_envs=2,
        config=StackCubeConfig(observations=[JointPositions(), ObjectVelocity()]),
        device="cpu",
        seed=0,
    )
    env.reset(seed=0)
    env.qvel[env._world_rows[:, None], env._a_dadr[:, None] + torch.arange(6)] = torch.tensor(
        [[0.1, 0.2, 0.3, -0.1, -0.2, -0.3]] * 2
    )
    got = env._compute_state_vector()[:, component_slice(env, ObjectVelocity)]
    torch.testing.assert_close(got, torch.tensor([[0.1, 0.2, 0.3, -0.1, -0.2, -0.3]] * 2))


@pytest.mark.parametrize("component_name", ["GazeState", "GazeDirection", "ObjectVelocity"])
def test_move_task_rejects_object_components(component_name):
    """Move has no target object, so these must fail at construction."""
    import so101_nexus.observations as obs_module
    from so101_nexus.config import MoveConfig
    from so101_nexus.observations import JointPositions
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    component = getattr(obs_module, component_name)()
    config = MoveConfig(observations=[JointPositions(), component])
    with pytest.raises(NotImplementedError, match=component_name):
        WarpMoveVectorEnv(num_envs=2, config=config, device="cpu")


def test_gaze_angle_matches_mujoco_at_matched_state():
    """Both backends read their own camera arrays; at identical simulator state
    and identical camera extrinsics they must report the same bearing."""
    import gymnasium as gym
    import mujoco
    import torch

    import so101_nexus.mujoco  # noqa: F401 - registers MuJoCo*-v1
    from so101_nexus.config import PickConfig

    w = _pick_env(num_envs=1)
    m_env = gym.make("MuJoCoPickLift-v1", config=PickConfig())
    m = m_env.unwrapped
    try:
        w.reset(seed=0)
        m.reset(seed=0)
        m.data.qpos[:] = w.qpos[0].detach().cpu().numpy().astype(np.float64)
        m.data.qvel[:] = w.qvel[0].detach().cpu().numpy().astype(np.float64)
        mujoco.mj_forward(m.model, m.data)

        # Off-axis (out of frame on both backends) and on-axis (in frame), so the
        # shared FOV boundary is cross-checked in both directions.
        cam_pos = w._cam_xpos[0, w._wrist_cam_id].numpy().astype(np.float64)
        axis = -w._cam_xmat[0, w._wrist_cam_id, :, 2].numpy().astype(np.float64)
        for target, expected_in_view in (
            (np.array([0.25, 0.05, 0.10]), 0.0),
            (cam_pos + axis * 0.15, 1.0),
        ):
            m._gaze_target_pos = lambda t=target: t
            w._gaze_target_pos = lambda t=target: torch.tensor(t[None], dtype=torch.float32)

            # Warp carries the camera pose in float32, so the bearings agree to
            # single precision; a wiring mismatch (wrong anchor or axis) is ~0.1 rad.
            assert float(w._gaze_angle_rad()[0]) == pytest.approx(m._gaze_angle_rad(), abs=2e-4)
            assert float(w._is_looking_at()[0]) == m._is_looking_at() == expected_in_view
    finally:
        m_env.close()
