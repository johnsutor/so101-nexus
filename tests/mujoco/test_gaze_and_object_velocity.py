"""MuJoCo tests for the gaze observations and the object-velocity observation.

The gaze cases place the target object at a known bearing from the real wrist
camera and read the observation back, so they pin the geometry the predicate
claims (a frustum test anchored at the camera) rather than re-deriving it.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import so101_nexus.mujoco  # noqa: F401 - registers envs
from so101_nexus.config import LookAtConfig, PickConfig
from so101_nexus.objects import CubeObject
from so101_nexus.observations import GazeDirection, GazeState, ObjectVelocity
from so101_nexus.testing import component_slice


def _pick_env(observations=None):
    config = PickConfig(
        objects=[CubeObject(half_size=0.012, color="red")], observations=observations
    )
    return gym.make("MuJoCoPickLift-v1", config=config)


def _place_target(env, pos):
    """Teleport the target object to *pos* and refresh derived state."""
    slot = env._slots[env._target_slot_idx]
    env.data.qpos[slot.qpos_addr : slot.qpos_addr + 3] = pos
    env.data.qpos[slot.qpos_addr + 3 : slot.qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]
    env.data.qvel[slot.dof_addr : slot.dof_addr + 6] = 0.0
    mujoco.mj_forward(env.model, env.data)


def _camera(env):
    """Return the wrist camera's world pose, with kinematics refreshed.

    ``mj_step`` leaves ``cam_xpos``/``cam_xmat`` one integration behind ``qpos``,
    so a test that predicts the observation has to forward first.
    """
    mujoco.mj_forward(env.model, env.data)
    cam_pos = env.data.cam_xpos[env._wrist_cam_id].copy()
    axis = -env.data.cam_xmat[env._wrist_cam_id].reshape(3, 3)[:, 2].copy()
    return cam_pos, axis


def test_gaze_state_is_one_on_axis_and_zero_behind_the_camera():
    env = _pick_env()
    inner = env.unwrapped
    try:
        env.reset(seed=0)
        cam_pos, axis = _camera(inner)
        gaze = component_slice(env, GazeState)

        _place_target(inner, cam_pos + axis * 0.15)
        assert float(inner._compute_obs_components()[gaze][0]) == 1.0

        _place_target(inner, cam_pos - axis * 0.15)
        assert float(inner._compute_obs_components()[gaze][0]) == 0.0
    finally:
        env.close()


def test_gaze_is_anchored_at_the_camera_not_the_gripper_tip():
    """The wrist camera sits ~8 cm from the TCP, which is the same order as the
    distance to a graspable object: a TCP-anchored ray reports a target as
    centred in frame when the camera is looking tens of degrees away from it.
    """
    env = _pick_env()
    inner = env.unwrapped
    try:
        env.reset(seed=0)
        cam_pos, axis = _camera(inner)
        tcp = inner._get_tcp_pose()[:3]
        # On the optical axis as measured from the TCP, close in.
        _place_target(inner, tcp + axis * 0.10)

        obj = inner._gaze_target_pos()
        from_tcp = obj - tcp
        from_cam = obj - cam_pos
        angle_tcp = np.arccos(np.clip(axis @ (from_tcp / np.linalg.norm(from_tcp)), -1, 1))
        angle_cam = np.arccos(np.clip(axis @ (from_cam / np.linalg.norm(from_cam)), -1, 1))
        assert angle_cam > np.radians(5.0)  # the two anchors genuinely disagree

        assert inner._gaze_angle_rad() == pytest.approx(float(angle_cam), abs=1e-6)
        assert inner._gaze_angle_rad() != pytest.approx(float(angle_tcp), abs=1e-3)
    finally:
        env.close()


def test_gaze_direction_is_a_unit_vector_from_the_camera():
    env = _pick_env(observations=[GazeDirection(), GazeState()])
    inner = env.unwrapped
    try:
        env.reset(seed=1)
        cam_pos, _ = _camera(inner)  # forwards kinematics; read the obs after
        direction = inner._compute_obs_components()[component_slice(env, GazeDirection)]
        expected = inner._gaze_target_pos() - cam_pos
        expected = expected / np.linalg.norm(expected)
        np.testing.assert_allclose(direction, expected, rtol=0, atol=1e-6)
        assert float(np.linalg.norm(direction)) == pytest.approx(1.0, abs=1e-6)
    finally:
        env.close()


def test_gaze_state_tracks_the_live_camera_fov():
    """A fixed bearing flips with the model's own FOV, which wrist-camera domain
    randomization rewrites per episode."""

    def _gaze_state(fovy_deg: float) -> float:
        env = _pick_env()
        inner = env.unwrapped
        try:
            env.reset(seed=0)
            cam_pos, axis = _camera(inner)
            perp = np.array([axis[1], -axis[0], 0.0])
            perp = perp / np.linalg.norm(perp)
            angle = np.radians(18.0)
            _place_target(inner, cam_pos + (np.cos(angle) * axis + np.sin(angle) * perp) * 0.12)
            inner.model.cam_fovy[inner._wrist_cam_id] = fovy_deg
            return inner._is_looking_at()
        finally:
            env.close()

    assert _gaze_state(10.0) == 0.0  # half-FOV 5 deg < 18 deg bearing
    assert _gaze_state(50.0) == 1.0  # half-FOV 25 deg > 18 deg bearing


def test_lookat_fov_deg_pins_the_gaze_state_boundary():
    """``LookAtConfig.fov_deg`` overrides the live camera for success, and
    ``GazeState`` reports that same pinned boundary so the two cannot disagree."""
    env = gym.make("MuJoCoLookAt-v1", config=LookAtConfig(fov_deg=10.0)).unwrapped
    try:
        env.reset(seed=0)
        cam_pos, axis = _camera(env)
        perp = np.array([axis[1], -axis[0], 0.0])
        perp = perp / np.linalg.norm(perp)
        angle = np.radians(18.0)  # inside the real 48.5 deg fovy, outside the 10 deg pin
        env.data.mocap_pos[env._look_target_mocap_id] = (
            cam_pos + (np.cos(angle) * axis + np.sin(angle) * perp) * 0.12
        )
        mujoco.mj_forward(env.model, env.data)
        assert env._half_fov_rad() == pytest.approx(np.radians(5.0))
        assert env._is_looking_at() == 0.0
        assert env._get_info()["success"] is False
    finally:
        env.close()


def test_lookat_gaze_state_equals_task_success():
    """LookAt scores success on exactly the predicate the observation reports."""
    env = gym.make("MuJoCoLookAt-v1")
    try:
        obs, info = env.reset(seed=3)
        gaze = component_slice(env, GazeState)
        assert float(obs[gaze][0]) == float(info["success"])
        for _ in range(20):
            obs, _, _, _, info = env.step(env.action_space.sample())
            assert float(obs[gaze][0]) == float(info["success"])
    finally:
        env.close()


def test_object_velocity_reports_the_target_free_joint_velocity():
    env = gym.make("MuJoCoPickAndPlace-v1")
    inner = env.unwrapped
    try:
        obs, _ = env.reset(seed=0)
        vel = component_slice(env, ObjectVelocity)
        slot = inner._slots[inner._target_slot_idx]
        inner.data.qvel[slot.dof_addr : slot.dof_addr + 6] = [0.3, -0.2, 0.1, 1.0, -1.5, 0.5]
        mujoco.mj_forward(inner.model, inner.data)
        np.testing.assert_allclose(
            inner._compute_obs_components()[vel],
            np.array([0.3, -0.2, 0.1, 1.0, -1.5, 0.5], dtype=np.float32),
            rtol=0,
            atol=1e-6,
        )
    finally:
        env.close()


def test_object_velocity_is_zero_for_a_settled_object():
    """The place success predicate gates on this being near zero, so a settled
    scene must actually report zero rather than reset noise."""
    env = gym.make("MuJoCoStackCube-v1")
    inner = env.unwrapped
    try:
        obs, _ = env.reset(seed=0)
        vel = obs[component_slice(env, ObjectVelocity)]
        assert float(np.linalg.norm(vel[:3])) < inner.config.cube_static_lin_threshold
        assert float(np.linalg.norm(vel[3:])) < inner.config.cube_static_ang_threshold
    finally:
        env.close()


@pytest.mark.parametrize("component", [GazeState, GazeDirection, ObjectVelocity])
def test_move_task_has_no_object_to_observe(component):
    """Move has no target object, so these components must fail loudly."""
    from so101_nexus.config import MoveConfig
    from so101_nexus.observations import JointPositions

    config = MoveConfig(observations=[JointPositions(), component()])
    env = gym.make("MuJoCoMove-v1", config=config)
    try:
        with pytest.raises(NotImplementedError):
            env.reset(seed=0)
    finally:
        env.close()
