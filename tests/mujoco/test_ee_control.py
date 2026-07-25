"""End-effector control modes for the MuJoCo backend.

Covers the ``pd_ee_pose`` / ``pd_ee_delta_pose`` action-space contract and the
damped least-squares IK behind it. Per-task coverage of the two modes lives in
``test_envs.py``; this file is about the end-effector contract itself.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

from contextlib import contextmanager

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import so101_nexus.mujoco  # noqa: F401 - registers envs
from so101_nexus.config import (
    EE_CONTROL_MODES,
    SO101_JOINT_NAMES,
    RobotConfig,
    TouchConfig,
)
from so101_nexus.kinematics import (
    EE_ACTION_DIM,
    EE_DELTA_ACTION_SCALE,
    orientation_error,
    quat_to_rotvec,
)

# Half-width of the pd_ee_pose position box, mirroring
# so101_nexus.mujoco.base_env._EE_WORKSPACE_RADIUS. The measured maximum TCP
# reach is 0.5457 m.
_EXPECTED_WORKSPACE_RADIUS = 0.55

# A normalized +1 pd_ee_delta_pose action maps to this physical step: 2 cm per
# position axis, 0.1 rad per rotation-vector axis, 0.2 rad on the gripper.
# Mirrors so101_nexus.kinematics.EE_DELTA_ACTION_SCALE.
_EXPECTED_EE_DELTA_SCALE = np.array([0.02, 0.02, 0.02, 0.1, 0.1, 0.1, 0.2])

# The named "extended" pose is the working configuration the tracking contract is
# asserted from. The default rest pose folds the arm back over its own base,
# where the position Jacobian has sigma_min = 0.047; three damped iterations at
# EE_IK_DAMPING = 0.05 then leave ~3 mm of a commanded 1 cm step along that weak
# direction. Extended has sigma_min = 0.092 and lands the same command inside
# 0.14 mm. Both are honest solver behavior, but only the second says anything
# about tracking rather than about a singularity.
_WORKING_POSE = "extended"


@contextmanager
def _touch_env(
    control_mode: str,
    *,
    init_pose: str | None = None,
    robot: RobotConfig | None = None,
):
    """Yield an unwrapped Touch env in ``control_mode``, closed on exit."""
    config = TouchConfig(robot=robot if robot is not None else RobotConfig(init_pose=init_pose))
    env = gym.make("MuJoCoTouch-v1", config=config, control_mode=control_mode)
    try:
        yield env.unwrapped
    finally:
        env.close()


def _tcp_at(env, arm_qpos: np.ndarray) -> np.ndarray:
    """World TCP position with the arm placed at ``arm_qpos``, evaluated off-sim."""
    data = mujoco.MjData(env.model)
    data.qpos[:] = env.data.qpos
    data.qpos[env._arm_qpos_addrs] = arm_qpos
    mujoco.mj_kinematics(env.model, data)
    return data.site_xpos[env._tcp_site_id].copy()


@pytest.mark.parametrize("control_mode", EE_CONTROL_MODES)
def test_ee_action_space_is_seven_dimensional(control_mode):
    """Both EE modes expose the 7-dim [x, y, z, wx, wy, wz, gripper] action."""
    with _touch_env(control_mode) as env:
        assert env.action_space.shape == (EE_ACTION_DIM,)
        assert env.action_space.shape == (7,)


def test_ee_delta_pose_action_space_is_normalized():
    """pd_ee_delta_pose is the normalized [-1, 1] box the delta contract requires."""
    with _touch_env("pd_ee_delta_pose") as env:
        np.testing.assert_array_equal(env.action_space.low, np.full(7, -1.0, dtype=np.float32))
        np.testing.assert_array_equal(env.action_space.high, np.full(7, 1.0, dtype=np.float32))


def test_ee_pose_action_space_bounds():
    """pd_ee_pose bounds the workspace, the rotation vector, and the gripper target."""
    with _touch_env("pd_ee_pose") as env:
        low, high = env.action_space.low, env.action_space.high
        np.testing.assert_allclose(low[:3], -_EXPECTED_WORKSPACE_RADIUS, atol=1e-6)
        np.testing.assert_allclose(high[:3], _EXPECTED_WORKSPACE_RADIUS, atol=1e-6)
        np.testing.assert_allclose(low[3:6], -np.pi, atol=1e-6)
        np.testing.assert_allclose(high[3:6], np.pi, atol=1e-6)
        assert low[-1] == pytest.approx(env._target_low[-1], abs=1e-6)
        assert high[-1] == pytest.approx(env._target_high[-1], abs=1e-6)


def test_ee_pose_gripper_bounds_match_pd_joint_pos():
    """The absolute gripper element carries pd_joint_pos's units and bounds."""
    with _touch_env("pd_ee_pose") as ee_env, _touch_env("pd_joint_pos") as joint_env:
        assert ee_env.action_space.low[-1] == joint_env.action_space.low[-1]
        assert ee_env.action_space.high[-1] == joint_env.action_space.high[-1]


@pytest.mark.parametrize("sign", [1.0, -1.0])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_ee_delta_pose_tracks_a_one_centimetre_axis_step(axis, sign):
    """A commanded 1 cm world-axis step lands the TCP within 1 mm of the target.

    Asserted on the IK solution rather than the settled pose: the arm is driven
    by PD position actuators, so a single control step only starts moving toward
    the commanded joint targets. The joint targets ``step()`` actually wrote are
    read back out of ``data.ctrl`` and evaluated through forward kinematics, which
    measures the solver rather than the tracking lag of the actuators.
    """
    step_m = 0.01
    with _touch_env("pd_ee_delta_pose", init_pose=_WORKING_POSE) as env:
        env.reset(seed=0)
        tcp_before = env._get_tcp_pose()[:3].copy()

        action = np.zeros(EE_ACTION_DIM, dtype=np.float32)
        action[axis] = sign * step_m / _EXPECTED_EE_DELTA_SCALE[axis]
        env.step(action)

        arm_targets = env.data.ctrl[env._actuator_ids][:-1].copy()
        expected = tcp_before.copy()
        expected[axis] += sign * step_m
        error_m = float(np.linalg.norm(_tcp_at(env, arm_targets) - expected))
        assert error_m < 1e-3, f"TCP tracking error {error_m * 1000:.4f} mm exceeds 1 mm"


def test_ee_delta_pose_zero_action_holds_tcp():
    """A zero delta re-commands the measured configuration and holds the TCP."""
    with _touch_env("pd_ee_delta_pose") as env:
        env.reset(seed=0)
        zero = np.zeros(EE_ACTION_DIM, dtype=np.float32)
        np.testing.assert_allclose(env._action_to_ctrl(zero), env._get_current_qpos(), atol=1e-3)

        tcp_before = env._get_tcp_pose()[:3].copy()
        for _ in range(5):
            env.step(zero)
        drift_m = float(np.linalg.norm(env._get_tcp_pose()[:3] - tcp_before))
        assert drift_m < 2e-3, f"TCP drifted {drift_m * 1000:.4f} mm under a zero action"


def test_ee_pose_current_pose_is_a_fixed_point():
    """Commanding the pose the TCP already holds leaves the joint targets put."""
    with _touch_env("pd_ee_pose") as env:
        env.reset(seed=0)
        tcp = env._get_tcp_pose()
        qpos = env._get_current_qpos()
        action = np.concatenate([tcp[:3], quat_to_rotvec(tcp[3:]), qpos[-1:]])
        np.testing.assert_allclose(env._action_to_ctrl(action), qpos, atol=2e-3)


def test_ee_pose_out_of_reach_target_clamps_instead_of_raising():
    """An unreachable absolute target resolves to the closest achievable pose."""
    with _touch_env("pd_ee_pose") as env:
        env.reset(seed=0)
        # Clipped by the action space to the workspace corner, still 0.95 m from
        # the base against a measured maximum reach of 0.5457 m.
        action = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        ctrl = env._action_to_ctrl(action)
        assert np.all(np.isfinite(ctrl))
        assert np.all(ctrl >= env._target_low - 1e-9)
        assert np.all(ctrl <= env._target_high + 1e-9)


def test_ee_pose_gripper_target_passes_through():
    """The absolute gripper element lands verbatim in the actuator target."""
    with _touch_env("pd_ee_pose") as env:
        env.reset(seed=0)
        tcp = env._get_tcp_pose()
        gripper_rad = 0.5
        action = np.concatenate([tcp[:3], quat_to_rotvec(tcp[3:]), [gripper_rad]])
        assert env._action_to_ctrl(action)[-1] == pytest.approx(gripper_rad, abs=1e-9)


def test_ee_delta_pose_gripper_moves_by_the_delta_scale():
    """A +1 gripper element opens the gripper by EE_DELTA_ACTION_SCALE[-1]."""
    assert EE_DELTA_ACTION_SCALE[-1] == _EXPECTED_EE_DELTA_SCALE[-1]
    with _touch_env("pd_ee_delta_pose") as env:
        env.reset(seed=0)
        before = env._get_current_qpos()[-1]
        action = np.zeros(EE_ACTION_DIM, dtype=np.float32)
        action[-1] = 1.0
        expected = min(before + _EXPECTED_EE_DELTA_SCALE[-1], env._target_high[-1])
        assert env._action_to_ctrl(action)[-1] == pytest.approx(expected, abs=1e-6)


def test_joint_pos_mode_is_unaffected_by_ee_support():
    """Adding the EE branches left the joint-space control path alone."""
    with _touch_env("pd_joint_pos") as env:
        env.reset(seed=0)
        assert env.action_space.shape == (len(SO101_JOINT_NAMES),)
        np.testing.assert_allclose(env.action_space.low, env._target_low, atol=1e-6)

        action = np.clip(env._get_current_qpos() + 0.01, env._target_low, env._target_high)
        env.step(action.astype(np.float32))
        np.testing.assert_allclose(env.data.ctrl[env._actuator_ids], action, atol=1e-6)


def test_ee_orientation_weight_is_a_live_knob():
    """``RobotConfig.ee_orientation_weight`` reaches the solve, not just the docs.

    A pure rotation command is the discriminating case: the weight scales both
    sides of the damped least-squares solve, so raising it buys tool rotation at
    the cost of position tracking. Asserted on the realized TCP rotation rather
    than on the joint targets, because the joint targets could differ for
    reasons that never reach the tool frame.
    """
    action = np.zeros(EE_ACTION_DIM, dtype=np.float32)
    action[5] = 1.0

    realized = {}
    for weight in (0.01, 0.5):
        robot = RobotConfig(init_pose=_WORKING_POSE, ee_orientation_weight=weight)
        with _touch_env("pd_ee_delta_pose", robot=robot) as env:
            env.reset(seed=0)
            before = env._get_tcp_pose()[3:].copy()
            arm_targets = env._action_to_ctrl(action)[:-1]
            data = mujoco.MjData(env.model)
            data.qpos[:] = env.data.qpos
            data.qpos[env._arm_qpos_addrs] = arm_targets
            mujoco.mj_kinematics(env.model, data)
            after = np.zeros(4)
            mujoco.mju_mat2Quat(after, data.site_xmat[env._tcp_site_id])
            realized[weight] = float(np.linalg.norm(orientation_error(before, after)))

    assert realized[0.5] > 3.0 * realized[0.01], realized


def test_ee_delta_action_scale_is_a_live_knob():
    """``RobotConfig.ee_delta_action_scale`` sets the physical step of a +/-1 action."""
    scale = (0.04, 0.04, 0.04, 0.1, 0.1, 0.1, 0.05)
    robot = RobotConfig(init_pose=_WORKING_POSE, ee_delta_action_scale=scale)
    with _touch_env("pd_ee_delta_pose", robot=robot) as env:
        env.reset(seed=0)
        tcp_before = env._get_tcp_pose()[:3].copy()
        gripper_before = env._get_current_qpos()[-1]

        action = np.zeros(EE_ACTION_DIM, dtype=np.float32)
        action[0] = 1.0
        action[6] = 1.0
        ctrl = env._action_to_ctrl(action)

        expected = tcp_before + np.array([scale[0], 0.0, 0.0])
        error_m = float(np.linalg.norm(_tcp_at(env, ctrl[:-1]) - expected))
        assert error_m < 1e-3, f"TCP tracking error {error_m * 1000:.4f} mm exceeds 1 mm"
        assert ctrl[-1] == pytest.approx(
            min(gripper_before + scale[6], env._target_high[-1]), abs=1e-6
        )
