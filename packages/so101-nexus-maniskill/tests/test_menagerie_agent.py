"""Tests for the menagerie-backed ManiSkill SO101 agent."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401 — registers envs
from so101_nexus_core.config import ReachConfig, RobotConfig
from so101_nexus_maniskill import menagerie_constants as mc


def test_agent_loads_menagerie_link_names(so101_reach_env):
    robot = so101_reach_env.agent.robot
    names = {link.name for link in robot.links}
    assert {"gripper", "moving_jaw_so101_v1", "camera_mount"} <= names
    assert "gripper_link" not in names
    assert "gripper_frame_link" not in names


def test_agent_active_joint_order(so101_reach_env):
    robot = so101_reach_env.agent.robot
    assert [j.name for j in robot.active_joints] == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]


def test_rest_keyframe_derives_from_core(so101_reach_env):
    expected = np.radians(RobotConfig().rest_qpos_deg)
    np.testing.assert_allclose(so101_reach_env.agent.keyframes["rest"].qpos, expected, atol=1e-9)


def test_tcp_pose_composes_offset_on_gripper_link(so101_reach_env):
    env = so101_reach_env
    env.reset(seed=0)
    gripper = env.agent.robot.links_map["gripper"]
    from mani_skill.utils.structs.pose import Pose

    offset = Pose.create_from_pq(
        p=torch.tensor(mc.TCP_OFFSET_POS, dtype=torch.float32),
        q=torch.tensor(mc.TCP_OFFSET_QUAT, dtype=torch.float32),
    )
    expected = gripper.pose * offset
    np.testing.assert_allclose(
        env.agent.tcp_pose.raw_pose.cpu().numpy(), expected.raw_pose.cpu().numpy(), atol=1e-5
    )


def test_link_masses_applied_on_every_subscene(so101_reach_env_vec):
    robot = so101_reach_env_vec.agent.robot
    for name, inertial in mc.LINK_INERTIALS.items():
        link = robot.links_map[name]
        for obj in link._objs:
            assert obj.mass == pytest.approx(inertial.mass, rel=1e-6), name


def test_link_inertia_and_com_applied(so101_reach_env_vec):
    from transforms3d.quaternions import quat2mat

    robot = so101_reach_env_vec.agent.robot
    for name, inertial in mc.LINK_INERTIALS.items():
        link = robot.links_map[name]
        for obj in link._objs:
            np.testing.assert_allclose(
                sorted(np.asarray(obj.inertia)), sorted(inertial.principal_moments), atol=1e-10
            )
            np.testing.assert_allclose(
                np.asarray(obj.cmass_local_pose.p), inertial.com_pos, atol=1e-9
            )
            # The principal-axis orientation is load-bearing: compare the full
            # applied inertia tensor (moments rotated by the applied cmass quat)
            # to the intended tensor, so a wrong quat cannot pass on sorted
            # moments alone. (Comparing quats directly is sign/axis ambiguous.)
            applied_q = np.asarray(obj.cmass_local_pose.q)  # (w, x, y, z)
            rot_applied = quat2mat(applied_q)
            tensor_applied = rot_applied @ np.diag(np.asarray(obj.inertia)) @ rot_applied.T
            rot_intended = quat2mat(inertial.principal_quat)
            tensor_intended = rot_intended @ np.diag(inertial.principal_moments) @ rot_intended.T
            np.testing.assert_allclose(tensor_applied, tensor_intended, atol=1e-10)


def test_gripper_friction_applied_on_every_collision_shape(so101_reach_env_vec):
    robot = so101_reach_env_vec.agent.robot
    for name in mc.GRIPPER_FRICTION_LINKS:
        link = robot.links_map[name]
        for obj in link._objs:
            shapes = obj.get_collision_shapes()
            assert shapes, f"{name} has no collision shapes"
            for shape in shapes:
                mat = shape.get_physical_material()
                assert mat.get_static_friction() == pytest.approx(mc.GRIPPER_STATIC_FRICTION)
                assert mat.get_dynamic_friction() == pytest.approx(mc.GRIPPER_DYNAMIC_FRICTION)
                assert shape.get_patch_radius() == pytest.approx(mc.GRIPPER_PATCH_RADIUS)
                assert shape.get_min_patch_radius() == pytest.approx(mc.GRIPPER_MIN_PATCH_RADIUS)


def test_joint_armature_applied(so101_reach_env_vec):
    robot = so101_reach_env_vec.agent.robot
    for joint in robot.active_joints:
        for jobj in joint._objs:
            np.testing.assert_allclose(np.asarray(jobj.armature), [mc.JOINT_ARMATURE], atol=1e-9)


def test_qlimits_equal_menagerie_joint_limits(so101_reach_env_vec):
    robot = so101_reach_env_vec.agent.robot
    qlimits = robot.get_qlimits()[0].cpu().numpy()  # (dof, 2)
    for i, joint in enumerate(robot.active_joints):
        lo, hi = mc.JOINT_LIMITS[joint.name]
        np.testing.assert_allclose(qlimits[i], [lo, hi], atol=1e-5)


def test_pd_joint_pos_action_space_equals_joint_limits():
    """The absolute (non-delta) controller's action space must equal the
    menagerie joint limits (lower/upper=None resolves to compiled limits)."""
    env = gym.make(
        "ManiSkillReachSO101-v1",
        config=ReachConfig(),
        num_envs=1,
        obs_mode="state",
        render_mode=None,
        control_mode="pd_joint_pos",
    )
    try:
        space = env.unwrapped.single_action_space
        joints = [j.name for j in env.unwrapped.agent.robot.active_joints]
        lows = np.array([mc.JOINT_LIMITS[n][0] for n in joints], dtype=np.float32)
        highs = np.array([mc.JOINT_LIMITS[n][1] for n in joints], dtype=np.float32)
        np.testing.assert_allclose(space.low, lows, atol=1e-4)
        np.testing.assert_allclose(space.high, highs, atol=1e-4)
    finally:
        env.close()


def test_delta_action_space_is_normalized():
    """Delta modes keep ManiSkill's default normalized [-1, 1] action space
    (unchanged from the pre-menagerie behavior). The +/-0.05 arm / +/-0.2
    gripper delta scale is the controller's internal range, not the action
    space; normalize_action stays at its default (True) for delta modes."""
    for mode in ("pd_joint_delta_pos", "pd_joint_target_delta_pos"):
        env = gym.make(
            "ManiSkillReachSO101-v1",
            config=ReachConfig(),
            num_envs=1,
            obs_mode="state",
            render_mode=None,
            control_mode=mode,
        )
        try:
            space = env.unwrapped.single_action_space
            np.testing.assert_allclose(space.low, [-1.0] * 6, atol=1e-6)
            np.testing.assert_allclose(space.high, [1.0] * 6, atol=1e-6)
        finally:
            env.close()
