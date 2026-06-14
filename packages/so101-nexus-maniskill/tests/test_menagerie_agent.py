"""Tests for the menagerie-backed ManiSkill SO101 agent."""

from __future__ import annotations

import numpy as np
import torch

import so101_nexus_maniskill  # noqa: F401 — registers envs
from so101_nexus_core.config import RobotConfig
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
