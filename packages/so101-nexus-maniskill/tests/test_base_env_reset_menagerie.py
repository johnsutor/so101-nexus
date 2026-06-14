"""Reset regressions for the menagerie-backed ManiSkill SO101 backend.

Mirrors the MuJoCo backend's test_base_env_init_qpos coverage.
"""

from __future__ import annotations

import logging

import gymnasium as gym
import numpy as np
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401 — registers envs
from so101_nexus_core.config import PickConfig, ReachConfig, RobotConfig

_BASE = {"obs_mode": "state", "num_envs": 1, "render_mode": None}


def _qlimits(env):
    return env.agent.robot.get_qlimits()[0].cpu().numpy()  # (dof, 2)


def test_default_rest_clamps_gripper_to_menagerie_lower_limit():
    env = gym.make(
        "ManiSkillPickLiftSO101-v1",
        config=PickConfig(reset_settle_frames=0),
        robot_init_qpos_noise=0.0,
        **_BASE,
    ).unwrapped
    try:
        env.reset(seed=0)
        qpos = env.agent.robot.get_qpos()[0].cpu().numpy()
        lo = _qlimits(env)[:, 0]
        assert qpos[5] == pytest.approx(lo[5], abs=1e-4)
        assert np.all(qpos >= lo - 1e-6)
        assert np.all(qpos <= _qlimits(env)[:, 1] + 1e-6)
    finally:
        env.close()


def test_custom_rest_qpos_deg_is_honored():
    robot = RobotConfig(rest_qpos_deg=(10.0, -80.0, 80.0, 30.0, 0.0, 0.0))
    env = gym.make(
        "ManiSkillPickLiftSO101-v1",
        config=PickConfig(robot=robot, reset_settle_frames=0),
        robot_init_qpos_noise=0.0,
        **_BASE,
    ).unwrapped
    try:
        env.reset(seed=0)
        qpos = env.agent.robot.get_qpos()[0].cpu().numpy()
        expected = np.clip(
            np.radians(robot.rest_qpos_deg), _qlimits(env)[:, 0], _qlimits(env)[:, 1]
        )
        np.testing.assert_allclose(qpos, expected, atol=1e-4)
    finally:
        env.close()


def test_reset_noise_cannot_exceed_joint_limits():
    robot = RobotConfig(rest_qpos_deg=(0.0, -90.0, 90.0, 0.0, 157.0, -10.0))
    env = gym.make(
        "ManiSkillPickLiftSO101-v1",
        config=PickConfig(robot=robot, reset_settle_frames=0),
        robot_init_qpos_noise=0.2,
        **_BASE,
    ).unwrapped
    try:
        lo, hi = _qlimits(env)[:, 0], _qlimits(env)[:, 1]
        for seed in range(20):
            env.reset(seed=seed)
            qpos = env.agent.robot.get_qpos()[0].cpu().numpy()
            assert np.all(qpos >= lo - 1e-6)
            assert np.all(qpos <= hi + 1e-6)
    finally:
        env.close()


def test_out_of_range_init_qpos_clamps_and_warns_once(caplog):
    env = gym.make(
        "ManiSkillPickLiftSO101-v1",
        config=PickConfig(reset_settle_frames=0),
        **_BASE,
    ).unwrapped
    try:
        huge = np.full(6, 1e9, dtype=np.float32)
        with caplog.at_level(logging.WARNING, logger="so101_nexus_maniskill.base_env"):
            env.reset(options={"init_qpos": huge})
            env.reset(options={"init_qpos": huge})
        qpos = env.agent.robot.get_qpos()[0].cpu().numpy()
        np.testing.assert_allclose(qpos, _qlimits(env)[:, 1], atol=1e-4)
        assert sum("init_qpos" in r.message for r in caplog.records) == 1
    finally:
        env.close()


def test_settle_window_holds_clamped_reset_target_in_target_delta_mode():
    env = gym.make(
        "ManiSkillPickLiftSO101-v1",
        config=PickConfig(reset_settle_frames=3),
        control_mode="pd_joint_target_delta_pos",
        robot_init_qpos_noise=0.0,
        **_BASE,
    ).unwrapped
    try:
        init = np.array([0.2, -0.5, 0.5, 0.2, 0.0, 0.5], dtype=np.float32)
        env.reset(seed=0, options={"init_qpos": init})
        after_settle = env.agent.robot.get_qpos()[0].cpu().numpy()
        np.testing.assert_allclose(after_settle, init, atol=0.05)
        env.step(np.zeros(6, dtype=np.float32))
        after_zero = env.agent.robot.get_qpos()[0].cpu().numpy()
        np.testing.assert_allclose(after_zero, init, atol=0.05)
    finally:
        env.close()


def test_reset_writes_drive_targets_to_clamped_qpos():
    """controller.reset() alone does not write PhysX drive targets; _reset_robot
    must set them explicitly. Direct assertion on get_drive_targets so the fix
    is proven without relying on settle-physics tolerance.
    """
    env = gym.make(
        "ManiSkillPickLiftSO101-v1",
        config=PickConfig(reset_settle_frames=0),
        control_mode="pd_joint_target_delta_pos",
        robot_init_qpos_noise=0.0,
        **_BASE,
    ).unwrapped
    try:
        init = np.array([0.2, -0.5, 0.5, 0.2, 0.0, 0.5], dtype=np.float32)
        env.reset(seed=0, options={"init_qpos": init})
        targets = env.agent.robot.get_drive_targets()[0].cpu().numpy()
        np.testing.assert_allclose(targets, init, atol=1e-4)
    finally:
        env.close()


def test_partial_reset_clamps_only_reset_env():
    """A partial reset (env_idx=[0]) with out-of-range init_qpos clamps env 0
    and leaves env 1 untouched. Guards the env_idx slicing in _clamp_to_qlimits.
    """
    try:
        env = gym.make(
            "ManiSkillReachSO101-v1",
            config=ReachConfig(reset_settle_frames=0),
            obs_mode="state",
            num_envs=2,
            render_mode=None,
        ).unwrapped
    except Exception as exc:  # pragma: no cover - runtime/GPU availability
        pytest.skip(f"ManiSkill vectorized runtime unavailable: {exc}")
    try:
        env.reset(seed=0)
        qpos_before = env.agent.robot.get_qpos().clone()
        env_idx = torch.tensor([0], device=env.device)
        huge = torch.full((1, 6), 1e9, dtype=torch.float32, device=env.device)
        env.reset(options={"env_idx": env_idx, "init_qpos": huge})
        qpos_after = env.agent.robot.get_qpos()
        hi = env.agent.robot.get_qlimits()[0, :, 1].cpu().numpy()
        np.testing.assert_allclose(qpos_after[0].cpu().numpy(), hi, atol=1e-4)
        torch.testing.assert_close(qpos_after[1], qpos_before[1])
    finally:
        env.close()
