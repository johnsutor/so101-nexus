"""Test that base_env.reset() respects options['init_qpos']."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_reset_uses_init_qpos_from_options():
    """Env reset should set joints to the provided init_qpos, not REST_QPOS."""
    import gymnasium as gym

    importlib.import_module("so101_nexus_mujoco")

    env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")
    try:
        custom_qpos = np.array([0.1, -0.5, 0.8, 0.2, 0.0, 0.05], dtype=np.float64)
        obs, _ = env.reset(options={"init_qpos": custom_qpos})
        actual_qpos = env.unwrapped._get_current_qpos()
        np.testing.assert_allclose(actual_qpos, custom_qpos, atol=1e-6)
    finally:
        env.close()


def test_reset_without_init_qpos_uses_rest_pose():
    """Without init_qpos, reset should use the default REST_QPOS (within noise)."""
    import gymnasium as gym

    importlib.import_module("so101_nexus_mujoco")
    from so101_nexus_core.config import EnvironmentConfig

    rest = np.array(EnvironmentConfig().robot.rest_qpos_rad, dtype=np.float64)
    env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=0)
        actual_qpos = env.unwrapped._get_current_qpos()
        np.testing.assert_allclose(actual_qpos, rest, atol=0.025)
    finally:
        env.close()


def test_reset_with_init_pose_rest():
    """When init_pose='rest', joints should be within REST_POSE ranges."""
    import gymnasium as gym

    importlib.import_module("so101_nexus_mujoco")
    from so101_nexus_core.config import PickConfig, RobotConfig

    robot = RobotConfig(init_pose="rest")
    config = PickConfig(robot=robot, robot_init_qpos_noise=0.0)
    env = gym.make("MuJoCoPickLift-v1", config=config, render_mode="rgb_array")
    try:
        env.reset(seed=42)
        qpos_deg = np.degrees(env.unwrapped._get_current_qpos())
        # shoulder_lift is fixed at -90
        assert qpos_deg[1] == pytest.approx(-90.0, abs=0.1)
        # Free joint: shoulder_pan within full range
        assert -110.0 <= qpos_deg[0] <= 110.0
    finally:
        env.close()


def test_reset_with_init_pose_extended():
    """When init_pose='extended', fixed joints should match EXTENDED_POSE."""
    import gymnasium as gym

    importlib.import_module("so101_nexus_mujoco")
    from so101_nexus_core.config import PickConfig, RobotConfig

    robot = RobotConfig(init_pose="extended")
    config = PickConfig(robot=robot, robot_init_qpos_noise=0.0)
    env = gym.make("MuJoCoPickLift-v1", config=config, render_mode="rgb_array")
    try:
        env.reset(seed=42)
        qpos_deg = np.degrees(env.unwrapped._get_current_qpos())
        # shoulder_lift fixed at -30 for extended
        assert qpos_deg[1] == pytest.approx(-30.0, abs=0.1)
        # elbow_flex fixed at 20 for extended
        assert qpos_deg[2] == pytest.approx(20.0, abs=0.1)
    finally:
        env.close()


def test_reset_init_pose_none_backward_compat():
    """When init_pose=None (default), behavior is the same as before."""
    import gymnasium as gym

    importlib.import_module("so101_nexus_mujoco")
    from so101_nexus_core.config import EnvironmentConfig

    rest = np.array(EnvironmentConfig().robot.rest_qpos_rad, dtype=np.float64)
    env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        actual_qpos = env.unwrapped._get_current_qpos()
        np.testing.assert_allclose(actual_qpos, rest, atol=0.025)
    finally:
        env.close()
