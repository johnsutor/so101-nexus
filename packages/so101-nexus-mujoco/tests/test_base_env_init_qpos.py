"""Test that base_env.reset() respects options['init_qpos']."""
from __future__ import annotations

import numpy as np


def test_reset_uses_init_qpos_from_options():
    """Env reset should set joints to the provided init_qpos, not REST_QPOS."""
    import so101_nexus_mujoco  # noqa: F401 — registers envs
    import gymnasium as gym

    env = gym.make("MuJoCoPickCubeGoal-v1", camera_mode="wrist", render_mode="rgb_array")
    try:
        custom_qpos = np.array([0.1, -0.5, 0.8, 0.2, 0.0, 0.05], dtype=np.float64)
        obs, _ = env.reset(options={"init_qpos": custom_qpos})
        actual_qpos = env.unwrapped._get_current_qpos()
        np.testing.assert_allclose(actual_qpos, custom_qpos, atol=1e-6)
    finally:
        env.close()


def test_reset_without_init_qpos_uses_rest_pose():
    """Without init_qpos, reset should use the default REST_QPOS (within noise)."""
    import so101_nexus_mujoco  # noqa: F401
    import gymnasium as gym
    from so101_nexus_core.config import EnvironmentConfig

    rest = np.array(EnvironmentConfig().robot.rest_qpos_rad, dtype=np.float64)
    env = gym.make("MuJoCoPickCubeGoal-v1", camera_mode="wrist", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=0)
        actual_qpos = env.unwrapped._get_current_qpos()
        # Should be close to rest pose (within default noise 0.02 rad)
        np.testing.assert_allclose(actual_qpos, rest, atol=0.025)
    finally:
        env.close()
