"""Tests for ManiSkill reset options parity with MuJoCo teleop."""

from __future__ import annotations

import numpy as np
import pytest

import so101_nexus_maniskill  # noqa: F401 - registers gym envs

BASE_KWARGS = {"obs_mode": "state", "num_envs": 1, "render_mode": None}


def _current_qpos(env) -> np.ndarray:
    qpos = env.unwrapped.agent.robot.get_qpos()
    return qpos.cpu().numpy()[0]


def test_reset_uses_init_qpos_from_options() -> None:
    """ManiSkill reset should match MuJoCo and honor options['init_qpos']."""
    import gymnasium as gym

    env = gym.make("ManiSkillPickLiftSO101-v1", **BASE_KWARGS)
    try:
        custom_qpos = np.array([0.1, -0.5, 0.8, 0.2, 0.0, 0.05], dtype=np.float32)
        env.reset(options={"init_qpos": custom_qpos})

        np.testing.assert_allclose(_current_qpos(env), custom_qpos, atol=1e-6)
    finally:
        env.close()


def test_reset_rejects_invalid_init_qpos_shape() -> None:
    """Invalid init_qpos should fail before applying a malformed robot reset."""
    import gymnasium as gym

    env = gym.make("ManiSkillPickLiftSO101-v1", **BASE_KWARGS)
    try:
        with pytest.raises(ValueError, match="init_qpos shape"):
            env.reset(options={"init_qpos": np.zeros(5, dtype=np.float32)})
    finally:
        env.close()
