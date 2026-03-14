"""Tests for MuJoCoLookAt-v1."""

import gymnasium as gym
import pytest

import so101_nexus_mujoco  # noqa: F401


@pytest.fixture(scope="module")
def look_at_env():
    env = gym.make("MuJoCoLookAt-v1")
    yield env
    env.close()


class TestLookAtEnv:
    def test_make(self, look_at_env):
        assert look_at_env is not None

    def test_obs_shape(self, look_at_env):
        obs, _ = look_at_env.reset()
        assert obs.shape == (10,)

    def test_step_five_tuple(self, look_at_env):
        look_at_env.reset()
        assert len(look_at_env.step(look_at_env.action_space.sample())) == 5

    def test_reward_range(self, look_at_env):
        look_at_env.reset()
        _, r, _, _, _ = look_at_env.step(look_at_env.action_space.sample())
        assert -0.1 <= r <= 1.0

    def test_info_keys(self, look_at_env):
        look_at_env.reset()
        _, _, _, _, info = look_at_env.step(look_at_env.action_space.sample())
        assert "orientation_error" in info and "success" in info

    def test_task_description(self, look_at_env):
        assert isinstance(look_at_env.unwrapped.task_description, str)
