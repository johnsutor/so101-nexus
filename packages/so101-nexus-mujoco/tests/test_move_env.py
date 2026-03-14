"""Tests for MuJoCoMove-v1."""

import gymnasium as gym
import pytest

import so101_nexus_mujoco  # noqa: F401


@pytest.fixture(scope="module")
def move_env():
    env = gym.make("MuJoCoMove-v1")
    yield env
    env.close()


class TestMoveEnv:
    def test_make(self, move_env):
        assert move_env is not None

    def test_obs_shape(self, move_env):
        obs, _ = move_env.reset()
        assert obs.shape == (10,)

    def test_step_five_tuple(self, move_env):
        move_env.reset()
        assert len(move_env.step(move_env.action_space.sample())) == 5

    def test_reward_range(self, move_env):
        move_env.reset()
        _, r, _, _, _ = move_env.step(move_env.action_space.sample())
        assert -0.1 <= r <= 1.0

    def test_info_keys(self, move_env):
        move_env.reset()
        _, _, _, _, info = move_env.step(move_env.action_space.sample())
        assert "tcp_to_target_dist" in info and "success" in info

    def test_task_description(self, move_env):
        assert isinstance(move_env.unwrapped.task_description, str)
