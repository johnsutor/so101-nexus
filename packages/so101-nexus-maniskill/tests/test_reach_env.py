"""Tests for ManiSkill Reach environments."""

import gymnasium as gym
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401

BASE_KWARGS = dict(obs_mode="state", num_envs=1, render_mode=None)

REACH_ENV_IDS = [
    ("ManiSkillReachSO100-v1", "so100"),
    ("ManiSkillReachSO101-v1", "so101"),
]


@pytest.fixture(scope="module")
def reach_so100_env():
    env = gym.make("ManiSkillReachSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def reach_so101_env():
    env = gym.make("ManiSkillReachSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillReachSO100-v1": "reach_so100_env",
        "ManiSkillReachSO101-v1": "reach_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestEnvCreation:
    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_env_creates(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert isinstance(env, gym.Env)

    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_env_reset_and_step(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        assert isinstance(obs, torch.Tensor)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward is not None

    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_action_space_shape(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert env.action_space.shape == (6,)


class TestEpisodeLogic:
    EVALUATE_KEYS = {"tcp_to_target_dist", "success"}

    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_evaluate_keys(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        info = env.unwrapped.evaluate()
        assert self.EVALUATE_KEYS <= set(info.keys())

    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_reward_range(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert (reward >= 0).all()
        assert (reward <= 1).all()


class TestTaskDescription:
    def test_task_description_nonempty(self, reach_so100_env):
        assert reach_so100_env.unwrapped.task_description
