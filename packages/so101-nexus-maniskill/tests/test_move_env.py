"""Tests for ManiSkill Move environments."""

import gymnasium as gym
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import MoveConfig
from so101_nexus_core.observations import (
    EndEffectorPose,
    JointPositions,
    TargetOffset,
)

BASE_KWARGS = {"obs_mode": "state", "num_envs": 1, "render_mode": None}

MOVE_ENV_IDS = [
    ("ManiSkillMoveSO100-v1", "so100"),
    ("ManiSkillMoveSO101-v1", "so101"),
]


@pytest.fixture(scope="module")
def move_so100_env():
    env = gym.make("ManiSkillMoveSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def move_so101_env():
    env = gym.make("ManiSkillMoveSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillMoveSO100-v1": "move_so100_env",
        "ManiSkillMoveSO101-v1": "move_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestEnvCreation:
    @pytest.mark.parametrize("env_id,robot", MOVE_ENV_IDS)
    def test_env_creates(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert isinstance(env, gym.Env)

    @pytest.mark.parametrize("env_id,robot", MOVE_ENV_IDS)
    def test_env_reset_and_step(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        assert isinstance(obs, torch.Tensor)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward is not None

    @pytest.mark.parametrize("env_id,robot", MOVE_ENV_IDS)
    def test_action_space_shape(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert env.action_space.shape == (6,)


class TestEpisodeLogic:
    EVALUATE_KEYS = {"tcp_to_target_dist", "success"}

    @pytest.mark.parametrize("env_id,robot", MOVE_ENV_IDS)
    def test_evaluate_keys(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        info = env.unwrapped.evaluate()
        assert set(info.keys()) >= self.EVALUATE_KEYS

    @pytest.mark.parametrize("env_id,robot", MOVE_ENV_IDS)
    def test_reward_range(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert (reward >= 0).all()
        assert (reward <= 1).all()


class TestTaskDescription:
    def test_task_description_nonempty(self, move_so100_env):
        desc = move_so100_env.unwrapped.task_description
        assert desc
        assert "up" in desc


class TestCustomObservations:
    def test_target_offset_obs(self):
        """TargetOffset component produces a 3-dim obs_extra entry."""
        config = MoveConfig(observations=[JointPositions(), TargetOffset()])
        env = gym.make("ManiSkillMoveSO100-v1", config=config, **BASE_KWARGS)
        try:
            obs, info = env.reset()
            _, _, _, _, info = env.step(env.action_space.sample())
            extra = env.unwrapped._get_obs_extra(info)
            assert "target_offset" in extra
            assert extra["target_offset"].shape[-1] == 3
        finally:
            env.close()

    def test_end_effector_pose_obs(self):
        """EndEffectorPose component produces tcp_pose in obs_extra."""
        config = MoveConfig(observations=[JointPositions(), EndEffectorPose(), TargetOffset()])
        env = gym.make("ManiSkillMoveSO100-v1", config=config, **BASE_KWARGS)
        try:
            obs, info = env.reset()
            _, _, _, _, info = env.step(env.action_space.sample())
            extra = env.unwrapped._get_obs_extra(info)
            assert "tcp_pose" in extra
            assert "target_offset" in extra
        finally:
            env.close()
