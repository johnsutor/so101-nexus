"""Tests for ManiSkill LookAt environments."""

import gymnasium as gym
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import LookAtConfig
from so101_nexus_core.observations import (
    EndEffectorPose,
    GazeDirection,
    JointPositions,
)

BASE_KWARGS = dict(obs_mode="state", num_envs=1, render_mode=None)

LOOK_AT_ENV_IDS = [
    ("ManiSkillLookAtSO100-v1", "so100"),
    ("ManiSkillLookAtSO101-v1", "so101"),
]


@pytest.fixture(scope="module")
def look_at_so100_env():
    env = gym.make("ManiSkillLookAtSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def look_at_so101_env():
    env = gym.make("ManiSkillLookAtSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillLookAtSO100-v1": "look_at_so100_env",
        "ManiSkillLookAtSO101-v1": "look_at_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestEnvCreation:
    @pytest.mark.parametrize("env_id,robot", LOOK_AT_ENV_IDS)
    def test_env_creates(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert isinstance(env, gym.Env)

    @pytest.mark.parametrize("env_id,robot", LOOK_AT_ENV_IDS)
    def test_env_reset_and_step(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        assert isinstance(obs, torch.Tensor)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward is not None

    @pytest.mark.parametrize("env_id,robot", LOOK_AT_ENV_IDS)
    def test_action_space_shape(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert env.action_space.shape == (6,)


class TestEpisodeLogic:
    EVALUATE_KEYS = {"orientation_error", "success"}

    @pytest.mark.parametrize("env_id,robot", LOOK_AT_ENV_IDS)
    def test_evaluate_keys(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        info = env.unwrapped.evaluate()
        assert self.EVALUATE_KEYS <= set(info.keys())

    @pytest.mark.parametrize("env_id,robot", LOOK_AT_ENV_IDS)
    def test_reward_range(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert (reward >= 0).all()
        assert (reward <= 1).all()


class TestTaskDescription:
    def test_task_description_nonempty(self, look_at_so100_env):
        assert look_at_so100_env.unwrapped.task_description


class TestCustomObservations:
    def test_gaze_direction_obs(self):
        """GazeDirection component produces a normalized 3-dim obs_extra entry."""
        config = LookAtConfig(observations=[JointPositions(), GazeDirection()])
        env = gym.make("ManiSkillLookAtSO100-v1", config=config, **BASE_KWARGS)
        try:
            obs, info = env.reset()
            _, _, _, _, info = env.step(env.action_space.sample())
            extra = env.unwrapped._get_obs_extra(info)
            assert "gaze_direction" in extra
            assert extra["gaze_direction"].shape[-1] == 3
        finally:
            env.close()

    def test_end_effector_pose_obs(self):
        """EndEffectorPose component produces tcp_pose in obs_extra."""
        config = LookAtConfig(observations=[JointPositions(), EndEffectorPose(), GazeDirection()])
        env = gym.make("ManiSkillLookAtSO100-v1", config=config, **BASE_KWARGS)
        try:
            obs, info = env.reset()
            _, _, _, _, info = env.step(env.action_space.sample())
            extra = env.unwrapped._get_obs_extra(info)
            assert "tcp_pose" in extra
            assert "gaze_direction" in extra
        finally:
            env.close()
