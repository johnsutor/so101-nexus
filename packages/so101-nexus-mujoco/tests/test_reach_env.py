"""Tests for MuJoCoReach-v1."""

import gymnasium as gym
import pytest

import so101_nexus_mujoco  # noqa: F401


@pytest.fixture(scope="module")
def reach_env():
    env = gym.make("MuJoCoReach-v1")
    yield env
    env.close()


class TestReachEnv:
    def test_make(self, reach_env):
        assert reach_env is not None

    def test_obs_shape(self, reach_env):
        obs, _ = reach_env.reset()
        assert obs.shape == (6,)

    def test_step_five_tuple(self, reach_env):
        reach_env.reset()
        assert len(reach_env.step(reach_env.action_space.sample())) == 5

    def test_reward_range(self, reach_env):
        reach_env.reset()
        _, r, _, _, _ = reach_env.step(reach_env.action_space.sample())
        assert -0.1 <= r <= 1.0

    def test_info_keys(self, reach_env):
        reach_env.reset()
        _, _, _, _, info = reach_env.step(reach_env.action_space.sample())
        assert "tcp_to_target_dist" in info
        assert "success" in info

    def test_task_description(self, reach_env):
        assert isinstance(reach_env.unwrapped.task_description, str)

    def test_custom_observations(self):
        from so101_nexus_core.config import ReachConfig
        from so101_nexus_core.observations import EndEffectorPose, JointPositions

        config = ReachConfig(observations=[JointPositions(), EndEffectorPose()])
        env = gym.make("MuJoCoReach-v1", config=config)
        obs, _ = env.reset()
        assert obs.shape == (13,)  # 6 + 7
        env.close()
