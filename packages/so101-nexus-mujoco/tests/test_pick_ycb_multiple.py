import os

import gymnasium as gym
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus_mujoco  # noqa: F401, E402
from so101_nexus_core.config import YCB_OBJECTS, PickYCBMultipleConfig
from so101_nexus_mujoco.pick_ycb_multiple import PickYCBMultipleEnv

_CFG = PickYCBMultipleConfig()


@pytest.fixture(scope="module")
def goal_env():
    env = gym.make("MuJoCoPickYCBMultipleGoal-v1")
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_env():
    env = gym.make("MuJoCoPickYCBMultipleLift-v1")
    yield env
    env.close()


class TestConstructionValidation:
    def test_invalid_model_id(self):
        with pytest.raises(ValueError, match="model_id"):
            PickYCBMultipleEnv(model_id="invalid_object")

    def test_invalid_num_distractors(self):
        with pytest.raises(ValueError, match="num_distractors"):
            PickYCBMultipleConfig(num_distractors=0)


class TestEnvCreation:
    def test_goal_env_creates(self, goal_env):
        assert isinstance(goal_env, gym.Env)

    def test_lift_env_creates(self, lift_env):
        assert isinstance(lift_env, gym.Env)

    def test_goal_env_reset(self, goal_env):
        obs, info = goal_env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_lift_env_reset(self, lift_env):
        obs, info = lift_env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_goal_env_step(self, goal_env):
        goal_env.reset()
        action = goal_env.action_space.sample()
        obs, reward, terminated, truncated, info = goal_env.step(action)
        assert isinstance(obs, np.ndarray)
        assert reward is not None
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_lift_env_step(self, lift_env):
        lift_env.reset()
        action = lift_env.action_space.sample()
        obs, reward, terminated, truncated, info = lift_env.step(action)
        assert isinstance(obs, np.ndarray)
        assert reward is not None

    def test_observation_space(self, goal_env):
        obs, _ = goal_env.reset()
        assert goal_env.observation_space.contains(obs)
        assert obs.shape == (24,)

    def test_action_space_shape(self, goal_env):
        assert goal_env.action_space.shape == (6,)


class TestEpisodeLogic:
    EXPECTED_INFO_KEYS = {
        "obj_to_goal_dist",
        "is_obj_placed",
        "is_grasped",
        "is_robot_static",
        "lift_height",
        "success",
        "tcp_to_obj_dist",
    }

    def test_info_keys_goal(self, goal_env):
        _, info = goal_env.reset()
        assert set(info.keys()) == self.EXPECTED_INFO_KEYS

    def test_info_keys_lift(self, lift_env):
        _, info = lift_env.reset()
        assert set(info.keys()) == self.EXPECTED_INFO_KEYS

    def test_target_spawns_in_bounds(self, goal_env):
        cx, cy = _CFG.spawn_center
        hs = _CFG.spawn_half_size
        for _ in range(5):
            goal_env.reset()
            obj_pose = goal_env.unwrapped._get_obj_pose()
            assert cx - hs <= obj_pose[0] <= cx + hs
            assert cy - hs <= obj_pose[1] <= cy + hs

    def test_reward_range_goal(self, goal_env):
        goal_env.reset()
        action = goal_env.action_space.sample()
        _, reward, _, _, _ = goal_env.step(action)
        assert 0.0 <= reward <= 1.0

    def test_reward_range_lift(self, lift_env):
        lift_env.reset()
        action = lift_env.action_space.sample()
        _, reward, _, _, _ = lift_env.step(action)
        assert 0.0 <= reward <= 1.0


class TestMultipleObjects:
    def test_correct_number_of_distractors(self):
        env = PickYCBMultipleEnv()
        assert len(env._distractor_qpos_addrs) == _CFG.num_distractors
        env.close()

    def test_distractor_models_differ_from_target(self):
        env = PickYCBMultipleEnv(model_id="058_golf_ball")
        for mid in env.distractor_model_ids:
            assert mid != "058_golf_ball"
            assert mid in YCB_OBJECTS
        env.close()

    def test_per_ycb_multiple_goal_creates(self):
        env = gym.make("MuJoCoPickGolfBallMultipleGoal-v1")
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()

    def test_per_ycb_multiple_lift_creates(self):
        env = gym.make("MuJoCoPickGolfBallMultipleLift-v1")
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()


class TestTaskDescription:
    def test_task_description_exists(self):
        env = PickYCBMultipleEnv(model_id="058_golf_ball")
        assert isinstance(env.task_description, str)
        assert "golf ball" in env.task_description
        env.close()

    def test_task_description_starts_with_capital(self):
        env = PickYCBMultipleEnv()
        assert env.task_description[0].isupper()
        env.close()
