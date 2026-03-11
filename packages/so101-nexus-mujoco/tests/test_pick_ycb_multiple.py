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
    def test_lift_env_creates(self, lift_env):
        assert isinstance(lift_env, gym.Env)

    def test_lift_env_reset(self, lift_env):
        obs, info = lift_env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_lift_env_step(self, lift_env):
        lift_env.reset()
        action = lift_env.action_space.sample()
        obs, reward, terminated, truncated, info = lift_env.step(action)
        assert isinstance(obs, np.ndarray)
        assert reward is not None
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_observation_space(self, lift_env):
        obs, _ = lift_env.reset()
        assert lift_env.observation_space.contains(obs)

    def test_action_space_shape(self, lift_env):
        assert lift_env.action_space.shape == (6,)


class TestEpisodeLogic:
    EXPECTED_INFO_KEYS = {
        "is_grasped",
        "is_robot_static",
        "lift_height",
        "success",
        "tcp_to_obj_dist",
    }

    def test_info_keys_lift(self, lift_env):
        _, info = lift_env.reset()
        assert set(info.keys()) == self.EXPECTED_INFO_KEYS

    def test_target_spawns_in_radius_bounds(self, lift_env):
        min_r = _CFG.spawn_min_radius
        max_r = _CFG.spawn_max_radius
        for _ in range(5):
            lift_env.reset()
            obj_pose = lift_env.unwrapped._get_obj_pose()
            r = float(np.sqrt(obj_pose[0] ** 2 + obj_pose[1] ** 2))
            assert min_r <= r <= max_r

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
