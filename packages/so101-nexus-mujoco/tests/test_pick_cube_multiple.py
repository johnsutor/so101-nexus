import os

import gymnasium as gym
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus_mujoco  # noqa: F401, E402
from so101_nexus_core.config import CUBE_COLOR_MAP, PickCubeMultipleConfig
from so101_nexus_mujoco.pick_cube_multiple import PickCubeMultipleEnv

_CFG = PickCubeMultipleConfig()


@pytest.fixture(scope="module")
def lift_env():
    env = gym.make("MuJoCoPickCubeMultipleLift-v1")
    yield env
    env.close()


class TestConstructionValidation:
    def test_invalid_cube_color(self):
        with pytest.raises(ValueError, match="cube_color"):
            PickCubeMultipleEnv(cube_color="neon")

    def test_invalid_cube_half_size(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickCubeMultipleEnv(config=PickCubeMultipleConfig(cube_half_size=0.001))

    def test_invalid_num_distractors(self):
        with pytest.raises(ValueError, match="num_distractors"):
            PickCubeMultipleConfig(num_distractors=0)


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
        assert isinstance(info, dict)

    def test_observation_space_lift(self, lift_env):
        obs, _ = lift_env.reset()
        assert lift_env.observation_space.contains(obs)

    def test_action_space_shape_lift(self, lift_env):
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
            cube_pose = lift_env.unwrapped._get_cube_pose()
            r = float(np.sqrt(cube_pose[0] ** 2 + cube_pose[1] ** 2))
            assert min_r <= r <= max_r

    def test_reward_range_lift(self, lift_env):
        lift_env.reset()
        action = lift_env.action_space.sample()
        _, reward, _, _, _ = lift_env.step(action)
        assert 0.0 <= reward <= 1.0


class TestMultipleObjects:
    def test_correct_number_of_distractors(self):
        env = PickCubeMultipleEnv()
        assert len(env._distractor_geom_ids) == _CFG.num_distractors
        env.close()

    def test_distractor_colors_differ_from_target(self):
        env = PickCubeMultipleEnv(cube_color="red")
        env.reset()
        target_rgba = CUBE_COLOR_MAP["red"]
        for geom_id in env._distractor_geom_ids:
            distractor_rgba = list(env.model.geom_rgba[geom_id])
            assert distractor_rgba != target_rgba
        env.close()

    def test_custom_num_distractors(self):
        cfg = PickCubeMultipleConfig(num_distractors=5)
        env = PickCubeMultipleEnv(config=cfg)
        assert len(env._distractor_geom_ids) == 5
        env.close()


class TestTaskDescription:
    def test_task_description_exists(self):
        env = PickCubeMultipleEnv(cube_color="red")
        assert isinstance(env.task_description, str)
        assert "red" in env.task_description
        env.close()

    def test_task_description_starts_with_capital(self):
        env = PickCubeMultipleEnv(cube_color="blue")
        assert env.task_description[0].isupper()
        env.close()
