import os

import gymnasium as gym
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus_mujoco  # noqa: F401
from so101_nexus_core.config import PickAndPlaceConfig
from so101_nexus_mujoco.pick_and_place import PickAndPlaceEnv

_CFG = PickAndPlaceConfig()


@pytest.fixture(scope="module")
def env():
    env = gym.make("MuJoCoPickAndPlace-v1")
    yield env
    env.close()


class TestConstructionValidation:
    def test_invalid_cube_colors(self):
        with pytest.raises(ValueError, match="cube_colors"):
            PickAndPlaceConfig(cube_colors="neon")

    def test_invalid_target_colors(self):
        with pytest.raises(ValueError, match="target_colors"):
            PickAndPlaceConfig(target_colors="neon")

    def test_same_cube_and_target_color_warns(self):
        with pytest.warns(UserWarning, match="overlap"):
            PickAndPlaceConfig(cube_colors="red", target_colors="red")

    def test_invalid_cube_half_size(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickAndPlaceConfig(cube_half_size=0.001)


class TestSharedConstants:
    def test_default_cube_half_size_matches_core(self):
        env = PickAndPlaceEnv()
        assert env.cube_half_size == _CFG.cube_half_size
        env.close()

    def test_disc_radius_matches_core(self):
        env = PickAndPlaceEnv()
        assert env.target_disc_radius == _CFG.target_disc_radius
        env.close()


class TestEnvCreation:
    def test_env_creates(self, env):
        assert isinstance(env, gym.Env)

    def test_env_reset(self, env):
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_env_step(self, env):
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert reward is not None
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_observation_space(self, env):
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (6,)


class TestObservationVector:
    def test_default_shape_is_24(self):
        env = PickAndPlaceEnv()
        obs, _ = env.reset()
        assert obs.shape == (24,)
        env.close()

    def test_target_z_near_ground(self):
        env = PickAndPlaceEnv()
        obs, _ = env.reset()
        target_pos = obs[8:11]
        assert target_pos[2] < 0.01
        env.close()


class TestEpisodeLogic:
    EXPECTED_INFO_KEYS = {
        "obj_to_target_dist",
        "is_obj_placed",
        "is_grasped",
        "is_robot_static",
        "lift_height",
        "success",
        "tcp_to_obj_dist",
    }

    def test_info_keys(self, env):
        _, info = env.reset()
        assert set(info.keys()) == self.EXPECTED_INFO_KEYS

    def test_cube_spawns_in_bounds(self, env):
        min_r = _CFG.spawn_min_radius
        max_r = _CFG.spawn_max_radius
        cx, cy = _CFG.spawn_center
        for _ in range(5):
            env.reset()
            cube_pos = env.unwrapped._get_cube_pose()[:2]
            r = float(np.sqrt((cube_pos[0] - cx) ** 2 + (cube_pos[1] - cy) ** 2))
            assert min_r <= r <= max_r

    def test_minimum_cube_target_separation(self, env):
        for _ in range(10):
            env.reset()
            cube_xy = env.unwrapped._get_cube_pose()[:2]
            target_xy = env.unwrapped._get_target_pos()[:2]
            dist = np.linalg.norm(cube_xy - target_xy)
            assert dist >= _CFG.min_cube_target_separation - 1e-6

    def test_reward_range(self, env):
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert 0.0 <= reward <= 1.0

    def test_success_false_at_reset(self, env):
        _, info = env.reset()
        assert not info["success"]


class TestTaskDescription:
    def test_task_description_exists(self):
        env = PickAndPlaceEnv(config=PickAndPlaceConfig(cube_colors="red", target_colors="blue"))
        assert isinstance(env.task_description, str)
        assert "red" in env.task_description
        assert "blue" in env.task_description
        env.close()

    def test_task_description_starts_with_capital(self):
        env = PickAndPlaceEnv()
        assert env.task_description[0].isupper()
        env.close()

    def test_task_description_is_instance_attr(self):
        env = PickAndPlaceEnv()
        assert "task_description" in env.__dict__
        env.close()


class TestRobotInitQposNoise:
    def test_noise_param_exists(self):
        env = PickAndPlaceEnv(robot_init_qpos_noise=0.05)
        assert env.robot_init_qpos_noise == 0.05
        env.close()

    def test_noise_produces_different_qpos(self):
        env = PickAndPlaceEnv(robot_init_qpos_noise=0.02)
        qpos_list = []
        for seed in range(5):
            env.reset(seed=seed)
            qpos_list.append(env._get_current_qpos().copy())
        env.close()
        all_same = all(np.allclose(qpos_list[0], q) for q in qpos_list[1:])
        assert not all_same


class TestGoalThreshConfig:
    def test_goal_thresh_from_config(self):
        env = PickAndPlaceEnv()
        assert env.config.goal_thresh == _CFG.goal_thresh
        env.close()


class TestVisibleTarget:
    def test_target_disc_geom_exists(self):
        env = PickAndPlaceEnv()
        env.reset()
        import mujoco

        geom_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "target_disc")
        assert geom_id >= 0
        env.close()

    def test_no_mocap_goal_body(self):
        env = PickAndPlaceEnv()
        env.reset()
        import mujoco

        body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        assert body_id == -1
        env.close()


class TestControlModes:
    ALL_MODES = ["pd_joint_pos", "pd_joint_delta_pos", "pd_joint_target_delta_pos"]

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_action_shape(self, mode):
        env = PickAndPlaceEnv(control_mode=mode)
        assert env.action_space.shape == (6,)
        env.close()

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_reset_and_step(self, mode):
        env = PickAndPlaceEnv(control_mode=mode)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert reward is not None
        env.close()


class TestRenderModes:
    def test_rgb_array(self):
        env = gym.make("MuJoCoPickAndPlace-v1", render_mode="rgb_array")
        env.reset()
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        env.close()


def test_spawn_center_offsets_cube_and_target():
    """Cube and target should be offset by spawn_center."""
    config = PickAndPlaceConfig(spawn_angle_half_range_deg=30.0)
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    cube_xs = []
    for seed in range(20):
        env.reset(seed=seed)
        cube_pos = env.unwrapped._get_cube_pose()[:2]
        cube_xs.append(cube_pos[0])
    env.close()
    assert np.mean(cube_xs) > 0.10
