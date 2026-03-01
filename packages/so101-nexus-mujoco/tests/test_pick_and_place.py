import os

import gymnasium as gym
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus_mujoco  # noqa: F401, E402
from so101_nexus_core.types import (
    DEFAULT_CUBE_HALF_SIZE,
    DEFAULT_CUBE_SPAWN_HALF_SIZE,
    DEFAULT_GOAL_THRESH,
    DEFAULT_MIN_CUBE_TARGET_SEPARATION,
    DEFAULT_TARGET_DISC_RADIUS,
)
from so101_nexus_mujoco.pick_and_place import PickAndPlaceEnv


@pytest.fixture(scope="module")
def env():
    env = gym.make("MuJoCoPickAndPlace-v1")
    yield env
    env.close()


class TestConstructionValidation:
    def test_invalid_cube_color(self):
        with pytest.raises(ValueError, match="cube_color"):
            PickAndPlaceEnv(cube_color="neon")

    def test_invalid_target_color(self):
        with pytest.raises(ValueError, match="target_color"):
            PickAndPlaceEnv(target_color="neon")

    def test_same_cube_and_target_color_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            PickAndPlaceEnv(cube_color="red", target_color="red")

    def test_invalid_cube_half_size(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickAndPlaceEnv(cube_half_size=0.001)


class TestSharedConstants:
    def test_default_cube_half_size_matches_core(self):
        env = PickAndPlaceEnv()
        assert env.cube_half_size == DEFAULT_CUBE_HALF_SIZE
        env.close()

    def test_disc_radius_matches_core(self):
        env = PickAndPlaceEnv()
        assert env.target_disc_radius == DEFAULT_TARGET_DISC_RADIUS
        env.close()

    def test_spawn_half_size_matches_core(self):
        assert DEFAULT_CUBE_SPAWN_HALF_SIZE == 0.05

    def test_goal_thresh_matches_core(self):
        assert DEFAULT_GOAL_THRESH == 0.025


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
    def test_state_only_shape_is_24(self):
        env = PickAndPlaceEnv(camera_mode="state_only")
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
        cx, cy = 0.15, 0.0
        hs = DEFAULT_CUBE_SPAWN_HALF_SIZE
        for _ in range(5):
            env.reset()
            cube_pose = env.unwrapped._get_cube_pose()
            assert cx - hs <= cube_pose[0] <= cx + hs
            assert cy - hs <= cube_pose[1] <= cy + hs

    def test_minimum_cube_target_separation(self, env):
        for _ in range(10):
            env.reset()
            cube_xy = env.unwrapped._get_cube_pose()[:2]
            target_xy = env.unwrapped._get_target_pos()[:2]
            dist = np.linalg.norm(cube_xy - target_xy)
            assert dist >= DEFAULT_MIN_CUBE_TARGET_SEPARATION - 1e-6

    def test_reward_range(self, env):
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert 0.0 <= reward <= 1.0

    def test_success_false_at_reset(self, env):
        _, info = env.reset()
        assert not info["success"]


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


class TestCameraModes:
    def test_state_only_returns_flat_array(self):
        env = PickAndPlaceEnv(camera_mode="state_only")
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (24,)
        env.close()

    def test_wrist_mode_returns_dict(self):
        env = PickAndPlaceEnv(camera_mode="wrist")
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert "state" in obs
        assert "wrist_camera" in obs
        assert obs["state"].shape == (24,)
        assert obs["wrist_camera"].shape == (224, 224, 3)
        assert obs["wrist_camera"].dtype == np.uint8
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
