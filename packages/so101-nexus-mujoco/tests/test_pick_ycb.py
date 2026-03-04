import os

import gymnasium as gym
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus_mujoco  # noqa: F401, E402
from so101_nexus_core.types import (
    DEFAULT_CUBE_SPAWN_HALF_SIZE,
    DEFAULT_GOAL_THRESH,
    DEFAULT_LIFT_THRESHOLD,
    DEFAULT_MAX_GOAL_HEIGHT,
)
from so101_nexus_mujoco.pick_ycb import PickYCBEnv

GOAL_ENV_IDS = [
    "MuJoCoPickYCBGoal-v1",
    "MuJoCoPickBananaGoal-v1",
    "MuJoCoPickGolfBallGoal-v1",
    "MuJoCoPickForkGoal-v1",
]
LIFT_ENV_IDS = [
    "MuJoCoPickYCBLift-v1",
    "MuJoCoPickBananaLift-v1",
    "MuJoCoPickGolfBallLift-v1",
]
ALL_ENV_IDS = GOAL_ENV_IDS + LIFT_ENV_IDS

PER_ROBOT_GOAL_IDS = [
    "MuJoCoPickGolfBallGoalSO101-v1",
    "MuJoCoPickBananaGoalSO101-v1",
]
PER_ROBOT_LIFT_IDS = [
    "MuJoCoPickGolfBallLiftSO101-v1",
]


@pytest.fixture(scope="module")
def goal_env():
    env = gym.make("MuJoCoPickGolfBallGoal-v1")
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_env():
    env = gym.make("MuJoCoPickGolfBallLift-v1")
    yield env
    env.close()


class TestConstructionValidation:
    def test_invalid_model_id(self):
        with pytest.raises(ValueError, match="model_id"):
            PickYCBEnv(model_id="invalid_object")

    def test_invalid_control_mode(self):
        with pytest.raises(ValueError, match="control_mode"):
            PickYCBEnv(control_mode="invalid_mode")


class TestSharedConstants:
    def test_spawn_half_size_matches_core(self):
        assert DEFAULT_CUBE_SPAWN_HALF_SIZE == 0.05

    def test_goal_thresh_matches_core(self):
        assert DEFAULT_GOAL_THRESH == 0.025

    def test_lift_threshold_matches_core(self):
        assert DEFAULT_LIFT_THRESHOLD == 0.05

    def test_max_goal_height_matches_core(self):
        assert DEFAULT_MAX_GOAL_HEIGHT == 0.08


class TestEnvCreation:
    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_env_creates(self, env_id):
        env = gym.make(env_id)
        assert isinstance(env, gym.Env)
        env.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_env_reset(self, env_id):
        env = gym.make(env_id)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        env.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_env_step(self, env_id):
        env = gym.make(env_id)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert reward is not None
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        env.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_observation_space(self, env_id):
        env = gym.make(env_id)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        env.close()

    @pytest.mark.parametrize("env_id", ALL_ENV_IDS)
    def test_action_space_shape(self, env_id):
        env = gym.make(env_id)
        assert env.action_space.shape == (6,)
        env.close()


class TestPerRobotEnvIDs:
    @pytest.mark.parametrize("env_id", PER_ROBOT_GOAL_IDS)
    def test_per_robot_goal_env_creates(self, env_id):
        env = gym.make(env_id)
        assert isinstance(env, gym.Env)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()

    @pytest.mark.parametrize("env_id", PER_ROBOT_LIFT_IDS)
    def test_per_robot_lift_env_creates(self, env_id):
        env = gym.make(env_id)
        assert isinstance(env, gym.Env)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()


class TestObservationVector:
    def test_state_shape(self, goal_env):
        obs, _ = goal_env.reset()
        assert obs.shape == (24,)

    def test_lift_state_shape(self, lift_env):
        obs, _ = lift_env.reset()
        assert obs.shape == (24,)


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

    def test_obj_spawns_in_bounds(self, goal_env):
        cx, cy = 0.15, 0.0
        hs = DEFAULT_CUBE_SPAWN_HALF_SIZE
        for _ in range(5):
            goal_env.reset()
            obj_pose = goal_env.unwrapped._get_obj_pose()
            assert cx - hs <= obj_pose[0] <= cx + hs
            assert cy - hs <= obj_pose[1] <= cy + hs

    def test_goal_spawns_in_bounds(self, goal_env):
        cx, cy = 0.15, 0.0
        hs = DEFAULT_CUBE_SPAWN_HALF_SIZE
        for _ in range(5):
            goal_env.reset()
            goal_pos = goal_env.unwrapped._get_goal_pos()
            assert cx - hs <= goal_pos[0] <= cx + hs
            assert cy - hs <= goal_pos[1] <= cy + hs

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

    def test_success_false_at_reset(self, goal_env):
        _, info = goal_env.reset()
        assert not info["success"]


class TestTaskDescription:
    def test_task_description_exists(self):
        env = PickYCBEnv(model_id="058_golf_ball")
        assert isinstance(env.task_description, str)
        assert "golf ball" in env.task_description
        env.close()

    def test_task_description_starts_with_capital(self):
        env = PickYCBEnv()
        assert env.task_description[0].isupper()
        env.close()


class TestRobotInitQposNoise:
    def test_noise_param_exists(self):
        env = PickYCBEnv(robot_init_qpos_noise=0.05)
        assert env.robot_init_qpos_noise == 0.05
        env.close()

    def test_noise_produces_different_qpos(self):
        env = PickYCBEnv(robot_init_qpos_noise=0.02)
        qpos_list = []
        for seed in range(5):
            env.reset(seed=seed)
            qpos_list.append(env._get_current_qpos().copy())
        env.close()
        all_same = all(np.allclose(qpos_list[0], q) for q in qpos_list[1:])
        assert not all_same


class TestLiftThreshold:
    def test_lift_threshold_class_attr(self):
        assert hasattr(PickYCBEnv, "LIFT_THRESHOLD")
        assert PickYCBEnv.LIFT_THRESHOLD == DEFAULT_LIFT_THRESHOLD


class TestGoalThreshConfig:
    def test_goal_thresh_stored(self):
        env = PickYCBEnv()
        assert env._goal_thresh == DEFAULT_GOAL_THRESH
        env.close()


class TestCameraModes:
    def test_state_only_returns_flat_array(self):
        env = PickYCBEnv(camera_mode="state_only")
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (24,)
        env.close()

    def test_wrist_mode_returns_dict(self):
        env = PickYCBEnv(camera_mode="wrist")
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
        env = PickYCBEnv(control_mode=mode)
        assert env.action_space.shape == (6,)
        env.close()

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_reset_and_step(self, mode):
        env = PickYCBEnv(control_mode=mode)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert reward is not None
        env.close()


class TestRenderModes:
    def test_rgb_array(self):
        env = gym.make("MuJoCoPickGolfBallGoal-v1", render_mode="rgb_array")
        env.reset()
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        env.close()
