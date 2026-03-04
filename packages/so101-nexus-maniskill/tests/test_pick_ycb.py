import gymnasium as gym
import numpy as np
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.types import (
    DEFAULT_CUBE_SPAWN_HALF_SIZE,
    DEFAULT_GOAL_THRESH,
    DEFAULT_MAX_GOAL_HEIGHT,
)
from so101_nexus_maniskill.pick_ycb import PICK_YCB_CONFIGS

BASE_KWARGS = dict(obs_mode="state", num_envs=1, render_mode=None)

GOAL_ENV_IDS = [
    ("ManiSkillPickGolfBallGoalSO100-v1", "so100"),
    ("ManiSkillPickGolfBallGoalSO101-v1", "so101"),
    ("ManiSkillPickBananaGoalSO101-v1", "so101"),
]
LIFT_ENV_IDS = [
    ("ManiSkillPickGolfBallLiftSO100-v1", "so100"),
    ("ManiSkillPickGolfBallLiftSO101-v1", "so101"),
]
ALL_ENV_IDS = GOAL_ENV_IDS + LIFT_ENV_IDS


@pytest.fixture(scope="module")
def goal_so100_env():
    env = gym.make("ManiSkillPickGolfBallGoalSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def goal_so101_env():
    env = gym.make("ManiSkillPickGolfBallGoalSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_so100_env():
    env = gym.make("ManiSkillPickGolfBallLiftSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_so101_env():
    env = gym.make("ManiSkillPickGolfBallLiftSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def banana_so101_env():
    env = gym.make("ManiSkillPickBananaGoalSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillPickGolfBallGoalSO100-v1": "goal_so100_env",
        "ManiSkillPickGolfBallGoalSO101-v1": "goal_so101_env",
        "ManiSkillPickGolfBallLiftSO100-v1": "lift_so100_env",
        "ManiSkillPickGolfBallLiftSO101-v1": "lift_so101_env",
        "ManiSkillPickBananaGoalSO101-v1": "banana_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestConstructionValidation:
    def test_invalid_model_id(self):
        with pytest.raises(ValueError, match="model_id"):
            gym.make("ManiSkillPickYCBGoal-v1", model_id="invalid_object", **BASE_KWARGS)

    def test_invalid_robot_uid(self):
        with pytest.raises(ValueError, match="robot_uids"):
            gym.make("ManiSkillPickYCBGoal-v1", robot_uids="panda", **BASE_KWARGS)


class TestSharedConstants:
    def test_goal_thresh_matches_core(self):
        for robot_key, cfg in PICK_YCB_CONFIGS.items():
            assert cfg["goal_thresh"] == DEFAULT_GOAL_THRESH

    def test_spawn_half_size_matches_core(self):
        for robot_key, cfg in PICK_YCB_CONFIGS.items():
            assert cfg["cube_spawn_half_size"] == DEFAULT_CUBE_SPAWN_HALF_SIZE

    def test_max_goal_height_matches_core(self):
        for robot_key, cfg in PICK_YCB_CONFIGS.items():
            assert cfg["max_goal_height"] == DEFAULT_MAX_GOAL_HEIGHT

    def test_cube_spawn_center_at_origin_relative(self):
        for robot_key, cfg in PICK_YCB_CONFIGS.items():
            cx, cy = cfg["cube_spawn_center"]
            assert cx == pytest.approx(0.15), f"{robot_key} spawn center x mismatch"
            assert cy == pytest.approx(0.0), f"{robot_key} spawn center y mismatch"


class TestEnvCreation:
    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_env_creates(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert isinstance(env, gym.Env)

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_env_reset(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        assert isinstance(obs, torch.Tensor)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_env_step(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        action = env.action_space.sample()
        result = env.step(action)
        obs, reward, terminated, truncated, info = result
        assert isinstance(obs, torch.Tensor)
        assert reward is not None
        assert isinstance(terminated, (bool, torch.Tensor))
        assert isinstance(truncated, (bool, torch.Tensor))
        assert isinstance(info, dict)

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_observation_space(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_action_space_shape(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert env.action_space.shape == (6,)


class TestTaskDescription:
    def test_task_description_starts_with_capital(self):
        env = gym.make("ManiSkillPickGolfBallGoalSO100-v1", **BASE_KWARGS)
        assert env.unwrapped.task_description[0].isupper()
        env.close()


class TestEpisodeLogic:
    EVALUATE_KEYS = {
        "obj_to_goal_dist",
        "is_obj_placed",
        "is_grasped",
        "is_robot_static",
        "lift_height",
        "success",
        "tcp_to_obj_dist",
    }

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_evaluate_keys(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        info = env.unwrapped.evaluate()
        assert set(info.keys()) == self.EVALUATE_KEYS

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_obj_spawns_in_bounds(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        cfg = PICK_YCB_CONFIGS[robot]
        cx, cy = cfg["cube_spawn_center"]
        hs = cfg["cube_spawn_half_size"]
        obj_p = env.unwrapped.obj.pose.p[0].cpu()
        assert cx - hs <= obj_p[0].item() <= cx + hs
        assert cy - hs <= obj_p[1].item() <= cy + hs

    @pytest.mark.parametrize("env_id,robot", GOAL_ENV_IDS)
    def test_reward_range_goal(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert (reward >= 0).all()
        assert (reward <= 1).all()

    @pytest.mark.parametrize("env_id,robot", LIFT_ENV_IDS)
    def test_reward_range_lift(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert (reward >= 0).all()
        assert (reward <= 1).all()


class TestRobotOrientation:
    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_robot_base_uses_keyframe_rotation(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        inner = env.unwrapped
        expected_q = inner.agent.keyframes["rest"].pose.q
        actual_q = inner.agent.robot.pose.q[0].cpu().numpy()
        np.testing.assert_allclose(actual_q, expected_q, atol=1e-4)


class TestRobotSubclasses:
    def test_so100_goal_env_uses_so100(self, goal_so100_env):
        inner = goal_so100_env.unwrapped
        assert inner.robot_uids == "so100"

    def test_so101_goal_env_uses_so101(self, goal_so101_env):
        inner = goal_so101_env.unwrapped
        assert inner.robot_uids == "so101"

    def test_so100_lift_env_uses_so100(self, lift_so100_env):
        inner = lift_so100_env.unwrapped
        assert inner.robot_uids == "so100"

    def test_so101_lift_env_uses_so101(self, lift_so101_env):
        inner = lift_so101_env.unwrapped
        assert inner.robot_uids == "so101"


class TestCameraModes:
    @pytest.fixture(scope="class")
    def fixed_cam_env(self):
        env = gym.make("ManiSkillPickGolfBallGoalSO100-v1", camera_mode="fixed", **BASE_KWARGS)
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def wrist_cam_env(self):
        env = gym.make("ManiSkillPickGolfBallGoalSO100-v1", camera_mode="wrist", **BASE_KWARGS)
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def both_cam_env(self):
        env = gym.make("ManiSkillPickGolfBallGoalSO100-v1", camera_mode="both", **BASE_KWARGS)
        env.reset()
        yield env
        env.close()

    def test_fixed_camera_mode(self, fixed_cam_env):
        sensor_names = [cfg.uid for cfg in fixed_cam_env.unwrapped._default_sensor_configs]
        assert "base_camera" in sensor_names
        assert "wrist_camera" not in sensor_names

    def test_wrist_camera_mode(self, wrist_cam_env):
        sensor_names = [cfg.uid for cfg in wrist_cam_env.unwrapped._default_sensor_configs]
        assert "wrist_camera" in sensor_names
        assert "base_camera" not in sensor_names

    def test_both_camera_mode(self, both_cam_env):
        sensor_names = [cfg.uid for cfg in both_cam_env.unwrapped._default_sensor_configs]
        assert "base_camera" in sensor_names
        assert "wrist_camera" in sensor_names
