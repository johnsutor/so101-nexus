import gymnasium as gym
import numpy as np
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import PickCubeConfig
from so101_nexus_maniskill.pick_cube import (
    PICK_CUBE_CONFIGS,
    PickCubeGoalSO100Env,
    PickCubeGoalSO101Env,
    PickCubeLiftSO100Env,
    PickCubeLiftSO101Env,
)

_CFG = PickCubeConfig()
BASE_KWARGS = dict(obs_mode="state", num_envs=1, render_mode=None)

GOAL_ENV_IDS = [
    ("ManiSkillPickCubeGoalSO100-v1", "so100"),
    ("ManiSkillPickCubeGoalSO101-v1", "so101"),
]
LIFT_ENV_IDS = [
    ("ManiSkillPickCubeLiftSO100-v1", "so100"),
    ("ManiSkillPickCubeLiftSO101-v1", "so101"),
]
ALL_ENV_IDS = GOAL_ENV_IDS + LIFT_ENV_IDS


@pytest.fixture(scope="module")
def goal_so100_env():
    env = gym.make("ManiSkillPickCubeGoalSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def goal_so101_env():
    env = gym.make("ManiSkillPickCubeGoalSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_so100_env():
    env = gym.make("ManiSkillPickCubeLiftSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_so101_env():
    env = gym.make("ManiSkillPickCubeLiftSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillPickCubeGoalSO100-v1": "goal_so100_env",
        "ManiSkillPickCubeGoalSO101-v1": "goal_so101_env",
        "ManiSkillPickCubeLiftSO100-v1": "lift_so100_env",
        "ManiSkillPickCubeLiftSO101-v1": "lift_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestConstructionValidation:
    def test_invalid_cube_color(self):
        with pytest.raises(ValueError, match="cube_color"):
            PickCubeConfig(cube_color="neon")

    def test_invalid_cube_half_size(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickCubeConfig(cube_half_size=0.001)

    def test_invalid_robot_uid(self):
        with pytest.raises(ValueError, match="robot_uids"):
            gym.make("ManiSkillPickCubeGoal-v1", robot_uids="panda", **BASE_KWARGS)


class TestSharedConstants:
    def test_cube_half_size_matches_core(self):
        for robot_key, cfg in PICK_CUBE_CONFIGS.items():
            assert cfg["cube_half_size"] == _CFG.cube_half_size, (
                f"{robot_key} cube_half_size mismatch"
            )

    def test_goal_thresh_matches_core(self):
        for robot_key, cfg in PICK_CUBE_CONFIGS.items():
            assert cfg["goal_thresh"] == _CFG.goal_thresh

    def test_spawn_half_size_matches_core(self):
        for robot_key, cfg in PICK_CUBE_CONFIGS.items():
            assert cfg["cube_spawn_half_size"] == _CFG.spawn_half_size

    def test_max_goal_height_matches_core(self):
        for robot_key, cfg in PICK_CUBE_CONFIGS.items():
            assert cfg["max_goal_height"] == _CFG.max_goal_height

    def test_cube_spawn_center_at_origin_relative(self):
        for robot_key, cfg in PICK_CUBE_CONFIGS.items():
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
        env = gym.make("ManiSkillPickCubeGoalSO100-v1", **BASE_KWARGS)
        assert env.unwrapped.task_description[0].isupper()
        env.close()

    def test_task_description_includes_color(self):
        env = gym.make("ManiSkillPickCubeGoal-v1", config=PickCubeConfig(cube_color="green"), **BASE_KWARGS)
        assert "green" in env.unwrapped.task_description
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
    def test_cube_spawns_in_bounds(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        cfg = PICK_CUBE_CONFIGS[robot]
        cx, cy = cfg["cube_spawn_center"]
        hs = cfg["cube_spawn_half_size"]
        cube_p = env.unwrapped.obj.pose.p[0].cpu()
        assert cx - hs <= cube_p[0].item() <= cx + hs
        assert cy - hs <= cube_p[1].item() <= cy + hs

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_goal_spawns_in_bounds(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        cfg = PICK_CUBE_CONFIGS[robot]
        cx, cy = cfg["cube_spawn_center"]
        hs = cfg["cube_spawn_half_size"]
        goal_p = env.unwrapped.goal_site.pose.p[0].cpu()
        assert cx - hs <= goal_p[0].item() <= cx + hs
        assert cy - hs <= goal_p[1].item() <= cy + hs

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
        """Robot base pose must use the keyframe's Z-rotation so it faces the workspace."""
        env = _get_env(request, env_id)
        env.reset()
        inner = env.unwrapped
        expected_q = inner.agent.keyframes["rest"].pose.q
        actual_q = inner.agent.robot.pose.q[0].cpu().numpy()
        np.testing.assert_allclose(actual_q, expected_q, atol=1e-4)


class TestRobotSubclasses:
    def test_so100_goal_env_uses_so100(self, goal_so100_env):
        inner = goal_so100_env.unwrapped
        assert isinstance(inner, PickCubeGoalSO100Env)
        assert inner.robot_uids == "so100"

    def test_so101_goal_env_uses_so101(self, goal_so101_env):
        inner = goal_so101_env.unwrapped
        assert isinstance(inner, PickCubeGoalSO101Env)
        assert inner.robot_uids == "so101"

    def test_so100_lift_env_uses_so100(self, lift_so100_env):
        inner = lift_so100_env.unwrapped
        assert isinstance(inner, PickCubeLiftSO100Env)
        assert inner.robot_uids == "so100"

    def test_so101_lift_env_uses_so101(self, lift_so101_env):
        inner = lift_so101_env.unwrapped
        assert isinstance(inner, PickCubeLiftSO101Env)
        assert inner.robot_uids == "so101"


class TestCameraModes:
    @pytest.fixture(scope="class")
    def fixed_cam_env(self):
        env = gym.make("ManiSkillPickCubeGoalSO100-v1", config=PickCubeConfig(camera_mode="fixed"), **BASE_KWARGS)
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def wrist_cam_env(self):
        env = gym.make("ManiSkillPickCubeGoalSO100-v1", config=PickCubeConfig(camera_mode="wrist"), **BASE_KWARGS)
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def both_cam_env(self):
        env = gym.make("ManiSkillPickCubeGoalSO100-v1", config=PickCubeConfig(camera_mode="both"), **BASE_KWARGS)
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


class TestNoTable:
    def test_no_table_scene_builder(self, goal_so101_env):
        inner = goal_so101_env.unwrapped
        assert not hasattr(inner, "table_scene"), "Should not have a table_scene attribute"

    def test_robot_base_at_origin(self, goal_so101_env):
        inner = goal_so101_env.unwrapped
        inner.reset()
        base_pos = inner.agent.robot.pose.p[0].cpu()
        assert base_pos[0].item() == pytest.approx(0.0, abs=0.01)
        assert base_pos[1].item() == pytest.approx(0.0, abs=0.01)
        assert base_pos[2].item() == pytest.approx(0.0, abs=0.01)
