import gymnasium as gym
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import PickCubeMultipleConfig
from so101_nexus_maniskill.pick_cube_multiple import (
    PICK_CUBE_MULTIPLE_CONFIGS,
    PickCubeMultipleGoalSO100Env,
    PickCubeMultipleGoalSO101Env,
    PickCubeMultipleLiftSO100Env,
    PickCubeMultipleLiftSO101Env,
)

_CFG = PickCubeMultipleConfig()
BASE_KWARGS = dict(obs_mode="state", num_envs=1, render_mode=None)

GOAL_ENV_IDS = [
    ("ManiSkillPickCubeMultipleGoalSO100-v1", "so100"),
    ("ManiSkillPickCubeMultipleGoalSO101-v1", "so101"),
]
LIFT_ENV_IDS = [
    ("ManiSkillPickCubeMultipleLiftSO100-v1", "so100"),
    ("ManiSkillPickCubeMultipleLiftSO101-v1", "so101"),
]
ALL_ENV_IDS = GOAL_ENV_IDS + LIFT_ENV_IDS


@pytest.fixture(scope="module")
def goal_so100_env():
    env = gym.make("ManiSkillPickCubeMultipleGoalSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def goal_so101_env():
    env = gym.make("ManiSkillPickCubeMultipleGoalSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_so100_env():
    env = gym.make("ManiSkillPickCubeMultipleLiftSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_so101_env():
    env = gym.make("ManiSkillPickCubeMultipleLiftSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillPickCubeMultipleGoalSO100-v1": "goal_so100_env",
        "ManiSkillPickCubeMultipleGoalSO101-v1": "goal_so101_env",
        "ManiSkillPickCubeMultipleLiftSO100-v1": "lift_so100_env",
        "ManiSkillPickCubeMultipleLiftSO101-v1": "lift_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestConstructionValidation:
    def test_invalid_cube_color(self):
        with pytest.raises(ValueError, match="cube_color"):
            PickCubeMultipleConfig(cube_color="neon")

    def test_invalid_cube_half_size(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickCubeMultipleConfig(cube_half_size=0.001)

    def test_invalid_num_distractors(self):
        with pytest.raises(ValueError, match="num_distractors"):
            PickCubeMultipleConfig(num_distractors=0)

    def test_invalid_robot_uid(self):
        with pytest.raises(ValueError, match="robot_uids"):
            gym.make("ManiSkillPickCubeMultipleGoal-v1", robot_uids="panda", **BASE_KWARGS)


class TestSharedConstants:
    def test_cube_half_size_matches_core(self):
        for robot_key, cfg in PICK_CUBE_MULTIPLE_CONFIGS.items():
            assert cfg["cube_half_size"] == _CFG.cube_half_size

    def test_goal_thresh_matches_core(self):
        for robot_key, cfg in PICK_CUBE_MULTIPLE_CONFIGS.items():
            assert cfg["goal_thresh"] == _CFG.goal_thresh


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
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, torch.Tensor)
        assert reward is not None

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_observation_space(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_action_space_shape(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert env.action_space.shape == (6,)


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
    def test_target_spawns_in_bounds(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        cfg = PICK_CUBE_MULTIPLE_CONFIGS[robot]
        cx, cy = cfg["cube_spawn_center"]
        hs = cfg["cube_spawn_half_size"]
        cube_p = env.unwrapped.obj.pose.p[0].cpu()
        assert cx - hs <= cube_p[0].item() <= cx + hs
        assert cy - hs <= cube_p[1].item() <= cy + hs

    @pytest.mark.parametrize("env_id,robot", GOAL_ENV_IDS)
    def test_reward_range_goal(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert (reward >= 0).all()
        assert (reward <= 1).all()

    @pytest.mark.parametrize("env_id,robot", LIFT_ENV_IDS)
    def test_reward_range_lift(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert (reward >= 0).all()
        assert (reward <= 1).all()


class TestMultipleObjects:
    def test_correct_number_of_distractors(self, goal_so100_env):
        inner = goal_so100_env.unwrapped
        assert len(inner.distractors) == _CFG.num_distractors

    def test_custom_num_distractors(self):
        cfg = PickCubeMultipleConfig(num_distractors=5)
        env = gym.make("ManiSkillPickCubeMultipleGoalSO100-v1", config=cfg, **BASE_KWARGS)
        assert len(env.unwrapped.distractors) == 5
        env.close()


class TestTaskDescription:
    def test_task_description_starts_with_capital(self, goal_so100_env):
        assert goal_so100_env.unwrapped.task_description[0].isupper()

    def test_task_description_includes_color(self):
        cfg = PickCubeMultipleConfig(cube_color="green")
        env = gym.make("ManiSkillPickCubeMultipleGoal-v1", config=cfg, **BASE_KWARGS)
        assert "green" in env.unwrapped.task_description
        env.close()


class TestRobotSubclasses:
    def test_so100_goal_env_uses_so100(self, goal_so100_env):
        inner = goal_so100_env.unwrapped
        assert isinstance(inner, PickCubeMultipleGoalSO100Env)

    def test_so101_goal_env_uses_so101(self, goal_so101_env):
        inner = goal_so101_env.unwrapped
        assert isinstance(inner, PickCubeMultipleGoalSO101Env)

    def test_so100_lift_env_uses_so100(self, lift_so100_env):
        inner = lift_so100_env.unwrapped
        assert isinstance(inner, PickCubeMultipleLiftSO100Env)

    def test_so101_lift_env_uses_so101(self, lift_so101_env):
        inner = lift_so101_env.unwrapped
        assert isinstance(inner, PickCubeMultipleLiftSO101Env)
