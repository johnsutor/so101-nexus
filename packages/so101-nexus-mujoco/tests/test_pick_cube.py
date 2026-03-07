import os

import gymnasium as gym
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus_mujoco  # noqa: F401, E402
from so101_nexus_core.config import PickCubeConfig, RobotConfig
from so101_nexus_mujoco.pick_cube import PickCubeEnv

_CFG = PickCubeConfig()


@pytest.fixture(scope="module")
def goal_env():
    env = gym.make("MuJoCoPickCubeGoal-v1")
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_env():
    env = gym.make("MuJoCoPickCubeLift-v1")
    yield env
    env.close()


class TestConstructionValidation:
    def test_invalid_cube_colors(self):
        with pytest.raises(ValueError, match="cube_colors"):
            PickCubeEnv(config=PickCubeConfig(cube_colors="neon"))

    def test_invalid_cube_half_size(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickCubeConfig(cube_half_size=0.001)


class TestSharedConstants:
    def test_default_cube_half_size_matches_core(self):
        env = PickCubeEnv()
        assert env.cube_half_size == _CFG.cube_half_size
        env.close()


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

    def test_observation_space_goal(self, goal_env):
        obs, _ = goal_env.reset()
        assert goal_env.observation_space.contains(obs)

    def test_observation_space_lift(self, lift_env):
        obs, _ = lift_env.reset()
        assert lift_env.observation_space.contains(obs)

    def test_action_space_shape_goal(self, goal_env):
        assert goal_env.action_space.shape == (6,)

    def test_action_space_shape_lift(self, lift_env):
        assert lift_env.action_space.shape == (6,)


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

    def test_cube_spawns_in_bounds(self, goal_env):
        cx, cy = _CFG.spawn_center
        hs = _CFG.spawn_half_size
        for _ in range(5):
            goal_env.reset()
            cube_pose = goal_env.unwrapped._get_cube_pose()
            assert cx - hs <= cube_pose[0] <= cx + hs
            assert cy - hs <= cube_pose[1] <= cy + hs

    def test_goal_spawns_in_bounds(self, goal_env):
        cx, cy = _CFG.spawn_center
        hs = _CFG.spawn_half_size
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


class TestRewardBudget:
    def test_goal_reaching_only_reward_bounded_by_reaching_weight(self, goal_env):
        goal_env.reset()
        info = goal_env.unwrapped._get_info()
        assert not info["is_grasped"]
        reward = goal_env.unwrapped._compute_reward(info)
        assert reward <= _CFG.reward.reaching + 1e-6

    def test_lift_reaching_only_reward_bounded_by_reaching_weight(self, lift_env):
        lift_env.reset()
        info = lift_env.unwrapped._get_info()
        assert not info["is_grasped"]
        reward = lift_env.unwrapped._compute_reward(info)
        assert reward <= _CFG.reward.reaching + 1e-6


class TestTaskDescription:
    def test_task_description_exists(self):
        env = PickCubeEnv(config=PickCubeConfig(cube_colors="red"))
        assert isinstance(env.task_description, str)
        assert "red" in env.task_description
        env.close()

    def test_task_description_starts_with_capital(self):
        env = PickCubeEnv(config=PickCubeConfig(cube_colors="blue"))
        assert env.task_description[0].isupper()
        env.close()

    def test_task_description_is_instance_attr(self):
        env = PickCubeEnv()
        assert "task_description" in env.__dict__
        env.close()


class TestRobotInitQposNoise:
    def test_noise_param_exists(self):
        env = PickCubeEnv(robot_init_qpos_noise=0.05)
        assert env.robot_init_qpos_noise == 0.05
        env.close()

    def test_noise_produces_different_qpos(self):
        env = PickCubeEnv(robot_init_qpos_noise=0.02)
        qpos_list = []
        for seed in range(5):
            env.reset(seed=seed)
            qpos_list.append(env._get_current_qpos().copy())
        env.close()
        all_same = all(np.allclose(qpos_list[0], q) for q in qpos_list[1:])
        assert not all_same


class TestCameraModes:
    def test_state_only_returns_flat_array(self):
        env = PickCubeEnv(camera_mode="state_only")
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (24,)
        env.close()

    def test_wrist_mode_returns_dict(self):
        env = PickCubeEnv(camera_mode="wrist")
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert "state" in obs
        assert "wrist_camera" in obs
        assert obs["state"].shape == (24,)
        assert obs["wrist_camera"].shape == (224, 224, 3)
        assert obs["wrist_camera"].dtype == np.uint8
        env.close()

    def test_wrist_mode_step_returns_dict(self):
        env = PickCubeEnv(camera_mode="wrist")
        env.reset()
        action = env.action_space.sample()
        obs, _, _, _, _ = env.step(action)
        assert isinstance(obs, dict)
        assert obs["wrist_camera"].shape == (224, 224, 3)
        assert obs["wrist_camera"].dtype == np.uint8
        env.close()


class TestControlModes:
    ALL_MODES = ["pd_joint_pos", "pd_joint_delta_pos", "pd_joint_target_delta_pos"]

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_action_shape(self, mode):
        env = PickCubeEnv(control_mode=mode)
        assert env.action_space.shape == (6,)
        env.close()

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_reset_and_step(self, mode):
        env = PickCubeEnv(control_mode=mode)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, np.ndarray)
        assert reward is not None
        env.close()

    @pytest.mark.parametrize("mode", ["pd_joint_delta_pos", "pd_joint_target_delta_pos"])
    def test_zero_action_stays_near_rest(self, mode):
        env = PickCubeEnv(control_mode=mode, robot_init_qpos_noise=0.0)
        env.reset()
        rest = np.array(RobotConfig().rest_qpos_rad, dtype=np.float64)
        zero = np.zeros(6, dtype=np.float32)
        for _ in range(10):
            env.step(zero)
        qpos = env._get_current_qpos()[:5]
        assert np.allclose(qpos, rest[:5], atol=0.1)
        env.close()

    def test_target_delta_accumulates(self):
        env = PickCubeEnv(control_mode="pd_joint_target_delta_pos", robot_init_qpos_noise=0.0)
        env.reset()
        rest = np.array(RobotConfig().rest_qpos_rad, dtype=np.float64)
        small_delta = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        for _ in range(5):
            env.step(small_delta)
        expected_target = rest.copy()
        expected_target[0] += 0.05
        expected_target = np.clip(expected_target, env._ctrl_low, env._ctrl_high)
        assert np.allclose(env._prev_target, expected_target, atol=1e-6)
        env.close()

    def test_invalid_control_mode(self):
        with pytest.raises(ValueError, match="control_mode"):
            PickCubeEnv(control_mode="invalid_mode")


class TestRenderModes:
    def test_rgb_array(self):
        env = gym.make("MuJoCoPickCubeGoal-v1", render_mode="rgb_array")
        env.reset()
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        env.close()
