import gymnasium as gym
import numpy as np
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.types import (
    DEFAULT_MIN_CUBE_TARGET_SEPARATION,
)
from so101_nexus_maniskill.pick_and_place import (
    PickAndPlaceSO100Env,
    PickAndPlaceSO101Env,
)

BASE_KWARGS = dict(obs_mode="state", num_envs=1, render_mode=None)

ENV_IDS = [
    ("ManiSkillPickAndPlaceSO100-v1", "so100"),
    ("ManiSkillPickAndPlaceSO101-v1", "so101"),
]


@pytest.fixture(scope="module")
def so100_env():
    env = gym.make("ManiSkillPickAndPlaceSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def so101_env():
    env = gym.make("ManiSkillPickAndPlaceSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillPickAndPlaceSO100-v1": "so100_env",
        "ManiSkillPickAndPlaceSO101-v1": "so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestConstructionValidation:
    def test_invalid_cube_color(self):
        with pytest.raises(ValueError, match="cube_color"):
            gym.make("ManiSkillPickAndPlace-v1", cube_color="neon", **BASE_KWARGS)

    def test_invalid_target_color(self):
        with pytest.raises(ValueError, match="target_color"):
            gym.make("ManiSkillPickAndPlace-v1", target_color="neon", **BASE_KWARGS)

    def test_same_cube_and_target_color_raises(self):
        with pytest.raises(ValueError, match="must differ"):
            gym.make(
                "ManiSkillPickAndPlace-v1", cube_color="red", target_color="red", **BASE_KWARGS
            )

    def test_invalid_robot_uid(self):
        with pytest.raises(ValueError, match="robot_uids"):
            gym.make("ManiSkillPickAndPlace-v1", robot_uids="panda", **BASE_KWARGS)


class TestEnvCreation:
    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_env_creates(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert isinstance(env, gym.Env)

    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_env_reset(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        assert isinstance(obs, torch.Tensor)
        assert isinstance(info, dict)

    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_env_step(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs, torch.Tensor)
        assert reward is not None
        assert isinstance(terminated, (bool, torch.Tensor))
        assert isinstance(truncated, (bool, torch.Tensor))
        assert isinstance(info, dict)

    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_observation_space(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_action_space_shape(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert env.action_space.shape == (6,)


class TestEpisodeLogic:
    EVALUATE_KEYS = {
        "obj_to_target_dist",
        "is_obj_placed",
        "is_grasped",
        "is_robot_static",
        "lift_height",
        "success",
    }

    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_evaluate_keys(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        info = env.unwrapped.evaluate()
        assert set(info.keys()) == self.EVALUATE_KEYS

    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_target_on_ground(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        target_z = env.unwrapped.target_site.pose.p[0, 2].cpu().item()
        assert target_z < 0.01

    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_target_visible(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        assert env.unwrapped.target_site not in env.unwrapped._hidden_objects

    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_cube_target_separation(self, request, env_id, robot):
        env = _get_env(request, env_id)
        for _ in range(5):
            env.reset()
            cube_xy = env.unwrapped.obj.pose.p[0, :2].cpu()
            target_xy = env.unwrapped.target_site.pose.p[0, :2].cpu()
            dist = torch.linalg.norm(cube_xy - target_xy).item()
            assert dist >= DEFAULT_MIN_CUBE_TARGET_SEPARATION - 1e-4

    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_reward_range(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert (reward >= 0).all()
        assert (reward <= 1).all()


class TestTaskDescription:
    def test_includes_cube_color(self):
        env = gym.make(
            "ManiSkillPickAndPlace-v1", cube_color="green", target_color="blue", **BASE_KWARGS
        )
        assert "green" in env.unwrapped.task_description
        env.close()

    def test_includes_target_color(self):
        env = gym.make(
            "ManiSkillPickAndPlace-v1", cube_color="green", target_color="blue", **BASE_KWARGS
        )
        assert "blue" in env.unwrapped.task_description
        env.close()


class TestCameraModes:
    @pytest.fixture(scope="class")
    def fixed_cam_env(self):
        env = gym.make("ManiSkillPickAndPlaceSO100-v1", camera_mode="fixed", **BASE_KWARGS)
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def wrist_cam_env(self):
        env = gym.make("ManiSkillPickAndPlaceSO100-v1", camera_mode="wrist", **BASE_KWARGS)
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def both_cam_env(self):
        env = gym.make("ManiSkillPickAndPlaceSO100-v1", camera_mode="both", **BASE_KWARGS)
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


class TestRobotOrientation:
    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_robot_base_uses_keyframe_rotation(self, request, env_id, robot):
        """Robot base pose must use the keyframe's Z-rotation so it faces the workspace."""
        env = _get_env(request, env_id)
        env.reset()
        inner = env.unwrapped
        expected_q = inner.agent.keyframes["rest"].pose.q
        actual_q = inner.agent.robot.pose.q[0].cpu().numpy()
        np.testing.assert_allclose(actual_q, expected_q, atol=1e-4)


class TestTargetDiscOrientation:
    @pytest.mark.parametrize("env_id,robot", ENV_IDS)
    def test_target_disc_lies_flat(self, request, env_id, robot):
        """Target disc cylinder must be rotated so it lies flat on the ground (axis along Z).

        After rotation, the cylinder's Y-axis should map to Z.
        Quaternion for 90deg around X: [cos(pi/4), sin(pi/4), 0, 0].
        """
        env = _get_env(request, env_id)
        env.reset()
        target_q = env.unwrapped.target_site.pose.q[0].cpu().numpy()
        expected_q = np.array([0.7071068, 0.7071068, 0.0, 0.0])
        np.testing.assert_allclose(np.abs(target_q), np.abs(expected_q), atol=1e-3)


class TestRobotSubclasses:
    def test_so100_env_uses_so100(self, so100_env):
        inner = so100_env.unwrapped
        assert isinstance(inner, PickAndPlaceSO100Env)
        assert inner.robot_uids == "so100"

    def test_so101_env_uses_so101(self, so101_env):
        inner = so101_env.unwrapped
        assert isinstance(inner, PickAndPlaceSO101Env)
        assert inner.robot_uids == "so101"
