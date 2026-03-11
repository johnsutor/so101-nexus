import os

import gymnasium as gym
import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus_mujoco  # noqa: F401, E402
from so101_nexus_core.config import PickYCBConfig
from so101_nexus_mujoco.pick_ycb import PickYCBEnv

_CFG = PickYCBConfig()

LIFT_ENV_IDS = [
    "MuJoCoPickYCBLift-v1",
    "MuJoCoPickBananaLift-v1",
    "MuJoCoPickGolfBallLift-v1",
]

PER_ROBOT_LIFT_IDS = [
    "MuJoCoPickGolfBallLiftSO101-v1",
]


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


class TestEnvCreation:
    @pytest.mark.parametrize("env_id", LIFT_ENV_IDS)
    def test_env_creates(self, env_id):
        env = gym.make(env_id)
        assert isinstance(env, gym.Env)
        env.close()

    @pytest.mark.parametrize("env_id", LIFT_ENV_IDS)
    def test_env_reset(self, env_id):
        env = gym.make(env_id)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        env.close()

    @pytest.mark.parametrize("env_id", LIFT_ENV_IDS)
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

    @pytest.mark.parametrize("env_id", LIFT_ENV_IDS)
    def test_observation_space(self, env_id):
        env = gym.make(env_id)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
        env.close()

    @pytest.mark.parametrize("env_id", LIFT_ENV_IDS)
    def test_action_space_shape(self, env_id):
        env = gym.make(env_id)
        assert env.action_space.shape == (6,)
        env.close()


class TestPerRobotEnvIDs:
    @pytest.mark.parametrize("env_id", PER_ROBOT_LIFT_IDS)
    def test_per_robot_lift_env_creates(self, env_id):
        env = gym.make(env_id)
        assert isinstance(env, gym.Env)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        env.close()


class TestObservationVector:
    def test_lift_state_shape(self, lift_env):
        obs, _ = lift_env.reset()
        assert obs.shape == (18,)


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

    def test_obj_spawns_in_radius_bounds(self, lift_env):
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


class TestCameraModes:
    def test_state_only_returns_flat_array(self):
        env = PickYCBEnv(camera_mode="state_only")
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (18,)
        env.close()

    def test_wrist_mode_returns_dict(self):
        env = PickYCBEnv(camera_mode="wrist")
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert "state" in obs
        assert "wrist_camera" in obs
        assert obs["state"].shape == (18,)
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
        env = gym.make("MuJoCoPickGolfBallLift-v1", render_mode="rgb_array")
        env.reset()
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
        env.close()
