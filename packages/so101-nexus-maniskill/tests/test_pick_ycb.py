from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from mani_skill import ASSET_DIR

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import PickYCBConfig
from so101_nexus_maniskill.pick_ycb import PICK_YCB_CONFIGS

_CFG = PickYCBConfig()
BASE_KWARGS = dict(obs_mode="state", num_envs=1, render_mode=None)

LIFT_ENV_IDS = [
    ("ManiSkillPickGolfBallLiftSO100-v1", "so100"),
    ("ManiSkillPickGolfBallLiftSO101-v1", "so101"),
]
ALL_ENV_IDS = LIFT_ENV_IDS


def _has_ycb_assets(asset_root: Path | None = None) -> bool:
    root = asset_root if asset_root is not None else ASSET_DIR
    manifest = root / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
    return manifest.exists()


def _skip_if_missing_ycb_assets() -> None:
    if _has_ycb_assets():
        return
    pytest.skip(
        "Missing ManiSkill YCB assets at "
        f"{ASSET_DIR / 'assets' / 'mani_skill2_ycb' / 'info_pick_v0.json'}"
    )


def _make_ycb_env(env_id: str, **kwargs):
    _skip_if_missing_ycb_assets()
    return gym.make(env_id, **kwargs)


@pytest.fixture(scope="module")
def lift_so100_env():
    env = _make_ycb_env("ManiSkillPickGolfBallLiftSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_so101_env():
    env = _make_ycb_env("ManiSkillPickGolfBallLiftSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillPickGolfBallLiftSO100-v1": "lift_so100_env",
        "ManiSkillPickGolfBallLiftSO101-v1": "lift_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestYCBAssets:
    def test_has_ycb_assets_false_when_missing_manifest(self, tmp_path):
        assert not _has_ycb_assets(tmp_path)

    def test_has_ycb_assets_true_when_manifest_exists(self, tmp_path):
        manifest = tmp_path / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest.write_text("{}", encoding="utf-8")
        assert _has_ycb_assets(tmp_path)


class TestConstructionValidation:
    def test_invalid_model_id(self):
        with pytest.raises(ValueError, match="model_id"):
            PickYCBConfig(model_id="invalid_object")

    def test_invalid_robot_uid(self):
        with pytest.raises(ValueError, match="robot_uids"):
            gym.make("ManiSkillPickYCBLift-v1", robot_uids="panda", **BASE_KWARGS)


class TestSharedConstants:
    def test_spawn_min_radius_matches_core(self):
        for robot_key, cfg in PICK_YCB_CONFIGS.items():
            assert cfg["spawn_min_radius"] == _CFG.spawn_min_radius

    def test_spawn_max_radius_matches_core(self):
        for robot_key, cfg in PICK_YCB_CONFIGS.items():
            assert cfg["spawn_max_radius"] == _CFG.spawn_max_radius


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
        env = _make_ycb_env("ManiSkillPickGolfBallLiftSO100-v1", **BASE_KWARGS)
        assert env.unwrapped.task_description[0].isupper()
        env.close()


class TestEpisodeLogic:
    EVALUATE_KEYS = {
        "is_grasped",
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
    def test_obj_spawns_in_radius_bounds(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        cfg = PICK_YCB_CONFIGS[robot]
        min_r = cfg["spawn_min_radius"]
        max_r = cfg["spawn_max_radius"]
        obj_p = env.unwrapped.obj.pose.p[0].cpu()
        r = float(torch.sqrt(obj_p[0] ** 2 + obj_p[1] ** 2))
        assert min_r <= r <= max_r

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
    def test_so100_lift_env_uses_so100(self, lift_so100_env):
        inner = lift_so100_env.unwrapped
        assert inner.robot_uids == "so100"

    def test_so101_lift_env_uses_so101(self, lift_so101_env):
        inner = lift_so101_env.unwrapped
        assert inner.robot_uids == "so101"


class TestCameraModes:
    @pytest.fixture(scope="class")
    def fixed_cam_env(self):
        env = _make_ycb_env(
            "ManiSkillPickGolfBallLiftSO100-v1",
            config=PickYCBConfig(camera_mode="fixed"),
            **BASE_KWARGS,
        )
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def wrist_cam_env(self):
        env = _make_ycb_env(
            "ManiSkillPickGolfBallLiftSO100-v1",
            config=PickYCBConfig(camera_mode="wrist"),
            **BASE_KWARGS,
        )
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def both_cam_env(self):
        env = _make_ycb_env(
            "ManiSkillPickGolfBallLiftSO100-v1",
            config=PickYCBConfig(camera_mode="both"),
            **BASE_KWARGS,
        )
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
