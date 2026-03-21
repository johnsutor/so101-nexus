"""Tests for the unified ManiSkill pick environment (PickEnv / PickLiftEnv)."""

from pathlib import Path

import gymnasium as gym
import pytest
import torch
from mani_skill import ASSET_DIR

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import PickConfig
from so101_nexus_core.objects import CubeObject, YCBObject
from so101_nexus_maniskill.pick_env import (
    PickLiftSO100Env,
    PickLiftSO101Env,
)

BASE_KWARGS = {"obs_mode": "state", "num_envs": 1, "render_mode": None}

LIFT_ENV_IDS = [
    ("ManiSkillPickLiftSO100-v1", "so100"),
    ("ManiSkillPickLiftSO101-v1", "so101"),
]
ALL_ENV_IDS = LIFT_ENV_IDS


def _has_ycb_assets(asset_root: Path | None = None) -> bool:
    root = asset_root if asset_root is not None else ASSET_DIR
    manifest = root / "assets" / "mani_skill2_ycb" / "info_pick_v0.json"
    return manifest.exists()


@pytest.fixture(scope="module")
def lift_so100_env():
    env = gym.make("ManiSkillPickLiftSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def lift_so101_env():
    env = gym.make("ManiSkillPickLiftSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillPickLiftSO100-v1": "lift_so100_env",
        "ManiSkillPickLiftSO101-v1": "lift_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestConstructionValidation:
    def test_invalid_robot_uid(self):
        with pytest.raises(ValueError, match="robot_uids"):
            gym.make("ManiSkillPickLiftSO100-v1", robot_uids="panda", **BASE_KWARGS)

    def test_empty_objects_raises(self):
        with pytest.raises(ValueError, match="objects must not be empty"):
            PickConfig(objects=[])

    def test_mesh_object_raises_not_supported(self):
        """MeshObject is not supported on the ManiSkill backend."""
        from so101_nexus_core.objects import MeshObject

        obj = MeshObject(
            collision_mesh_path="/tmp/fake.obj",
            visual_mesh_path="/tmp/fake.obj",
            mass=0.01,
            name="fake mesh",
        )
        cfg = PickConfig(objects=[obj])
        with pytest.raises(TypeError, match="Unsupported object type"):
            gym.make("ManiSkillPickLiftSO100-v1", config=cfg, **BASE_KWARGS)


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
    def test_task_description_nonempty(self, lift_so100_env):
        assert lift_so100_env.unwrapped.task_description

    def test_task_description_starts_with_capital(self, lift_so100_env):
        assert lift_so100_env.unwrapped.task_description[0].isupper()

    def test_task_description_cube_includes_color(self):
        cfg = PickConfig(objects=[CubeObject(color="green")])
        env = gym.make("ManiSkillPickLiftSO100-v1", config=cfg, **BASE_KWARGS)
        assert "green" in env.unwrapped.task_description
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

    @pytest.mark.parametrize("env_id,robot", LIFT_ENV_IDS)
    def test_reward_range_lift(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        action = env.action_space.sample()
        _, reward, _, _, _ = env.step(action)
        assert (reward >= 0).all()
        assert (reward <= 1).all()

    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_obj_spawns_in_radius_bounds(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        inner = env.unwrapped
        min_r = inner._robot_cfg["spawn_min_radius"]
        max_r = inner._robot_cfg["spawn_max_radius"]
        obj_p = inner.obj.pose.p[0].cpu()
        r = float(torch.sqrt(obj_p[0] ** 2 + obj_p[1] ** 2))
        assert min_r <= r <= max_r


class TestRobotOrientation:
    @pytest.mark.parametrize("env_id,robot", ALL_ENV_IDS)
    def test_robot_base_uses_keyframe_rotation(self, request, env_id, robot):
        import numpy as np

        env = _get_env(request, env_id)
        env.reset()
        inner = env.unwrapped
        expected_q = inner.agent.keyframes["rest"].pose.q
        actual_q = inner.agent.robot.pose.q[0].cpu().numpy()
        np.testing.assert_allclose(actual_q, expected_q, atol=1e-4)


class TestRobotSubclasses:
    def test_so100_lift_env_uses_so100(self, lift_so100_env):
        inner = lift_so100_env.unwrapped
        assert isinstance(inner, PickLiftSO100Env)
        assert inner.robot_uids == "so100"

    def test_so101_lift_env_uses_so101(self, lift_so101_env):
        inner = lift_so101_env.unwrapped
        assert isinstance(inner, PickLiftSO101Env)
        assert inner.robot_uids == "so101"


class TestCameraModes:
    @pytest.fixture(scope="class")
    def fixed_cam_env(self):
        env = gym.make(
            "ManiSkillPickLiftSO100-v1",
            config=PickConfig(camera_mode="fixed"),
            **BASE_KWARGS,
        )
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def wrist_cam_env(self):
        env = gym.make(
            "ManiSkillPickLiftSO100-v1",
            config=PickConfig(camera_mode="wrist"),
            **BASE_KWARGS,
        )
        env.reset()
        yield env
        env.close()

    @pytest.fixture(scope="class")
    def both_cam_env(self):
        env = gym.make(
            "ManiSkillPickLiftSO100-v1",
            config=PickConfig(camera_mode="both"),
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


class TestNoTable:
    def test_no_table_scene_builder(self, lift_so101_env):
        inner = lift_so101_env.unwrapped
        assert not hasattr(inner, "table_scene"), "Should not have a table_scene attribute"

    def test_robot_base_at_origin(self, lift_so101_env):
        inner = lift_so101_env.unwrapped
        inner.reset()
        base_pos = inner.agent.robot.pose.p[0].cpu()
        assert base_pos[0].item() == pytest.approx(0.0, abs=0.01)
        assert base_pos[1].item() == pytest.approx(0.0, abs=0.01)
        assert base_pos[2].item() == pytest.approx(0.0, abs=0.01)


class TestYCBObjects:
    """Tests for YCB object support in the unified pick env."""

    def test_ycb_object_skips_if_assets_missing(self):
        if not _has_ycb_assets():
            pytest.skip(
                "Missing ManiSkill YCB assets at "
                f"{ASSET_DIR / 'assets' / 'mani_skill2_ycb' / 'info_pick_v0.json'}"
            )
        ycb_obj = YCBObject(model_id="011_banana")
        cfg = PickConfig(objects=[ycb_obj])
        env = gym.make("ManiSkillPickLiftSO100-v1", config=cfg, **BASE_KWARGS)
        obs, info = env.reset()
        assert isinstance(obs, torch.Tensor)
        assert "banana" in env.unwrapped.task_description.lower()
        env.close()
