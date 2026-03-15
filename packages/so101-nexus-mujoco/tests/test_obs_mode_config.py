"""Tests for obs_mode config validation."""

import numpy as np
import pytest
from so101_nexus_core import PickConfig
from so101_nexus_core.config import EnvironmentConfig, PickAndPlaceConfig
from so101_nexus_mujoco.pick_env import PickLiftEnv
from so101_nexus_mujoco.pick_and_place import PickAndPlaceEnv
from so101_nexus_mujoco.reach_env import ReachConfig, ReachEnv
from so101_nexus_mujoco.look_at_env import LookAtConfig, LookAtEnv
from so101_nexus_mujoco.move_env import MoveConfig, MoveEnv


class TestObsModeConfig:
    def test_default_obs_mode_is_state(self):
        cfg = EnvironmentConfig()
        assert cfg.obs_mode == "state"

    def test_obs_mode_state_accepts_any_camera_mode(self):
        for cm in ("fixed", "wrist", "both"):
            cfg = EnvironmentConfig(obs_mode="state", camera_mode=cm)
            assert cfg.obs_mode == "state"

    def test_obs_mode_visual_requires_wrist(self):
        cfg = EnvironmentConfig(obs_mode="visual", camera_mode="wrist")
        assert cfg.obs_mode == "visual"

    def test_obs_mode_visual_rejects_fixed(self):
        with pytest.raises(ValueError, match="obs_mode.*visual.*requires.*camera_mode.*wrist"):
            EnvironmentConfig(obs_mode="visual", camera_mode="fixed")

    def test_obs_mode_visual_rejects_both(self):
        with pytest.raises(ValueError, match="obs_mode.*visual.*requires.*camera_mode.*wrist"):
            EnvironmentConfig(obs_mode="visual", camera_mode="both")

    def test_invalid_obs_mode_rejected(self):
        with pytest.raises(ValueError, match="obs_mode"):
            EnvironmentConfig(obs_mode="invalid")


class TestObsModeVisualPickEnv:
    """Use direct construction with camera_mode='wrist' — gym.make does not
    forward camera_mode from config to the env constructor parameter."""

    def test_visual_obs_state_is_6d(self):
        cfg = PickConfig(camera_mode="wrist", obs_mode="visual")
        env = PickLiftEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = PickConfig(camera_mode="wrist", obs_mode="visual")
        env = PickLiftEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert "privileged_state" in info
        # PickEnv privileged: tcp_pose(7) + is_grasped(1) + obj_pose(7) + tcp_to_obj(3) = 18
        assert info["privileged_state"].shape == (18,)
        env.close()

    def test_state_mode_unchanged(self):
        """Default obs_mode='state' still returns full privileged state vector."""
        cfg = PickConfig(camera_mode="wrist")
        env = PickLiftEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (18,)
        assert "privileged_state" not in info
        env.close()


class TestObsModeVisualPickAndPlace:
    def test_visual_obs_state_is_6d(self):
        cfg = PickAndPlaceConfig(camera_mode="wrist", obs_mode="visual")
        env = PickAndPlaceEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = PickAndPlaceConfig(camera_mode="wrist", obs_mode="visual")
        env = PickAndPlaceEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert "privileged_state" in info
        # PickAndPlace: tcp(7)+grasped(1)+target(3)+obj(7)+tcp_to_obj(3)+obj_to_target(3) = 24
        assert info["privileged_state"].shape == (24,)
        env.close()

    def test_state_mode_unchanged(self):
        cfg = PickAndPlaceConfig(camera_mode="wrist")
        env = PickAndPlaceEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert obs["state"].shape == (24,)
        assert "privileged_state" not in info
        env.close()


class TestObsModeVisualReachEnv:
    def test_visual_obs_state_is_6d(self):
        cfg = ReachConfig(camera_mode="wrist", obs_mode="visual")
        env = ReachEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = ReachConfig(camera_mode="wrist", obs_mode="visual")
        env = ReachEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert "privileged_state" in info
        assert info["privileged_state"].shape == (10,)
        env.close()


class TestObsModeVisualLookAtEnv:
    def test_visual_obs_state_is_6d(self):
        cfg = LookAtConfig(camera_mode="wrist", obs_mode="visual")
        env = LookAtEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = LookAtConfig(camera_mode="wrist", obs_mode="visual")
        env = LookAtEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert "privileged_state" in info
        assert info["privileged_state"].shape == (10,)
        env.close()


class TestObsModeVisualMoveEnv:
    def test_visual_obs_state_is_6d(self):
        cfg = MoveConfig(camera_mode="wrist", obs_mode="visual")
        env = MoveEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = MoveConfig(camera_mode="wrist", obs_mode="visual")
        env = MoveEnv(config=cfg, camera_mode="wrist")
        obs, info = env.reset()
        assert "privileged_state" in info
        assert info["privileged_state"].shape == (10,)
        env.close()
