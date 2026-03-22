"""Tests for obs_mode config validation."""

import pytest

from so101_nexus_core import PickConfig
from so101_nexus_core.config import EnvironmentConfig, PickAndPlaceConfig
from so101_nexus_core.observations import (
    EndEffectorPose,
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    TargetOffset,
    TargetPosition,
    WristCamera,
)
from so101_nexus_mujoco.look_at_env import LookAtConfig, LookAtEnv
from so101_nexus_mujoco.move_env import MoveConfig, MoveEnv
from so101_nexus_mujoco.pick_and_place import PickAndPlaceEnv
from so101_nexus_mujoco.pick_env import PickLiftEnv
from so101_nexus_mujoco.reach_env import ReachConfig, ReachEnv

# Default state observation components for each env type (matching test_e2e.py)
_PICK_STATE_OBS = [JointPositions, EndEffectorPose, GraspState, ObjectPose, ObjectOffset]
_PICK_AND_PLACE_STATE_OBS = [
    JointPositions,
    EndEffectorPose,
    GraspState,
    TargetPosition,
    ObjectPose,
    ObjectOffset,
    TargetOffset,
]
_REACH_STATE_OBS = [JointPositions, EndEffectorPose, TargetOffset]

# Sizes: sum of component sizes
# Pick: 6+7+1+7+3 = 24
_PICK_STATE_SIZE = 24
# PickAndPlace: 6+7+1+3+7+3+3 = 30
_PICK_AND_PLACE_STATE_SIZE = 30
# Reach: 6+7+3 = 16
_REACH_STATE_SIZE = 16


class TestObsModeConfig:
    def test_default_obs_mode_is_state(self):
        cfg = EnvironmentConfig()
        assert cfg.obs_mode == "state"

    def test_invalid_obs_mode_rejected(self):
        with pytest.raises(ValueError, match="obs_mode"):
            EnvironmentConfig(obs_mode="invalid")


class TestObsModeVisualPickEnv:
    """Visual obs_mode tests using WristCamera in observations."""

    def _pick_obs_with_camera(self):
        return [cls() for cls in _PICK_STATE_OBS] + [WristCamera(width=64, height=48)]

    def test_visual_obs_state_is_6d(self):
        cfg = PickConfig(
            obs_mode="visual",
            observations=self._pick_obs_with_camera(),
        )
        env = PickLiftEnv(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = PickConfig(
            obs_mode="visual",
            observations=self._pick_obs_with_camera(),
        )
        env = PickLiftEnv(config=cfg)
        obs, info = env.reset()
        assert "privileged_state" in info
        assert info["privileged_state"].shape == (_PICK_STATE_SIZE,)
        env.close()

    def test_state_mode_unchanged(self):
        """Default obs_mode='state' still returns full state vector."""
        cfg = PickConfig(
            observations=self._pick_obs_with_camera(),
        )
        env = PickLiftEnv(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (_PICK_STATE_SIZE,)
        assert "privileged_state" not in info
        env.close()


class TestObsModeVisualPickAndPlace:
    def _pnp_obs_with_camera(self):
        return [cls() for cls in _PICK_AND_PLACE_STATE_OBS] + [WristCamera(width=64, height=48)]

    def test_visual_obs_state_is_6d(self):
        cfg = PickAndPlaceConfig(
            obs_mode="visual",
            observations=self._pnp_obs_with_camera(),
        )
        env = PickAndPlaceEnv(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = PickAndPlaceConfig(
            obs_mode="visual",
            observations=self._pnp_obs_with_camera(),
        )
        env = PickAndPlaceEnv(config=cfg)
        obs, info = env.reset()
        assert "privileged_state" in info
        assert info["privileged_state"].shape == (_PICK_AND_PLACE_STATE_SIZE,)
        env.close()

    def test_state_mode_unchanged(self):
        cfg = PickAndPlaceConfig(
            observations=self._pnp_obs_with_camera(),
        )
        env = PickAndPlaceEnv(config=cfg)
        obs, info = env.reset()
        assert obs["state"].shape == (_PICK_AND_PLACE_STATE_SIZE,)
        assert "privileged_state" not in info
        env.close()


class TestObsModeVisualReachEnv:
    def _reach_obs_with_camera(self):
        return [cls() for cls in _REACH_STATE_OBS] + [WristCamera(width=64, height=48)]

    def test_visual_obs_state_is_6d(self):
        cfg = ReachConfig(
            obs_mode="visual",
            observations=self._reach_obs_with_camera(),
        )
        env = ReachEnv(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = ReachConfig(
            obs_mode="visual",
            observations=self._reach_obs_with_camera(),
        )
        env = ReachEnv(config=cfg)
        obs, info = env.reset()
        assert "privileged_state" in info
        assert info["privileged_state"].shape == (_REACH_STATE_SIZE,)
        env.close()


class TestObsModeVisualLookAtEnv:
    def test_visual_obs_state_is_6d(self):
        cfg = LookAtConfig(
            obs_mode="visual",
            observations=[JointPositions(), WristCamera(width=64, height=48)],
        )
        env = LookAtEnv(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = LookAtConfig(
            obs_mode="visual",
            observations=[JointPositions(), WristCamera(width=64, height=48)],
        )
        env = LookAtEnv(config=cfg)
        obs, info = env.reset()
        assert "privileged_state" in info
        assert info["privileged_state"].shape == (6,)
        env.close()


class TestObsModeVisualMoveEnv:
    def test_visual_obs_state_is_6d(self):
        cfg = MoveConfig(
            obs_mode="visual",
            observations=[JointPositions(), WristCamera(width=64, height=48)],
        )
        env = MoveEnv(config=cfg)
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        env.close()

    def test_visual_obs_privileged_state_in_info(self):
        cfg = MoveConfig(
            obs_mode="visual",
            observations=[JointPositions(), WristCamera(width=64, height=48)],
        )
        env = MoveEnv(config=cfg)
        obs, info = env.reset()
        assert "privileged_state" in info
        assert info["privileged_state"].shape == (6,)
        env.close()
