"""Tests for obs_mode config validation."""

import pytest
from so101_nexus_core.config import EnvironmentConfig


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
