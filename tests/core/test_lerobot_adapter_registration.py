"""Tests for LeRobot adapter registration."""

from __future__ import annotations

import pytest

pytest.importorskip("lerobot")


def test_adapter_registers_robot_and_camera_configs() -> None:
    from lerobot.cameras import CameraConfig
    from lerobot.robots.config import RobotConfig

    from so101_nexus import lerobot_adapter as _adapter
    from so101_nexus.lerobot_adapter import SimCameraConfig, SimSOFollowerConfig

    assert _adapter is not None
    assert RobotConfig.get_choice_class("sim_so_follower") is SimSOFollowerConfig
    assert CameraConfig.get_choice_class("sim") is SimCameraConfig


def test_explicit_discover_packages_path_imports_adapter() -> None:
    """LeRobot's parser loads plugin packages before draccus parses configs."""
    from lerobot.configs.parser import load_plugin
    from lerobot.robots.config import RobotConfig

    load_plugin("so101_nexus.lerobot_adapter")

    assert RobotConfig.get_choice_class("sim_so_follower").__name__ == "SimSOFollowerConfig"


def test_sim_follower_defaults_to_degree_body_joints(tmp_path) -> None:
    from so101_nexus.lerobot_adapter import SimSOFollowerConfig

    config = SimSOFollowerConfig(
        id="sim_default_units",
        calibration_dir=tmp_path,
        env_id="MuJoCoReach-v1",
    )

    assert config.use_degrees is True
