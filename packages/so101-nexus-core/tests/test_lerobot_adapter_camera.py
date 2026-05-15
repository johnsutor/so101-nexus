"""Tests for simulator cameras exposed through the LeRobot camera interface."""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np
import pytest

pytest.importorskip("lerobot")


class _FakeEnv:
    def __init__(self) -> None:
        self.unwrapped = self

    def _get_obs(self) -> dict[str, Any]:
        return {
            "wrist_camera": np.full((6, 8, 3), 127, dtype=np.uint8),
            "sensor_data": {
                "overhead_camera": {"rgb": np.full((1, 5, 7, 3), 64, dtype=np.uint8)}
            },
        }

    def render(self) -> np.ndarray:
        return np.full((4, 9, 3), 32, dtype=np.uint8)


def test_sim_camera_connect_signature_matches_lerobot_base() -> None:
    from lerobot.cameras import Camera

    from so101_nexus_core.lerobot_adapter.sim_camera import SimCamera

    assert inspect.signature(SimCamera.connect) == inspect.signature(Camera.connect)


def test_sim_camera_reads_top_level_source() -> None:
    from so101_nexus_core.lerobot_adapter import SimCameraConfig
    from so101_nexus_core.lerobot_adapter.sim_camera import SimCamera

    camera = SimCamera(SimCameraConfig(source="wrist_camera", width=8, height=6, fps=30))
    camera.bind_env(_FakeEnv())
    camera.connect()

    frame = camera.read_latest()

    assert camera.is_connected
    assert frame.shape == (6, 8, 3)
    assert frame.dtype == np.uint8
    assert int(frame.mean()) == 127


def test_sim_camera_reads_maniskill_sensor_data() -> None:
    from so101_nexus_core.lerobot_adapter import SimCameraConfig
    from so101_nexus_core.lerobot_adapter.sim_camera import SimCamera

    camera = SimCamera(SimCameraConfig(source="overhead_camera", width=7, height=5, fps=30))
    camera.bind_env(_FakeEnv())
    camera.connect()

    frame = camera.async_read()

    assert frame.shape == (5, 7, 3)
    assert frame.dtype == np.uint8
    assert int(frame.mean()) == 64


def test_sim_camera_requires_env_before_connect() -> None:
    from so101_nexus_core.lerobot_adapter import SimCameraConfig
    from so101_nexus_core.lerobot_adapter.sim_camera import SimCamera

    camera = SimCamera(SimCameraConfig(source="wrist_camera", width=8, height=6, fps=30))

    with pytest.raises(RuntimeError, match="bind_env"):
        camera.connect()


def test_sim_camera_disconnect_clears_env_reference() -> None:
    from so101_nexus_core.lerobot_adapter import SimCameraConfig
    from so101_nexus_core.lerobot_adapter.sim_camera import SimCamera

    camera = SimCamera(SimCameraConfig(source="wrist_camera", width=8, height=6, fps=30))
    camera.bind_env(_FakeEnv())
    camera.connect()

    camera.disconnect()

    assert not camera.is_connected
    with pytest.raises(RuntimeError, match="not connected"):
        camera.read()


def test_sim_camera_validates_configured_shape() -> None:
    from so101_nexus_core.lerobot_adapter import SimCameraConfig
    from so101_nexus_core.lerobot_adapter.sim_camera import SimCamera

    camera = SimCamera(SimCameraConfig(source="wrist_camera", width=7, height=6, fps=30))
    camera.bind_env(_FakeEnv())
    camera.connect(warmup=False)

    with pytest.raises(ValueError, match="shape"):
        camera.read()
