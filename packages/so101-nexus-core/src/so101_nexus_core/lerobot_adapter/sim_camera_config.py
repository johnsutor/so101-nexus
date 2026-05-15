"""LeRobot camera config for simulator-rendered frames."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.cameras import CameraConfig


@CameraConfig.register_subclass("sim")
@dataclass
class SimCameraConfig(CameraConfig):
    """Configuration for a camera rendered from the active simulator env."""

    source: str
