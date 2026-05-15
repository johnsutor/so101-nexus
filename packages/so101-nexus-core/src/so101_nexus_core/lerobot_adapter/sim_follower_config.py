"""LeRobot robot config for the SO101-Nexus simulator follower."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("sim_so_follower")
@dataclass
class SimSOFollowerConfig(RobotConfig):
    """Configuration for a simulated SO follower controlled through LeRobot."""

    env_id: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    use_degrees: bool = True
    max_relative_target: float | dict[str, float] | None = None
    disable_torque_on_disconnect: bool = True
