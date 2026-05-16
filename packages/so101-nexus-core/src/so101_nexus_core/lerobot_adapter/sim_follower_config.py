"""LeRobot robot config for the SO101-Nexus simulator follower."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lerobot.cameras import CameraConfig  # noqa: TC002
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("sim_so_follower")
@dataclass(kw_only=True)
class SimSOFollowerConfig(RobotConfig):
    """Configuration for a simulated SO follower controlled through LeRobot."""

    env_id: str
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Keep the SO body joints in LeRobot degree units by default. MolmoAct2
    # SO100/101 checkpoints were normalized against degree body joints plus a
    # RANGE_0_100 gripper, while ``False`` switches body joints to percent mode.
    use_degrees: bool = True
    max_relative_target: float | dict[str, float] | None = None
    disable_torque_on_disconnect: bool = True
