"""LeRobot plugin adapter for SO101-Nexus simulation environments.

Importing this package requires ``lerobot`` from the ``teleop`` extra. The import
runs LeRobot ChoiceRegistry decorators so ``--robot.type=sim_so_follower`` and
``--robot.cameras='{... type: sim ...}'`` can be parsed after plugin discovery.
"""

from __future__ import annotations

from so101_nexus.lerobot_adapter.normalization import read_gripper_limits_rad
from so101_nexus.lerobot_adapter.sim_camera import SimCamera
from so101_nexus.lerobot_adapter.sim_camera_config import SimCameraConfig
from so101_nexus.lerobot_adapter.sim_follower import SimSOFollower
from so101_nexus.lerobot_adapter.sim_follower_config import SimSOFollowerConfig
from so101_nexus.lerobot_dataset import (
    SO101_GRIPPER_LIMITS_RAD,
    dataset_row_to_sim_qpos,
    sim_qpos_to_dataset_row,
)

__all__ = [
    "SO101_GRIPPER_LIMITS_RAD",
    "SimCamera",
    "SimCameraConfig",
    "SimSOFollower",
    "SimSOFollowerConfig",
    "dataset_row_to_sim_qpos",
    "read_gripper_limits_rad",
    "sim_qpos_to_dataset_row",
]
