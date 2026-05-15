"""LeRobot plugin adapter for SO101-Nexus simulation environments.

Importing this package requires ``lerobot`` from the ``teleop`` extra. The import
runs LeRobot ChoiceRegistry decorators so ``--robot.type=sim_so_follower`` and
``--robot.cameras='{... type: sim ...}'`` can be parsed after plugin discovery.
"""

from __future__ import annotations

from so101_nexus_core.lerobot_adapter.sim_camera import SimCamera
from so101_nexus_core.lerobot_adapter.sim_camera_config import SimCameraConfig
from so101_nexus_core.lerobot_adapter.sim_follower import SimSOFollower
from so101_nexus_core.lerobot_adapter.sim_follower_config import SimSOFollowerConfig

__all__ = ["SimCamera", "SimCameraConfig", "SimSOFollower", "SimSOFollowerConfig"]
