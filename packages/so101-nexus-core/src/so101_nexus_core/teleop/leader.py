"""Leader-arm factory and env/robot validation helpers for teleop.

All ``lerobot`` imports are deferred into :func:`get_leader` so this module
can be imported without the ``teleop`` extra installed.
"""

from __future__ import annotations

import importlib
from typing import Protocol

from so101_nexus_core.config import SO101_JOINT_NAMES

DEFAULT_WRIST_ROLL_OFFSET_DEG = -90.0

ROBOT_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "so100": SO101_JOINT_NAMES,
    "so101": SO101_JOINT_NAMES,
}


class LeaderProtocol(Protocol):
    """Minimal interface the teleop loop requires from a leader-arm driver."""

    def connect(self) -> None:
        """Open the connection to the leader arm."""
        ...

    def disconnect(self) -> None:
        """Close the connection to the leader arm."""
        ...

    def get_action(self) -> dict:
        """Return the latest joint readings from the leader arm."""
        ...


def get_leader(robot_type: str, port: str, leader_id: str) -> LeaderProtocol:
    """Create and return the appropriate ``SOLeader`` for *robot_type*."""
    if robot_type == "so100":
        from lerobot.teleoperators.so_leader.config_so_leader import SO100LeaderConfig
        from lerobot.teleoperators.so_leader.so_leader import SO100Leader

        return SO100Leader(SO100LeaderConfig(port=port, use_degrees=True, id=leader_id))

    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    return SO101Leader(SO101LeaderConfig(port=port, use_degrees=True, id=leader_id))


def check_robot_env_mismatch(env_id: str, robot_type: str) -> str | None:
    """Return a warning string if *env_id* encodes a robot that contradicts *robot_type*."""
    upper = env_id.upper()
    if robot_type == "so100" and "SO101" in upper:
        return f"Robot type is so100 but env '{env_id}' appears to target SO101."
    if robot_type == "so101" and "SO100" in upper:
        return f"Robot type is so101 but env '{env_id}' appears to target SO100."
    return None


def import_backend_for_env_id(env_id: str) -> None:
    """Import the simulator backend that matches the *env_id* prefix."""
    if env_id.startswith("ManiSkill"):
        importlib.import_module("so101_nexus_maniskill")
    elif env_id.startswith("MuJoCo"):
        importlib.import_module("so101_nexus_mujoco")
    else:
        raise ValueError("env_id must start with 'ManiSkill' or 'MuJoCo'")
