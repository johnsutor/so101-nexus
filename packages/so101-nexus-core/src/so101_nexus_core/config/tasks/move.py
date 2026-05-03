"""MoveConfig: config for the directional move primitive task."""

from __future__ import annotations

from so101_nexus_core.config._types import DIRECTION_VECTORS, MoveDirection
from so101_nexus_core.config.base import EnvironmentConfig
from so101_nexus_core.observations import JointPositions


class MoveConfig(EnvironmentConfig):
    """Config for the directional move primitive task.

    Args:
        direction: Cardinal direction to move the TCP.
        target_distance: Distance in metres to travel from the initial TCP position.
        success_threshold: Max residual distance (m) to count as success.
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        direction: MoveDirection = "up",
        target_distance: float = 0.10,
        success_threshold: float = 0.01,
        **kwargs,
    ) -> None:
        kwargs.setdefault("max_episode_steps", 256)
        if direction not in DIRECTION_VECTORS:
            raise ValueError(
                f"direction must be one of {list(DIRECTION_VECTORS)}, got {direction!r}"
            )
        super().__init__(**kwargs)
        self.direction = direction
        self.target_distance = target_distance
        self.success_threshold = success_threshold
        if self.observations is None:
            self.observations = [JointPositions()]
