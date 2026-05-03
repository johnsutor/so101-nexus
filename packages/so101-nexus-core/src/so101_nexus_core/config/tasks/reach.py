"""ReachConfig: config for the reach-to-target primitive task."""

from __future__ import annotations

from so101_nexus_core.config.base import EnvironmentConfig
from so101_nexus_core.observations import JointPositions


class ReachConfig(EnvironmentConfig):
    """Config for the reach-to-target primitive task.

    Args:
        target_radius: Visual radius of the target site sphere (metres).
        target_workspace_half_extent: Half-width of the cubic workspace to
            sample target positions from (metres).
        success_threshold: TCP-to-target distance (m) that counts as success.
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        target_radius: float = 0.02,
        target_workspace_half_extent: float = 0.15,
        success_threshold: float = 0.02,
        **kwargs,
    ) -> None:
        kwargs.setdefault("max_episode_steps", 512)
        super().__init__(**kwargs)
        self.target_radius = target_radius
        self.target_workspace_half_extent = target_workspace_half_extent
        self.success_threshold = success_threshold
        if self.observations is None:
            self.observations = [JointPositions()]
