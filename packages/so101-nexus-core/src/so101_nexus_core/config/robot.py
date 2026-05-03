"""RobotConfig: configurable robot parameters (not joint names; those live in URDF)."""

from __future__ import annotations

import numpy as np

from so101_nexus_core.config.pose import POSES, Pose


class RobotConfig:
    """Configurable robot parameters.

    Joint names are intentionally not included here — they are structural
    identifiers that must match the URDF/MJCF and should not be overridden.

    Parameters
    ----------
    rest_qpos_deg : tuple[float, ...]
        Rest joint positions in degrees.
    init_pose : str | Pose | None
        Initial pose for resets. A string looks up from ``POSES``,
        a ``Pose`` instance is used directly, ``None`` uses the legacy
        ``rest_qpos_deg`` + noise path.
    grasp_force_threshold : float
        Force threshold for grasp detection.
    static_vel_threshold : float
        Velocity threshold for static detection.
    """

    def __init__(
        self,
        rest_qpos_deg: tuple[float, ...] = (0.0, -90.0, 90.0, 37.8152144786, 0.0, -63.0253574644),
        init_pose: str | Pose | None = None,
        grasp_force_threshold: float = 0.5,
        static_vel_threshold: float = 0.2,
    ) -> None:
        self.rest_qpos_deg = rest_qpos_deg
        self.grasp_force_threshold = grasp_force_threshold
        self.static_vel_threshold = static_vel_threshold
        if len(self.rest_qpos_deg) != 6:
            raise ValueError(
                f"rest_qpos_deg must have exactly 6 elements, got {len(self.rest_qpos_deg)}"
            )
        if isinstance(init_pose, str) and init_pose not in POSES:
            raise ValueError(f"Unknown pose name {init_pose!r}. Available: {list(POSES)}")
        self.init_pose: str | Pose | None = init_pose

    def resolve_pose(self) -> Pose | None:
        """Return the resolved Pose object, or None if not set."""
        if self.init_pose is None:
            return None
        if isinstance(self.init_pose, Pose):
            return self.init_pose
        return POSES[self.init_pose]

    @property
    def rest_qpos_rad(self) -> tuple[float, ...]:
        """Rest joint positions in radians."""
        return tuple(float(np.radians(v)) for v in self.rest_qpos_deg)

    @property
    def rest_qpos(self) -> tuple[float, ...]:
        """Backward-compatible alias returning radians."""
        return self.rest_qpos_rad

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RobotConfig(init_pose={self.init_pose!r}, "
            f"grasp_force_threshold={self.grasp_force_threshold}, "
            f"static_vel_threshold={self.static_vel_threshold})"
        )
