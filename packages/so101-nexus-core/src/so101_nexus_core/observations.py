"""Composable observation components for SO101-Nexus environments.

Observation components are lightweight descriptor classes that tell
environments which data to include in the observation vector. They
follow the same pattern as ``SceneObject`` subclasses: pure-data
config objects consumed by backend-specific environment code.

State components produce fixed-size slices of the flat observation
vector. Camera components add image tensors to a dict-style
observation space.

Typical usage::

    from so101_nexus_core.observations import JointPositions, TargetOffset

    config = ReachConfig(observations=[JointPositions(), TargetOffset()])
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Observation(ABC):
    """Abstract base for observation components.

    Every concrete component must define ``name`` (unique string key)
    and ``size`` (number of scalar dimensions for state components,
    or ``0`` for camera components whose shape depends on resolution).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this observation component."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of scalar dimensions (0 for camera components)."""

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}()"


class JointPositions(Observation):
    """Current angle of each robot joint (6-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "joint_positions"

    @property
    def size(self) -> int:  # noqa: D102
        return 6


class EndEffectorPose(Observation):
    """Gripper tip position and orientation in world coordinates (7-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "end_effector_pose"

    @property
    def size(self) -> int:  # noqa: D102
        return 7


class TargetOffset(Observation):
    """3D vector pointing from the gripper tip to the goal position (3-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "target_offset"

    @property
    def size(self) -> int:  # noqa: D102
        return 3


class GazeDirection(Observation):
    """Unit vector from the gripper tip toward the target object (3-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "gaze_direction"

    @property
    def size(self) -> int:  # noqa: D102
        return 3


class GraspState(Observation):
    """Whether the robot is currently holding an object: 1.0 = yes, 0.0 = no (1-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "grasp_state"

    @property
    def size(self) -> int:  # noqa: D102
        return 1


class ObjectPose(Observation):
    """Target object position and orientation in world coordinates (7-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "object_pose"

    @property
    def size(self) -> int:  # noqa: D102
        return 7


class ObjectOffset(Observation):
    """3D vector pointing from the gripper tip to the target object (3-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "object_offset"

    @property
    def size(self) -> int:  # noqa: D102
        return 3


class TargetPosition(Observation):
    """Absolute position (x, y, z) of the goal location in world coordinates (3-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "target_position"

    @property
    def size(self) -> int:  # noqa: D102
        return 3


# ---------------------------------------------------------------------------
# Camera components — add image tensors to dict-style observation spaces
# ---------------------------------------------------------------------------


class _CameraObservation(Observation):
    """Base for camera observation components.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
    """

    _name: str  # set by subclasses

    def __init__(self, width: int = 640, height: int = 480) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Camera dimensions must be > 0, got {width}x{height}")
        self.width = width
        self.height = height

    @property
    def name(self) -> str:
        return self._name

    @property
    def size(self) -> int:
        return 0

    def __repr__(self) -> str:
        return f"{type(self).__name__}(width={self.width}, height={self.height})"


class WristCamera(_CameraObservation):
    """RGB image from the camera mounted on the robot's wrist.

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    fov_deg_range : tuple[float, float]
        Min/max field-of-view in degrees for domain randomisation.
    pitch_deg_range : tuple[float, float]
        Min/max pitch angle in degrees for domain randomisation.
    pos_x_noise : float
        Noise magnitude for camera x-position.
    pos_y_center : float
        Nominal y-offset of the camera from the wrist.
    pos_y_noise : float
        Noise magnitude for camera y-position.
    pos_z_center : float
        Nominal z-offset of the camera from the wrist.
    pos_z_noise : float
        Noise magnitude for camera z-position.
    """

    _name = "wrist_camera"

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fov_deg_range: tuple[float, float] = (60.0, 90.0),
        pitch_deg_range: tuple[float, float] = (-34.4, 0.0),
        pos_x_noise: float = 0.005,
        pos_y_center: float = 0.04,
        pos_y_noise: float = 0.01,
        pos_z_center: float = -0.04,
        pos_z_noise: float = 0.01,
    ) -> None:
        super().__init__(width=width, height=height)
        self.fov_deg_range = fov_deg_range
        self.pitch_deg_range = pitch_deg_range
        self.pos_x_noise = pos_x_noise
        self.pos_y_center = pos_y_center
        self.pos_y_noise = pos_y_noise
        self.pos_z_center = pos_z_center
        self.pos_z_noise = pos_z_noise

    @property
    def fov_rad_range(self) -> tuple[float, float]:
        """Field-of-view range converted to radians."""
        return (
            float(np.radians(self.fov_deg_range[0])),
            float(np.radians(self.fov_deg_range[1])),
        )

    @property
    def pitch_rad_range(self) -> tuple[float, float]:
        """Pitch angle range converted to radians."""
        return (
            float(np.radians(self.pitch_deg_range[0])),
            float(np.radians(self.pitch_deg_range[1])),
        )


class OverheadCamera(_CameraObservation):
    """RGB image from the stationary camera above the workspace.

    Parameters
    ----------
    width : int
        Image width in pixels.
    height : int
        Image height in pixels.
    fov_deg : float
        Vertical field-of-view in degrees.
    """

    _name = "overhead_camera"

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fov_deg: float = 45.0,
    ) -> None:
        super().__init__(width=width, height=height)
        self.fov_deg = fov_deg
