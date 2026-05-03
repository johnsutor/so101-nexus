"""Canonical, typed configuration objects for SO101-Nexus.

This package provides a HuggingFace-style configuration surface.
Each environment type gets its own config that inherits from a shared base.
Configs are shared between MuJoCo and ManiSkill backends.
"""

from so101_nexus_core.config._types import (
    DIRECTION_VECTORS,
    SO101_JOINT_NAMES,
    ControlMode,
    JointSpec,
    MoveDirection,
    ObsMode,
    YcbModelId,
)
from so101_nexus_core.config.base import EnvironmentConfig
from so101_nexus_core.config.cameras import ROBOT_CAMERA_PRESETS, SQRT_HALF, RobotCameraPreset
from so101_nexus_core.config.pose import EXTENDED_POSE, POSES, REST_POSE, Pose
from so101_nexus_core.config.render import RenderConfig
from so101_nexus_core.config.reward import RewardConfig
from so101_nexus_core.config.robot import RobotConfig
from so101_nexus_core.config.tasks import (
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    ReachConfig,
)

__all__ = [
    "DIRECTION_VECTORS",
    "EXTENDED_POSE",
    "POSES",
    "REST_POSE",
    "ROBOT_CAMERA_PRESETS",
    "SO101_JOINT_NAMES",
    "SQRT_HALF",
    "ControlMode",
    "EnvironmentConfig",
    "JointSpec",
    "LookAtConfig",
    "MoveConfig",
    "MoveDirection",
    "ObsMode",
    "PickAndPlaceConfig",
    "PickConfig",
    "Pose",
    "ReachConfig",
    "RenderConfig",
    "RewardConfig",
    "RobotCameraPreset",
    "RobotConfig",
    "YcbModelId",
]
