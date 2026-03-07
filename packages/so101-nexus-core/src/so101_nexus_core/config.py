"""Canonical, typed configuration objects for SO101-Nexus.

This module provides an immutable, HuggingFace-style configuration surface.
Each environment type gets its own config that inherits from a shared base.
Configs are shared between MuJoCo and ManiSkill backends.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Union

import numpy as np

ColorName = Literal["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "gray"]
ColorConfig = Union[ColorName, list[ColorName]]
ControlMode = Literal["pd_joint_pos", "pd_joint_delta_pos", "pd_joint_target_delta_pos"]
CameraMode = Literal["fixed", "wrist", "both"]

YcbModelId = Literal[
    "009_gelatin_box",
    "011_banana",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "037_scissors",
    "040_large_marker",
    "043_phillips_screwdriver",
    "058_golf_ball",
]

SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

COLOR_MAP: dict[str, list[float]] = {
    "red": [1.0, 0.0, 0.0, 1.0],
    "orange": [1.0, 0.5, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0, 1.0],
    "green": [0.0, 1.0, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
    "purple": [0.5, 0.0, 0.5, 1.0],
    "black": [0.0, 0.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0, 1.0],
    "gray": [0.5, 0.5, 0.5, 1.0],
}


def _validate_color_config(colors: ColorConfig, field_name: str) -> None:
    names = [colors] if isinstance(colors, str) else colors
    for name in names:
        if name not in COLOR_MAP:
            raise ValueError(f"{field_name} must be one of {list(COLOR_MAP)}, got {name!r}")


def sample_color(colors: ColorConfig, rng: np.random.Generator | None = None) -> list[float]:
    """Resolve a ColorConfig to an RGBA list. Samples uniformly if given a list."""
    if isinstance(colors, str):
        return COLOR_MAP[colors]
    if rng is None:
        rng = np.random.default_rng()
    chosen = rng.choice(colors)
    return COLOR_MAP[chosen]


@dataclass(frozen=True)
class CameraConfig:
    """Camera resolution and wrist field-of-view defaults."""

    width: int = 224
    height: int = 224
    wrist_fov_deg_range: tuple[float, float] = (60.0, 90.0)

    @property
    def wrist_fov_rad_range(self) -> tuple[float, float]:
        lo, hi = self.wrist_fov_deg_range
        return (float(np.radians(lo)), float(np.radians(hi)))


@dataclass(frozen=True)
class RobotConfig:
    """Configurable robot parameters.

    Joint names are intentionally not included here — they are structural
    identifiers that must match the URDF/MJCF and should not be overridden.
    """

    rest_qpos_deg: tuple[float, ...] = (0.0, -90.0, 90.0, 37.8152144786, 0.0, -63.0253574644)

    @property
    def rest_qpos_rad(self) -> tuple[float, ...]:
        return tuple(float(np.radians(v)) for v in self.rest_qpos_deg)

    @property
    def rest_qpos(self) -> tuple[float, ...]:
        """Backward-compatible alias returning radians."""
        return self.rest_qpos_rad


@dataclass(frozen=True)
class RewardConfig:
    """Normalized reward budget.

    The four component weights must sum to 1.0. The action_delta_penalty
    is applied separately: it scales the L2 norm of consecutive action
    differences and subtracts from the total (inspired by Walk These Ways).
    """

    reaching: float = 0.25
    grasping: float = 0.25
    task_objective: float = 0.40
    completion_bonus: float = 0.10
    action_delta_penalty: float = 0.0

    def compute(
        self,
        reach_progress: float,
        is_grasped: bool,
        task_progress: float,
        is_complete: bool,
        action_delta_norm: float = 0.0,
    ) -> float:
        """Compute a normalized reward in [0, 1] using this config's weights."""
        base = (
            self.reaching * reach_progress
            + self.grasping * float(is_grasped)
            + self.task_objective * task_progress
            + self.completion_bonus * float(is_complete)
        )
        return base - self.action_delta_penalty * action_delta_norm


@dataclass(frozen=True)
class RobotCameraPreset:
    """Robot-specific camera and mounting parameters."""

    base_quat: tuple[float, float, float, float]
    sensor_cam_eye_pos: tuple[float, float, float]
    sensor_cam_target_pos: tuple[float, float, float]
    human_cam_eye_pos: tuple[float, float, float]
    human_cam_target_pos: tuple[float, float, float]
    wrist_camera_mount_link: str
    wrist_cam_pos_center: tuple[float, float, float]
    wrist_cam_pos_noise: tuple[float, float, float]
    wrist_cam_euler_center_deg: tuple[float, float, float]
    wrist_cam_euler_noise_deg: tuple[float, float, float]

    @property
    def wrist_cam_euler_center_rad(self) -> tuple[float, float, float]:
        x, y, z = self.wrist_cam_euler_center_deg
        return (float(np.radians(x)), float(np.radians(y)), float(np.radians(z)))

    @property
    def wrist_cam_euler_noise_rad(self) -> tuple[float, float, float]:
        x, y, z = self.wrist_cam_euler_noise_deg
        return (float(np.radians(x)), float(np.radians(y)), float(np.radians(z)))


@dataclass(frozen=True)
class EnvironmentConfig:
    """Base config shared by all environments.

    Contains only parameters that every environment needs. Task-specific
    parameters live in subclass configs (PickCubeConfig, etc.).
    """

    camera: CameraConfig = CameraConfig()
    reward: RewardConfig = RewardConfig()
    robot: RobotConfig = RobotConfig()
    ground_colors: ColorConfig = "gray"
    max_episode_steps: int = 256
    goal_thresh: float = 0.025
    spawn_half_size: float = 0.05
    spawn_center: tuple[float, float] = (0.15, 0.0)
    camera_mode: CameraMode = "fixed"
    robot_colors: ColorConfig = "yellow"
    robot_init_qpos_noise: float = 0.02

    def __post_init__(self):
        if self.camera_mode not in ("fixed", "wrist", "both"):
            raise ValueError(f"camera_mode must be fixed|wrist|both, got {self.camera_mode!r}")
        if self.camera.width <= 0 or self.camera.height <= 0:
            raise ValueError(
                f"camera dimensions must be > 0, got {self.camera.width}x{self.camera.height}"
            )
        _validate_color_config(self.ground_colors, "ground_colors")
        _validate_color_config(self.robot_colors, "robot_colors")


@dataclass(frozen=True)
class PickCubeConfig(EnvironmentConfig):
    """Config for pick-cube and pick-cube-lift environments."""

    cube_colors: ColorConfig = "red"
    cube_half_size: float = 0.0125
    cube_mass: float = 0.01
    lift_threshold: float = 0.05
    max_goal_height: float = 0.08

    def __post_init__(self):
        super().__post_init__()
        _validate_color_config(self.cube_colors, "cube_colors")
        if not (0.01 <= self.cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {self.cube_half_size}")


@dataclass(frozen=True)
class PickAndPlaceConfig(EnvironmentConfig):
    """Config for pick-and-place environments."""

    cube_colors: ColorConfig = "red"
    target_colors: ColorConfig = "blue"
    cube_half_size: float = 0.0125
    cube_mass: float = 0.01
    target_disc_radius: float = 0.05
    min_cube_target_separation: float = 0.0375

    def __post_init__(self):
        super().__post_init__()
        _validate_color_config(self.cube_colors, "cube_colors")
        _validate_color_config(self.target_colors, "target_colors")
        cube_set = (
            {self.cube_colors} if isinstance(self.cube_colors, str) else set(self.cube_colors)
        )
        target_set = (
            {self.target_colors} if isinstance(self.target_colors, str) else set(self.target_colors)
        )
        overlap = cube_set & target_set
        if overlap:
            warnings.warn(
                f"cube_colors and target_colors overlap on {overlap}; "
                "the cube and target may be the same color in some episodes",
                stacklevel=2,
            )
        if not (0.01 <= self.cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {self.cube_half_size}")


@dataclass(frozen=True)
class PickYCBConfig(EnvironmentConfig):
    """Config for pick-YCB and pick-YCB-lift environments."""

    model_id: YcbModelId = "058_golf_ball"
    lift_threshold: float = 0.05
    max_goal_height: float = 0.08

    def __post_init__(self):
        super().__post_init__()
        if self.model_id not in YCB_OBJECTS:
            raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {self.model_id!r}")


@dataclass(frozen=True)
class PickCubeMultipleConfig(EnvironmentConfig):
    """Config for pick-cube-multiple environments with distractor cubes."""

    cube_color: ColorName = "red"
    cube_half_size: float = 0.0125
    cube_mass: float = 0.01
    lift_threshold: float = 0.05
    max_goal_height: float = 0.08
    num_distractors: int = 3
    min_object_separation: float = 0.04

    def __post_init__(self):
        super().__post_init__()
        if self.cube_color not in CUBE_COLOR_MAP:
            raise ValueError(
                f"cube_color must be one of {list(CUBE_COLOR_MAP)}, got {self.cube_color!r}"
            )
        if not (0.01 <= self.cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {self.cube_half_size}")
        if self.num_distractors < 1:
            raise ValueError(f"num_distractors must be >= 1, got {self.num_distractors}")


@dataclass(frozen=True)
class PickYCBMultipleConfig(EnvironmentConfig):
    """Config for pick-YCB-multiple environments with distractor YCB objects."""

    model_id: YcbModelId = "058_golf_ball"
    lift_threshold: float = 0.05
    max_goal_height: float = 0.08
    num_distractors: int = 3
    min_object_separation: float = 0.04

    def __post_init__(self):
        super().__post_init__()
        if self.model_id not in YCB_OBJECTS:
            raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {self.model_id!r}")
        if self.num_distractors < 1:
            raise ValueError(f"num_distractors must be >= 1, got {self.num_distractors}")


CUBE_COLOR_MAP: dict[str, list[float]] = {
    "red": [1.0, 0.0, 0.0, 1.0],
    "orange": [1.0, 0.5, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0, 1.0],
    "green": [0.0, 1.0, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
    "purple": [0.5, 0.0, 0.5, 1.0],
    "black": [0.0, 0.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0, 1.0],
}

TARGET_COLOR_MAP: dict[str, list[float]] = CUBE_COLOR_MAP

YCB_OBJECTS: dict[str, str] = {
    "009_gelatin_box": "gelatin box",
    "011_banana": "banana",
    "030_fork": "fork",
    "031_spoon": "spoon",
    "032_knife": "knife",
    "033_spatula": "spatula",
    "037_scissors": "scissors",
    "040_large_marker": "large marker",
    "043_phillips_screwdriver": "phillips screwdriver",
    "058_golf_ball": "golf ball",
}

YCB_ENV_NAME_MAP: dict[YcbModelId, str] = {
    "009_gelatin_box": "GelatinBox",
    "011_banana": "Banana",
    "030_fork": "Fork",
    "031_spoon": "Spoon",
    "032_knife": "Knife",
    "033_spatula": "Spatula",
    "037_scissors": "Scissors",
    "040_large_marker": "LargeMarker",
    "043_phillips_screwdriver": "PhillipsScrewdriver",
    "058_golf_ball": "GolfBall",
}


SQRT_HALF = float(np.sqrt(0.5))

ROBOT_CAMERA_PRESETS: dict[str, RobotCameraPreset] = {
    "so100": RobotCameraPreset(
        base_quat=(SQRT_HALF, 0.0, 0.0, SQRT_HALF),
        sensor_cam_eye_pos=(0.0, 0.3, 0.3),
        sensor_cam_target_pos=(0.15, 0.0, 0.02),
        human_cam_eye_pos=(0.0, 0.4, 0.4),
        human_cam_target_pos=(0.15, 0.0, 0.05),
        wrist_camera_mount_link="Fixed_Jaw",
        wrist_cam_pos_center=(0.0, -0.045, -0.045),
        wrist_cam_pos_noise=(0.0, 0.015, 0.015),
        wrist_cam_euler_center_deg=(-180.0, -37.5, -90.0),
        wrist_cam_euler_noise_deg=(0.0, 7.5, 0.0),
    ),
    "so101": RobotCameraPreset(
        base_quat=(1.0, 0.0, 0.0, 0.0),
        sensor_cam_eye_pos=(0.0, 0.3, 0.3),
        sensor_cam_target_pos=(0.15, 0.0, 0.02),
        human_cam_eye_pos=(0.0, 0.4, 0.4),
        human_cam_target_pos=(0.15, 0.0, 0.05),
        wrist_camera_mount_link="gripper_link",
        wrist_cam_pos_center=(0.0, 0.04, -0.04),
        wrist_cam_pos_noise=(0.005, 0.01, 0.01),
        wrist_cam_euler_center_deg=(-180.0, 37.5, -90.0),
        wrist_cam_euler_noise_deg=(0.0, 11.4591559026, 0.0),
    ),
}
