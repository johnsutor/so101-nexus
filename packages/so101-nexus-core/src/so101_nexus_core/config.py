"""Canonical, typed configuration objects for SO101-Nexus.

This module provides an immutable, HuggingFace-style configuration surface:
compose one `EnvironmentConfig` and derive legacy constants/presets from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ColorName = Literal["red", "orange", "yellow", "green", "blue", "purple", "black", "white"]
ControlMode = Literal["pd_joint_pos", "pd_joint_delta_pos", "pd_joint_target_delta_pos"]


@dataclass(frozen=True)
class RewardConfig:
    """Normalized reward budget used by all environments.

    The defaults sum to 1.0 and are interpreted as:
    reaching + grasping + task objective + completion bonus.
    """

    reaching: float = 0.25
    grasping: float = 0.25
    task_objective: float = 0.40
    completion_bonus: float = 0.10


@dataclass(frozen=True)
class CameraConfigSpec:
    """Shared camera defaults across backends."""

    width: int = 224
    height: int = 224
    wrist_fov_deg_range: tuple[float, float] = (60.0, 90.0)

    @property
    def wrist_fov_rad_range(self) -> tuple[float, float]:
        lo, hi = self.wrist_fov_deg_range
        return (float(np.radians(lo)), float(np.radians(hi)))


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
    wrist_cam_euler_center: tuple[float, float, float]
    wrist_cam_euler_noise: tuple[float, float, float]


@dataclass(frozen=True)
class TaskConfig:
    """Task-space defaults shared by pick, place, and YCB variants."""

    cube_half_size: float = 0.0125
    cube_mass: float = 0.01
    goal_thresh: float = 0.025
    lift_threshold: float = 0.05
    max_goal_height: float = 0.08
    cube_spawn_half_size: float = 0.05
    cube_spawn_center: tuple[float, float] = (0.15, 0.0)
    target_disc_radius: float = 0.05
    min_cube_target_separation: float = 0.0375
    max_episode_steps: int = 256


@dataclass(frozen=True)
class EnvironmentConfig:
    """Top-level base package config (HF-style).

    Consumers should treat this as the canonical source of defaults and derive
    backend-specific dictionaries/constants from it.
    """

    task: TaskConfig = TaskConfig()
    camera: CameraConfigSpec = CameraConfigSpec()
    reward: RewardConfig = RewardConfig()
    ground_color: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)


DEFAULT_ENV_CONFIG = EnvironmentConfig()


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

YCB_ENV_NAME_MAP: dict[str, str] = {
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
        wrist_cam_euler_center=(-np.pi, np.radians(-37.5), np.radians(-90.0)),
        wrist_cam_euler_noise=(0.0, np.radians(7.5), 0.0),
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
        wrist_cam_euler_center=(-np.pi, np.radians(37.5), np.radians(-90.0)),
        wrist_cam_euler_noise=(0.0, 0.2, 0.0),
    ),
}
