"""Canonical, typed configuration objects for SO101-Nexus.

This module provides a HuggingFace-style configuration surface.
Each environment type gets its own config that inherits from a shared base.
Configs are shared between MuJoCo and ManiSkill backends.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Literal, Union

import numpy as np

if TYPE_CHECKING:
    from so101_nexus_core.objects import SceneObject

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


class CameraConfig:
    """Camera resolution, wrist field-of-view, and wrist mount randomization defaults.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        wrist_fov_deg_range: (min, max) wrist camera field-of-view in degrees.
        wrist_pitch_deg_range: (min, max) wrist camera pitch in degrees.
        wrist_cam_pos_x_noise: Noise on wrist camera x position.
        wrist_cam_pos_y_center: Center of wrist camera y position.
        wrist_cam_pos_y_noise: Noise on wrist camera y position.
        wrist_cam_pos_z_center: Center of wrist camera z position.
        wrist_cam_pos_z_noise: Noise on wrist camera z position.
    """

    def __init__(
        self,
        width: int = 224,
        height: int = 224,
        wrist_fov_deg_range: tuple[float, float] = (60.0, 90.0),
        wrist_pitch_deg_range: tuple[float, float] = (-34.4, 0.0),
        wrist_cam_pos_x_noise: float = 0.005,
        wrist_cam_pos_y_center: float = 0.04,
        wrist_cam_pos_y_noise: float = 0.01,
        wrist_cam_pos_z_center: float = -0.04,
        wrist_cam_pos_z_noise: float = 0.01,
    ) -> None:
        self.width = width
        self.height = height
        self.wrist_fov_deg_range = wrist_fov_deg_range
        self.wrist_pitch_deg_range = wrist_pitch_deg_range
        self.wrist_cam_pos_x_noise = wrist_cam_pos_x_noise
        self.wrist_cam_pos_y_center = wrist_cam_pos_y_center
        self.wrist_cam_pos_y_noise = wrist_cam_pos_y_noise
        self.wrist_cam_pos_z_center = wrist_cam_pos_z_center
        self.wrist_cam_pos_z_noise = wrist_cam_pos_z_noise
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"camera dimensions must be > 0, got {self.width}x{self.height}")

    @property
    def wrist_fov_rad_range(self) -> tuple[float, float]:
        """Wrist FOV range converted to radians."""
        lo, hi = self.wrist_fov_deg_range
        return (float(np.radians(lo)), float(np.radians(hi)))

    @property
    def wrist_pitch_rad_range(self) -> tuple[float, float]:
        """Wrist pitch range converted to radians."""
        lo, hi = self.wrist_pitch_deg_range
        return (float(np.radians(lo)), float(np.radians(hi)))

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"CameraConfig(width={self.width}, height={self.height}, "
            f"wrist_fov_deg_range={self.wrist_fov_deg_range})"
        )


class RobotConfig:
    """Configurable robot parameters.

    Joint names are intentionally not included here — they are structural
    identifiers that must match the URDF/MJCF and should not be overridden.

    Args:
        rest_qpos_deg: Rest joint positions in degrees.
        grasp_force_threshold: Force threshold for grasp detection.
        static_vel_threshold: Velocity threshold for static detection.
    """

    def __init__(
        self,
        rest_qpos_deg: tuple[float, ...] = (0.0, -90.0, 90.0, 37.8152144786, 0.0, -63.0253574644),
        grasp_force_threshold: float = 0.5,
        static_vel_threshold: float = 0.2,
    ) -> None:
        self.rest_qpos_deg = rest_qpos_deg
        self.grasp_force_threshold = grasp_force_threshold
        self.static_vel_threshold = static_vel_threshold

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
            f"RobotConfig(grasp_force_threshold={self.grasp_force_threshold}, "
            f"static_vel_threshold={self.static_vel_threshold})"
        )


class RobotCameraPreset:
    """Robot-specific camera and mounting parameters.

    Args:
        base_quat: Base quaternion (w, x, y, z).
        sensor_cam_eye_pos: Sensor camera eye position.
        sensor_cam_target_pos: Sensor camera target position.
        human_cam_eye_pos: Human camera eye position.
        human_cam_target_pos: Human camera target position.
        wrist_camera_mount_link: Link name for wrist camera mounting.
        wrist_cam_pos_center: Center position for wrist camera.
        wrist_cam_pos_noise: Position noise for wrist camera.
        wrist_cam_euler_center_deg: Center Euler angles in degrees.
        wrist_cam_euler_noise_deg: Euler angle noise in degrees.
    """

    def __init__(
        self,
        base_quat: tuple[float, float, float, float],
        sensor_cam_eye_pos: tuple[float, float, float],
        sensor_cam_target_pos: tuple[float, float, float],
        human_cam_eye_pos: tuple[float, float, float],
        human_cam_target_pos: tuple[float, float, float],
        wrist_camera_mount_link: str,
        wrist_cam_pos_center: tuple[float, float, float],
        wrist_cam_pos_noise: tuple[float, float, float],
        wrist_cam_euler_center_deg: tuple[float, float, float],
        wrist_cam_euler_noise_deg: tuple[float, float, float],
    ) -> None:
        self.base_quat = base_quat
        self.sensor_cam_eye_pos = sensor_cam_eye_pos
        self.sensor_cam_target_pos = sensor_cam_target_pos
        self.human_cam_eye_pos = human_cam_eye_pos
        self.human_cam_target_pos = human_cam_target_pos
        self.wrist_camera_mount_link = wrist_camera_mount_link
        self.wrist_cam_pos_center = wrist_cam_pos_center
        self.wrist_cam_pos_noise = wrist_cam_pos_noise
        self.wrist_cam_euler_center_deg = wrist_cam_euler_center_deg
        self.wrist_cam_euler_noise_deg = wrist_cam_euler_noise_deg

    @property
    def wrist_cam_euler_center_rad(self) -> tuple[float, float, float]:
        """Wrist camera Euler center angles converted to radians."""
        x, y, z = self.wrist_cam_euler_center_deg
        return (float(np.radians(x)), float(np.radians(y)), float(np.radians(z)))

    @property
    def wrist_cam_euler_noise_rad(self) -> tuple[float, float, float]:
        """Wrist camera Euler noise angles converted to radians."""
        x, y, z = self.wrist_cam_euler_noise_deg
        return (float(np.radians(x)), float(np.radians(y)), float(np.radians(z)))

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RobotCameraPreset(wrist_camera_mount_link={self.wrist_camera_mount_link!r}, "
            f"wrist_cam_euler_center_deg={self.wrist_cam_euler_center_deg})"
        )


class RewardConfig:
    """Normalized reward budget.

    The four component weights must sum to 1.0. Penalty terms are applied
    additively and subtracted from the base reward.

    Args:
        reaching: Weight for TCP-to-object distance shaping.
        grasping: Weight for grasp binary signal.
        task_objective: Weight for task-specific progress.
        completion_bonus: Weight for episode completion signal.
        action_delta_penalty: Penalty coefficient on L2 norm of consecutive action deltas.
        energy_penalty: Penalty coefficient on L2 norm of the action vector (energy cost).
        tanh_shaping_scale: Scale factor for tanh distance shaping.
    """

    def __init__(
        self,
        reaching: float = 0.25,
        grasping: float = 0.25,
        task_objective: float = 0.40,
        completion_bonus: float = 0.10,
        action_delta_penalty: float = 0.0,
        energy_penalty: float = 0.0,
        tanh_shaping_scale: float = 5.0,
    ) -> None:
        total = reaching + grasping + task_objective + completion_bonus
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Reward weights must sum to 1.0, got {total:.6f}")
        self.reaching = reaching
        self.grasping = grasping
        self.task_objective = task_objective
        self.completion_bonus = completion_bonus
        self.action_delta_penalty = action_delta_penalty
        self.energy_penalty = energy_penalty
        self.tanh_shaping_scale = tanh_shaping_scale

    def compute(
        self,
        reach_progress: float,
        is_grasped: bool,
        task_progress: float,
        is_complete: bool,
        action_delta_norm: float = 0.0,
        energy_norm: float = 0.0,
    ) -> float:
        """Compute a normalized reward using this config's weights."""
        base = (
            self.reaching * reach_progress
            + self.grasping * float(is_grasped)
            + self.task_objective * task_progress
            + self.completion_bonus * float(is_complete)
        )
        penalty = self.action_delta_penalty * action_delta_norm + self.energy_penalty * energy_norm
        return base - penalty

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RewardConfig(reaching={self.reaching}, grasping={self.grasping}, "
            f"task_objective={self.task_objective}, completion_bonus={self.completion_bonus}, "
            f"action_delta_penalty={self.action_delta_penalty}, "
            f"energy_penalty={self.energy_penalty})"
        )


class EnvironmentConfig:
    """Base config shared by all environments.

    Contains only parameters that every environment needs. Task-specific
    parameters live in subclass configs (PickConfig, etc.).

    Args:
        camera: Camera configuration.
        reward: Reward configuration.
        robot: Robot configuration.
        ground_colors: Ground plane color(s).
        max_episode_steps: Maximum steps per episode.
        goal_thresh: Distance threshold for goal achievement.
        spawn_half_size: Half-size of spawn region.
        spawn_center: Center of spawn region (x, y).
        spawn_min_radius: Minimum spawn radius.
        spawn_max_radius: Maximum spawn radius.
        spawn_angle_half_range_deg: Half angular range for spawn angle in degrees.
        camera_mode: Camera mode (fixed, wrist, or both).
        robot_colors: Robot arm color(s).
        robot_init_qpos_noise: Initial joint position noise.
    """

    def __init__(
        self,
        camera: CameraConfig | None = None,
        reward: RewardConfig | None = None,
        robot: RobotConfig | None = None,
        ground_colors: ColorConfig = "gray",
        max_episode_steps: int = 1024,
        goal_thresh: float = 0.025,
        spawn_half_size: float = 0.05,
        spawn_center: tuple[float, float] = (0.15, 0.0),
        spawn_min_radius: float = 0.10,
        spawn_max_radius: float = 0.25,
        spawn_angle_half_range_deg: float = 90.0,
        camera_mode: CameraMode = "fixed",
        robot_colors: ColorConfig = "yellow",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        self.camera = camera if camera is not None else CameraConfig()
        self.reward = reward if reward is not None else RewardConfig()
        self.robot = robot if robot is not None else RobotConfig()
        self.ground_colors = ground_colors
        self.max_episode_steps = max_episode_steps
        self.goal_thresh = goal_thresh
        self.spawn_half_size = spawn_half_size
        self.spawn_center = spawn_center
        self.spawn_min_radius = spawn_min_radius
        self.spawn_max_radius = spawn_max_radius
        self.spawn_angle_half_range_deg = spawn_angle_half_range_deg
        self.camera_mode = camera_mode
        self.robot_colors = robot_colors
        self.robot_init_qpos_noise = robot_init_qpos_noise
        if self.camera_mode not in ("fixed", "wrist", "both"):
            raise ValueError(f"camera_mode must be fixed|wrist|both, got {self.camera_mode!r}")
        if self.camera.width <= 0 or self.camera.height <= 0:
            raise ValueError(
                f"camera dimensions must be > 0, got {self.camera.width}x{self.camera.height}"
            )
        _validate_color_config(self.ground_colors, "ground_colors")
        _validate_color_config(self.robot_colors, "robot_colors")
        if self.spawn_min_radius < 0:
            raise ValueError(f"spawn_min_radius must be >= 0, got {self.spawn_min_radius}")
        if self.spawn_max_radius <= self.spawn_min_radius:
            raise ValueError(
                f"spawn_max_radius ({self.spawn_max_radius}) must be > "
                f"spawn_min_radius ({self.spawn_min_radius})"
            )
        if not (0.0 <= self.spawn_angle_half_range_deg <= 180.0):
            raise ValueError(
                f"spawn_angle_half_range_deg must be in [0, 180], "
                f"got {self.spawn_angle_half_range_deg}"
            )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{type(self).__name__}(max_episode_steps={self.max_episode_steps}, "
            f"goal_thresh={self.goal_thresh}, camera_mode={self.camera_mode!r})"
        )


class PickConfig(EnvironmentConfig):
    """Config for the unified pick environment.

    The ``objects`` list defines the pool of scene objects to sample from each
    episode. One object is chosen as the target; ``n_distractors`` additional
    objects are sampled from the remaining pool and placed as distractors.
    Task descriptions are auto-generated from each object's ``__repr__``.

    Args:
        objects: Pool of scene objects to sample from. Accepts a single ``SceneObject``,
            a list of ``SceneObject``, or ``None`` (defaults to ``[CubeObject()]``).
            A single object is automatically wrapped in a list.
        n_distractors: Number of distractor objects to place. 0 means single-object scene.
        lift_threshold: Minimum height above initial z to count as lifted.
        max_goal_height: Height cap used to normalize lift progress to [0, 1].
        min_object_separation: Minimum distance between spawned objects (metres).
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        objects: list[SceneObject] | SceneObject | None = None,
        n_distractors: int = 0,
        lift_threshold: float = 0.05,
        max_goal_height: float = 0.08,
        min_object_separation: float = 0.04,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        from so101_nexus_core.objects import CubeObject, SceneObject as _SceneObject  # noqa: PLC0415, I001

        if objects is None:
            self.objects: list[SceneObject] = [CubeObject()]
        elif isinstance(objects, _SceneObject):
            self.objects = [objects]
        else:
            self.objects = list(objects)
        self.n_distractors = n_distractors
        self.lift_threshold = lift_threshold
        self.max_goal_height = max_goal_height
        self.min_object_separation = min_object_separation
        if not self.objects:
            raise ValueError("objects must not be empty")
        if self.n_distractors < 0:
            raise ValueError(f"n_distractors must be >= 0, got {self.n_distractors}")
        if self.n_distractors > 0 and len(self.objects) < self.n_distractors + 1:
            raise ValueError(
                f"objects pool must have at least n_distractors+1={self.n_distractors + 1} "
                f"entries to support {self.n_distractors} distractors, got {len(self.objects)}"
            )

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PickConfig(objects={self.objects!r}, n_distractors={self.n_distractors}, "
            f"lift_threshold={self.lift_threshold}, max_goal_height={self.max_goal_height})"
        )


class PickAndPlaceConfig(EnvironmentConfig):
    """Config for pick-and-place environments.

    Args:
        cube_colors: Cube color(s).
        target_colors: Target disc color(s).
        cube_half_size: Half-size of the cube in metres.
        cube_mass: Mass of the cube in kg.
        target_disc_radius: Radius of the target disc.
        min_cube_target_separation: Minimum separation between cube and target.
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        cube_colors: ColorConfig = "red",
        target_colors: ColorConfig = "blue",
        cube_half_size: float = 0.0125,
        cube_mass: float = 0.01,
        target_disc_radius: float = 0.05,
        min_cube_target_separation: float = 0.0375,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cube_colors = cube_colors
        self.target_colors = target_colors
        self.cube_half_size = cube_half_size
        self.cube_mass = cube_mass
        self.target_disc_radius = target_disc_radius
        self.min_cube_target_separation = min_cube_target_separation
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

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PickAndPlaceConfig(cube_colors={self.cube_colors!r}, "
            f"target_colors={self.target_colors!r}, cube_half_size={self.cube_half_size})"
        )


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
