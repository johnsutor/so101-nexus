from typing import Literal

from so101_nexus_core.config import (
    DEFAULT_ENV_CONFIG,
    ColorName,
)

CubeColorName = ColorName
TargetColorName = ColorName

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

DEFAULT_TARGET_DISC_RADIUS: float = DEFAULT_ENV_CONFIG.task.target_disc_radius
DEFAULT_MIN_CUBE_TARGET_SEPARATION: float = DEFAULT_ENV_CONFIG.task.min_cube_target_separation

DEFAULT_CUBE_HALF_SIZE: float = DEFAULT_ENV_CONFIG.task.cube_half_size
DEFAULT_CUBE_MASS: float = DEFAULT_ENV_CONFIG.task.cube_mass
DEFAULT_GOAL_THRESH: float = DEFAULT_ENV_CONFIG.task.goal_thresh
DEFAULT_LIFT_THRESHOLD: float = DEFAULT_ENV_CONFIG.task.lift_threshold
DEFAULT_MAX_GOAL_HEIGHT: float = DEFAULT_ENV_CONFIG.task.max_goal_height
DEFAULT_CUBE_SPAWN_HALF_SIZE: float = DEFAULT_ENV_CONFIG.task.cube_spawn_half_size
DEFAULT_MAX_EPISODE_STEPS: int = DEFAULT_ENV_CONFIG.task.max_episode_steps

REWARD_WEIGHT_REACHING: float = DEFAULT_ENV_CONFIG.reward.reaching
REWARD_WEIGHT_GRASPING: float = DEFAULT_ENV_CONFIG.reward.grasping
REWARD_WEIGHT_TASK_OBJECTIVE: float = DEFAULT_ENV_CONFIG.reward.task_objective
REWARD_WEIGHT_COMPLETION_BONUS: float = DEFAULT_ENV_CONFIG.reward.completion_bonus

SO101_REST_QPOS: list[float] = [0.0, -1.5708, 1.5708, 0.66, 0.0, -1.1]
SO101_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_GROUND_COLOR: tuple[float, float, float, float] = DEFAULT_ENV_CONFIG.ground_color

DEFAULT_CAMERA_WIDTH: int = DEFAULT_ENV_CONFIG.camera.width
DEFAULT_CAMERA_HEIGHT: int = DEFAULT_ENV_CONFIG.camera.height
DEFAULT_WRIST_CAM_FOV_RANGE: tuple[float, float] = DEFAULT_ENV_CONFIG.camera.wrist_fov_deg_range
DEFAULT_WRIST_CAM_FOV_DEG_RANGE: tuple[float, float] = DEFAULT_ENV_CONFIG.camera.wrist_fov_deg_range
DEFAULT_WRIST_CAM_FOV_RAD_RANGE: tuple[float, float] = DEFAULT_ENV_CONFIG.camera.wrist_fov_rad_range


def compute_normalized_reward(
    reach_progress: float,
    is_grasped: bool,
    task_progress: float,
    is_complete: bool,
) -> float:
    """Compute a normalized reward in [0, 1] using the standard reward budget."""
    return (
        REWARD_WEIGHT_REACHING * reach_progress
        + REWARD_WEIGHT_GRASPING * float(is_grasped)
        + REWARD_WEIGHT_TASK_OBJECTIVE * task_progress
        + REWARD_WEIGHT_COMPLETION_BONUS * float(is_complete)
    )
