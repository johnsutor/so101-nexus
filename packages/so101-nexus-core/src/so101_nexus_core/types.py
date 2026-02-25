from typing import Literal

CubeColorName = Literal["red", "orange", "yellow", "green", "blue", "purple", "black", "white"]

ControlMode = Literal["pd_joint_pos", "pd_joint_delta_pos", "pd_joint_target_delta_pos"]

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

DEFAULT_CUBE_HALF_SIZE: float = 0.0125
DEFAULT_CUBE_MASS: float = 0.01
DEFAULT_GOAL_THRESH: float = 0.025
DEFAULT_LIFT_THRESHOLD: float = 0.05
DEFAULT_MAX_GOAL_HEIGHT: float = 0.08
DEFAULT_CUBE_SPAWN_HALF_SIZE: float = 0.05
DEFAULT_MAX_EPISODE_STEPS: int = 256

REWARD_WEIGHT_REACHING: float = 0.25
REWARD_WEIGHT_GRASPING: float = 0.25
REWARD_WEIGHT_TASK_OBJECTIVE: float = 0.40
REWARD_WEIGHT_COMPLETION_BONUS: float = 0.10

SO101_REST_QPOS: list[float] = [0.0, -1.5708, 1.5708, 0.66, 0.0, -1.1]
SO101_JOINT_NAMES: list[str] = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

DEFAULT_GROUND_COLOR: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0)

DEFAULT_CAMERA_WIDTH: int = 224
DEFAULT_CAMERA_HEIGHT: int = 224
DEFAULT_WRIST_CAM_FOV_RANGE: tuple[float, float] = (60.0, 90.0)


def compute_normalized_reward(
    reach_progress: float,
    is_grasped: bool,
    task_progress: float,
    is_complete: bool,
) -> float:
    """Compute a normalized reward in [0, 1] using the standard reward budget.

    Parameters
    ----------
    reach_progress:
        Fraction of reaching completed, in [0, 1].
    is_grasped:
        Whether the object is currently grasped.
    task_progress:
        Task-specific progress fraction, in [0, 1] (e.g. lift height or placement accuracy).
    is_complete:
        Whether the task objective is fully achieved.

    Returns
    -------
    float
        Reward in [0, 1].
    """
    return (
        REWARD_WEIGHT_REACHING * reach_progress
        + REWARD_WEIGHT_GRASPING * float(is_grasped)
        + REWARD_WEIGHT_TASK_OBJECTIVE * task_progress
        + REWARD_WEIGHT_COMPLETION_BONUS * float(is_complete)
    )
