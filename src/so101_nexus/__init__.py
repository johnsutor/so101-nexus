"""Public API for the so101-nexus library."""

from __future__ import annotations

from pathlib import Path

from so101_nexus.config import (
    DIRECTION_VECTORS,
    EXTENDED_POSE,
    POSES,
    REST_POSE,
    ROBOT_CAMERA_PRESETS,
    SO101_JOINT_NAMES,
    ControlMode,
    EnvironmentConfig,
    LookAtConfig,
    MoveConfig,
    MoveDirection,
    ObsMode,
    PickAndPlaceConfig,
    PickConfig,
    Pose,
    RenderConfig,
    RewardConfig,
    RobotCameraPreset,
    RobotConfig,
    TouchConfig,
    YcbModelId,
    describe_pick_target,
)
from so101_nexus.constants import (
    COLOR_MAP,
    YCB_OBJECTS,
    ColorConfig,
    ColorName,
    sample_color,
)
from so101_nexus.lerobot_dataset import (
    SO101_GRIPPER_LIMITS_RAD,
    dataset_row_to_sim_qpos,
    sim_qpos_to_dataset_row,
)
from so101_nexus.objects import (
    CubeObject,
    MeshObject,
    SceneObject,
    YCBObject,
)
from so101_nexus.observations import (
    CameraObservation,
    EndEffectorPose,
    GazeDirection,
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    Observation,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
)
from so101_nexus.rewards import (
    lift_progress,
    orientation_progress,
    reach_progress,
    simple_reward,
)
from so101_nexus.ycb_assets import (
    ensure_ycb_assets,
    get_ycb_collision_mesh,
    get_ycb_mesh_dir,
    get_ycb_texture_file,
    get_ycb_visual_mesh,
)
from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SO101_DIR = ASSETS_DIR / "SO101"


def get_so101_simulation_dir() -> Path:
    """Return the path to the SO101 simulation assets directory."""
    return SO101_DIR


def get_so101_mujoco_model_dir() -> Path:
    """Return the directory holding the vendored MuJoCo Menagerie SO101 model.

    The MuJoCo backend loads this model
    (``SO101_menagerie/so101.xml``). The URDF/XML under ``SO101/`` (see
    ``get_so101_simulation_dir``) remains only for teleop calibration metadata.
    """
    return ASSETS_DIR / "SO101_menagerie"


def get_so101_mujoco_model_path() -> Path:
    """Return the path to the MJCF model used by the MuJoCo backend (menagerie)."""
    return get_so101_mujoco_model_dir() / "so101.xml"


__all__ = [
    "ASSETS_DIR",
    "COLOR_MAP",
    "DIRECTION_VECTORS",
    "EXTENDED_POSE",
    "POSES",
    "REST_POSE",
    "ROBOT_CAMERA_PRESETS",
    "SO101_DIR",
    "SO101_GRIPPER_LIMITS_RAD",
    "SO101_JOINT_NAMES",
    "YCB_OBJECTS",
    "CameraObservation",
    "ColorConfig",
    "ColorName",
    "ControlMode",
    "CubeObject",
    "EndEffectorPose",
    "EnvironmentConfig",
    "GazeDirection",
    "GraspState",
    "JointPositions",
    "LookAtConfig",
    "MeshObject",
    "MoveConfig",
    "MoveDirection",
    "ObjectOffset",
    "ObjectPose",
    "ObsMode",
    "Observation",
    "OverheadCamera",
    "PickAndPlaceConfig",
    "PickConfig",
    "Pose",
    "RenderConfig",
    "RewardConfig",
    "RobotCameraPreset",
    "RobotConfig",
    "SceneObject",
    "TargetOffset",
    "TargetPosition",
    "TouchConfig",
    "WristCamera",
    "YCBObject",
    "YcbModelId",
    "dataset_row_to_sim_qpos",
    "describe_pick_target",
    "ensure_ycb_assets",
    "get_mujoco_ycb_rest_pose",
    "get_so101_mujoco_model_dir",
    "get_so101_mujoco_model_path",
    "get_so101_simulation_dir",
    "get_ycb_collision_mesh",
    "get_ycb_mesh_dir",
    "get_ycb_texture_file",
    "get_ycb_visual_mesh",
    "lift_progress",
    "orientation_progress",
    "reach_progress",
    "sample_color",
    "sim_qpos_to_dataset_row",
    "simple_reward",
]
