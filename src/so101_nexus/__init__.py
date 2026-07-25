"""Public API for the so101-nexus library."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

try:
    __version__ = _package_version("so101-nexus")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

from so101_nexus.config import (
    ABSOLUTE_CONTROL_MODES,
    DELTA_CONTROL_MODES,
    DIRECTION_VECTORS,
    EE_CONTROL_MODES,
    EXTENDED_POSE,
    JOINT_CONTROL_MODES,
    POSES,
    REST_POSE,
    ROBOT_CAMERA_PRESETS,
    SO101_JOINT_NAMES,
    SO101_TCP_FRAME_NAME,
    SO101_TCP_SITE_NAME,
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
    StackCubeConfig,
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
from so101_nexus.kinematics import (
    EE_ACTION_DIM,
    EE_DELTA_ACTION_SCALE,
    EE_ORIENTATION_WEIGHT,
)
from so101_nexus.lerobot_dataset import (
    SO101_GRIPPER_LIMITS_RAD,
    dataset_row_to_sim_qpos,
    relabel_environment_state,
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
    JointVelocities,
    ObjectOffset,
    ObjectPose,
    Observation,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
    privileged_state_feature_names,
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


def get_so101_urdf_path() -> Path:
    """Return the path to the SO101 URDF used by LeRobot kinematics tooling.

    The URDF carries two tool frames. ``gripper_frame_link`` is the upstream
    LeRobot default and sits at the fixed fingertip. ``tcp_frame_link``
    (:data:`so101_nexus.config.SO101_TCP_FRAME_NAME`) matches the ``gripperframe``
    site of the MJCF model the simulator backends step, so end-effector poses
    resolve identically in simulation and on hardware.
    """
    return SO101_DIR / "so101_new_calib.urdf"


__all__ = [
    "ABSOLUTE_CONTROL_MODES",
    "ASSETS_DIR",
    "COLOR_MAP",
    "DELTA_CONTROL_MODES",
    "DIRECTION_VECTORS",
    "EE_ACTION_DIM",
    "EE_CONTROL_MODES",
    "EE_DELTA_ACTION_SCALE",
    "EE_ORIENTATION_WEIGHT",
    "EXTENDED_POSE",
    "JOINT_CONTROL_MODES",
    "POSES",
    "REST_POSE",
    "ROBOT_CAMERA_PRESETS",
    "SO101_DIR",
    "SO101_GRIPPER_LIMITS_RAD",
    "SO101_JOINT_NAMES",
    "SO101_TCP_FRAME_NAME",
    "SO101_TCP_SITE_NAME",
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
    "JointVelocities",
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
    "StackCubeConfig",
    "TargetOffset",
    "TargetPosition",
    "TouchConfig",
    "WristCamera",
    "YCBObject",
    "YcbModelId",
    "__version__",
    "dataset_row_to_sim_qpos",
    "describe_pick_target",
    "ensure_ycb_assets",
    "get_mujoco_ycb_rest_pose",
    "get_so101_mujoco_model_dir",
    "get_so101_mujoco_model_path",
    "get_so101_simulation_dir",
    "get_so101_urdf_path",
    "get_ycb_collision_mesh",
    "get_ycb_mesh_dir",
    "get_ycb_texture_file",
    "get_ycb_visual_mesh",
    "lift_progress",
    "orientation_progress",
    "privileged_state_feature_names",
    "reach_progress",
    "relabel_environment_state",
    "sample_color",
    "sim_qpos_to_dataset_row",
    "simple_reward",
]
