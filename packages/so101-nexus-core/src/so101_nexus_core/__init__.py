"""Public API for the so101-nexus-core package."""

from pathlib import Path

from so101_nexus_core.config import (
    COLOR_MAP as COLOR_MAP,
)
from so101_nexus_core.config import (
    ROBOT_CAMERA_PRESETS as ROBOT_CAMERA_PRESETS,
)
from so101_nexus_core.config import (
    SO101_JOINT_NAMES as SO101_JOINT_NAMES,
)
from so101_nexus_core.config import (
    YCB_OBJECTS as YCB_OBJECTS,
)
from so101_nexus_core.config import (
    CameraConfig as CameraConfig,
)
from so101_nexus_core.config import (
    ColorConfig as ColorConfig,
)
from so101_nexus_core.config import (
    ColorName as ColorName,
)
from so101_nexus_core.config import (
    ControlMode as ControlMode,
)
from so101_nexus_core.config import (
    EnvironmentConfig as EnvironmentConfig,
)
from so101_nexus_core.config import (
    PickAndPlaceConfig as PickAndPlaceConfig,
)
from so101_nexus_core.config import (
    PickConfig as PickConfig,
)
from so101_nexus_core.config import (
    PickCubeConfig as PickCubeConfig,
)
from so101_nexus_core.config import (
    PickCubeMultipleConfig as PickCubeMultipleConfig,
)
from so101_nexus_core.config import (
    PickYCBConfig as PickYCBConfig,
)
from so101_nexus_core.config import (
    PickYCBMultipleConfig as PickYCBMultipleConfig,
)
from so101_nexus_core.config import (
    RewardConfig as RewardConfig,
)
from so101_nexus_core.config import (
    RobotCameraPreset as RobotCameraPreset,
)
from so101_nexus_core.config import (
    RobotConfig as RobotConfig,
)
from so101_nexus_core.config import (
    YCBEnvironmentConfig as YCBEnvironmentConfig,
)
from so101_nexus_core.config import (
    YcbModelId as YcbModelId,
)
from so101_nexus_core.config import (
    sample_color as sample_color,
)
from so101_nexus_core.objects import (  # noqa: F401
    CubeObject,
    MeshObject,
    SceneObject,
    YCBObject,
)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
SO_ARM100_DIR = ASSETS_DIR / "SO-ARM100"
SO101_DIR = ASSETS_DIR / "SO101"


def get_so101_simulation_dir() -> Path:
    """Return the path to the SO101 simulation assets directory."""
    return SO101_DIR


def get_so100_simulation_dir() -> Path:
    """Return the path to the SO100 simulation assets directory."""
    return SO_ARM100_DIR / "Simulation" / "SO100"


from so101_nexus_core.ycb_assets import (  # noqa: E402
    ensure_ycb_assets as ensure_ycb_assets,
)
from so101_nexus_core.ycb_assets import (  # noqa: E402
    get_ycb_collision_mesh as get_ycb_collision_mesh,
)
from so101_nexus_core.ycb_assets import (  # noqa: E402
    get_ycb_mesh_dir as get_ycb_mesh_dir,
)
from so101_nexus_core.ycb_assets import (  # noqa: E402
    get_ycb_visual_mesh as get_ycb_visual_mesh,
)
from so101_nexus_core.ycb_geometry import (  # noqa: E402
    get_maniskill_ycb_spawn_z as get_maniskill_ycb_spawn_z,
)
from so101_nexus_core.ycb_geometry import (  # noqa: E402
    get_mujoco_ycb_rest_pose as get_mujoco_ycb_rest_pose,
)
