from pathlib import Path

from so101_nexus_core.config import (  # noqa: I001
    DEFAULT_ENV_CONFIG as DEFAULT_ENV_CONFIG,
)
from so101_nexus_core.config import (
    CameraConfigSpec as CameraConfigSpec,
)
from so101_nexus_core.config import (
    EnvironmentConfig as EnvironmentConfig,
)
from so101_nexus_core.config import (
    RewardConfig as RewardConfig,
)
from so101_nexus_core.config import (
    RobotCameraPreset as RobotCameraPreset,
)
from so101_nexus_core.config import (
    TaskConfig as TaskConfig,
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


from so101_nexus_core.ycb_assets import (  # noqa: E402, I001
    ensure_ycb_assets as ensure_ycb_assets,
    get_ycb_collision_mesh as get_ycb_collision_mesh,
    get_ycb_mesh_dir as get_ycb_mesh_dir,
    get_ycb_visual_mesh as get_ycb_visual_mesh,
)
from so101_nexus_core.ycb_geometry import (  # noqa: E402, I001
    get_maniskill_ycb_spawn_z as get_maniskill_ycb_spawn_z,
    get_mujoco_ycb_rest_pose as get_mujoco_ycb_rest_pose,
)
