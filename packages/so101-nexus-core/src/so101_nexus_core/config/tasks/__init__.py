"""Task-specific config subpackage."""

from so101_nexus_core.config.tasks.look_at import LookAtConfig
from so101_nexus_core.config.tasks.move import MoveConfig
from so101_nexus_core.config.tasks.pick import PickConfig
from so101_nexus_core.config.tasks.pick_and_place import PickAndPlaceConfig
from so101_nexus_core.config.tasks.reach import ReachConfig

__all__ = [
    "LookAtConfig",
    "MoveConfig",
    "PickAndPlaceConfig",
    "PickConfig",
    "ReachConfig",
]
