"""Task spec factories for primitive target environments."""

from so101_nexus_core.tasks.look_at import make_look_at_spec
from so101_nexus_core.tasks.move import make_move_spec
from so101_nexus_core.tasks.primitive_target import (
    NumpyContext,
    PrimitiveTargetSpec,
    TargetMarker,
    TorchContext,
    resolve_task_description,
)
from so101_nexus_core.tasks.reach import make_reach_spec

__all__ = [
    "NumpyContext",
    "PrimitiveTargetSpec",
    "TargetMarker",
    "TorchContext",
    "make_look_at_spec",
    "make_move_spec",
    "make_reach_spec",
    "resolve_task_description",
]
