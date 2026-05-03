"""LookAt primitive task: spec factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from so101_nexus_core.observations import GazeDirection
from so101_nexus_core.tasks.metrics import OrientationError
from so101_nexus_core.tasks.primitive_target import PrimitiveTargetSpec, TargetMarker
from so101_nexus_core.tasks.samplers import FloorObjectSampler
from so101_nexus_core.tasks.shapers import CosineSimilarityShaper

if TYPE_CHECKING:
    from so101_nexus_core.config import LookAtConfig
    from so101_nexus_core.objects import CubeObject


def make_look_at_spec(config: LookAtConfig) -> PrimitiveTargetSpec:  # noqa: D103
    target_obj: CubeObject = config.objects[0]  # type: ignore[assignment]
    return PrimitiveTargetSpec(
        marker=TargetMarker(
            name="look_target",
            shape="cube",
            size=target_obj.half_size,
            rgba=(0.0, 0.0, 0.0, 1.0),
            is_kinematic=False,
            mass=target_obj.mass,
            color_name=target_obj.color,
        ),
        sampler=FloorObjectSampler(),
        metric=OrientationError(),
        shaper=CosineSimilarityShaper(),
        obs_extra=GazeDirection,
        task_description=lambda cfg: f"Look at the {cfg.objects[0]!r}.",
        requires_tcp_pos_for_sampling=False,
        requires_tcp_forward_for_metric=True,
    )
