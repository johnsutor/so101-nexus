"""Move primitive task: spec factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from so101_nexus_core.observations import TargetOffset
from so101_nexus_core.tasks.metrics import DistanceToTarget
from so101_nexus_core.tasks.primitive_target import PrimitiveTargetSpec, TargetMarker
from so101_nexus_core.tasks.samplers import TcpRelativeSampler
from so101_nexus_core.tasks.shapers import TanhDistanceShaper

if TYPE_CHECKING:
    from so101_nexus_core.config import MoveConfig


def make_move_spec(config: MoveConfig) -> PrimitiveTargetSpec:  # noqa: D103
    return PrimitiveTargetSpec(
        marker=TargetMarker(
            name="move_target",
            shape="sphere",
            size=0.015,
            rgba=(0.0, 0.8, 0.2, 0.7),
            is_kinematic=True,
        ),
        sampler=TcpRelativeSampler(),
        metric=DistanceToTarget(),
        shaper=TanhDistanceShaper(scale=config.reward.tanh_shaping_scale),
        obs_extra=TargetOffset,
        task_description=(
            lambda cfg: f"Move the end-effector {cfg.direction} by {cfg.target_distance:.2f} m."
        ),
        requires_tcp_pos_for_sampling=True,
        requires_tcp_forward_for_metric=False,
    )
