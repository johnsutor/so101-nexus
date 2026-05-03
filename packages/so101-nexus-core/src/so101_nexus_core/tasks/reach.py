"""Reach primitive task: spec factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from so101_nexus_core.observations import TargetOffset
from so101_nexus_core.tasks.metrics import DistanceToTarget
from so101_nexus_core.tasks.primitive_target import PrimitiveTargetSpec, TargetMarker
from so101_nexus_core.tasks.samplers import WorkspaceTargetSampler
from so101_nexus_core.tasks.shapers import TanhDistanceShaper

if TYPE_CHECKING:
    from so101_nexus_core.config import ReachConfig


def make_reach_spec(config: ReachConfig) -> PrimitiveTargetSpec:  # noqa: D103
    return PrimitiveTargetSpec(
        marker=TargetMarker(
            name="reach_target",
            shape="sphere",
            size=config.target_radius,
            rgba=(1.0, 0.5, 0.0, 0.7),
            is_kinematic=True,
        ),
        sampler=WorkspaceTargetSampler(),
        metric=DistanceToTarget(),
        shaper=TanhDistanceShaper(scale=config.reward.tanh_shaping_scale),
        obs_extra=TargetOffset,
        task_description="Move the robot's end-effector to the target position.",
        requires_tcp_pos_for_sampling=False,
        requires_tcp_forward_for_metric=False,
    )
