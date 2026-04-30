"""Default ``DataProcessorPipeline`` factories for SO101-Nexus."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from lerobot.processor import DataProcessorPipeline

from so101_nexus_core.config import SO101_JOINT_NAMES
from so101_nexus_core.processors.action import (
    DegreesToRadiansActionStep,
    JointOffsetActionStep,
    LeaderActionToJointArrayStep,
)

if TYPE_CHECKING:
    import numpy as np


def _leader_action_to_transition(data: dict[str, Any]) -> dict[str, Any]:
    return {"action": data["action"]}


def _leader_action_to_output(transition: dict[str, Any]) -> "np.ndarray":  # noqa: UP037
    return transition["action"]


def make_default_leader_action_pipeline(
    joint_names: tuple[str, ...] = SO101_JOINT_NAMES,
    wrist_roll_offset_deg: float = -90.0,
) -> DataProcessorPipeline:
    """Build the default leader-arm action pipeline.

    Parameters
    ----------
    joint_names
        Joint names in the order of the output action vector. Must include
        ``"wrist_roll"`` (the calibration shift is applied at this index).
    wrist_roll_offset_deg
        Calibration offset applied to ``wrist_roll`` after the deg-to-rad
        conversion. Defaults to ``-90.0`` (degrees), matching the previous
        ``convert_leader_action`` helper.

    Returns
    -------
    DataProcessorPipeline
        Pipeline that accepts ``{"action": leader_dict}`` and returns a
        ``np.ndarray`` of shape ``(len(joint_names),)`` in radians, with the
        wrist-roll offset applied.
    """
    if "wrist_roll" not in joint_names:
        raise ValueError(
            "joint_names must include 'wrist_roll' for the default pipeline; "
            "build a custom pipeline if your joint set differs."
        )
    wrist_roll_index = joint_names.index("wrist_roll")
    return DataProcessorPipeline(
        steps=[
            LeaderActionToJointArrayStep(joint_names=joint_names),
            DegreesToRadiansActionStep(),
            JointOffsetActionStep(
                joint_index=wrist_roll_index,
                offset_rad=math.radians(wrist_roll_offset_deg),
            ),
        ],
        name="so101_leader_action_pipeline",
        to_transition=_leader_action_to_transition,
        to_output=_leader_action_to_output,
    )
