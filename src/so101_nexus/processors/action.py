"""Action processor steps for SO101 leader-arm input."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from lerobot.processor import ActionProcessorStep, ProcessorStepRegistry

from so101_nexus.config import SO101_JOINT_NAMES

if TYPE_CHECKING:
    from lerobot.configs.types import PipelineFeatureType, PolicyFeature
    from lerobot.processor import EnvAction, PolicyAction, RobotAction


@dataclass
@ProcessorStepRegistry.register(name="so101_leader_action_to_joint_array")
class LeaderActionToJointArrayStep(ActionProcessorStep):
    """Convert a leader-arm action dict to an ordered numpy array.

    The leader publishes a dict of ``{joint_name}.pos`` floats (degrees by default
    on SO leaders). This step gathers values in the order specified by
    ``joint_names`` and returns a ``np.ndarray`` of shape ``(len(joint_names),)``
    with dtype ``np.float64``. The output unit is preserved (degrees in, degrees
    out); a separate step performs the deg-to-rad conversion.

    Parameters
    ----------
    joint_names
        Tuple of joint names defining the output array order. Defaults to
        :data:`so101_nexus.config.SO101_JOINT_NAMES`.
    """

    joint_names: tuple[str, ...] = field(default_factory=lambda: SO101_JOINT_NAMES)

    def action(
        self,
        action: PolicyAction | RobotAction | EnvAction,
    ) -> PolicyAction | RobotAction | EnvAction:
        """Convert the leader-arm dict to an ordered ndarray of joint values."""
        leader = cast("RobotAction", action)
        return np.array(
            [leader[f"{name}.pos"] for name in self.joint_names],
            dtype=np.float64,
        )

    def get_config(self) -> dict[str, Any]:
        """Return init kwargs for serialization round-trips."""
        return {"joint_names": list(self.joint_names)}

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass features through unchanged; this step does not alter feature shapes."""
        return features


@dataclass
@ProcessorStepRegistry.register(name="so101_degrees_to_radians_action")
class DegreesToRadiansActionStep(ActionProcessorStep):
    """Convert an action vector from degrees to radians.

    Operates on a numpy array; the action is assumed to already be ordered (typically
    after :class:`LeaderActionToJointArrayStep`).
    """

    def action(
        self,
        action: PolicyAction | RobotAction | EnvAction,
    ) -> PolicyAction | RobotAction | EnvAction:
        """Convert the action vector from degrees to radians."""
        arr = cast("EnvAction", action)
        return np.deg2rad(arr)

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass features through unchanged; this step does not alter feature shapes."""
        return features


@dataclass
@ProcessorStepRegistry.register(name="so101_joint_offset_action")
class JointOffsetActionStep(ActionProcessorStep):
    """Add a constant offset (in radians) to a single joint index of the action vector.

    Generic over the target joint so the step can be reused for any per-joint
    calibration shift, not just the SO101 wrist_roll. The action vector is assumed
    to already be in radians by the time this step runs.

    Parameters
    ----------
    joint_index
        Zero-based index of the joint in the action vector.
    offset_rad
        Constant offset to add, in radians.
    """

    joint_index: int = 0
    offset_rad: float = 0.0

    def action(
        self,
        action: PolicyAction | RobotAction | EnvAction,
    ) -> PolicyAction | RobotAction | EnvAction:
        """Add ``offset_rad`` to ``joint_index`` of a copy of the action vector."""
        arr = cast("EnvAction", action)
        out = arr.copy()
        out[self.joint_index] = out[self.joint_index] + self.offset_rad
        return out

    def get_config(self) -> dict[str, Any]:
        """Return init kwargs for serialization round-trips."""
        return {"joint_index": self.joint_index, "offset_rad": self.offset_rad}

    def transform_features(
        self,
        features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass features through unchanged; this step does not alter feature shapes."""
        return features
