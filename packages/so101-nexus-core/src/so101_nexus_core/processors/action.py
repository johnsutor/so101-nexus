"""Action processor steps for SO101 leader-arm input."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from lerobot.processor import ActionProcessorStep, ProcessorStepRegistry

from so101_nexus_core.config import SO101_JOINT_NAMES

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
        :data:`so101_nexus_core.config.SO101_JOINT_NAMES`.
    """

    joint_names: tuple[str, ...] = field(default_factory=lambda: SO101_JOINT_NAMES)

    def action(
        self, action: PolicyAction | RobotAction | EnvAction,
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
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass features through unchanged; this step does not alter feature shapes."""
        return features
