"""Default ``DataProcessorPipeline`` factories for SO101-Nexus."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DataProcessorPipeline,
    DeviceProcessorStep,
    ProcessorStep,
    RenameObservationsProcessorStep,
)
from lerobot.processor.converters import create_transition

from so101_nexus_core.config import SO101_JOINT_NAMES
from so101_nexus_core.processors.action import (
    DegreesToRadiansActionStep,
    JointOffsetActionStep,
    LeaderActionToJointArrayStep,
)
from so101_nexus_core.processors.observation import Hwc2ChwImageObservationStep

if TYPE_CHECKING:
    from collections.abc import Iterable

    import gymnasium as gym
    import numpy as np


def _leader_action_to_transition(data: dict[str, Any]) -> dict[str, Any]:
    return create_transition(action=data["action"])


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


def infer_lerobot_rename_map(keys: Iterable[str]) -> dict[str, str]:
    """Infer the so101-nexus -> lerobot key rename map from a set of obs keys.

    - ``"state"`` becomes ``"observation.state"``.
    - ``"<name>_camera"`` becomes ``"observation.images.<name>"``.
    Other keys are left out of the map (they pass through unchanged).
    """
    rename: dict[str, str] = {}
    for key in keys:
        if key == "state":
            rename[key] = "observation.state"
        elif key.endswith("_camera"):
            camera_name = key[: -len("_camera")]
            rename[key] = f"observation.images.{camera_name}"
    return rename


def _env_observation_to_transition(data: dict[str, Any]) -> dict[str, Any]:
    return create_transition(observation=data["observation"])


def _env_observation_to_output(transition: dict[str, Any]) -> dict[str, Any]:
    return transition["observation"]


def make_default_env_observation_pipeline(
    observation_space: gym.spaces.Dict,
    *,
    device: str | Any = None,  # str | torch.device | None
    add_batch_dim: bool = False,
) -> DataProcessorPipeline:
    """Build the default env-observation pipeline.

    Steps:
    1. Rename ``state`` and ``*_camera`` keys to LeRobot conventions.
    2. Convert HWC uint8 images to CHW float32 tensors in ``[0, 1]``.
    3. (Optional) Add a leading batch dimension to state and image tensors.
    4. (Optional) Move tensors to ``device``.

    Parameters
    ----------
    observation_space
        The wrapped env's ``observation_space``. The rename map is inferred from
        its top-level keys.
    device
        If set, the default pipeline appends a ``DeviceProcessorStep`` to move
        tensors onto this device.
    add_batch_dim
        If true, the default pipeline appends an ``AddBatchDimensionProcessorStep``.
    """
    rename_map = infer_lerobot_rename_map(observation_space.spaces.keys())
    image_keys = tuple(v for k, v in rename_map.items() if k.endswith("_camera"))
    steps: list[ProcessorStep] = [
        RenameObservationsProcessorStep(rename_map=rename_map),
        Hwc2ChwImageObservationStep(image_keys=image_keys),
    ]
    if add_batch_dim:
        steps.append(AddBatchDimensionProcessorStep())
    if device is not None:
        steps.append(DeviceProcessorStep(device=str(device)))

    return DataProcessorPipeline(
        steps=steps,
        name="so101_env_observation_pipeline",
        to_transition=_env_observation_to_transition,
        to_output=_env_observation_to_output,
    )
