"""Observation processor steps for SO101-Nexus environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from lerobot.processor import ObservationProcessorStep, ProcessorStepRegistry

if TYPE_CHECKING:
    from lerobot.configs.types import PipelineFeatureType, PolicyFeature


@dataclass
@ProcessorStepRegistry.register(name="so101_hwc_to_chw_image_observation")
class Hwc2ChwImageObservationStep(ObservationProcessorStep):
    """Convert HWC ``uint8`` image observations to CHW ``float32`` tensors in [0, 1].

    Only the keys listed in ``image_keys`` are transformed; other observation
    entries pass through unchanged. This step is image-key-generic by design so it
    can be reused for any SO101-Nexus camera (current backends ship ``wrist`` and
    ``overhead``, but adding a new camera requires no step change).

    Parameters
    ----------
    image_keys
        Tuple of observation keys whose values are expected to be HWC ``uint8``
        ``np.ndarray`` images. Each is converted to a CHW ``torch.float32`` tensor
        normalized to ``[0, 1]``.
    """

    image_keys: tuple[str, ...] = field(default_factory=tuple)

    def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Convert HWC uint8 images at ``image_keys`` to CHW float32 tensors in [0, 1]."""
        for key in self.image_keys:
            if key not in observation:
                continue
            image = observation[key]
            if isinstance(image, np.ndarray):
                tensor = torch.from_numpy(image).to(torch.float32) / 255.0
            elif isinstance(image, torch.Tensor):
                tensor = image.to(torch.float32) / 255.0
            else:
                raise TypeError(
                    f"Image observation '{key}' has unsupported type {type(image)!r}; "
                    "expected np.ndarray or torch.Tensor."
                )
            if tensor.ndim != 3:
                raise ValueError(
                    f"Image observation '{key}' must be HWC (3D); got shape {tuple(tensor.shape)}."
                )
            observation[key] = tensor.permute(2, 0, 1).contiguous()
        return observation

    def get_config(self) -> dict[str, Any]:
        """Return init kwargs for serialization round-trips."""
        return {"image_keys": list(self.image_keys)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]],
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Pass features through unchanged; this step does not alter feature shapes."""
        return features
