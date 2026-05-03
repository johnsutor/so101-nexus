"""Tests for SO101 observation processor steps."""

from __future__ import annotations

import numpy as np
import torch


def test_hwc_to_chw_converts_only_specified_image_keys() -> None:
    from so101_nexus_core.processors.observation import Hwc2ChwImageObservationStep

    obs = {
        "observation.images.wrist": np.full((4, 5, 3), 255, dtype=np.uint8),
        "observation.images.overhead": np.zeros((4, 5, 3), dtype=np.uint8),
        "observation.state": np.array([1.0, 2.0, 3.0]),
    }
    step = Hwc2ChwImageObservationStep(
        image_keys=("observation.images.wrist", "observation.images.overhead"),
    )

    out = step({"observation": obs})["observation"]

    wrist = out["observation.images.wrist"]
    overhead = out["observation.images.overhead"]
    state = out["observation.state"]

    assert isinstance(wrist, torch.Tensor)
    assert wrist.shape == (3, 4, 5)
    assert wrist.dtype == torch.float32
    assert torch.allclose(wrist, torch.ones_like(wrist))

    assert isinstance(overhead, torch.Tensor)
    assert overhead.shape == (3, 4, 5)
    assert torch.allclose(overhead, torch.zeros_like(overhead))

    np.testing.assert_array_equal(state, np.array([1.0, 2.0, 3.0]))


def test_hwc_to_chw_skips_keys_not_in_image_keys() -> None:
    from so101_nexus_core.processors.observation import Hwc2ChwImageObservationStep

    obs = {
        "observation.images.wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        "observation.images.overhead": np.zeros((2, 2, 3), dtype=np.uint8),
    }
    step = Hwc2ChwImageObservationStep(image_keys=("observation.images.wrist",))

    out = step({"observation": obs})["observation"]

    assert isinstance(out["observation.images.wrist"], torch.Tensor)
    assert isinstance(out["observation.images.overhead"], np.ndarray)


def test_hwc_to_chw_registered_in_registry() -> None:
    from lerobot.processor.pipeline import ProcessorStepRegistry

    from so101_nexus_core.processors.observation import Hwc2ChwImageObservationStep

    assert (
        ProcessorStepRegistry.get("so101_hwc_to_chw_image_observation")
        is Hwc2ChwImageObservationStep
    )
