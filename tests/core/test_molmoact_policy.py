"""Tests for the MolmoAct2 chunked policy adapter."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest


@dataclass
class _PredictOutput:
    actions: np.ndarray


class _FakeMolmoModel:
    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = chunks
        self.calls: list[dict[str, Any]] = []

    def predict_action(self, **kwargs: Any) -> _PredictOutput:
        self.calls.append(kwargs)
        if not self._chunks:
            raise AssertionError("predict_action called more times than expected")
        return _PredictOutput(actions=self._chunks.pop(0))


def _batch() -> dict[str, Any]:
    return {
        "observation.state": np.array([0, 10, 20, 30, 40, 50], dtype=np.float64),
        "observation.images.overhead": np.full((4, 5, 3), 10, dtype=np.uint8),
        "observation.images.wrist": np.full((4, 5, 3), 20, dtype=np.uint8),
        "task": "pick the cube",
    }


def test_select_action_calls_model_with_images_in_configured_order_and_float32_state() -> None:
    from so101_nexus.policy_adapters import MolmoActPolicy

    chunk = np.array([[[1, 2, 3, 4, 5, 6]]], dtype=np.float64)
    model = _FakeMolmoModel([chunk])
    processor = object()
    policy = MolmoActPolicy(model, processor)
    batch = _batch()

    action = policy.select_action(batch)

    assert action.dtype == np.float32
    assert action.shape == (6,)
    assert action.tolist() == [1, 2, 3, 4, 5, 6]
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["processor"] is processor
    assert call["images"][0] is batch["observation.images.overhead"]
    assert call["images"][1] is batch["observation.images.wrist"]
    assert call["task"] == "pick the cube"
    assert call["state"].dtype == np.float32
    assert call["state"].tolist() == [0, 10, 20, 30, 40, 50]
    assert call["norm_tag"] == "so100_so101_molmoact2"
    assert call["action_mode"] == "continuous"
    assert call["enable_depth_reasoning"] is False
    assert call["num_steps"] == 10
    assert call["normalize_language"] is True
    assert call["enable_cuda_graph"] is True


def test_chunk_size_controls_requery_cadence() -> None:
    from so101_nexus.policy_adapters import MolmoActPolicy

    first_chunk = np.array(
        [
            [
                [1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3],
            ]
        ],
        dtype=np.float32,
    )
    second_chunk = np.array([[[4, 4, 4, 4, 4, 4]]], dtype=np.float32)
    model = _FakeMolmoModel([first_chunk, second_chunk])
    policy = MolmoActPolicy(model, object(), chunk_size=2)

    np.testing.assert_array_equal(policy.select_action(_batch()), np.full(6, 1, dtype=np.float32))
    np.testing.assert_array_equal(policy.select_action(_batch()), np.full(6, 2, dtype=np.float32))
    np.testing.assert_array_equal(policy.select_action(_batch()), np.full(6, 4, dtype=np.float32))

    assert len(model.calls) == 2


def test_reset_clears_cached_chunk() -> None:
    from so101_nexus.policy_adapters import MolmoActPolicy

    first_chunk = np.array(
        [
            [
                [1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2],
            ]
        ],
        dtype=np.float32,
    )
    second_chunk = np.array([[[9, 9, 9, 9, 9, 9]]], dtype=np.float32)
    model = _FakeMolmoModel([first_chunk, second_chunk])
    policy = MolmoActPolicy(model, object(), chunk_size=2)

    np.testing.assert_array_equal(policy.select_action(_batch()), np.full(6, 1, dtype=np.float32))
    policy.reset()
    np.testing.assert_array_equal(policy.select_action(_batch()), np.full(6, 9, dtype=np.float32))

    assert len(model.calls) == 2


def test_custom_image_keys_are_respected() -> None:
    from so101_nexus.policy_adapters import MolmoActPolicy

    batch = {
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.images.wrist": np.full((2, 2, 3), 1, dtype=np.uint8),
        "observation.images.overhead": np.full((2, 2, 3), 2, dtype=np.uint8),
        "task": "move",
    }
    model = _FakeMolmoModel([np.zeros((1, 1, 6), dtype=np.float32)])
    policy = MolmoActPolicy(
        model,
        object(),
        image_keys=("observation.images.wrist", "observation.images.overhead"),
    )

    policy.select_action(batch)

    call = model.calls[0]
    assert call["images"] == [
        batch["observation.images.wrist"],
        batch["observation.images.overhead"],
    ]


def test_discrete_mode_requires_action_tokenizer() -> None:
    from so101_nexus.policy_adapters import MolmoActPolicy

    with pytest.raises(ValueError, match="action_tokenizer"):
        MolmoActPolicy(object(), object(), action_mode="discrete")


def test_discrete_mode_passes_action_tokenizer() -> None:
    from so101_nexus.policy_adapters import MolmoActPolicy

    tokenizer = object()
    model = _FakeMolmoModel([np.zeros((1, 1, 6), dtype=np.float32)])
    policy = MolmoActPolicy(
        model,
        object(),
        action_mode="discrete",
        action_tokenizer=tokenizer,
    )

    policy.select_action(_batch())

    assert model.calls[0]["action_tokenizer"] is tokenizer


def test_importing_policy_adapters_does_not_require_molmoact_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for module_name in list(sys.modules):
        if module_name == "so101_nexus.policy_adapters" or module_name.startswith(
            "so101_nexus.policy_adapters."
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    module = importlib.import_module("so101_nexus.policy_adapters")

    assert module.MolmoActPolicy.__name__ == "MolmoActPolicy"


def test_from_pretrained_wraps_missing_molmoact_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from so101_nexus.policy_adapters import MolmoActPolicy

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    with pytest.raises(RuntimeError, match="molmoact"):
        MolmoActPolicy.from_pretrained()
