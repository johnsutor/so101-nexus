"""Tests for model factories and LeRobot checkpoint inference adapters."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from so101_nexus.policy_adapters.collection import LeRobotPolicyAdapter, load_collection_policy


def test_custom_factory_accepts_structural_policy(monkeypatch):
    policy = SimpleNamespace(reset=lambda: None, select_action=lambda batch: np.zeros(6))
    monkeypatch.setitem(sys.modules, "collection_example", SimpleNamespace(build=lambda: policy))
    assert load_collection_policy(factory="collection_example:build") is policy


@pytest.mark.parametrize("kwargs", [{"path": "model", "factory": "a:b"}, {"factory": "invalid"}])
def test_invalid_model_configuration_fails_early(kwargs):
    with pytest.raises(ValueError, match=r"Choose either|module:function"):
        load_collection_policy(**kwargs)


def test_custom_factory_requires_policy_interface(monkeypatch):
    monkeypatch.setitem(sys.modules, "collection_example", SimpleNamespace(build=object))
    with pytest.raises(TypeError, match=r"select_action.*reset"):
        load_collection_policy(factory="collection_example:build")


def test_lerobot_adapter_uses_saved_processors_and_resets_state():
    import torch

    calls = []

    class Processor:
        def __init__(self, name):
            self.name = name

        def __call__(self, value):
            calls.append(self.name)
            return value

        def reset(self):
            calls.append(f"reset {self.name}")

    def select_action(batch):
        calls.append("select")
        assert batch["observation.images.wrist"].shape == (1, 3, 2, 2)
        assert batch["observation.images.wrist"].max() == 1
        assert batch["observation.state"].shape == (1, 6)
        assert batch["task"] == "Pick"
        return torch.full((1, 6), 12.0)

    policy = SimpleNamespace(
        select_action=select_action,
        reset=lambda: calls.append("reset model"),
        config=SimpleNamespace(use_amp=False),
    )
    adapter = LeRobotPolicyAdapter(policy, Processor("pre"), Processor("post"), device="cpu")
    result = adapter.select_action(
        {
            "observation.state": np.zeros(6, dtype=np.float32),
            "observation.images.wrist": np.full((2, 2, 3), 255, dtype=np.uint8),
            "task": "Pick",
        }
    )
    np.testing.assert_array_equal(result, np.full(6, 12))
    adapter.reset()
    assert calls == ["pre", "select", "post", "reset model", "reset pre", "reset post"]


def test_real_act_checkpoint_loads_saved_normalization_and_predicts(tmp_path):
    import torch
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    config = ACTConfig(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.environment_state": PolicyFeature(FeatureType.ENV, (3,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        normalization_mapping={
            "STATE": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.MEAN_STD,
        },
        device="cpu",
        chunk_size=2,
        n_action_steps=2,
        use_vae=False,
        dim_model=16,
        n_heads=2,
        dim_feedforward=32,
        n_encoder_layers=1,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        policy = ACTPolicy(config).eval()
    stats = {"action": {"mean": torch.full((6,), 10.0), "std": torch.full((6,), 2.0)}}
    pre, post = make_pre_post_processors(config, dataset_stats=stats)
    policy.save_pretrained(tmp_path)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    adapter = LeRobotPolicyAdapter.from_pretrained(str(tmp_path))
    batch = {
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.environment_state": np.zeros(3, dtype=np.float32),
        "task": "Pick",
    }
    with torch.inference_mode():
        raw = policy.select_action(
            {
                "observation.state": torch.zeros(1, 6),
                "observation.environment_state": torch.zeros(1, 3),
            }
        )
    expected = raw.squeeze(0).numpy() * 2 + 10
    actual = adapter.select_action(batch)
    assert actual.shape == (6,)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    adapter.reset()
    np.testing.assert_allclose(adapter.select_action(batch), expected, rtol=1e-5, atol=1e-5)
