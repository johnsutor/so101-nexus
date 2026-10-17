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
    adapter = LeRobotPolicyAdapter.from_pretrained(str(tmp_path), fps=20)
    assert adapter.fps == 20
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


def test_camera_mapping_and_policy_rate_contract():
    from so101_nexus.policy_adapters.collection import parse_policy_camera_map, validate_policy_fps

    assert parse_policy_camera_map(["overhead=front", "wrist=wrist"]) == {
        "observation.images.overhead": "observation.images.front",
        "observation.images.wrist": "observation.images.wrist",
    }
    with pytest.raises(ValueError, match="30"):
        validate_policy_fps(SimpleNamespace(fps=30), 50)
    with pytest.raises(ValueError, match="policy-fps"):
        validate_policy_fps(SimpleNamespace(fps=None), 30)
    validate_policy_fps(SimpleNamespace(fps=30), 30)


@pytest.mark.parametrize("mapping", [["bad"], ["other=front"], ["wrist=front", "side=front"]])
def test_invalid_camera_mappings_fail_before_model_loading(mapping):
    from so101_nexus.policy_adapters.collection import parse_policy_camera_map

    with pytest.raises(ValueError, match="camera"):
        parse_policy_camera_map(mapping)


@pytest.mark.parametrize("policy_type", [None, "molmoact2"])
@pytest.mark.parametrize("norm_tag", [None, "so100_so101_molmoact2"])
def test_molmoact2_original_checkpoint_uses_native_config_and_norm_tag(
    monkeypatch, policy_type, norm_tag
):
    from lerobot.policies import factory
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config

    seen = {}
    monkeypatch.setattr(
        "so101_nexus.policy_adapters.collection._checkpoint_json",
        lambda path, filename: (
            {"model_type": "molmoact2"}
            if filename == "config.json"
            else {
                "metadata_by_tag": {
                    "so100_so101_molmoact2": {
                        "normalize_gripper": True,
                        "control_mode": "absolute joint pose",
                        "action_stats": {
                            "names": [
                                "shoulder_pan",
                                "shoulder_lift",
                                "elbow_flex",
                                "wrist_flex",
                                "wrist_roll",
                                "gripper",
                            ]
                        },
                        "state_stats": {
                            "names": [
                                "shoulder_pan",
                                "shoulder_lift",
                                "elbow_flex",
                                "wrist_flex",
                                "wrist_roll",
                                "gripper",
                            ]
                        },
                    }
                },
            }
        ),
    )

    class Policy:
        def __init__(self, config):
            seen["config"] = config
            self.config = config

        def to(self, device):
            seen["device"] = device
            return self

        def eval(self):
            return self

    monkeypatch.setattr(factory, "get_policy_class", lambda kind: Policy)
    monkeypatch.setattr(
        factory,
        "make_pre_post_processors",
        lambda config, **kwargs: (
            seen.setdefault("pre", object()),
            seen.setdefault("post", object()),
        ),
    )
    adapter = LeRobotPolicyAdapter.from_pretrained(
        "allenai/MolmoAct2-SO100_101",
        policy_type=policy_type,
        norm_tag=norm_tag,
        device="cuda",
    )
    config = seen["config"]
    assert isinstance(config, MolmoAct2Config)
    assert config.checkpoint_path == "allenai/MolmoAct2-SO100_101"
    assert config.norm_tag == "so100_so101_molmoact2"
    assert config.inference_action_mode == "continuous"
    assert config.normalize_gripper is True
    assert config.joint_signs == [1, -1, 1, 1, 1, 1]
    assert config.joint_offsets == [0, 90, 90, 0, 0, 0]
    assert config.action_feature.shape == (6,)
    assert config.image_keys == ["observation.images.overhead", "observation.images.side"]
    assert adapter.fps == 30


def test_native_molmoact2_replays_one_action_per_tick_and_resets_chunks(monkeypatch):
    import torch
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    config = MolmoAct2Config(
        input_features={
            "observation.state": PolicyFeature(FeatureType.STATE, (6,)),
            "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 2, 2)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        inference_action_mode="continuous",
        device="cpu",
        chunk_size=30,
        n_action_steps=30,
        train_mode_vlm="fft",
    )
    generations = []
    monkeypatch.setattr(MolmoAct2Policy, "_load_hf_model", lambda self: None)

    def predict(self, batch):
        generations.append(batch)
        return torch.arange(30).reshape(1, 30, 1).expand(1, 30, 6) + 100 * len(generations)

    monkeypatch.setattr(MolmoAct2Policy, "predict_action_chunk", predict)
    policy = MolmoAct2Policy(config)
    for index in range(30):
        torch.testing.assert_close(policy.select_action({}), torch.full((1, 6), 100 + index))
    assert len(generations) == 1
    policy.reset()
    torch.testing.assert_close(policy.select_action({}), torch.full((1, 6), 200))
    assert len(generations) == 2


def test_native_molmoact2_postprocessor_restores_gripper_percent():
    import torch
    from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.molmoact2.processor_molmoact2 import (
        MolmoAct2MaskedUnnormalizerProcessorStep,
    )
    from lerobot.processor import create_transition

    step = MolmoAct2MaskedUnnormalizerProcessorStep(
        features={"action": PolicyFeature(FeatureType.ACTION, (6,))},
        norm_map={"ACTION": NormalizationMode.QUANTILES},
        stats={
            "action": {
                "q01": torch.zeros(6),
                "q99": torch.full((6,), 100.0),
                "mask": torch.ones(6, dtype=torch.bool),
            }
        },
    )
    restored = step(create_transition(action=torch.zeros(1, 6)))["action"]
    torch.testing.assert_close(restored, torch.full((1, 6), 50.0))


def test_camera_mapping_is_applied_before_preprocessing_without_collisions():
    import torch
    from lerobot.configs import FeatureType, PolicyFeature

    observed = []

    def select(batch):
        observed.append(batch)
        return torch.zeros(1, 6)

    class Processor:
        def __call__(self, value):
            return value

    policy = SimpleNamespace(
        select_action=select,
        config=SimpleNamespace(
            use_amp=False,
            input_features={
                "observation.images.front": PolicyFeature(FeatureType.VISUAL, (3, 2, 2)),
            },
        ),
    )
    adapter = LeRobotPolicyAdapter(
        policy,
        Processor(),
        Processor(),
        camera_map={"observation.images.wrist": "observation.images.front"},
    )
    adapter.select_action(
        {
            "observation.state": np.zeros(6, dtype=np.float32),
            "observation.images.front": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation.images.wrist": np.full((2, 2, 3), 255, dtype=np.uint8),
            "task": "Pick",
        }
    )
    torch.testing.assert_close(observed[0]["observation.images.front"], torch.ones(1, 3, 2, 2))
    assert "observation.images.wrist" not in observed[0]


def test_native_molmoact2_joint_frame_transform_is_invertible_and_preserves_gripper():
    import torch
    from lerobot.policies.molmoact2.processor_molmoact2 import (
        MolmoAct2ActionFrameTransformStep,
        MolmoAct2StateFrameTransformStep,
    )
    from lerobot.processor import create_transition

    options = {"joint_signs": [1, -1, 1, 1, 1, 1], "joint_offsets": [0, 90, 90, 0, 0, 0]}
    follower = torch.tensor([[10.0, -30.0, 30.0, 40.0, 50.0, 60.0]])
    encoded = MolmoAct2StateFrameTransformStep(**options)(
        create_transition(observation={"observation.state": follower})
    )["observation"]["observation.state"]
    torch.testing.assert_close(encoded, torch.tensor([[10.0, 120.0, 120.0, 40.0, 50.0, 60.0]]))
    decoded = MolmoAct2ActionFrameTransformStep(**options)(create_transition(action=encoded))[
        "action"
    ]
    torch.testing.assert_close(decoded, follower)
