"""Native LeRobot DM0.5 loading and SO101 processor contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from so101_nexus.policy_adapters.collection import load_collection_policy


@pytest.mark.parametrize("policy_type", [None, "dm05"])
def test_published_dm05_routes_to_native_loader(monkeypatch, policy_type):
    seen = {}
    expected = object()

    def load(path, **kwargs):
        seen.update(path=path, **kwargs)
        return expected

    monkeypatch.setattr("so101_nexus.policy_adapters.dm05.load_native_dm05_policy", load)
    result = load_collection_policy(
        path="Dexmal/DM05-SO101-Pick-Cube", policy_type=policy_type, device="cuda", fps=30
    )
    assert result is expected
    assert seen == {
        "path": "Dexmal/DM05-SO101-Pick-Cube",
        "device": "cuda",
        "fps": 30,
        "camera_map": None,
    }


def test_published_dm05_rejects_another_policy_family(monkeypatch):
    def load(*args, **kwargs):
        pytest.fail("A mismatched policy family must fail before loading weights")

    monkeypatch.setattr("so101_nexus.policy_adapters.dm05.load_native_dm05_policy", load)
    with pytest.raises(ValueError, match="policy-type"):
        load_collection_policy(path="Dexmal/DM05-SO101-Pick-Cube", policy_type="act", fps=30)


@pytest.fixture
def raw_dm05_checkpoint(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
    (tmp_path / "meta.yaml").write_text(
        json.dumps(
            {
                "model_type": "dm05",
                "base_model": "Dexmal/DM05",
                "checkpoint_type": "lora",
                "robot": "SO101",
                "task": "pick_cube",
            }
        )
    )
    (tmp_path / "norm_stats.json").write_text(
        json.dumps(
            {
                "norm_stats": {
                    "state": {"q01": [-100] * 6, "q99": [100] * 6},
                    "action": {"q01": [-10] * 6, "q99": [10] * 6},
                }
            }
        )
    )
    return tmp_path


def test_raw_dm05_uses_lerobot_core_lora_and_checkpoint_processors(
    monkeypatch, raw_dm05_checkpoint
):
    import torch
    from lerobot.policies.dm05.modeling_dm05 import DM05Policy
    from lerobot.policies.dm05.processor_dm05 import DM05TokenizerProcessorStep
    from peft import PeftModel

    configs = []
    chunks = []
    merges = []

    def initialize(self, config):
        torch.nn.Module.__init__(self)
        self.config = config
        self.model = torch.nn.Linear(1, 1)
        configs.append(config)
        self.reset()

    def predict(self, batch):
        chunks.append(batch)
        return torch.zeros(1, 2, 6)

    def merge(model, path, **kwargs):
        merges.append((model, path))
        return SimpleNamespace(merge_and_unload=lambda: model)

    monkeypatch.setattr(DM05Policy, "__init__", initialize)
    monkeypatch.setattr(DM05Policy, "predict_action_chunk", predict)
    monkeypatch.setattr(PeftModel, "from_pretrained", merge)
    monkeypatch.setattr(DM05TokenizerProcessorStep, "observation", lambda self, obs: obs)
    policy = load_collection_policy(path=str(raw_dm05_checkpoint), policy_type="dm05", fps=30)
    config = configs[0]
    assert config.pretrained_name_or_path == "Dexmal/DM05"
    assert config.processor_name_or_path == str(raw_dm05_checkpoint)
    assert config.use_relative_actions is True
    assert config.relative_exclude_joints == ["gripper"]
    assert config.action_feature_names == [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    assert config.chunk_size == config.n_action_steps == 50
    assert config.image_keys == ["observation.images.images_1", "observation.images.images_2"]
    assert merges == [(policy.policy.model, str(raw_dm05_checkpoint))]
    batch = {
        "observation.state": np.array([10, 20, 30, 40, 50, 60], dtype=np.float32),
        "observation.images.overhead": np.full((8, 12, 3), 255, dtype=np.uint8),
        "observation.images.wrist": np.zeros((8, 12, 3), dtype=np.uint8),
        "task": "Pick the cube",
    }
    np.testing.assert_allclose(policy.select_action(batch), [10, 20, 30, 40, 50, 0], atol=1e-5)
    np.testing.assert_allclose(
        chunks[0]["observation.state"].cpu(), [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], atol=1e-5
    )
    batch["observation.state"] = batch["observation.state"] + 5
    np.testing.assert_allclose(policy.select_action(batch), [10, 20, 30, 40, 50, 0], atol=1e-5)
    assert len(chunks) == 1
    policy.reset()
    np.testing.assert_allclose(policy.select_action(batch), [15, 25, 35, 45, 55, 0], atol=1e-5)
    assert len(chunks) == 2


@pytest.mark.parametrize("field", ["state", "action"])
def test_native_dm05_rejects_invalid_statistics_before_loading_weights(raw_dm05_checkpoint, field):
    path = raw_dm05_checkpoint / "norm_stats.json"
    stats = json.loads(path.read_text())
    stats["norm_stats"][field]["q99"] = [0] * 5
    path.write_text(json.dumps(stats))
    with pytest.raises(ValueError, match=r"six.*quantiles"):
        load_collection_policy(path=str(raw_dm05_checkpoint), policy_type="dm05", fps=30)


def test_native_dm05_rejects_other_embodiments(raw_dm05_checkpoint):
    path = raw_dm05_checkpoint / "meta.yaml"
    metadata = json.loads(path.read_text())
    metadata["robot"] = "ALOHA"
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="SO101"):
        load_collection_policy(path=str(raw_dm05_checkpoint), policy_type="dm05", fps=30)


def test_raw_so101_prompt_preserves_embodiment_and_camera_labels():
    from so101_nexus.policy_adapters.dm05_processors import DM05SO101TokenizerProcessorStep

    step = DM05SO101TokenizerProcessorStep(processor_name_or_path="unused", processor=object())
    content = step.tokenization._build_user_content(
        prompt="Pick the cube",
        images=[object(), object()],
        state_bins=[1] * 6,
        meta_data={"dataset_meta": {"image_keys": ["images_1", "images_2"]}},
        speed_text=None,
    )
    assert content[0]["text"] == (
        "Robot: SO101\nOverall speed: 0.5\nTask: Pick the cube.\nHead image: "
    )
    assert content[2]["text"] == "Left wrist image: "
    assert content[-1]["text"] == "States: 1 1 1 1 1 1"


def test_native_dm05_loads_real_lora_weights_and_predicts_without_opendm(raw_dm05_checkpoint):
    import torch
    from lerobot.policies.dm05.core.modeling import DM05CoreModelConfig, DM05ForCausalLM
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file, save_file

    text = {
        "model_type": "gemma3_text",
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "max_position_embeddings": 256,
        "layer_types": ["full_attention", "full_attention"],
        "sliding_window": 128,
        "pad_token_id": 0,
        "bos_token_id": 2,
        "eos_token_id": 1,
    }
    core_config = DM05CoreModelConfig(
        vlm_config={
            "model_type": "gemma3",
            "text_config": text,
            "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 32,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 16,
                "patch_size": 8,
            },
            "mm_tokens_per_image": 4,
            "image_token_index": 127,
            "boi_token_index": 125,
            "eoi_token_index": 126,
        },
        action_config={**text, "vocab_size": 16},
        action_dim=32,
        chunk_size=50,
        bf16=False,
        gradient_checkpointing=False,
        vlm_gradient_checkpointing=False,
        ae_gradient_checkpointing=False,
        freeze_vlm_embedding=False,
        llm_attn_implementation="eager",
        vision_attn_implementation="eager",
        action_attn_implementation="eager",
    )
    base = raw_dm05_checkpoint / "base"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        core = DM05ForCausalLM(core_config)
        expected_vision = (
            core.model.vlm.model.vision_tower.embeddings.patch_embedding.weight.detach().clone()
        )
        core.save_pretrained(base)
        lora = get_peft_model(
            core, LoraConfig(r=2, target_modules=["q_proj"], modules_to_save=["action_out_proj"])
        )
        dense = lora.get_base_model().model.action_out_proj.modules_to_save["default"]
        with torch.no_grad():
            dense.weight.fill_(0.03)
            for name, parameter in lora.named_parameters():
                if "lora_B" in name:
                    parameter.fill_(0.02)
        expected = dense.weight.detach().clone()
        lora.save_pretrained(raw_dm05_checkpoint)
        merged = lora.merge_and_unload()
        expected_vision_query = (
            merged.model.vlm.model.vision_tower.encoder.layers[0]
            .self_attn.q_proj.weight.detach()
            .clone()
        )
    adapter_config_path = raw_dm05_checkpoint / "adapter_config.json"
    adapter_config = json.loads(adapter_config_path.read_text())
    adapter_config["task_type"] = "CAUSAL_LM"
    adapter_config_path.write_text(json.dumps(adapter_config))
    for weights_path in (
        base / "model.safetensors",
        raw_dm05_checkpoint / "adapter_model.safetensors",
    ):
        legacy_weights = {
            key.replace(".vision_tower.", ".vision_tower.vision_model."): value
            for key, value in load_file(weights_path).items()
        }
        save_file(legacy_weights, weights_path, metadata={"format": "pt"})
    meta_path = raw_dm05_checkpoint / "meta.yaml"
    metadata = json.loads(meta_path.read_text())
    metadata["base_model"] = str(base)
    meta_path.write_text(json.dumps(metadata))

    _save_tiny_processor(raw_dm05_checkpoint)

    policy = load_collection_policy(path=str(raw_dm05_checkpoint), policy_type="dm05", fps=30)
    torch.testing.assert_close(policy.policy.model.model.action_out_proj.weight, expected)
    torch.testing.assert_close(
        policy.policy.model.model.vlm.model.vision_tower.embeddings.patch_embedding.weight,
        expected_vision,
    )
    torch.testing.assert_close(
        policy.policy.model.model.vlm.model.vision_tower.encoder.layers[0].self_attn.q_proj.weight,
        expected_vision_query,
    )
    batch = {
        "observation.state": np.zeros(6, dtype=np.float32),
        "observation.images.overhead": np.full((8, 12, 3), 255, dtype=np.uint8),
        "observation.images.wrist": np.zeros((8, 12, 3), dtype=np.uint8),
        "task": "Pick the cube",
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        action = policy.select_action(batch)
    assert action.shape == (6,)
    assert np.isfinite(action).all()
    assert policy.policy.count_queued_actions() == 49
    policy.reset()
    assert policy.policy.count_queued_actions() == 0


def _save_tiny_processor(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import Gemma3ImageProcessor, Gemma3Processor, PreTrainedTokenizerFast

    tokens = [f"unused_{index}" for index in range(128)]
    for index, token in {
        0: "<pad>",
        1: "<eos>",
        2: "<bos>",
        3: "<unk>",
        4: "<start_of_turn>",
        5: "<end_of_turn>",
        125: "<start_of_image>",
        126: "<end_of_image>",
        127: "<image_soft_token>",
    }.items():
        tokens[index] = token
    backend = Tokenizer(WordLevel(dict(zip(tokens, range(128), strict=True)), unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        bos_token="<bos>",
        unk_token="<unk>",
        image_token="<image_soft_token>",
        boi_token="<start_of_image>",
        eoi_token="<end_of_image>",
        additional_special_tokens=tokens[4:6] + tokens[125:],
    )
    template = (
        "{% for message in messages %}<start_of_turn>{{ message['role'] }}\n"
        "{% for item in message['content'] %}{% if item['type'] == 'image' %}"
        "<start_of_image>{% else %}{{ item['text'] }}{% endif %}{% endfor %}<end_of_turn>\n"
        "{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
    )
    Gemma3Processor(
        Gemma3ImageProcessor(size={"height": 16, "width": 16}, do_pan_and_scan=False),
        tokenizer,
        chat_template=template,
        image_seq_length=4,
    ).save_pretrained(path)
