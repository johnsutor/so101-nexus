"""Dependency contracts for the supported LeRobot release and policy extras."""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement


def test_dm05_extra_pins_native_lerobot_and_secure_transformers():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    dependencies = project["project"]["optional-dependencies"]["dm05"]
    requirements = {Requirement(item).name: Requirement(item) for item in dependencies}
    assert project["tool"]["uv"]["sources"]["lerobot"] == [
        {
            "url": (
                "https://github.com/huggingface/lerobot/archive/"
                "2577da0ef39b47f870592d62d81edc8bce922cc4.tar.gz"
            ),
            "extra": "dm05",
        }
    ]
    assert requirements["lerobot"].extras == {"peft-dep"}
    assert "5.10.0" in requirements["transformers"].specifier
    assert "5.5.4" not in requirements["transformers"].specifier


def test_lerobot_061_includes_dataset_dependencies_and_opt_in_molmoact2():
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    requirement = "lerobot[dataset,feetech,placo-dep]>=0.6.1,<0.7"
    assert requirement in extras["teleop"]
    assert (
        "lerobot[dataset,feetech,peft-dep,placo-dep,scipy-dep]>=0.6.1,<0.7"
        in project["dependency-groups"]["test"]
    )
    assert "lerobot[peft-dep,scipy-dep]>=0.6.1,<0.7" in extras["molmoact2"]
    assert not any("git+" in dependency for dependency in extras["molmoact2"])


def test_molmoact2_transformers_requirements_exclude_vulnerable_releases():
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text())
    for dependencies in (
        project["project"]["optional-dependencies"]["molmoact2"],
        project["dependency-groups"]["test"],
    ):
        requirement = next(
            Requirement(dependency)
            for dependency in dependencies
            if Requirement(dependency).name == "transformers"
        )
        assert "5.5.4" not in requirement.specifier
        assert "5.8.1" not in requirement.specifier
        assert "5.10.0" in requirement.specifier
        assert "5.11.0" not in requirement.specifier
    lock = tomllib.loads((root / "uv.lock").read_text())
    transformers = next(package for package in lock["package"] if package["name"] == "transformers")
    assert transformers["version"] in requirement.specifier


def test_secure_transformers_runs_molmoact2_visual_cached_and_continuous_inference():
    import torch
    from lerobot.policies.molmoact2.molmoact2_hf_model.configuration_molmoact2 import (
        MolmoAct2Config,
    )
    from lerobot.policies.molmoact2.molmoact2_hf_model.modeling_molmoact2 import (
        MolmoAct2ForConditionalGeneration,
    )

    config = MolmoAct2Config(
        text_config={
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "vocab_size": 32,
            "additional_vocab_size": 0,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "max_position_embeddings": 32,
        },
        vit_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "image_default_input_size": (4, 4),
            "image_patch_size": 2,
            "image_num_pos": 4,
        },
        adapter_config={
            "vit_layers": (-1,),
            "hidden_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "intermediate_size": 32,
            "text_hidden_size": 16,
        },
        action_expert_config={
            "hidden_size": 16,
            "num_layers": 1,
            "num_heads": 2,
            "ffn_multiple_of": 8,
            "timestep_embed_dim": 8,
        },
        max_action_horizon=2,
        max_action_dim=6,
        flow_matching_num_steps=1,
        action_mode="continuous",
        image_patch_id=30,
        image_end_token_id=31,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = MolmoAct2ForConditionalGeneration(config).eval()
    batch = {
        "input_ids": torch.tensor([[1, 30, 31, 2]]),
        "pixel_values": torch.zeros(1, 4, 12, dtype=torch.uint8),
        "image_token_pooling": torch.tensor([[0, 1, 2, 3]]),
        "image_grids": torch.tensor([[1, 1, 0, 0]]),
        "image_num_crops": torch.tensor([1]),
    }
    with torch.inference_mode():
        result = model(**batch, use_cache=True)
        cached = model(
            input_ids=torch.tensor([[4]]), past_key_values=result.past_key_values, use_cache=True
        )
        actions = model.model.generate_actions_from_inputs(
            **batch, generator=torch.Generator().manual_seed(0)
        )
    assert result.logits.shape == (1, 4, 32)
    assert cached.logits.shape == (1, 1, 32)
    assert actions.shape == (1, 2, 6)
    assert all(torch.isfinite(value).all() for value in (result.logits, cached.logits, actions))
