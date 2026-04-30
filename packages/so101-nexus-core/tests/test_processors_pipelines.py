"""Tests for default SO101-Nexus processor pipelines."""

from __future__ import annotations

import math

import numpy as np
import pytest

from so101_nexus_core.config import SO101_JOINT_NAMES


def _legacy_convert(
    leader_action: dict, joint_names: tuple[str, ...], offset_deg: float
) -> np.ndarray:
    """Reference implementation of the previous ``convert_leader_action`` helper."""
    converted: list[float] = []
    offset_rad = math.radians(offset_deg)
    for name in joint_names:
        value = math.radians(leader_action[f"{name}.pos"])
        if name == "wrist_roll":
            value += offset_rad
        converted.append(value)
    return np.array(converted, dtype=np.float64)


def test_default_leader_pipeline_byte_equivalent_to_legacy_convert() -> None:
    from so101_nexus_core.processors.pipelines import make_default_leader_action_pipeline

    pipeline = make_default_leader_action_pipeline(wrist_roll_offset_deg=-90.0)

    leader_action = {
        "shoulder_pan.pos": 12.5,
        "shoulder_lift.pos": -47.0,
        "elbow_flex.pos": 30.0,
        "wrist_flex.pos": 5.5,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 22.0,
    }

    out = pipeline({"action": leader_action})
    expected = _legacy_convert(leader_action, SO101_JOINT_NAMES, offset_deg=-90.0)

    assert isinstance(out, np.ndarray)
    assert out.shape == (6,)
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-12)


def test_default_leader_pipeline_custom_offset_zero_keeps_wrist_roll_unchanged() -> None:
    from so101_nexus_core.processors.pipelines import make_default_leader_action_pipeline

    pipeline = make_default_leader_action_pipeline(wrist_roll_offset_deg=0.0)
    leader_action = {f"{n}.pos": 0.0 for n in SO101_JOINT_NAMES}
    out = pipeline({"action": leader_action})

    np.testing.assert_array_equal(out, np.zeros(6))


def test_default_leader_pipeline_supports_custom_joint_order() -> None:
    from so101_nexus_core.processors.pipelines import make_default_leader_action_pipeline

    custom_order = (
        "gripper",
        "wrist_roll",
        "wrist_flex",
        "elbow_flex",
        "shoulder_lift",
        "shoulder_pan",
    )
    pipeline = make_default_leader_action_pipeline(
        joint_names=custom_order,
        wrist_roll_offset_deg=0.0,
    )
    leader_action = {f"{n}.pos": float(i) for i, n in enumerate(custom_order)}
    out = pipeline({"action": leader_action})

    np.testing.assert_allclose(out, np.deg2rad(np.arange(6, dtype=np.float64)))


def test_default_leader_pipeline_rejects_joint_names_without_wrist_roll() -> None:
    from so101_nexus_core.processors.pipelines import make_default_leader_action_pipeline

    with pytest.raises(ValueError, match="wrist_roll"):
        make_default_leader_action_pipeline(joint_names=("a", "b", "c"))


def test_default_leader_pipeline_save_and_load_round_trip(tmp_path) -> None:
    from lerobot.processor.pipeline import DataProcessorPipeline

    from so101_nexus_core.processors.pipelines import make_default_leader_action_pipeline

    pipeline = make_default_leader_action_pipeline(wrist_roll_offset_deg=-90.0)
    pipeline.save_pretrained(str(tmp_path), config_filename="processor.json")

    reloaded = DataProcessorPipeline.from_pretrained(
        str(tmp_path),
        config_filename="processor.json",
    )
    assert len(reloaded.steps) == len(pipeline.steps)
    assert [type(s).__name__ for s in reloaded.steps] == [type(s).__name__ for s in pipeline.steps]


def test_default_env_pipeline_renames_state_and_camera_keys() -> None:
    import gymnasium as gym

    from so101_nexus_core.processors.pipelines import make_default_env_observation_pipeline

    space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            "wrist_camera": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
            "overhead_camera": gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
        }
    )
    pipeline = make_default_env_observation_pipeline(space)

    obs = {
        "state": np.zeros(6, dtype=np.float32),
        "wrist_camera": np.full((64, 64, 3), 128, dtype=np.uint8),
        "overhead_camera": np.full((64, 64, 3), 64, dtype=np.uint8),
    }
    out = pipeline({"observation": obs})

    assert set(out.keys()) >= {
        "observation.state",
        "observation.images.wrist",
        "observation.images.overhead",
    }
    assert out["observation.images.wrist"].shape == (3, 64, 64)
    assert out["observation.images.overhead"].shape == (3, 64, 64)


def test_default_env_pipeline_passes_unknown_keys_through() -> None:
    import gymnasium as gym

    from so101_nexus_core.processors.pipelines import make_default_env_observation_pipeline

    space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            "task_description": gym.spaces.Text(max_length=128),
        }
    )
    pipeline = make_default_env_observation_pipeline(space)
    obs = {"state": np.zeros(6, dtype=np.float32), "task_description": "pick up the cube"}
    out = pipeline({"observation": obs})

    assert "task_description" in out
    assert out["task_description"] == "pick up the cube"


def test_infer_rename_map_state_and_cameras() -> None:
    from so101_nexus_core.processors.pipelines import _infer_rename_map

    keys = ("state", "wrist_camera", "overhead_camera", "extra")
    mapping = _infer_rename_map(keys)
    assert mapping == {
        "state": "observation.state",
        "wrist_camera": "observation.images.wrist",
        "overhead_camera": "observation.images.overhead",
    }


def test_default_env_pipeline_with_add_batch_dim_does_not_raise() -> None:
    """Regression: AddBatchDimensionProcessorStep does bracket-access on action/comp data
    keys, so the to_transition helper must produce a complete EnvTransition.
    """
    import gymnasium as gym

    from so101_nexus_core.processors.pipelines import make_default_env_observation_pipeline

    space = gym.spaces.Dict(
        {
            "state": gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32),
            "wrist_camera": gym.spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8),
        }
    )
    # This should not raise KeyError when AddBatchDimensionProcessorStep does bracket access
    # on the transition produced by _env_observation_to_transition.
    pipeline = make_default_env_observation_pipeline(space, add_batch_dim=True)

    obs = {
        "state": np.zeros(6, dtype=np.float32),
        "wrist_camera": np.full((8, 8, 3), 255, dtype=np.uint8),
    }
    out = pipeline({"observation": obs})

    # Verify that the pipeline ran successfully and produced output
    assert "observation.state" in out
    assert "observation.images.wrist" in out
