"""Tests for SO101 action processor steps."""

from __future__ import annotations

import numpy as np
import pytest

from so101_nexus_core.config import SO101_JOINT_NAMES


def test_processors_subpackage_imports() -> None:
    """The subpackage and module stubs are importable on a base install."""
    import so101_nexus_core.processors  # noqa: F401
    from so101_nexus_core.processors import action, observation, pipelines  # noqa: F401


def test_leader_action_to_joint_array_orders_by_joint_names() -> None:
    from so101_nexus_core.processors.action import LeaderActionToJointArrayStep

    leader_dict = {f"{name}.pos": float(i) for i, name in enumerate(SO101_JOINT_NAMES)}
    step = LeaderActionToJointArrayStep(joint_names=SO101_JOINT_NAMES)

    transition = {"action": leader_dict}
    out = step(transition)

    arr = out["action"]
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (6,)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, np.arange(6, dtype=np.float64))


def test_leader_action_to_joint_array_raises_on_missing_joint() -> None:
    from so101_nexus_core.processors.action import LeaderActionToJointArrayStep

    leader_dict = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES if name != "wrist_roll"}
    step = LeaderActionToJointArrayStep(joint_names=SO101_JOINT_NAMES)

    with pytest.raises(KeyError):
        step({"action": leader_dict})


def test_leader_action_to_joint_array_registered_in_registry() -> None:
    from lerobot.processor.pipeline import ProcessorStepRegistry

    from so101_nexus_core.processors.action import LeaderActionToJointArrayStep

    cls = ProcessorStepRegistry.get("so101_leader_action_to_joint_array")
    assert cls is LeaderActionToJointArrayStep


def test_degrees_to_radians_action_step() -> None:
    from so101_nexus_core.processors.action import DegreesToRadiansActionStep

    step = DegreesToRadiansActionStep()
    out = step({"action": np.array([0.0, 90.0, 180.0, -90.0])})

    np.testing.assert_allclose(
        out["action"],
        np.deg2rad(np.array([0.0, 90.0, 180.0, -90.0])),
    )


def test_degrees_to_radians_registered_in_registry() -> None:
    from lerobot.processor.pipeline import ProcessorStepRegistry

    from so101_nexus_core.processors.action import DegreesToRadiansActionStep

    cls = ProcessorStepRegistry.get("so101_degrees_to_radians_action")
    assert cls is DegreesToRadiansActionStep


def test_joint_offset_action_step_offsets_only_target_index() -> None:
    from so101_nexus_core.processors.action import JointOffsetActionStep

    step = JointOffsetActionStep(joint_index=4, offset_rad=-np.pi / 2)
    inp = np.zeros(6)
    out = step({"action": inp})

    expected = np.zeros(6)
    expected[4] = -np.pi / 2
    np.testing.assert_allclose(out["action"], expected)


def test_joint_offset_action_step_does_not_mutate_input() -> None:
    from so101_nexus_core.processors.action import JointOffsetActionStep

    step = JointOffsetActionStep(joint_index=0, offset_rad=1.5)
    inp = np.zeros(6)
    _ = step({"action": inp})

    np.testing.assert_array_equal(inp, np.zeros(6))


def test_joint_offset_action_step_registered() -> None:
    from lerobot.processor.pipeline import ProcessorStepRegistry

    from so101_nexus_core.processors.action import JointOffsetActionStep

    assert ProcessorStepRegistry.get("so101_joint_offset_action") is JointOffsetActionStep
