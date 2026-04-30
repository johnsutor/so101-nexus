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
