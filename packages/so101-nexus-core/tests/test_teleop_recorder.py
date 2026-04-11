"""Tests for teleop recorder pure helpers."""

from __future__ import annotations

import numpy as np

from so101_nexus_core.config import SO101_JOINT_NAMES
from so101_nexus_core.teleop.recorder import (
    compute_delta_actions,
    convert_leader_action,
)


def test_compute_delta_actions_zero_prefix() -> None:
    a0 = np.array([1.0, 2.0, 3.0])
    a1 = np.array([1.5, 2.5, 3.5])
    a2 = np.array([2.0, 2.0, 4.0])
    deltas = compute_delta_actions([a0, a1, a2])

    assert len(deltas) == 3
    np.testing.assert_allclose(deltas[0], np.zeros(3))
    np.testing.assert_allclose(deltas[1], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(deltas[2], [0.5, -0.5, 0.5])


def test_convert_leader_action_degrees_to_radians() -> None:
    action = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}
    action["shoulder_pan.pos"] = 90.0
    action["wrist_roll.pos"] = 0.0

    out = convert_leader_action(
        action,
        SO101_JOINT_NAMES,
        wrist_roll_offset_deg=0.0,
    )

    assert out.shape == (len(SO101_JOINT_NAMES),)
    np.testing.assert_allclose(out[0], np.pi / 2)


def test_convert_leader_action_applies_wrist_roll_offset_only_to_wrist_roll() -> None:
    action = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}

    out = convert_leader_action(
        action,
        SO101_JOINT_NAMES,
        wrist_roll_offset_deg=-90.0,
    )

    wrist_roll_idx = SO101_JOINT_NAMES.index("wrist_roll")
    for i, value in enumerate(out):
        if i == wrist_roll_idx:
            np.testing.assert_allclose(value, -np.pi / 2)
        else:
            np.testing.assert_allclose(value, 0.0)
