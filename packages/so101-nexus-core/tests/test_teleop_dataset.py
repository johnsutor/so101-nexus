"""Tests for teleop dataset field selection, features, and frame builders."""

from __future__ import annotations

import numpy as np
import pytest

from so101_nexus_core.config import SO101_JOINT_NAMES
from so101_nexus_core.teleop.dataset import (
    FieldSelection,
    build_features,
    build_frame,
)


def test_field_selection_forces_state_and_action_on() -> None:
    sel = FieldSelection(
        wrist_image=False,
        overhead_image=False,
        task=False,
    )
    assert sel.state is True
    assert sel.action is True


def test_build_features_default_contains_all_keys() -> None:
    sel = FieldSelection()
    features = build_features(sel, SO101_JOINT_NAMES, (480, 480), (480, 480))

    assert set(features) == {
        "observation.state",
        "action",
        "observation.images.wrist_cam",
        "observation.images.overhead_cam",
    }
    assert features["observation.state"]["shape"] == (len(SO101_JOINT_NAMES),)
    assert features["observation.images.wrist_cam"]["shape"] == (480, 480, 3)
    assert features["observation.images.overhead_cam"]["shape"] == (480, 480, 3)


def test_build_features_omits_deselected_image_keys() -> None:
    sel = FieldSelection(wrist_image=True, overhead_image=False)
    features = build_features(sel, SO101_JOINT_NAMES, (320, 240), (640, 360))

    assert "observation.images.wrist_cam" in features
    assert "observation.images.overhead_cam" not in features
    assert features["observation.images.wrist_cam"]["shape"] == (240, 320, 3)


def test_build_features_state_and_action_always_present() -> None:
    sel = FieldSelection(wrist_image=False, overhead_image=False, task=False)
    features = build_features(sel, SO101_JOINT_NAMES, (64, 64), (64, 64))

    assert "observation.state" in features
    assert "action" in features


def test_build_frame_default_includes_all_selected_fields() -> None:
    sel = FieldSelection()
    state = np.zeros(6, dtype=np.float32)
    action = np.ones(6, dtype=np.float32)
    wrist = np.zeros((64, 64, 3), dtype=np.uint8)
    overhead = np.ones((64, 64, 3), dtype=np.uint8) * 255

    frame = build_frame(
        sel,
        state=state,
        action=action,
        task="pick the cube",
        wrist_image=wrist,
        overhead_image=overhead,
    )

    assert set(frame) == {
        "observation.state",
        "action",
        "observation.images.wrist_cam",
        "observation.images.overhead_cam",
        "task",
    }
    assert frame["task"] == "pick the cube"


def test_build_frame_omits_deselected_fields() -> None:
    sel = FieldSelection(wrist_image=False, overhead_image=False, task=False)
    state = np.zeros(6, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)

    frame = build_frame(
        sel,
        state=state,
        action=action,
        task="ignored",
        wrist_image=None,
        overhead_image=None,
    )

    assert set(frame) == {"observation.state", "action"}


def test_build_frame_raises_when_selected_image_missing() -> None:
    sel = FieldSelection(wrist_image=True)
    with pytest.raises(ValueError, match="wrist"):
        build_frame(
            sel,
            state=np.zeros(6, dtype=np.float32),
            action=np.zeros(6, dtype=np.float32),
            task="t",
            wrist_image=None,
            overhead_image=None,
        )
