"""Tests for teleop dataset field selection, features, and frame builders."""

from __future__ import annotations

import numpy as np
import pytest

from so101_nexus_core.config import SO101_JOINT_NAMES
from so101_nexus_core.teleop.dataset import (
    OVERHEAD_KEY,
    WRIST_KEY,
    FieldSelection,
    build_features,
    build_frame,
)


def _motor_features() -> dict[str, type]:
    return {f"{name}.pos": float for name in SO101_JOINT_NAMES}


def _follower_features(
    wrist_shape: tuple[int, int, int] = (480, 480, 3),
    overhead_shape: tuple[int, int, int] = (480, 480, 3),
) -> dict[str, object]:
    return {**_motor_features(), "wrist": wrist_shape, "overhead": overhead_shape}


def test_image_keys_are_lerobot_canonical() -> None:
    assert WRIST_KEY == "observation.images.wrist"
    assert OVERHEAD_KEY == "observation.images.overhead"


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
    features = build_features(sel, _follower_features(), _motor_features())

    assert set(features) == {
        "observation.state",
        "action",
        WRIST_KEY,
        OVERHEAD_KEY,
    }
    assert features["observation.state"]["shape"] == (len(SO101_JOINT_NAMES),)
    assert features["observation.state"]["dtype"] == "float32"
    assert features["observation.state"]["names"] == [f"{name}.pos" for name in SO101_JOINT_NAMES]
    assert features["action"]["shape"] == (len(SO101_JOINT_NAMES),)
    assert features["action"]["dtype"] == "float32"
    assert features["action"]["names"] == [f"{name}.pos" for name in SO101_JOINT_NAMES]
    assert features[WRIST_KEY]["dtype"] == "video"
    assert features[WRIST_KEY]["shape"] == (480, 480, 3)
    assert features[WRIST_KEY]["names"] == ["height", "width", "channels"]
    assert features[OVERHEAD_KEY]["shape"] == (480, 480, 3)


def test_build_features_omits_deselected_image_keys() -> None:
    sel = FieldSelection(wrist_image=True, overhead_image=False)
    features = build_features(
        sel,
        _follower_features(wrist_shape=(240, 320, 3), overhead_shape=(360, 640, 3)),
        _motor_features(),
    )

    assert WRIST_KEY in features
    assert OVERHEAD_KEY not in features
    assert features[WRIST_KEY]["shape"] == (240, 320, 3)


def test_build_features_state_and_action_always_present() -> None:
    sel = FieldSelection(wrist_image=False, overhead_image=False, task=False)
    features = build_features(sel, _follower_features(), _motor_features())

    assert "observation.state" in features
    assert "action" in features


def test_build_features_requires_selected_camera_feature() -> None:
    sel = FieldSelection(wrist_image=True, overhead_image=False)
    follower_features = _motor_features()

    with pytest.raises(ValueError, match="wrist"):
        build_features(sel, follower_features, _motor_features())


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
        WRIST_KEY,
        OVERHEAD_KEY,
        "task",
    }
    assert frame["task"] == "pick the cube"


def test_build_frame_keeps_task_when_task_deselected_for_lerobot_v3() -> None:
    sel = FieldSelection(wrist_image=False, overhead_image=False, task=False)
    state = np.zeros(6, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)

    frame = build_frame(
        sel,
        state=state,
        action=action,
        task="required by LeRobotDataset.add_frame",
        wrist_image=None,
        overhead_image=None,
    )

    assert set(frame) == {"observation.state", "action", "task"}
    assert frame["task"] == "required by LeRobotDataset.add_frame"


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
