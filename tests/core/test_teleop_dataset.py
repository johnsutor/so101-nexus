"""Tests for teleop dataset field selection, features, and frame builders."""

from __future__ import annotations

import builtins
import importlib
import sys
import types

import numpy as np
import pytest

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.teleop.dataset import (
    OVERHEAD_KEY,
    REWARD_KEY,
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
        REWARD_KEY,
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

    assert "observation.state" in features
    assert "action" in features
    assert REWARD_KEY in features
    assert features[REWARD_KEY] == {"dtype": "float32", "shape": (1,), "names": None}


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
        reward=0.25,
        wrist_image=wrist,
        overhead_image=overhead,
    )

    assert set(frame) == {
        "observation.state",
        "action",
        REWARD_KEY,
        WRIST_KEY,
        OVERHEAD_KEY,
        "task",
    }
    assert frame["task"] == "pick the cube"
    assert frame[REWARD_KEY].dtype == np.float32
    assert frame[REWARD_KEY].shape == (1,)
    np.testing.assert_allclose(frame[REWARD_KEY], [0.25])


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

    assert set(frame) == {"observation.state", "action", REWARD_KEY, "task"}
    assert frame["task"] == "required by LeRobotDataset.add_frame"
    assert frame[REWARD_KEY].shape == (1,)


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


def _reload_dataset_module():
    """Re-import teleop.dataset so the import shim runs against current stubs."""
    sys.modules.pop("so101_nexus.teleop.dataset", None)
    return importlib.import_module("so101_nexus.teleop.dataset")


def _install_stub(monkeypatch, module_path: str, attr: str, value) -> None:
    parent_path, _, leaf = module_path.rpartition(".")
    stub = types.ModuleType(module_path)
    setattr(stub, attr, value)
    monkeypatch.setitem(sys.modules, module_path, stub)

    parent = sys.modules.get(parent_path)
    if parent is not None:
        monkeypatch.setattr(parent, leaf, stub, raising=False)


def test_dataset_module_import_does_not_require_lerobot(monkeypatch) -> None:
    """MuJoCo-only installs import teleop.app without installing LeRobot."""
    with monkeypatch.context() as mp:
        for name in list(sys.modules):
            if name == "lerobot" or name.startswith("lerobot."):
                mp.delitem(sys.modules, name, raising=False)

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lerobot" or name.startswith("lerobot."):
                raise ModuleNotFoundError("simulated missing lerobot")
            return real_import(name, globals, locals, fromlist, level)

        mp.setattr(builtins, "__import__", fake_import)

        module = _reload_dataset_module()

        assert module.FieldSelection().state is True
    _reload_dataset_module()


def test_dataset_prefers_lerobot_feature_utils(monkeypatch) -> None:
    """When LeRobot exposes feature_utils.hw_to_dataset_features, use it."""
    with monkeypatch.context() as mp:
        sentinel = lambda *a, **kw: {"sentinel": "feature_utils"}  # noqa: E731
        _install_stub(
            mp,
            "lerobot.datasets.feature_utils",
            "hw_to_dataset_features",
            sentinel,
        )

        module = _reload_dataset_module()

        assert module._hw_to_dataset_features() is sentinel
    _reload_dataset_module()


def test_dataset_falls_back_to_lerobot_datasets_utils(monkeypatch) -> None:
    """When feature_utils is absent, fall back to the LeRobot 0.5.0 path."""
    with monkeypatch.context() as mp:
        sentinel = lambda *a, **kw: {"sentinel": "datasets.utils"}  # noqa: E731
        mp.delitem(sys.modules, "lerobot.datasets.feature_utils", raising=False)
        _install_stub(mp, "lerobot.datasets.utils", "hw_to_dataset_features", sentinel)

        real_import = builtins.__import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "lerobot.datasets.feature_utils":
                raise ImportError("simulated missing feature_utils")
            return real_import(name, globals, locals, fromlist, level)

        mp.setattr(builtins, "__import__", fake_import)

        module = _reload_dataset_module()

        assert module._hw_to_dataset_features() is sentinel
    _reload_dataset_module()
