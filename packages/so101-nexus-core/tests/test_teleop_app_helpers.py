"""Unit tests for pure helpers in so101_nexus_core.teleop.app.

The ``app`` module is designed to be importable on a base install; gradio,
lerobot, and cv2 are imported lazily inside ``main()`` and individual
callbacks. These tests exercise the pure-logic helpers that need no
gradio runtime.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from so101_nexus_core.teleop.app import (
    _build_field_selection,
    _cb_poll_init,
    _cb_poll_recording,
    _cb_retry_init,
    _connect_leader,
    _create_dataset,
    _progress_text,
)
from so101_nexus_core.teleop.dataset import OVERHEAD_KEY, WRIST_KEY
from so101_nexus_core.teleop.recorder import RecordingState


@pytest.fixture
def fake_gradio(monkeypatch):
    class _Walkthrough:
        def __init__(self, *, selected):
            self.selected = selected

    def _walkthrough_factory(**kwargs):
        return _Walkthrough(**kwargs)

    fake = types.SimpleNamespace(
        update=lambda **kwargs: kwargs,
        Walkthrough=_walkthrough_factory,
        Warning=lambda _msg: None,
    )
    monkeypatch.setitem(sys.modules, "gradio", fake)
    return fake


def test_progress_text_formats_episode_count() -> None:
    assert _progress_text(2, 5) == "**Episode 2 / 5**"


def test_progress_text_formats_zero_completed() -> None:
    assert _progress_text(0, 10) == "**Episode 0 / 10**"


def test_build_field_selection_all_keys() -> None:
    selection = _build_field_selection([WRIST_KEY, OVERHEAD_KEY, "task"])

    assert selection.wrist_image is True
    assert selection.overhead_image is True
    assert selection.task is True


def test_build_field_selection_empty() -> None:
    selection = _build_field_selection([])

    assert selection.wrist_image is False
    assert selection.overhead_image is False
    assert selection.task is False


def test_build_field_selection_only_wrist() -> None:
    selection = _build_field_selection([WRIST_KEY])

    assert selection.wrist_image is True
    assert selection.overhead_image is False
    assert selection.task is False


def test_connect_leader_wraps_connect_failure_in_runtime_error(monkeypatch) -> None:
    """If leader.connect() raises, _connect_leader wraps it with a hint message."""

    class _FailingLeader:
        def connect(self) -> None:
            raise OSError("permission denied")

    def _fake_get_leader(_robot_type, _port, _leader_id):
        return _FailingLeader()

    monkeypatch.setattr("so101_nexus_core.teleop.app.get_leader", _fake_get_leader)

    with pytest.raises(RuntimeError, match="Failed to connect on /dev/ttyACM0") as excinfo:
        _connect_leader("so101", "/dev/ttyACM0", "leader_a")

    assert "lerobot-find-port" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, OSError)


def test_connect_leader_includes_permission_recovery_commands(monkeypatch) -> None:
    class _FailingLeader:
        def connect(self) -> None:
            raise OSError("permission denied")

    monkeypatch.setattr(
        "so101_nexus_core.teleop.app.get_leader",
        lambda *_a, **_kw: _FailingLeader(),
    )

    with pytest.raises(RuntimeError) as excinfo:
        _connect_leader("so101", "/dev/ttyACM0", "leader_a")

    assert "chmod 666 /dev/ttyACM0" in str(excinfo.value)


def test_connect_leader_returns_connected_leader_on_success(monkeypatch) -> None:
    """Happy path: _connect_leader returns the leader after a successful connect."""
    state = {"connected": False}

    class _OkLeader:
        def connect(self) -> None:
            state["connected"] = True

    monkeypatch.setattr(
        "so101_nexus_core.teleop.app.get_leader",
        lambda *_a, **_kw: _OkLeader(),
    )

    leader = _connect_leader("so101", "/dev/ttyACM0", "leader_a")

    assert state["connected"] is True
    assert isinstance(leader, _OkLeader)


def test_create_dataset_disconnects_leader_on_failure(monkeypatch) -> None:
    """If LeRobotDataset.create raises, the leader is disconnected and a RuntimeError is raised."""
    disconnect_calls = {"n": 0}

    class _StubLeader:
        def disconnect(self) -> None:
            disconnect_calls["n"] += 1

    class _RaisingDataset:
        @classmethod
        def create(cls, **_kwargs):
            raise ValueError("schema mismatch")

    fake_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    fake_module.LeRobotDataset = _RaisingDataset  # type: ignore[attr-defined]

    for name, mod in [
        ("lerobot", types.ModuleType("lerobot")),
        ("lerobot.datasets", types.ModuleType("lerobot.datasets")),
        ("lerobot.datasets.lerobot_dataset", fake_module),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    leader = _StubLeader()
    with pytest.raises(RuntimeError, match="Failed to create dataset"):
        _create_dataset("local/test", 30, "so101", {}, leader)

    assert disconnect_calls["n"] == 1


def test_create_dataset_returns_dataset_on_success(monkeypatch) -> None:
    """Happy path: _create_dataset returns the LeRobotDataset instance."""
    seen = {}

    class _OkDataset:
        @classmethod
        def create(cls, **kwargs):
            seen.update(kwargs)
            return cls()

    fake_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
    fake_module.LeRobotDataset = _OkDataset  # type: ignore[attr-defined]

    for name, mod in [
        ("lerobot", types.ModuleType("lerobot")),
        ("lerobot.datasets", types.ModuleType("lerobot.datasets")),
        ("lerobot.datasets.lerobot_dataset", fake_module),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)

    class _StubLeader:
        def disconnect(self) -> None:
            pass

    ds = _create_dataset("local/test", 30, "so101", {"action": {}}, _StubLeader())

    assert isinstance(ds, _OkDataset)
    assert seen["repo_id"] == "local/test"
    assert seen["fps"] == 30
    assert seen["robot_type"] == "so101"
    assert seen["features"] == {"action": {}}


def test_poll_init_surfaces_error_and_retry_button(fake_gradio) -> None:
    init_state = {
        "done": True,
        "processed": False,
        "error": "Failed to connect on /dev/ttyACM0: permission denied",
        "tee_stdout": types.SimpleNamespace(get_output=lambda: "Connecting leader arm..."),
        "tee_stderr": None,
        "running": True,
        "warning": None,
    }

    outputs = _cb_poll_init({}, init_state)

    assert "permission denied" in outputs[0]["value"].lower()
    assert outputs[1]["visible"] is True
    assert init_state["processed"] is True


def test_retry_init_resets_failed_state(fake_gradio) -> None:
    init_state = {
        "running": True,
        "done": True,
        "processed": True,
        "error": "boom",
    }

    outputs = _cb_retry_init(init_state)

    assert init_state == {
        "running": False,
        "done": False,
        "processed": False,
        "error": None,
    }
    assert outputs[1]["value"] == ""
    assert outputs[2]["visible"] is False


def test_poll_recording_countdown_uses_dedicated_countdown_area(fake_gradio) -> None:
    session = {
        "state": RecordingState(countdown_value=3),
        "fps": 30,
    }

    outputs = _cb_poll_recording(session)

    assert outputs[0]["value"] == ""
    assert outputs[2]["visible"] is True
    assert "Get ready" in outputs[2]["value"]


def test_poll_recording_shows_live_feeds_only_while_recording(fake_gradio, monkeypatch) -> None:
    fake_cv2 = types.SimpleNamespace(
        resize=lambda frame, size, interpolation=None: frame, INTER_LINEAR=1
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)

    state = RecordingState(is_recording=True, num_episodes=5, episodes_completed=1)
    state.live_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    state.live_overhead_frame = np.zeros((4, 4, 3), dtype=np.uint8)
    state.episode_actions.append(np.zeros(6, dtype=np.float32))
    session = {
        "state": state,
        "fps": 30,
        "wrist_wh": (4, 4),
        "overhead_wh": (4, 4),
    }

    outputs = _cb_poll_recording(session)

    assert "Recording episode 2/5" in outputs[0]["value"]
    assert outputs[2]["visible"] is False
    assert outputs[3]["visible"] is True
    assert outputs[4]["visible"] is True
    assert outputs[5]["visible"] is True
