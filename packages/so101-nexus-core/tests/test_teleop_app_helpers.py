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
    _build_record_step,
    _cb_approve_episode,
    _cb_discard_episode,
    _cb_poll_init,
    _cb_poll_recording,
    _cb_retry_init,
    _connect_leader,
    _create_dataset,
    _default_env_id,
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
        Error=RuntimeError,
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


def test_default_env_id_prefers_matching_robot_variant() -> None:
    env_ids = [
        "ManiSkillLookAtSO100-v1",
        "ManiSkillLookAtSO101-v1",
        "ManiSkillReachSO100-v1",
    ]

    assert _default_env_id(env_ids, "so101") == "ManiSkillLookAtSO101-v1"


def test_default_env_id_falls_back_to_first_env() -> None:
    env_ids = ["ManiSkillLookAtSO100-v1", "ManiSkillReachSO100-v1"]

    assert _default_env_id(env_ids, "so101") == "ManiSkillLookAtSO100-v1"


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
    assert len(outputs) == 9
    assert outputs[2]["visible"] is True
    assert "Get ready" in outputs[2]["value"]


def test_poll_recording_emits_single_preview_during_recording(fake_gradio) -> None:
    state = RecordingState(is_recording=True, num_episodes=5, episodes_completed=1)
    state.live_preview = np.full((90, 160, 3), 200, dtype=np.uint8)
    state.episode_actions.append(np.zeros(6, dtype=np.float32))
    session = {
        "state": state,
        "fps": 30,
    }

    outputs = _cb_poll_recording(session)

    assert len(outputs) == 9
    assert "Recording episode 2/5" in outputs[0]["value"]
    assert outputs[2]["visible"] is False
    assert outputs[3]["visible"] is True
    np.testing.assert_array_equal(outputs[3]["value"], state.live_preview)
    assert outputs[4]["visible"] is True


def test_poll_recording_waits_for_real_preview_before_showing_image(fake_gradio) -> None:
    state = RecordingState(is_recording=True, num_episodes=1)
    session = {
        "state": state,
        "fps": 30,
    }

    outputs = _cb_poll_recording(session)

    assert len(outputs) == 9
    assert "Waiting for camera frame" in outputs[0]["value"]
    assert outputs[3]["visible"] is False
    assert "value" not in outputs[3]
    assert outputs[4]["visible"] is True


def test_record_step_countdown_area_starts_blank() -> None:
    """The hidden countdown markdown must not ship a stale 'Get ready' value."""

    class _FakeComponent:
        def __init__(self, value=None, **kwargs):
            self.value = value
            for key, val in kwargs.items():
                setattr(self, key, val)

    class _FakeRow:
        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

    def _markdown(value="", **kwargs):
        return _FakeComponent(value, **kwargs)

    def _button(value="", **kwargs):
        return _FakeComponent(value, **kwargs)

    def _component(**kwargs):
        return _FakeComponent(**kwargs)

    fake_gr = types.SimpleNamespace(
        Markdown=_markdown,
        Button=_button,
        Image=_component,
        Timer=_component,
        Row=_FakeRow,
    )

    components = _build_record_step(fake_gr)

    countdown_area = components[2]
    assert countdown_area.value == ""
    assert countdown_area.visible is False


def test_discard_episode_restores_record_controls(fake_gradio) -> None:
    class _Dataset:
        def __init__(self) -> None:
            self.clear_calls = 0

        def clear_episode_buffer(self) -> None:
            self.clear_calls += 1

    state = RecordingState(num_episodes=3, episodes_completed=1)
    state.live_preview = np.full((10, 10, 3), 255, dtype=np.uint8)
    dataset = _Dataset()
    session = {"state": state, "dataset": dataset}

    outputs = _cb_discard_episode(session)

    assert dataset.clear_calls == 1
    assert len(outputs) == 5
    assert outputs[1]["value"].startswith("Episode discarded.")
    assert outputs[3]["visible"] is True
    assert outputs[4]["visible"] is False
    assert outputs[4]["value"] is None


def test_approve_episode_restores_record_controls_for_next_episode(fake_gradio) -> None:
    class _Dataset:
        repo_id = "local/test"

        def __init__(self) -> None:
            self.frames = []
            self.saved = 0

        def add_frame(self, frame) -> None:
            self.frames.append(frame)

        def save_episode(self) -> None:
            self.saved += 1

    from so101_nexus_core.teleop.dataset import FieldSelection

    state = RecordingState(num_episodes=2, episodes_completed=0)
    state.episode_actions.append(np.zeros(6, dtype=np.float32))
    state.episode_states.append(np.zeros(6, dtype=np.float32))
    dataset = _Dataset()
    session = {
        "state": state,
        "dataset": dataset,
        "action_space": "joint_pos",
        "field_selection": FieldSelection(wrist_image=False, overhead_image=False, task=False),
        "env_id": "ManiSkillLookAtSO101-v1",
        "fps": 30,
    }

    outputs = _cb_approve_episode(session)

    assert dataset.saved == 1
    assert len(outputs) == 6
    assert outputs[1]["value"].startswith("Episode saved!")
    assert outputs[4]["visible"] is True
    assert outputs[5]["visible"] is False
    assert outputs[5]["value"] is None


def test_cb_push_to_hub_finalizes_before_uploading(fake_gradio) -> None:
    """Push to Hub must flush the v3.0 per-episode metadata before upload."""
    from so101_nexus_core.teleop.app import _cb_push_to_hub

    calls: list[str] = []

    class _RecordingDataset:
        def finalize(self) -> None:
            calls.append("finalize")

        def push_to_hub(self) -> None:
            calls.append("push_to_hub")

    result = _cb_push_to_hub({"dataset": _RecordingDataset()})

    assert calls == ["finalize", "push_to_hub"], (
        f"expected finalize -> push_to_hub, got {calls}"
    )
    assert "pushed" in result.lower()


def test_cb_push_to_hub_skips_upload_when_finalize_raises(monkeypatch) -> None:
    """If finalize() raises, push_to_hub() must not run and the error is wrapped."""

    class _FakeGrError(Exception):
        pass

    fake_gr = types.SimpleNamespace(Error=_FakeGrError)
    monkeypatch.setitem(sys.modules, "gradio", fake_gr)

    from so101_nexus_core.teleop.app import _cb_push_to_hub

    calls: list[str] = []

    class _BadFinalizeDataset:
        def finalize(self) -> None:
            calls.append("finalize")
            raise RuntimeError("writer already closed")

        def push_to_hub(self) -> None:
            calls.append("push_to_hub")

    with pytest.raises(_FakeGrError) as excinfo:
        _cb_push_to_hub({"dataset": _BadFinalizeDataset()})

    assert calls == ["finalize"]
    assert "Failed to push to Hub" in str(excinfo.value)
    assert "writer already closed" in str(excinfo.value)


def test_finalize_and_close_after_push_is_idempotent(fake_gradio) -> None:
    """Push -> Finalize & Close should not error even though finalize ran twice."""
    from so101_nexus_core.teleop.app import _cb_finalize_and_close, _cb_push_to_hub

    finalize_calls = {"n": 0}
    disconnect_calls = {"n": 0}

    class _Dataset:
        def finalize(self) -> None:
            finalize_calls["n"] += 1

        def push_to_hub(self) -> None:
            pass

    class _Leader:
        def disconnect(self) -> None:
            disconnect_calls["n"] += 1

    session = {"dataset": _Dataset(), "leader": _Leader()}

    _cb_push_to_hub(session)
    result = _cb_finalize_and_close(session)

    assert finalize_calls["n"] == 2
    assert disconnect_calls["n"] == 1
    assert "finalized" in result.lower()


def test_poll_recording_failed_empty_episode_warns_without_plot_error(
    fake_gradio, monkeypatch
) -> None:
    import so101_nexus_core.teleop.app as app_mod

    warnings: list[str] = []
    fake_gradio.Warning = warnings.append

    monkeypatch.setattr(app_mod, "make_review_video", lambda _images, _fps: None)

    def _fail_on_empty_plot(*_args, **_kwargs):
        raise AssertionError("empty failed recordings should not build a plot")

    monkeypatch.setattr(app_mod, "make_state_plot", _fail_on_empty_plot)

    state = RecordingState(recording_finished=True, error="RuntimeError: boom", num_episodes=1)
    session = {
        "state": state,
        "fps": 30,
        "joint_names": ("joint_a",),
    }

    outputs = _cb_poll_recording(session)

    assert len(outputs) == 9
    assert warnings == ["Recording failed: RuntimeError: boom"]
    assert state.error is None
    assert outputs[7]["value"] is None
