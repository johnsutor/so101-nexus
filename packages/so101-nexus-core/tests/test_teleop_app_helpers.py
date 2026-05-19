"""Unit tests for pure helpers in so101_nexus_core.teleop.app.

The ``app`` module is designed to be importable on a base install; gradio,
lerobot, and cv2 are imported lazily inside ``main()`` and individual
callbacks. These tests exercise the pure-logic helpers that need no
gradio runtime.
"""

from __future__ import annotations

import logging
import sys
import types

import numpy as np
import pytest

import so101_nexus_core.teleop.app as teleop_app
from so101_nexus_core.config import PickAndPlaceConfig, PickConfig
from so101_nexus_core.objects import CubeObject, YCBObject
from so101_nexus_core.teleop.app import (
    _build_field_selection,
    _build_record_step,
    _build_setup_screen,
    _cb_approve_episode,
    _cb_discard_episode,
    _cb_poll_init,
    _cb_poll_recording,
    _cb_retry_init,
    _connect_leader,
    _create_dataset,
    _default_env_id,
    _import_env_modules,
    _merge_extra_env_ids,
    _normalized_init_config,
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


def test_default_follower_calibration_dir_uses_env_override(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HF_LEROBOT_CALIBRATION", str(tmp_path))

    assert teleop_app._default_follower_calibration_dir() == (
        tmp_path / "robots" / "sim_so_follower"
    )


def test_run_init_worker_creates_canonical_lerobot_features(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class _Leader:
        def disconnect(self) -> None:
            pass

    class _Dataset:
        pass

    def _fake_create_dataset(repo_id, fps, robot_type, features, leader):
        seen.update(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            leader=leader,
        )
        return _Dataset()

    monkeypatch.setattr(teleop_app, "import_backend_for_env_id", lambda _env_id: None)
    monkeypatch.setattr(teleop_app, "_connect_leader", lambda *_args: _Leader())
    monkeypatch.setattr(teleop_app, "_create_dataset", _fake_create_dataset)

    session: dict = {}
    init_state: dict = {}

    teleop_app._run_init_worker(
        session,
        init_state,
        "/dev/ttyACM0",
        "MuJoCoReach-v1",
        "so101",
        "leader",
        30,
        (320, 240),
        (640, 360),
        "local/test",
        1,
        "joint_pos",
        10,
        0,
        -90.0,
        teleop_app.FieldSelection(),
        None,
    )

    features = seen["features"]
    assert set(features) >= {
        "action",
        "observation.state",
        WRIST_KEY,
        OVERHEAD_KEY,
    }
    assert features["action"]["names"][0] == "shoulder_pan.pos"
    assert features[WRIST_KEY]["shape"] == (240, 320, 3)
    assert session["dataset"].__class__ is _Dataset
    assert init_state["done"] is True
    assert init_state.get("error") is None


def test_start_recording_passes_follower_config_kwargs(monkeypatch, tmp_path, fake_gradio) -> None:
    captured: dict[str, object] = {}

    class _Thread:
        def __init__(self, *, target, args=(), kwargs=None, daemon=False):
            captured.update(target=target, args=args, kwargs=kwargs or {}, daemon=daemon)

        def start(self) -> None:
            captured["started"] = True

    monkeypatch.setattr(teleop_app.threading, "Thread", _Thread)
    monkeypatch.setattr(teleop_app, "_default_follower_calibration_dir", lambda: tmp_path)

    state = RecordingState(num_episodes=1)
    session = {
        "state": state,
        "env_id": "MuJoCoReach-v1",
        "leader": object(),
        "joint_names": ("a", "b"),
        "fps": 30,
        "max_steps": 5,
        "countdown": 0,
        "wrist_roll_offset_deg": -90.0,
        "wrist_wh": (320, 240),
        "overhead_wh": (640, 360),
    }

    teleop_app._cb_start_recording(session)

    assert captured["started"] is True
    assert captured["args"] == ()
    kwargs = captured["kwargs"]
    assert kwargs["follower_calibration_dir"] == tmp_path
    assert kwargs["follower_robot_id"] == "teleop_sim"
    assert "action_pipeline" not in kwargs


def test_setup_screen_defaults_to_absolute_joint_position(monkeypatch) -> None:
    class _FakeComponent:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.value = kwargs.get("value")
            self.label = kwargs.get("label")

    class _Context:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_exc_info):
            return False

    radios: list[_FakeComponent] = []
    sliders: list[_FakeComponent] = []

    def _radio(*args, **kwargs):
        component = _FakeComponent(*args, **kwargs)
        radios.append(component)
        return component

    def _slider(*args, **kwargs):
        component = _FakeComponent(*args, **kwargs)
        sliders.append(component)
        return component

    fake_gr = types.SimpleNamespace(
        Markdown=_FakeComponent,
        Dropdown=_FakeComponent,
        Radio=_radio,
        Row=_Context,
        Number=_FakeComponent,
        Slider=_slider,
        Textbox=_FakeComponent,
        CheckboxGroup=_FakeComponent,
        Checkbox=_FakeComponent,
        Accordion=_Context,
        Group=_Context,
        Button=_FakeComponent,
    )
    monkeypatch.setattr(
        teleop_app,
        "_customization_ui_state_for_env",
        lambda _env_id: teleop_app.CustomizationUIState(),
    )

    _build_setup_screen(fake_gr, ["MuJoCoReach-v1"], "leader", -90.0)

    action_space_radio = next(radio for radio in radios if radio.label == "Action Space")
    assert action_space_radio.value == "joint_pos"
    reset_settle_slider = next(
        slider for slider in sliders if slider.label == "Reset Settle Frames"
    )
    assert reset_settle_slider.value == 5


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


def test_import_env_modules_imports_each_module(monkeypatch) -> None:
    imported: list[str] = []
    monkeypatch.setattr("so101_nexus_core.teleop.app.importlib.import_module", imported.append)

    _import_env_modules(["custom_a", "custom_b"])

    assert imported == ["custom_a", "custom_b"]


def test_merge_extra_env_ids_appends_unique_ids() -> None:
    assert _merge_extra_env_ids(["A-v1"], ["A-v1", "B-v1"]) == ["A-v1", "B-v1"]


def test_normalized_init_config_includes_env_overrides() -> None:
    config = _normalized_init_config(
        "leader",
        "MuJoCoPickLift-v1",
        "so101",
        "",
        30,
        320,
        240,
        640,
        480,
        "",
        1,
        "joint_pos",
        100,
        0,
        -90,
        [],
        True,
        ["cube:green", "ycb:011_banana"],
        1,
        ["white"],
        ["yellow", "orange"],
        0.15,
        0.25,
        60,
        5,
        ["red", "green"],
        ["blue"],
    )

    overrides = config["env_overrides"]
    assert overrides.object_specs == ("cube:green", "ycb:011_banana")
    assert overrides.n_distractors == 1
    assert overrides.ground_colors == ("white",)
    assert overrides.robot_colors == ("yellow", "orange")
    assert overrides.spawn_min_radius == 0.15
    assert overrides.spawn_max_radius == 0.25
    assert overrides.spawn_angle_half_range_deg == 60
    assert overrides.reset_settle_frames == 5
    assert overrides.cube_colors == ("red", "green")
    assert overrides.target_colors == ("blue",)


def test_normalized_init_config_rounds_reset_settle_frames_from_ui_float() -> None:
    config = _normalized_init_config(
        "leader",
        "MuJoCoPickLift-v1",
        "so101",
        "",
        30,
        320,
        240,
        640,
        480,
        "",
        1,
        "joint_pos",
        100,
        0,
        -90,
        [],
        True,
        ["cube:red"],
        0,
        ["gray"],
        ["yellow"],
        0.10,
        0.30,
        90,
        2.999,
        ["red"],
        ["blue"],
    )

    assert config["env_overrides"].reset_settle_frames == 3


def test_normalized_init_config_leaves_env_overrides_disabled_by_default() -> None:
    config = _normalized_init_config(
        "leader",
        "MuJoCoPickLift-v1",
        "so101",
        "",
        30,
        320,
        240,
        640,
        480,
        "",
        1,
        "joint_pos",
        100,
        0,
        -90,
        [],
        False,
        ["cube:red"],
        0,
        ["gray"],
        ["yellow"],
        0.10,
        0.30,
        90,
        5,
        ["red"],
        ["blue"],
    )

    assert config["env_overrides"] is None


def test_customization_ui_state_for_pick_config_uses_base_config_defaults() -> None:
    state = teleop_app._customization_ui_state_from_config(
        PickConfig(
            objects=[CubeObject(color="green"), YCBObject(model_id="011_banana")],
            n_distractors=1,
            ground_colors=["white"],
            robot_colors="yellow",
            spawn_min_radius=0.12,
            spawn_max_radius=0.28,
            spawn_angle_half_range_deg=45,
            reset_settle_frames=7,
        )
    )

    assert state.customize_visible is True
    assert state.customize_value is True
    assert state.common_visible is True
    assert state.pick_visible is True
    assert state.pick_and_place_visible is False
    assert state.object_specs == ["cube:green", "ycb:011_banana"]
    assert state.n_distractors == 1
    assert state.ground_colors == ["white"]
    assert state.robot_colors == ["yellow"]
    assert state.spawn_min_radius == 0.12
    assert state.spawn_max_radius == 0.28
    assert state.spawn_angle_half_range_deg == 45
    assert state.reset_settle_frames == 7


def test_customization_ui_state_for_pick_and_place_hides_pick_controls() -> None:
    state = teleop_app._customization_ui_state_from_config(
        PickAndPlaceConfig(
            cube_colors=["red", "green"],
            target_colors="blue",
            ground_colors="gray",
            robot_colors=["yellow", "orange"],
        )
    )

    assert state.customize_visible is True
    assert state.customize_value is True
    assert state.common_visible is True
    assert state.pick_visible is False
    assert state.pick_and_place_visible is True
    assert state.cube_colors == ["red", "green"]
    assert state.target_colors == ["blue"]
    assert state.ground_colors == ["gray"]
    assert state.robot_colors == ["yellow", "orange"]


def test_customization_ui_state_without_config_hides_customization() -> None:
    state = teleop_app._customization_ui_state_from_config(None)

    assert state.customize_visible is False
    assert state.customize_value is False
    assert state.common_visible is False
    assert state.pick_visible is False
    assert state.pick_and_place_visible is False


def test_customization_ui_state_for_env_logs_resolution_failure(monkeypatch, caplog) -> None:
    def _raise_on_resolve(_env_id: str):
        raise RuntimeError("broken default config")

    monkeypatch.setattr(teleop_app, "_resolve_env_ctor", _raise_on_resolve)

    with caplog.at_level(logging.WARNING, logger=teleop_app.__name__):
        state = teleop_app._customization_ui_state_for_env("BrokenEnv-v1")

    assert state == teleop_app.CustomizationUIState()
    assert "BrokenEnv-v1" in caplog.text
    assert "broken default config" in caplog.text


def test_normalized_init_config_rejects_invalid_ui_color() -> None:
    with pytest.raises(ValueError, match="unknown ground_colors"):
        _normalized_init_config(
            "leader",
            "MuJoCoPickLift-v1",
            "so101",
            "",
            30,
            320,
            240,
            640,
            480,
            "",
            1,
            "joint_pos",
            100,
            0,
            -90,
            [],
            True,
            ["cube:red"],
            0,
            ["not-a-color"],
            ["yellow"],
            0.10,
            0.30,
            90,
            5,
            ["red"],
            ["blue"],
        )


def test_normalized_init_config_rejects_gray_pick_and_place_ui_color() -> None:
    with pytest.raises(ValueError, match="unknown cube_colors"):
        _normalized_init_config(
            "leader",
            "MuJoCoPickLift-v1",
            "so101",
            "",
            30,
            320,
            240,
            640,
            480,
            "",
            1,
            "joint_pos",
            100,
            0,
            -90,
            [],
            True,
            ["cube:red"],
            0,
            ["gray"],
            ["yellow"],
            0.10,
            0.30,
            90,
            5,
            ["gray"],
            ["blue"],
        )


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
        "log_text": "Connecting leader arm...",
        "running": True,
        "warning": None,
    }

    outputs = _cb_poll_init({}, init_state)

    assert "Connecting leader arm" in outputs[0]["value"]
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
    assert len(outputs) == 13
    assert outputs[2]["visible"] is True
    assert "Get ready" in outputs[2]["value"]
    assert "Task will appear" in outputs[5]["value"]


def test_poll_recording_emits_single_preview_during_recording(fake_gradio) -> None:
    state = RecordingState(is_recording=True, num_episodes=5, episodes_completed=1)
    state.live_preview = np.full((90, 160, 3), 200, dtype=np.uint8)
    state.episode_actions.append(np.zeros(6, dtype=np.float32))
    session = {
        "state": state,
        "fps": 30,
    }

    outputs = _cb_poll_recording(session)

    assert len(outputs) == 13
    assert "Recording episode 2/5" in outputs[0]["value"]
    assert "Task will appear" in outputs[5]["value"]
    assert outputs[2]["visible"] is False
    assert outputs[3]["visible"] is True
    np.testing.assert_array_equal(outputs[3]["value"], state.live_preview)
    assert outputs[4]["visible"] is True


def test_poll_recording_shows_current_task_during_recording(fake_gradio) -> None:
    state = RecordingState(is_recording=True, num_episodes=3, episodes_completed=0)
    state.task_description = "Pick up the red cube."
    state.episode_actions.append(np.zeros(6, dtype=np.float32))
    session = {
        "state": state,
        "fps": 30,
    }

    outputs = _cb_poll_recording(session)

    assert "Task:" not in outputs[0]["value"]
    assert outputs[5]["value"] == "**Task:** Pick up the red cube."


def test_poll_recording_waits_for_real_preview_before_showing_image(fake_gradio) -> None:
    state = RecordingState(is_recording=True, num_episodes=1)
    session = {
        "state": state,
        "fps": 30,
    }

    outputs = _cb_poll_recording(session)

    assert len(outputs) == 13
    assert "Waiting for camera frame" in outputs[0]["value"]
    assert "Task will appear" in outputs[5]["value"]
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
    task_status = components[5]
    assert countdown_area.value == ""
    assert countdown_area.visible is False
    assert "Task will appear" in task_status.value


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
    assert len(outputs) == 6
    assert outputs[1]["value"].startswith("Episode discarded.")
    assert outputs[3]["visible"] is True
    assert outputs[4]["visible"] is False
    assert outputs[4]["value"] is None
    assert outputs[5]["visible"] is False
    assert outputs[5]["value"] == ""


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
    assert len(outputs) == 9
    assert outputs[1]["value"].startswith("Episode saved!")
    assert outputs[4]["visible"] is True
    assert outputs[5]["visible"] is False
    assert outputs[5]["value"] is None
    assert outputs[6]["visible"] is False
    assert outputs[7]["interactive"] is True
    assert outputs[8]["interactive"] is True


def test_approve_episode_failure_reenables_review_controls(fake_gradio) -> None:
    class _Dataset:
        def __init__(self) -> None:
            self.frames = []
            self.saved = 0

        def add_frame(self, frame) -> None:
            self.frames.append(frame)

        def save_episode(self) -> None:
            self.saved += 1

    from so101_nexus_core.teleop.dataset import FieldSelection

    state = RecordingState(num_episodes=1, episodes_completed=0)
    state.episode_actions.append(np.zeros(6, dtype=np.float32))
    state.episode_states.append(np.zeros(6, dtype=np.float32))
    dataset = _Dataset()
    session = {
        "state": state,
        "dataset": dataset,
        "action_space": "joint_pos",
        "field_selection": FieldSelection(wrist_image=True, overhead_image=False, task=False),
        "env_id": "ManiSkillLookAtSO101-v1",
        "fps": 30,
    }

    outputs = _cb_approve_episode(session)

    assert dataset.frames == []
    assert dataset.saved == 0
    assert len(outputs) == 9
    assert outputs[6]["visible"] is True
    assert "Failed to save episode" in outputs[6]["value"]
    assert "wrist_image selected" in outputs[6]["value"]
    assert outputs[7]["interactive"] is True
    assert outputs[8]["interactive"] is True


def test_prepare_episode_approval_shows_saving_status(fake_gradio) -> None:
    outputs = teleop_app._cb_prepare_episode_approval()

    assert outputs[0]["visible"] is True
    assert "Saving episode" in outputs[0]["value"]
    assert outputs[1]["interactive"] is False
    assert outputs[2]["interactive"] is False


def test_prepare_push_and_finalize_show_busy_status(fake_gradio) -> None:
    push_status = teleop_app._cb_prepare_push_to_hub()
    finalize_status = teleop_app._cb_prepare_finalize_and_close()

    assert "Pushing dataset" in push_status["value"]
    assert "Finalizing dataset" in finalize_status["value"]


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

    assert calls == ["finalize", "push_to_hub"], f"expected finalize -> push_to_hub, got {calls}"
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

    assert len(outputs) == 13
    assert warnings == ["Recording failed: RuntimeError: boom"]
    assert state.error is None
    assert outputs[7]["value"] is None
