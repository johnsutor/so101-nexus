"""Dynamics callbacks recover completed attempts without touching real motor buses."""

from types import SimpleNamespace

import numpy as np
import pytest

from so101_nexus.calibration.session import CalibrationSession
from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.profiles import RobotProfile


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    gr = pytest.importorskip("gradio")
    import plotly.graph_objects as go

    from so101_nexus.calibration.app import build_app

    profile = RobotProfile(
        "fake-recovery",
        {
            name: {
                "id": index,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095,
            }
            for index, name in enumerate(SO101_JOINT_NAMES, 1)
        },
        joint_offsets_deg=(1, 2, 3, 4, 5),
    )
    torque = dict.fromkeys(SO101_JOINT_NAMES, 0)
    positions = np.array([5, -20, 35, 40, 10, 50.0])
    monitors = []
    releases = []

    class Monitor:
        def __init__(self, port, directory):
            assert port == "/dev/FAKE-recovery"
            self.profile, self.torque, self.closed = profile, torque, False
            self.calibration_path = tmp_path / "fake.json"
            monitors.append(self)

        def connect(self):
            return self.profile

        def read(self):
            assert not self.closed
            return positions.copy()

        def close(self):
            self.closed = True

    class Session(CalibrationSession):
        def _make_robot(self, profile, directory):
            return SimpleNamespace(bus=SimpleNamespace(is_connected=False))

        def release(self, *, supported):
            self._idle()
            if not supported:
                raise ValueError("Support the fake arm before releasing motors.")
            self._close_monitor()
            releases.append(True)
            torque.update(dict.fromkeys(SO101_JOINT_NAMES, 0))
            self.torque_possible = False
            self.phase = "disconnected"

    session = Session(monitor_factory=Monitor)
    session.connect("/dev/FAKE-recovery", str(tmp_path))
    session.confirmed = set(range(5))
    session.gripper_confirmed = True
    saved = session.save_profile(tmp_path / "profile.json")
    monkeypatch.setattr("so101_nexus.calibration.session.preflight_excitation", lambda *args: None)
    monkeypatch.setattr("so101_nexus.calibration.app.render_overlay", lambda *a, **k: go.Figure())
    app = build_app(session)
    config = app.get_config_file()
    tick_dependency = next(
        item
        for item in config["dependencies"]
        if any(event == "tick" for _, event in item["targets"])
    )
    tick = app.fns[tick_dependency["id"]].fn
    guide, viewport = tick.args[1:3]
    guide.stage = 3

    def callback(label):
        component = next(
            item
            for item in config["components"]
            if item["type"] == "button" and item["props"].get("value") == label
        )
        dependency = next(
            item for item in config["dependencies"] if (component["id"], "click") in item["targets"]
        )
        return app.fns[dependency["id"]].fn

    settings = [0, -15, 30, 45, 0, 50, *([8] * 6), 1, 0, 0, 0.02]
    callback("Dynamics tuning")()
    return SimpleNamespace(
        gr=gr,
        session=session,
        callback=callback,
        tick=lambda: tick("multisine", 0, 1.5),
        viewport=viewport,
        positions=positions,
        torque=torque,
        monitors=monitors,
        releases=releases,
        saved=saved,
        settings=settings,
        directory=tmp_path,
    )


def test_motion_check_failure_can_retry_using_current_live_model_pose(workflow, monkeypatch):
    ui = workflow
    check = ui.callback("Preview motion without enabling hold")
    monkeypatch.setattr(
        "so101_nexus.calibration.session.preflight_excitation",
        lambda *args: (_ for _ in ()).throw(ValueError("tool clearance violated")),
    )
    with pytest.raises(ui.gr.Error, match="tool clearance"):
        check(False, *ui.settings)
    assert ui.session.monitor is ui.monitors[-1]
    assert not ui.session.verified
    assert "Adjust the supported pose" in ui.session.status
    ui.positions[1] -= 10
    monkeypatch.setattr("so101_nexus.calibration.session.preflight_excitation", lambda *args: None)
    result = check(False, *ui.settings)
    expected = [6, -28, 38, 44, 15, 45]
    assert list(result[2:8]) == expected
    assert ui.session.config.center_qpos_deg == tuple(expected)
    assert ui.session.verified
    assert result[1].selected == 0
    assert ui.releases == []


def test_partial_attempt_reconnect_requires_explicit_fake_release_before_new_preview(
    workflow, monkeypatch
):
    ui = workflow
    checked = ui.callback("Preview motion without enabling hold")(False, *ui.settings)
    settings = [*checked[2:8], *ui.settings[6:]]
    destination = ui.directory / "attempt"

    def collect(*args, **kwargs):
        destination.mkdir()
        (destination / "00_multisine.npz").write_bytes(b"completed fake episode")
        ui.torque.update(dict.fromkeys(SO101_JOINT_NAMES, 1))
        raise RuntimeError("fake tracking failure after completed episode")

    monkeypatch.setattr("so101_nexus.calibration.session.collect_excitation", collect)
    ui.callback("Arm supported, prepare robot")(str(destination), False, *settings)
    ui.session.worker.join(2)
    assert not ui.session.worker.is_alive()
    assert ui.session.phase == "stopped"
    assert ui.session.monitor is None
    assert (destination / "source_profile.json").is_file()
    assert ui.tick()[20]["visible"] is True
    retry = ui.callback("Reconnect and prepare another attempt")
    restored = retry("/dev/FAKE-recovery", str(ui.directory))
    assert restored.selected == 1
    assert ui.session.profile_path == ui.saved
    assert ui.tick()[21]["visible"] is True
    release = ui.callback("Release motors and prepare another attempt")
    with pytest.raises(ui.gr.Error, match="Support"):
        release(False, "/dev/FAKE-recovery", str(ui.directory))
    assert ui.releases == []
    result = release(True, "/dev/FAKE-recovery", str(ui.directory))
    assert result[0].selected == 0
    assert ui.releases == [True]
    assert not any(ui.torque.values())
    assert ui.tick()[21]["visible"] is False
    assert str(destination / "00_multisine.npz") in ui.viewport.previous_artifacts
    assert str(destination / "source_profile.json") in ui.viewport.previous_artifacts
    assert ui.callback("Preview motion without enabling hold")(False, *ui.settings)[1].selected == 0
    assert (destination / "00_multisine.npz").read_bytes() == b"completed fake episode"


def test_torque_free_retry_restores_profile_and_retains_prior_downloads(workflow):
    ui = workflow
    previous = ui.directory / "previous.npz"
    previous.write_bytes(b"previous recording")
    ui.session.artifacts.append(str(previous))
    ui.session._close_monitor()
    ui.session.phase = "stopped"
    restored = ui.callback("Reconnect and prepare another attempt")(
        "/dev/FAKE-recovery", str(ui.directory)
    )
    assert restored.selected == 0
    assert ui.session.profile_path == ui.saved
    assert ui.session.confirmed == set(range(5))
    assert not ui.session.verified
    assert str(previous) in ui.viewport.previous_artifacts
    assert ui.tick()[21]["visible"] is False
    assert ui.releases == []


def test_busy_collection_rejects_retry_without_closing_reader(workflow):
    ui = workflow
    monitor = ui.session.monitor
    ui.session.worker = SimpleNamespace(is_alive=lambda: True)
    with pytest.raises(ui.gr.Error, match="worker"):
        ui.callback("Reconnect and prepare another attempt")(
            "/dev/FAKE-recovery", str(ui.directory)
        )
    assert ui.session.monitor is monitor
    assert not monitor.closed
    assert ui.session.profile_path == ui.saved
    assert ui.releases == []


def test_one_start_records_training_and_validation_then_opens_tuning(workflow, monkeypatch):
    import threading

    ui = workflow
    ready = threading.Event()
    checks = []
    monkeypatch.setattr(
        "so101_nexus.calibration.session.preflight_excitation",
        lambda trajectories, *args: checks.append(tuple(t.name for t in trajectories)),
    )
    wait = ui.session.motion.wait
    monkeypatch.setattr(ui.session.motion, "wait", lambda timeout: (ready.set(), wait(timeout))[1])
    destination = ui.directory / "complete_dataset"

    def collect(robot, profile, config, **kwargs):
        kwargs["on_ready"]()
        destination.mkdir()
        for index, name in enumerate(("multisine", "joint_chirps", "heldout")):
            kwargs["on_progress"](name, 10, 10)
            (destination / f"{index:02d}_{name}.npz").write_bytes(b"fake completed recording")
        return ()

    monkeypatch.setattr("so101_nexus.calibration.session.collect_excitation", collect)
    prepared = ui.callback("Arm supported, prepare robot")(str(destination), False, *ui.settings)
    assert ready.wait(2)
    assert ui.session.phase == "holding"
    assert not ui.session.motion.is_set()
    assert not ui.session.verified
    assert checks == [("multisine", "joint_chirps", "heldout")]
    np.testing.assert_allclose(prepared[2:8], ui.session.profile.to_model_deg(ui.positions))
    actual_settings = [*prepared[2:8], *ui.settings[6:]]
    ui.callback("Hands clear, record calibration")(*actual_settings)
    ui.session.worker.join(2)
    assert not ui.session.worker.is_alive()
    assert len(list(destination.glob("*.npz"))) == 3
    assert ui.session.collection_complete
    updates = ui.tick()
    assert ui.viewport.tuning_step == 2
    assert updates[23].selected == 2
    assert updates[24] == str(destination)
    assert "3 of 3" in updates[25]
    assert updates[26]["interactive"]


@pytest.mark.parametrize("count,complete", [(1, False), (2, True), (3, False)])
def test_partial_dataset_never_advances_to_tuning(workflow, count, complete):
    ui = workflow
    destination = ui.directory / "partial_dataset"
    destination.mkdir()
    for index, name in enumerate(("multisine", "joint_chirps", "heldout")[:count]):
        (destination / f"{index:02d}_{name}.npz").write_bytes(b"partial dataset")
    ui.session.recordings_dir = destination
    ui.session.phase = "stopped"
    ui.session.collection_complete = complete
    ui.viewport.tuning_step = 1
    updates = ui.tick()
    assert ui.viewport.tuning_step == 1
    assert not updates[26]["interactive"]
    assert f"{count} of 3" in updates[25]


def test_prepare_never_enables_hold_when_motion_check_fails(workflow, monkeypatch):
    from unittest.mock import Mock

    ui = workflow
    arm = Mock(side_effect=AssertionError("must not arm"))
    monkeypatch.setattr(ui.session, "arm", arm)
    monkeypatch.setattr(
        "so101_nexus.calibration.session.preflight_excitation",
        lambda *args: (_ for _ in ()).throw(ValueError("table collision")),
    )
    with pytest.raises(ui.gr.Error, match="table collision"):
        ui.callback("Arm supported, prepare robot")(
            str(ui.directory / "never_created"), False, *ui.settings
        )
    arm.assert_not_called()
    assert ui.session.monitor is ui.monitors[-1]
    assert ui.session.phase == "monitoring"
    assert not ui.session.verified


def test_completed_dataset_can_be_tuned_after_disconnect(workflow):
    ui = workflow
    directory = ui.directory / "saved_dataset"
    directory.mkdir()
    for index, name in enumerate(("multisine", "joint_chirps", "heldout")):
        (directory / f"{index:02d}_{name}.npz").write_bytes(b"saved recording")
    ui.session.recordings_dir = directory
    ui.session.collection_complete = True
    ui.session.disconnect()
    assert ui.session.monitor is None
    updates = ui.tick()
    assert ui.viewport.tuning
    assert ui.viewport.tuning_step == 2
    assert updates[26]["interactive"]
    assert updates[24] == str(directory)


def test_worker_failure_publishes_queued_warning_and_persistent_alert(workflow, monkeypatch):
    from unittest.mock import Mock

    import gradio as gr

    ui = workflow

    def fail(*args, **kwargs):
        raise RuntimeError("hardware loop overrun: 70.0 ms late, limit 50.0 ms")

    monkeypatch.setattr("so101_nexus.calibration.session.collect_excitation", fail)
    ui.callback("Arm supported, prepare robot")(str(ui.directory / "failed"), False, *ui.settings)
    ui.session.worker.join(2)
    updates = ui.tick()
    assert "70.0 ms" in ui.session.collection_error
    event, alert = updates[-2:]
    assert event["message"] == ui.session.collection_error
    assert alert["visible"]
    assert 'role="alert"' in alert["value"]
    assert "Recording stopped" in alert["value"]
    warning = Mock()
    monkeypatch.setattr(gr, "Warning", warning)
    from so101_nexus.calibration.app import _notify_collection_error

    _notify_collection_error(event)
    warning.assert_called_once_with(event["message"], title="Recording stopped", duration=None)
    assert ui.tick()[-2] == event
    ui.callback("Reconnect and prepare another attempt")("/dev/FAKE-recovery", str(ui.directory))
    assert ui.tick()[-2] is None
    assert not ui.tick()[-1]["visible"]
    ui.callback("Arm supported, prepare robot")(str(ui.directory / "failed"), False, *ui.settings)
    ui.session.worker.join(2)
    repeated = ui.tick()[-2]
    assert repeated["sequence"] == event["sequence"] + 1
    assert repeated["message"] == event["message"]
