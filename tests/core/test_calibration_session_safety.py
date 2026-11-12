"""Session faults retain ownership and invalidate stale hardware approvals."""

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from so101_nexus.calibration.hardware import PortLease
from so101_nexus.calibration.session import CalibrationSession
from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.excitation import ExcitationConfig
from so101_nexus.profiles import RobotProfile


@pytest.fixture
def session():
    profile = RobotProfile(
        robot_id="safety",
        calibration={
            name: {
                "id": index,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095,
            }
            for index, name in enumerate(SO101_JOINT_NAMES, 1)
        },
    )

    class Monitor:
        def __init__(self, *args):
            self.profile = profile
            self.torque = dict.fromkeys(SO101_JOINT_NAMES, 0)
            self.calibration_path = Path("safety.json")
            self.fail_close = False
            self.closed = False

        def connect(self):
            return self.profile

        def read(self):
            return np.array([1, 2, 3, 4, 5, 50.0])

        def close(self):
            if self.fail_close:
                raise OSError("serial close failed")
            self.closed = True

    result = CalibrationSession(monitor_factory=Monitor)
    result.connect("/dev/fake-safety", "/tmp")
    result.confirmed = set(range(5))
    result.gripper_confirmed = True
    result.set_plan(ExcitationConfig(center_qpos_deg=(0, -15, 30, 45, 0, 50)))
    result.verified = True
    return result


def test_failed_monitor_close_retains_owner_and_discards_measurement(session):
    monitor = session.monitor
    monitor.fail_close = True
    with pytest.raises(OSError, match="serial close"):
        session.disconnect()
    assert session.monitor is monitor
    assert session.measured is None
    assert not session.verified
    monitor.fail_close = False
    session.disconnect()
    assert monitor.closed
    assert session.monitor is None


def test_stop_invalidates_preflight_and_idle_stop_does_not_claim_worker_cleanup(session):
    session.stop()
    assert not session.verified
    assert "waiting" not in session.status.lower()
    with pytest.raises(ValueError, match="preflight"):
        session.arm("/tmp/never-created-safety", supported=True)


def test_capture_checks_torque_after_fresh_read(session):
    original = session.monitor.read

    def read():
        session.monitor.torque["shoulder_pan"] = 1
        return original()

    session.monitor.read = read
    with pytest.raises(ValueError, match="torque off"):
        session.capture_joint(0, 0, confirmed=True)
    assert session.profile.joint_offsets_deg == (0, 0, 0, 0, 0)


def test_thread_start_failure_releases_serial_lease(session, monkeypatch, tmp_path):
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: (_ for _ in ()).throw(RuntimeError("cannot start"))
    )
    with pytest.raises(RuntimeError, match="cannot start"):
        session.arm(tmp_path / "recordings", supported=True)
    assert not session.verified
    assert session.phase == "stopped"
    with PortLease(session.port):
        pass


def test_failed_connect_cleanup_retains_monitor_reference(session):
    session.disconnect()
    monitor = session.monitor_factory()
    monitor.connect = lambda: (_ for _ in ()).throw(OSError("connect failed"))
    monitor.fail_close = True
    session.monitor_factory = lambda *args: monitor
    with pytest.raises(OSError, match="connect failed") as raised:
        session.connect("/dev/fake-safety", "/tmp")
    assert session.monitor is monitor
    assert session.measured is None
    assert any("cleanup" in note.lower() for note in raised.value.__notes__)
    monitor.fail_close = False
    session.disconnect()


def test_saved_profile_load_uses_live_matching_calibration(session, tmp_path):
    path = tmp_path / "saved.json"
    saved = replace(session.profile, joint_offsets_deg=(5, 4, 3, 2, 1), control_period_s=1 / 30)
    saved.save(path)
    session.confirmed.clear()
    session.gripper_confirmed = False
    session.load_profile(path)
    assert session.profile == saved
    assert session.confirmed == set(range(5))
    assert session.gripper_confirmed
    assert not session.verified
    assert session.profile_path == path.resolve()
    np.testing.assert_allclose(session.use_current_center(), [6, 6, 6, 6, 6, 45])


def test_saved_profile_mismatch_revokes_preflight_and_old_confirmation(session, tmp_path):
    path = tmp_path / "different.json"
    payload = session.profile.to_dict()
    payload["calibration"]["wrist_roll"]["homing_offset"] = 1
    RobotProfile(**payload).save(path)
    with pytest.raises(ValueError, match="calibration"):
        session.load_profile(path)
    assert not session.verified
    assert not session.confirmed


def test_loading_requires_live_monitor_and_center_requires_alignment(session, tmp_path):
    path = tmp_path / "saved.json"
    session.profile.save(path)
    session.disconnect()
    with pytest.raises(RuntimeError, match="read-only"):
        session.load_profile(path)
    with pytest.raises(RuntimeError, match="read-only"):
        session.use_current_center()


def test_period_change_invalidates_and_rejects_nonphysical_values(session):
    session.set_period(1 / 30)
    assert session.profile.control_period_s == 1 / 30
    assert not session.verified
    with pytest.raises(ValueError, match=r"positive|>"):
        session.set_period(0)


def test_fitting_does_not_block_readonly_polling(session, monkeypatch, tmp_path):
    import so101_nexus.sysid

    recording = tmp_path / "recordings"
    recording.mkdir()
    (recording / "00.npz").touch()
    monkeypatch.setattr(
        "so101_nexus.calibration.session.RecordedTrajectory.load", lambda path: object()
    )
    entered = threading.Event()
    finish = threading.Event()
    fit_errors = []

    def fit(*args, **kwargs):
        entered.set()
        assert finish.wait(2)
        return SimpleNamespace(episodes=(), diagnostics=(), validation_improved=False)

    monkeypatch.setattr(so101_nexus.sysid, "fit_robot_profile", fit)

    def run():
        try:
            session.fit(str(recording), str(tmp_path / "fit"), 1)
        except Exception as error:
            fit_errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    assert entered.wait(2)
    poll_done = threading.Event()
    poll = threading.Thread(target=lambda: (session.poll(), poll_done.set()))
    poll.start()
    try:
        assert poll_done.wait(0.2)
        session.disconnect()
        with pytest.raises(RuntimeError, match=r"fitting|worker"):
            session.set_period(0.02)
    finally:
        finish.set()
        worker.join(2)
        poll.join(2)
    assert not fit_errors


def test_collection_observations_update_live_pose_without_proxy(session):
    values = {f"{name}.pos": float(index) for index, name in enumerate(SO101_JOINT_NAMES)}

    class Robot:
        def get_observation(self):
            return values

    robot = Robot()
    session._track_observation(robot)
    assert isinstance(robot, Robot)
    assert robot.get_observation() is values
    np.testing.assert_array_equal(session.measured, np.arange(6))
    values["wrist_roll.pos"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        robot.get_observation()


def test_holding_read_failure_stops_before_motion_and_releases_port(session, monkeypatch, tmp_path):
    read_attempted = threading.Event()

    def read():
        read_attempted.set()
        raise OSError("holding encoder lost")

    robot = SimpleNamespace(get_observation=read)
    monkeypatch.setattr(session, "_make_robot", lambda *args: robot)
    monkeypatch.setattr(
        "so101_nexus.calibration.session.collect_excitation",
        lambda *args, **kwargs: kwargs["on_ready"](),
    )
    session.arm(tmp_path / "recordings", supported=True)
    assert read_attempted.wait(2)
    session.worker.join(2)
    assert session.phase == "stopped"
    assert "holding encoder lost" in session.status
    assert session.measured is None
    assert not session.motion.is_set()
    with PortLease(session.port):
        pass


def test_collection_failed_serial_close_retains_lease_until_disconnect(
    session, monkeypatch, tmp_path
):
    bus = SimpleNamespace(is_connected=True)

    def disconnect(*, disable_torque):
        assert disable_torque is False
        bus.is_connected = False

    bus.disconnect = disconnect
    monkeypatch.setattr(session, "_make_robot", lambda *args: SimpleNamespace(bus=bus))
    monkeypatch.setattr(
        "so101_nexus.calibration.session.collect_excitation",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("cleanup failed")),
    )
    session.arm(tmp_path / "recordings", supported=True)
    session.worker.join(2)
    with pytest.raises(RuntimeError, match="already owned"):
        PortLease(session.port).acquire()
    session.disconnect()
    assert not bus.is_connected
    with PortLease(session.port):
        pass


def test_explicit_release_failure_keeps_ownership_for_disconnect_retry(session, monkeypatch):
    import lerobot.motors

    bus = SimpleNamespace(is_connected=False, fail_close=True, releases=0)

    def connect(*, handshake):
        assert handshake is False
        bus.is_connected = True

    def release():
        bus.releases += 1

    def disconnect(*, disable_torque):
        assert disable_torque is False
        if bus.fail_close:
            raise OSError("release cleanup failed")
        bus.is_connected = False

    bus.connect, bus.disable_torque, bus.disconnect = connect, release, disconnect
    monkeypatch.setattr(lerobot.motors, "SerialMotorsBus", lambda **kwargs: bus)
    with pytest.raises(ValueError, match="Support"):
        session.release(supported=False)
    assert bus.releases == 0
    with pytest.raises(OSError, match="cleanup failed"):
        session.release(supported=True)
    assert bus.releases == 1
    with pytest.raises(RuntimeError, match="already owned"):
        PortLease(session.port).acquire()
    bus.fail_close = False
    session.disconnect()
    assert bus.releases == 1
    with PortLease(session.port):
        pass


def test_invalid_period_revokes_preflight_without_changing_profile(session):
    profile = session.profile
    assert session.verified
    with pytest.raises(ValueError, match="control_period_s"):
        session.set_period(float("nan"))
    assert session.profile is profile
    assert not session.verified


def test_follower_uses_bundled_calibration_gains_and_preserves_period(
    session, monkeypatch, tmp_path
):
    import json

    import lerobot.robots.so_follower

    captured = []
    monkeypatch.setattr(lerobot.robots.so_follower, "SO101Follower", captured.append)
    profile = replace(session.profile, control_period_s=1 / 30, servo_gains=(12, 3, 24))
    session._make_robot(profile, tmp_path)
    config = captured[0]
    assert config.calibration_dir == tmp_path
    assert config.id == profile.robot_id
    assert config.port == session.port
    assert config.use_degrees is True
    assert config.disable_torque_on_disconnect is False
    assert (
        config.position_p_coefficient,
        config.position_i_coefficient,
        config.position_d_coefficient,
    ) == profile.servo_gains
    assert json.loads((tmp_path / f"{profile.robot_id}.json").read_text()) == {
        name: dict(value) for name, value in profile.calibration.items()
    }
    assert profile.control_period_s == 1 / 30


def test_collection_ready_explicit_start_progress_and_completion(session, monkeypatch, tmp_path):
    entered = threading.Event()
    moved = threading.Event()
    wait_for_motion = session.motion.wait

    def wait(timeout):
        entered.set()
        return wait_for_motion(timeout)

    monkeypatch.setattr(session.motion, "wait", wait)
    robot = SimpleNamespace(
        get_observation=lambda: {
            f"{name}.pos": float(index) for index, name in enumerate(SO101_JOINT_NAMES)
        }
    )
    monkeypatch.setattr(session, "_make_robot", lambda *args: robot)

    def collect(robot, profile, config, **kwargs):
        kwargs["on_ready"]()
        moved.set()
        kwargs["on_progress"]("heldout", 10, 10)
        robot.get_observation()
        assert session.measured is not None
        return ()

    monkeypatch.setattr("so101_nexus.calibration.session.collect_excitation", collect)
    session.arm(tmp_path / "recordings", supported=True)
    assert entered.wait(2)
    with session.lock:
        assert session.phase == "holding"
    assert not moved.is_set()
    with pytest.raises(ValueError, match="hands"):
        session.start_motion(hands_clear=False)
    assert not moved.is_set()
    session.start_motion(hands_clear=True)
    session.worker.join(2)
    assert moved.is_set()
    assert session.phase == "stopped"
    assert session.progress == "heldout: 10/10 samples"
    assert "complete" in session.status
    assert not session.verified


def test_fitting_exposes_metrics_and_diagnostics_without_accepting_candidate(
    session, monkeypatch, tmp_path
):
    import so101_nexus.sysid
    from so101_nexus.sysid import EpisodeFit, FitResult, ParameterDiagnostic

    original = session.profile
    candidate = replace(original, joint_offsets_deg=(1, 2, 3, 4, 5))
    recording = tmp_path / "recordings"
    recording.mkdir()
    (recording / "00.npz").touch()
    monkeypatch.setattr(
        "so101_nexus.calibration.session.RecordedTrajectory.load", lambda path: object()
    )
    result = FitResult(
        candidate_profile=candidate,
        episodes=(EpisodeFit("heldout", "validation", (2.0,) * 6, (1.0,) * 6),),
        diagnostics=(
            ParameterDiagnostic("shoulder_pan.kp", 1.0, 2.0, (0.5, 3.0), False, 0.4, False, True),
        ),
        correlations=(),
        validation_improved=True,
    )
    monkeypatch.setattr(so101_nexus.sysid, "fit_robot_profile", lambda *args, **kwargs: result)
    errors, diagnostics, summary = session.fit(str(recording), str(tmp_path / "fit"), 10)
    assert len(errors) == 6
    assert errors[0] == ["heldout", "validation", "shoulder_pan", 2.0, 1.0]
    assert diagnostics == [["shoulder_pan.kp", 2.0, False, True, 0.4]]
    assert "not accepted" in summary
    assert session.profile is original
    assert session.artifacts[-2:] == [
        str(tmp_path / "fit" / name) for name in ("candidate_profile.json", "metrics.json")
    ]


def test_missing_recordings_rejected_before_fitting(session, monkeypatch, tmp_path):
    import so101_nexus.sysid

    monkeypatch.setattr(
        so101_nexus.sysid, "fit_robot_profile", lambda *args, **kwargs: pytest.fail("must not fit")
    )
    with pytest.raises(ValueError, match="No completed recordings"):
        session.fit(str(tmp_path), str(tmp_path / "fit"), 1)
    assert session.phase == "monitoring"


def test_arm_requires_support_before_closing_monitor_or_starting_worker(
    session, monkeypatch, tmp_path
):
    monitor = session.monitor
    monkeypatch.setattr(
        threading.Thread, "start", lambda self: pytest.fail("must not start worker")
    )
    with pytest.raises(ValueError, match="supported"):
        session.arm(tmp_path / "recordings", supported=False)
    assert session.monitor is monitor
    assert not monitor.closed
    assert not (tmp_path / "recordings").exists()


def test_release_success_changes_only_torque_after_support_confirmation(session, monkeypatch):
    import lerobot.motors

    events = []
    bus = SimpleNamespace(is_connected=False)

    def connect(*, handshake):
        events.append(("connect", handshake))
        bus.is_connected = True

    def disconnect(*, disable_torque):
        events.append(("disconnect", disable_torque))
        bus.is_connected = False

    bus.connect = connect
    bus.disable_torque = lambda: events.append(("disable_torque",))
    bus.disconnect = disconnect
    monkeypatch.setattr(lerobot.motors, "SerialMotorsBus", lambda **kwargs: bus)
    with pytest.raises(ValueError, match="Support"):
        session.release(supported=False)
    assert events == []
    session.release(supported=True)
    assert events == [("connect", False), ("disable_torque",), ("disconnect", False)]
    assert session.phase == "disconnected"
    assert not session.torque_possible
    assert not session.verified
    with PortLease(session.port):
        pass


def test_new_connection_clears_prior_run_metadata_without_deleting_artifacts(session, tmp_path):
    profile_path = tmp_path / "previous-profile.json"
    recordings = tmp_path / "previous-recordings"
    profile_path.write_text("saved profile")
    recordings.mkdir()
    recording = recordings / "00.npz"
    recording.write_bytes(b"saved recording")
    session.profile_path = profile_path
    session.recordings_dir = recordings
    session.artifacts = [str(profile_path), str(recording)]
    session.progress = "previous multisine: 15/100 samples"
    session.connect("/dev/fake-new-arm", "/tmp")
    assert session.profile_path is None
    assert session.recordings_dir is None
    assert session.artifacts == []
    assert session.progress == ""
    assert profile_path.read_text() == "saved profile"
    assert recording.read_bytes() == b"saved recording"


def test_disconnect_clears_progress_after_successful_cleanup(session):
    session.progress = "previous heldout: 15/100 samples"
    session.disconnect()
    assert session.progress == ""


def test_arming_clears_previous_progress_before_new_worker_runs(session, monkeypatch, tmp_path):
    session.progress = "previous multisine: 15/100 samples"

    def start(worker):
        assert session.progress == ""
        raise RuntimeError("skip worker")

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="skip worker"):
        session.arm(tmp_path / "recordings", supported=True)
    assert session.progress == ""
