"""Real motor-free subprocesses preserve collection timing and cleanup ownership."""

import ctypes
import os
import sys
import threading
import time
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from so101_nexus.calibration.hardware import PortLease
from so101_nexus.calibration.session import CalibrationSession
from so101_nexus.calibration.worker import IsolatedCalibrationSession
from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.excitation import ExcitationConfig, RecordedTrajectory
from so101_nexus.profiles import RobotProfile


class _FakeRobot:
    def __init__(self, profile, center):
        self.config = SimpleNamespace(
            use_degrees=True,
            id=profile.robot_id,
            position_p_coefficient=profile.servo_gains[0],
            position_i_coefficient=profile.servo_gains[1],
            position_d_coefficient=profile.servo_gains[2],
            cameras={},
        )
        self.calibration = profile.calibration
        self.position = dict(
            zip(SO101_JOINT_NAMES, profile.from_model_deg(np.asarray(center)), strict=True)
        )
        self.is_connected = False
        self.is_calibrated = True

    def connect(self, *, calibrate):
        assert not calibrate
        self.is_connected = True

    def disconnect(self):
        self.is_connected = False

    def get_observation(self):
        return {f"{name}.pos": value for name, value in self.position.items()}

    def send_action(self, action):
        self.position = {name.removesuffix(".pos"): value for name, value in action.items()}
        return action


class _MotorFreeSession(CalibrationSession):
    def _make_robot(self, profile, directory):
        return _FakeRobot(profile, self.config.center_qpos_deg)

    def _collect(self, profile, config, destination, lease):
        from so101_nexus import excitation

        assert "gradio" not in sys.modules
        excitation.preflight_excitation = lambda *args: None
        excitation._prepare_follower_connection = lambda *args: None
        super()._collect(profile, config, destination, lease)


class _StartupSession(_MotorFreeSession):
    def _collect(self, profile, config, destination, lease):
        (destination.parent / "collector-started").touch()
        super()._collect(profile, config, destination, lease)


class _HungSession(CalibrationSession):
    def _collect(self, profile, config, destination, lease):
        destination.mkdir()
        (destination / "child-started").touch()
        time.sleep(30)


class _CrashingSession(CalibrationSession):
    def _collect(self, profile, config, destination, lease):
        destination.mkdir()
        (destination / "child-started").touch()
        time.sleep(0.3)
        os._exit(7)


@pytest.fixture
def worker_types(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parent))
    return import_module(Path(__file__).stem)


@pytest.fixture
def isolated(tmp_path, worker_types):
    session = IsolatedCalibrationSession()
    session._collector_type = worker_types._MotorFreeSession
    session.profile = RobotProfile(
        robot_id="isolated",
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
    session.config = ExcitationConfig(
        center_qpos_deg=(0, -15, 30, 45, 0, 50), duration_s=1, ramp_s=0.2
    )
    session.port = str(tmp_path / "fake-port")
    session.phase = "arming"
    session.recordings_dir = tmp_path / "recordings"
    session.torque_possible = True
    yield session
    session.cancel.set()
    if session.worker is not None:
        session.worker.join(timeout=8)
        assert not session.worker.is_alive()


def _start(session):
    lease = PortLease(session.port).acquire()
    session.worker = threading.Thread(
        target=session._collect,
        args=(session.profile, session.config, session.recordings_dir, lease),
        daemon=True,
    )
    session.worker.start()
    return lease


def _wait(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "Worker state did not arrive"
        time.sleep(0.01)


@pytest.mark.skipif(sys.platform != "linux", reason="Uses libc to hold the parent interpreter lock")
def test_spawned_collection_survives_parent_interpreter_stall(isolated):
    lease = _start(isolated)
    _wait(lambda: isolated.phase == "holding")
    isolated.start_motion(hands_clear=True)
    _wait(lambda: isolated.phase == "running")
    ctypes.PyDLL(None).usleep(150000)
    isolated.worker.join(timeout=8)
    assert not isolated.worker.is_alive()
    assert isolated.collection_complete
    assert isolated.collection_error is None
    assert isolated.measured is None
    assert not lease.acquired
    assert len(list(isolated.recordings_dir.glob("*.npz"))) == 3
    for path in isolated.recordings_dir.glob("*.npz"):
        recording = RecordedTrajectory.load(path)
        assert np.max(np.diff(recording.command_timestamps_s)) < 0.05
        assert np.min(np.diff(recording.command_timestamps_s)) >= 0.02 - 1e-9
    assert RobotProfile.load(isolated.recordings_dir / "source_profile.json") == isolated.profile
    assert isolated.failure_sequence == 0


@pytest.mark.parametrize("trigger", ["stop", "heartbeat"])
def test_isolated_hold_cancels_and_releases_port_after_exit(isolated, trigger):
    lease = _start(isolated)
    _wait(lambda: isolated.phase == "holding")
    with pytest.raises(RuntimeError, match="already owned"):
        PortLease(isolated.port).acquire()
    if trigger == "stop":
        isolated.stop()
    else:
        isolated.last_seen = time.monotonic() - 6
    isolated.worker.join(timeout=8)
    assert not isolated.worker.is_alive()
    assert not lease.acquired
    assert isolated.phase == "stopped"
    assert not isolated.collection_complete
    assert isolated.measured is None
    assert "Holding cancelled" in isolated.collection_error
    assert isolated.failure_sequence == 1
    assert isolated.torque_possible
    assert not isolated.recordings_dir.exists()
    with PortLease(isolated.port):
        pass


def test_child_crash_keeps_lease_until_process_exit_and_reports_error(isolated, worker_types):
    isolated._collector_type = worker_types._CrashingSession
    lease = _start(isolated)
    _wait((isolated.recordings_dir / "child-started").exists)
    assert lease.acquired
    isolated.worker.join(timeout=8)
    assert not isolated.worker.is_alive()
    assert not lease.acquired
    assert "code 7" in isolated.collection_error
    assert "Support before release" in isolated.status
    assert isolated.failure_sequence == 1
    assert isolated.measured is None
    assert isolated.torque_possible


def test_parent_pipe_close_cancels_child_holding(isolated):
    import multiprocessing

    from so101_nexus.calibration.worker import _run_collection

    context = multiprocessing.get_context("spawn")
    connection, child = context.Pipe()
    process = context.Process(
        target=_run_collection,
        args=(
            child,
            isolated.profile,
            isolated.config,
            isolated.recordings_dir,
            isolated.port,
            None,
            isolated._collector_type,
        ),
    )
    process.start()
    child.close()
    connection.send((time.monotonic(), False, False))
    try:
        assert connection.poll(8)
        while connection.recv()["phase"] != "holding":
            assert connection.poll(8)
        connection.close()
        process.join(timeout=3)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert not isolated.recordings_dir.exists()
    finally:
        connection.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


def test_spawn_start_failure_reports_error_and_releases_lease(isolated, monkeypatch):
    import multiprocessing
    from unittest.mock import Mock

    context = multiprocessing.get_context("spawn")
    process = Mock(pid=None)
    process.start.side_effect = OSError("spawn unavailable")
    monkeypatch.setattr(
        "so101_nexus.calibration.worker.multiprocessing.get_context",
        lambda method: SimpleNamespace(Pipe=context.Pipe, Process=lambda **kwargs: process),
    )
    lease = _start(isolated)
    isolated.worker.join(timeout=2)
    assert not isolated.worker.is_alive()
    assert not lease.acquired
    assert "spawn unavailable" in isolated.collection_error
    assert isolated.phase == "stopped"
    assert isolated.failure_sequence == 1


def test_unresponsive_child_is_terminated_before_releasing_port(isolated, worker_types):
    isolated._collector_type = worker_types._HungSession
    lease = _start(isolated)
    _wait((isolated.recordings_dir / "child-started").exists)
    isolated.stop()
    time.sleep(0.1)
    assert lease.acquired
    isolated.worker.join(timeout=5)
    assert not isolated.worker.is_alive()
    assert not lease.acquired
    assert not isolated.collection_complete
    assert "exited unexpectedly" in isolated.collection_error
    assert isolated.failure_sequence == 1
    assert isolated.torque_possible


def test_failed_process_termination_retains_lease_for_disconnect_retry(isolated, monkeypatch):
    import multiprocessing
    from unittest.mock import Mock

    context = multiprocessing.get_context("spawn")
    process = Mock(pid=123)
    process.is_alive.return_value = True
    monkeypatch.setattr(
        "so101_nexus.calibration.worker.multiprocessing.get_context",
        lambda method: SimpleNamespace(Pipe=context.Pipe, Process=lambda **kwargs: process),
    )
    monkeypatch.setattr(
        isolated, "_supervise", Mock(side_effect=OSError("supervisor connection failed"))
    )
    lease = _start(isolated)
    isolated.worker.join(timeout=2)
    assert not isolated.worker.is_alive()
    assert lease.acquired
    assert "Port remains owned" in isolated.collection_error
    assert isolated._pending_lease is lease
    with pytest.raises(RuntimeError, match="cleanup is incomplete"):
        isolated._idle()
    process.is_alive.return_value = False
    isolated._close_pending()
    assert not lease.acquired
    assert isolated._pending_bus is None
    process.close.assert_called_once()


@pytest.mark.parametrize("initial", ["stop", "stale", "eof", "stale_then_fresh", "stop_then_clear"])
def test_child_requires_current_initial_control_before_motor_preparation(
    isolated, worker_types, initial
):
    import multiprocessing

    from so101_nexus.calibration.worker import _run_collection

    context = multiprocessing.get_context("spawn")
    connection, child = context.Pipe()
    process = context.Process(
        target=_run_collection,
        args=(
            child,
            isolated.profile,
            isolated.config,
            isolated.recordings_dir,
            isolated.port,
            None,
            worker_types._StartupSession,
        ),
    )
    queued = "_then_" in initial
    heartbeat = time.monotonic() - (6 if initial.startswith("stale") else 0)
    first_control = (heartbeat, initial.startswith("stop"), False)
    if queued:
        connection.send(first_control)
        connection.send((time.monotonic(), False, False))
    process.start()
    child.close()
    marker = isolated.recordings_dir.parent / "collector-started"
    try:
        if not queued:
            assert not connection.poll(0.4)
            assert process.is_alive()
            assert not marker.exists()
        if initial == "eof":
            connection.close()
        else:
            if not queued:
                connection.send(first_control)
            assert connection.poll(3)
            state = connection.recv()
            if initial == "stale_then_fresh":
                assert state["phase"] == "holding"
                assert marker.exists()
                connection.send((time.monotonic(), True, False))
            else:
                assert state["final"]
                assert "cancelled before motor connection" in state["collection_error"]
        process.join(timeout=3)
        assert not process.is_alive()
        assert process.exitcode == 0
        assert marker.exists() == (initial == "stale_then_fresh")
        assert not isolated.recordings_dir.exists()
    finally:
        connection.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=2)
        process.close()


def test_completed_collection_survives_slow_interpreter_shutdown(isolated, monkeypatch):
    import multiprocessing
    from unittest.mock import Mock

    from so101_nexus.calibration.worker import _snapshot

    context = multiprocessing.get_context("spawn")
    process = Mock(pid=123, exitcode=None)
    process.is_alive.return_value = True

    def finish_shutdown(timeout):
        if timeout >= 0.3:
            process.exitcode = 0
            process.is_alive.return_value = False

    process.join.side_effect = finish_shutdown
    monkeypatch.setattr(
        "so101_nexus.calibration.worker.multiprocessing.get_context",
        lambda method: SimpleNamespace(Pipe=context.Pipe, Process=lambda **kwargs: process),
    )
    final = _snapshot(isolated, final=True)
    final.update(phase="stopped", collection_complete=True, status="Collection complete.")
    monkeypatch.setattr(isolated, "_supervise", lambda *args: final)
    lease = _start(isolated)
    isolated.worker.join(timeout=2)
    assert not isolated.worker.is_alive()
    assert isolated.collection_complete
    assert isolated.collection_error is None
    assert isolated.failure_sequence == 0
    assert not lease.acquired
    process.terminate.assert_not_called()
    process.kill.assert_not_called()
    process.close.assert_called_once()
