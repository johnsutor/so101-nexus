"""Guided alignment never treats provisional encoder coordinates as evidence."""

import threading
from dataclasses import replace

import numpy as np
import pytest

from so101_nexus.calibration.session import CalibrationSession
from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.excitation import ExcitationConfig
from so101_nexus.profiles import RobotProfile


@pytest.fixture
def profile():
    return RobotProfile(
        robot_id="test",
        calibration={
            name: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
            for i, name in enumerate(SO101_JOINT_NAMES, 1)
        },
    )


class Monitor:
    def __init__(self, profile):
        self.profile = profile
        self.closed = False
        self.torque = dict.fromkeys(SO101_JOINT_NAMES, 0)
        self.calibration_path = "test.json"

    def connect(self):
        return self.profile

    def read(self):
        return np.array([10, 20, 30, 40, 50, 60.0])

    def close(self):
        self.closed = True


@pytest.fixture
def session(profile):
    return CalibrationSession(monitor_factory=lambda *args: Monitor(profile))


def align(session):
    session.connect("/dev/fake", "/tmp")
    for index in range(5):
        session.capture_joint(index, index * 5, confirmed=True)
    session.set_gripper(-10, 100, confirmed=True)


def test_alignment_requires_physical_confirmation_and_reads_each_joint(session, tmp_path):
    session.connect("/dev/fake", "/tmp")
    with pytest.raises(ValueError, match="physical"):
        session.capture_joint(0, 0, confirmed=False)
    with pytest.raises(ValueError, match="five"):
        session.save_profile(tmp_path / "profile.json")
    for index in range(5):
        session.capture_joint(index, index * 5, confirmed=True)
    with pytest.raises(ValueError, match="gripper"):
        session.save_profile(tmp_path / "profile.json")
    session.set_gripper(-10, 100, confirmed=True)
    path = session.save_profile(tmp_path / "profile.json")
    assert RobotProfile.load(path).joint_offsets_deg == (-10, -15, -20, -25, -30)
    assert session.confirmed == set(range(5))
    with pytest.raises(FileExistsError):
        session.save_profile(path)


def test_capture_rejects_disconnected_and_nonfinite_reference(session):
    with pytest.raises(RuntimeError, match="Connect"):
        session.capture_joint(0, 0, confirmed=True)
    session.connect("/dev/fake", "/tmp")
    with pytest.raises(ValueError, match="finite"):
        session.capture_joint(0, float("nan"), confirmed=True)


def test_preflight_is_invalidated_by_alignment_and_plan_changes(session, monkeypatch):
    align(session)
    monkeypatch.setattr("so101_nexus.calibration.session.preflight_excitation", lambda *a: None)
    config = ExcitationConfig(center_qpos_deg=(0, -15, 30, 45, 0, 50))
    session.set_plan(config)
    session.preflight()
    assert session.verified
    session.set_plan(replace(config, seed=1))
    assert not session.verified
    session.preflight()
    session.capture_joint(0, 15, confirmed=True)
    assert not session.verified
    session.preflight()
    session.set_gripper(0, 90, confirmed=True)
    assert not session.verified


def test_monitor_failure_closes_and_discards_old_measurement(session):
    session.connect("/dev/fake", "/tmp")
    monitor = session.monitor
    monitor.read = lambda: (_ for _ in ()).throw(OSError("unplugged"))
    session.poll()
    assert monitor.closed
    assert session.monitor is None
    assert session.measured is None
    assert "unplugged" in session.status


def test_collection_releases_monitor_and_waits_for_explicit_start(session, monkeypatch, tmp_path):
    align(session)
    monkeypatch.setattr("so101_nexus.calibration.session.preflight_excitation", lambda *a: None)
    session.set_plan(ExcitationConfig(center_qpos_deg=(0, -15, 30, 45, 0, 50)))
    session.preflight()
    monitor = session.monitor
    ready = threading.Event()
    moved = threading.Event()

    def collect(robot, profile, config, **kwargs):
        assert monitor.closed
        ready.set()
        kwargs["on_ready"]()
        if kwargs["should_stop"]():
            raise InterruptedError("cancelled")
        moved.set()
        return ()

    monkeypatch.setattr("so101_nexus.calibration.session.collect_excitation", collect)
    monkeypatch.setattr(session, "_make_robot", lambda *a: object())
    session.arm(tmp_path / "recordings", supported=True)
    assert ready.wait(2)
    assert not moved.is_set()
    session.stop()
    session.worker.join(2)
    assert not session.worker.is_alive()
    assert not moved.is_set()
    assert session.monitor is None
    assert "torque" in session.status.lower()


def test_stale_browser_cancels_and_start_requires_holding(session):
    session.last_seen -= 10
    assert session.should_stop()
    with pytest.raises(RuntimeError, match="holding"):
        session.start_motion(hands_clear=True)
