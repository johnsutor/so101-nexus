"""Read-only profiling never changes motor state or shares a serial reader."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from so101_nexus.calibration.hardware import PortLease, ReadOnlyFollower, discover_ports
from so101_nexus.config import SO101_JOINT_NAMES


@pytest.fixture
def hardware(tmp_path, monkeypatch):
    monkeypatch.setenv("SO101_NEXUS_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.delenv("SO101_NEXUS_PROFILES_DIR", raising=False)
    monkeypatch.delenv("SO101_NEXUS_CALIBRATION_DIR", raising=False)
    from lerobot.motors import MotorCalibration

    calibration = {
        name: MotorCalibration(i, 0, 0, 100, 4000) for i, name in enumerate(SO101_JOINT_NAMES, 1)
    }
    (tmp_path / "arm.json").write_text(json.dumps({k: asdict(v) for k, v in calibration.items()}))

    class Bus:
        def __init__(self, **kwargs):
            self.motors = kwargs["motors"]
            self.calibration = {}
            self.is_connected = False
            self.events = []
            self.positions = dict.fromkeys(SO101_JOINT_NAMES, 2050)
            self.calibration_read = calibration
            self.fail_close = False

        def connect(self, *, handshake):
            self.events.append(("connect", handshake))
            self.is_connected = True

        def disconnect(self, *, disable_torque):
            self.events.append(("disconnect", disable_torque))
            if self.fail_close:
                raise OSError("lost serial connection")
            self.is_connected = False

        def ping(self, name):
            self.events.append(("ping", name))
            return 777

        def read_calibration(self):
            self.events.append(("read_calibration",))
            return self.calibration_read

        def sync_read(self, register, *, normalize):
            self.events.append(("read", register, normalize))
            if register == "Present_Position":
                return self.positions.copy()
            value = {
                "P_Coefficient": 16,
                "I_Coefficient": 0,
                "D_Coefficient": 32,
                "Torque_Enable": 0,
            }[register]
            return dict.fromkeys(SO101_JOINT_NAMES, value)

    bus = Bus(motors={})

    def factory(**kwargs):
        bus.motors = kwargs["motors"]
        return bus

    monitor = ReadOnlyFollower(str(tmp_path / "port"), tmp_path, bus_factory=factory)
    return monitor, bus


def test_readonly_connect_matches_saved_calibration_and_reads_units(hardware):
    monitor, bus = hardware
    profile = monitor.connect()
    try:
        assert profile.robot_id == "arm"
        assert profile.servo_gains == (16, 0, 32)
        assert monitor.calibration_path.name == "arm.json"
        assert monitor.torque == dict.fromkeys(SO101_JOINT_NAMES, 0)
        np.testing.assert_allclose(monitor.read(), [0, 0, 0, 0, 0, 50])
    finally:
        monitor.close()
    assert bus.events[0] == ("connect", False)
    assert bus.events[-1] == ("disconnect", False)
    assert all(
        event[0] in {"connect", "disconnect", "read", "read_calibration", "ping"}
        for event in bus.events
    )


def test_mismatched_calibration_closes_bus_and_releases_port(hardware):
    monitor, bus = hardware
    bus.calibration_read["shoulder_pan"].range_max = 3999
    with pytest.raises(ValueError, match="matching calibration"):
        monitor.connect()
    assert not bus.is_connected
    with PortLease(monitor.port):
        pass


def test_missing_motor_closes_without_torque_change(hardware):
    monitor, bus = hardware
    bus.ping = lambda name: None if name == "elbow_flex" else 777
    with pytest.raises(ConnectionError, match="elbow_flex"):
        monitor.connect()
    assert bus.events[-1] == ("disconnect", False)


def test_competing_readers_and_aliases_rejected(tmp_path, hardware):
    monitor, _ = hardware
    alias = tmp_path / "alias"
    alias.symlink_to(monitor.port)
    monitor.connect()
    try:
        with pytest.raises(RuntimeError, match="already owned"):
            PortLease(str(alias)).acquire()
        with pytest.raises(RuntimeError, match="already connected"):
            monitor.connect()
    finally:
        monitor.close()
    with PortLease(str(alias)):
        pass


def test_nonfinite_or_incomplete_measurement_rejected(hardware):
    monitor, bus = hardware
    monitor.connect()
    try:
        bus.positions.pop("wrist_roll")
        with pytest.raises(ValueError, match="six motors"):
            monitor.read()
        bus.positions["wrist_roll"] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            monitor.read()
    finally:
        monitor.close()


def test_ambiguous_software_direction_rejected(hardware):
    monitor, bus = hardware
    payload = json.loads((monitor.calibration_dir / "arm.json").read_text())
    payload["wrist_roll"]["drive_mode"] = 1
    (monitor.calibration_dir / "other.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Ambiguous"):
        monitor.connect()
    assert not bus.is_connected


def test_disconnect_failure_retains_ownership_until_closed(hardware):
    monitor, bus = hardware
    monitor.connect()
    bus.fail_close = True
    with pytest.raises(OSError, match="lost serial"):
        monitor.close()
    with pytest.raises(RuntimeError, match="already owned"):
        PortLease(monitor.port).acquire()
    bus.fail_close = False
    monitor.close()
    with PortLease(monitor.port):
        pass


def test_discovery_does_not_open_serial_ports(monkeypatch):
    monkeypatch.setattr(
        "serial.tools.list_ports.comports",
        lambda: [SimpleNamespace(device="/dev/ttyACM1"), SimpleNamespace(device="/dev/ttyACM0")],
    )
    assert discover_ports() == ["/dev/ttyACM0", "/dev/ttyACM1"]


def test_saved_profile_reconnects_after_disposable_calibration_is_removed(
    hardware, tmp_path, monkeypatch
):
    from so101_nexus import RobotProfile
    from so101_nexus.calibration.storage import profiles_dir

    monitor, bus = hardware
    monkeypatch.setenv("SO101_NEXUS_DATA_DIR", str(tmp_path / "durable"))
    source = monitor.calibration_dir / "arm.json"
    saved = profiles_dir() / "robot.json"
    original = RobotProfile.from_lerobot(source, robot_id="arm")
    original.save(saved)
    source.unlink()
    profile = monitor.connect()
    try:
        assert profile.calibration == original.calibration
        assert profile.servo_gains == (16, 0, 32)
        assert monitor.calibration_path == saved
        np.testing.assert_allclose(monitor.read(), [0, 0, 0, 0, 0, 50])
        assert all(
            event[0] in {"connect", "disconnect", "read", "read_calibration", "ping"}
            for event in bus.events
        )
    finally:
        monitor.close()


def test_durable_fallback_still_rejects_changed_motor_calibration(hardware, tmp_path, monkeypatch):
    from so101_nexus import RobotProfile
    from so101_nexus.calibration.storage import profiles_dir

    monitor, bus = hardware
    monkeypatch.setenv("SO101_NEXUS_DATA_DIR", str(tmp_path / "durable"))
    source = monitor.calibration_dir / "arm.json"
    RobotProfile.from_lerobot(source, robot_id="arm").save(profiles_dir() / "robot.json")
    source.unlink()
    bus.calibration_read["shoulder_pan"].range_max = 3999
    with pytest.raises(ValueError, match="matching calibration"):
        monitor.connect()
    assert not bus.is_connected


def test_durable_fallback_rejects_conflicting_software_directions(hardware):
    from dataclasses import replace

    from so101_nexus import RobotProfile
    from so101_nexus.calibration.storage import profiles_dir

    monitor, bus = hardware
    source = monitor.calibration_dir / "arm.json"
    profile = RobotProfile.from_lerobot(source, robot_id="arm")
    profile.save(profiles_dir() / "first.json")
    calibration = {name: dict(value) for name, value in profile.calibration.items()}
    calibration["wrist_roll"]["drive_mode"] = 1
    replace(profile, calibration=calibration).save(profiles_dir() / "second.json")
    source.unlink()
    with pytest.raises(ValueError, match="Ambiguous"):
        monitor.connect()
    assert not bus.is_connected
