"""Read-only SO-101 inspection with exclusive serial-port ownership."""

from __future__ import annotations

import os
import threading
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.profiles import RobotProfile

if TYPE_CHECKING:
    from collections.abc import Callable

_PORTS: set[str] = set()
_PORT_LOCK = threading.Lock()


class PortLease:
    """Exclude overlapping readers and collectors, including serial-path aliases."""

    def __init__(self, port: str) -> None:
        self.port = os.path.realpath(port)
        self.acquired = False

    def acquire(self) -> PortLease:
        """Reserve the port until explicit release."""
        with _PORT_LOCK:
            if self.acquired or self.port in _PORTS:
                raise RuntimeError(
                    f"Serial port already owned: {self.port}. Disconnect its owner first."
                )
            _PORTS.add(self.port)
            self.acquired = True
        return self

    def release(self) -> None:
        """Release this lease without changing hardware state."""
        with _PORT_LOCK:
            if self.acquired:
                _PORTS.remove(self.port)
                self.acquired = False

    def __enter__(self) -> PortLease:
        """Acquire ownership for a context."""
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        """Release ownership when leaving the context."""
        self.release()


def discover_ports() -> list[str]:
    """List available serial devices without opening them."""
    from serial.tools.list_ports import comports

    return sorted({port.device for port in comports()})


def default_calibration_dir() -> Path:
    """Return LeRobot's physical SO follower calibration directory."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

    root = Path(os.environ.get("HF_LEROBOT_CALIBRATION", HF_LEROBOT_CALIBRATION)).expanduser()
    return root / ROBOTS / "so_follower"


def _matches_calibration(profile: RobotProfile, actual: dict[str, Any]) -> bool:
    # Feetech drive direction lives in the saved file, not a motor register.
    fields = ("id", "homing_offset", "range_min", "range_max")
    return all(
        all(profile.calibration[name][key] == asdict(actual[name])[key] for key in fields)
        for name in SO101_JOINT_NAMES
    )


class ReadOnlyFollower:
    """Inspect encoders and registers without enabling torque or configuring motors.

    Parameters
    ----------
    port : str
        Follower serial device or stable serial alias.
    calibration_dir : Path
        Directory containing existing LeRobot calibration JSON files.
    bus_factory : callable, optional
        Motor-bus constructor for dependency injection in hardware-free tests.
    """

    def __init__(
        self,
        port: str,
        calibration_dir: Path,
        *,
        bus_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.port = port
        self.calibration_dir = Path(calibration_dir).expanduser()
        self.bus: Any = None
        self.profile: RobotProfile | None = None
        self.calibration_path: Path | None = None
        self.torque: dict[str, int] = {}
        self.motor_info: dict[str, int] = {}
        self._bus_factory = bus_factory
        self._lease = PortLease(port)

    def connect(self) -> RobotProfile:
        """Read motor state and identify the saved calibration that matches it."""
        if self.bus is not None:
            raise RuntimeError("Monitor already connected; disconnect before connecting again.")
        from lerobot.motors import MotorCalibration, SerialMotorsBus

        from so101_nexus.lerobot_adapter.normalization import build_so101_motors

        self._lease.acquire()
        self.profile = None
        self.calibration_path = None
        self.motor_info = {}
        self.torque = {}
        try:
            factory = self._bus_factory or SerialMotorsBus
            self.bus = factory(port=self.port, motors=build_so101_motors(use_degrees=True))
            self.bus.connect(handshake=False)
            for name in SO101_JOINT_NAMES:
                model = self.bus.ping(name)
                if model is None:
                    raise ConnectionError(f"Motor {name} did not respond. Check power and cabling.")
                self.motor_info[name] = int(model)
            actual = self.bus.read_calibration()
            if set(actual) != set(SO101_JOINT_NAMES):
                raise ValueError("Calibration read must contain all six motors.")
            gains = []
            for coefficient in ("P", "I", "D"):
                values = self._read_register(f"{coefficient}_Coefficient")
                if len(set(values.values())) != 1:
                    raise ValueError(
                        f"Motor {coefficient} gains differ; one profile requires common gains."
                    )
                gains.append(next(iter(values.values())))
            self.torque = self._read_register("Torque_Enable")
            self.calibration_path, self.profile = self._match_calibration(
                actual, (gains[0], gains[1], gains[2])
            )
            self.bus.calibration = {
                name: MotorCalibration(**dict(value))
                for name, value in self.profile.calibration.items()
            }
            self.read()
            return self.profile
        except BaseException as error:
            try:
                self.close()
            except Exception as cleanup_error:
                error.add_note(f"Read-only bus cleanup failed: {cleanup_error}")
            raise

    def _match_calibration(
        self, actual: dict[str, Any], gains: tuple[int, int, int]
    ) -> tuple[Path, RobotProfile]:
        matches = []
        errors = []
        for path in sorted(self.calibration_dir.glob("*.json")):
            try:
                profile = RobotProfile.from_lerobot(path, robot_id=path.stem, servo_gains=gains)
                if _matches_calibration(profile, actual):
                    matches.append((path, profile))
            except (ValueError, TypeError, OSError) as error:
                errors.append(f"{path.name}: {error}")
        if not matches:
            from so101_nexus.calibration.storage import saved_profiles

            for path in saved_profiles():
                try:
                    candidate = RobotProfile.load(path)
                    profile = RobotProfile(
                        candidate.robot_id, candidate.calibration, servo_gains=gains
                    )
                    if _matches_calibration(profile, actual):
                        matches.append((path, profile))
                except (ValueError, TypeError, OSError) as error:
                    errors.append(f"{path.name}: {error}")
        if not matches:
            detail = "; ".join(errors)
            raise ValueError(
                f"No matching calibration in {self.calibration_dir}. "
                f"Select the directory containing this follower's saved calibration. {detail}"
            )
        snapshots = [profile.to_dict()["calibration"] for _, profile in matches]
        if any(snapshot != snapshots[0] for snapshot in snapshots[1:]):
            raise ValueError(
                "Ambiguous matching calibrations with different software directions: "
                + ", ".join(path.name for path, _ in matches)
                + ". Select a directory containing only the verified calibration."
            )
        return matches[0]

    def _read_register(self, name: str) -> dict[str, int]:
        values = self.bus.sync_read(name, normalize=False)
        if set(values) != set(SO101_JOINT_NAMES):
            raise ValueError(f"{name} read must contain all six motors.")
        if any(not np.isfinite(value) or int(value) != value for value in values.values()):
            raise ValueError(f"{name} requires finite integer register values.")
        return {joint: int(value) for joint, value in values.items()}

    def read(self) -> np.ndarray:
        """Return five hardware body degrees and gripper percent, in joint order."""
        if self.bus is None or not self.bus.is_connected or self.profile is None:
            raise RuntimeError("Connect the read-only monitor before reading positions.")
        from so101_nexus.lerobot_adapter.normalization import normalize_ticks

        raw = self._read_register("Present_Position")
        normalized = normalize_ticks(raw, motors=self.bus.motors, calibration=self.bus.calibration)
        values = np.asarray([normalized[name] for name in SO101_JOINT_NAMES], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("Encoder readings must be finite.")
        self.torque = self._read_register("Torque_Enable")
        return values

    def close(self) -> None:
        """Close the reader without changing torque, including after failed setup."""
        try:
            if self.bus is not None:
                self.bus.disconnect(disable_torque=False)
        finally:
            # Keep ownership if a failed close leaves the serial reader alive.
            if self.bus is None or not self.bus.is_connected:
                self.bus = None
                self._lease.release()
