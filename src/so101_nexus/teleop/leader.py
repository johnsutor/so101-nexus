"""Leader-arm factory and env/robot validation helpers for teleop.

All ``lerobot`` imports are deferred into :func:`get_leader` so this module
can be imported without the ``teleop`` extra installed.
"""

from __future__ import annotations

import importlib
import math
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Protocol

from so101_nexus.config import SO101_JOINT_NAMES

DEFAULT_WRIST_ROLL_OFFSET_DEG = -90.0

ROBOT_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "so100": SO101_JOINT_NAMES,
    "so101": SO101_JOINT_NAMES,
}


class LeaderProtocol(Protocol):
    """Minimal interface the teleop loop requires from a leader-arm driver."""

    def connect(self) -> None:
        """Open the connection to the leader arm."""
        ...

    def disconnect(self) -> None:
        """Close the connection to the leader arm."""
        ...

    def get_action(self) -> dict:
        """Return the latest joint readings from the leader arm."""
        ...


@dataclass(frozen=True)
class LeaderPortDiagnostic:
    """Structured diagnosis for a requested leader-arm serial path."""

    kind: str
    message: str
    recovery_hint: str = ""


def apply_wrist_roll_offset_deg(action: dict[str, float], offset_deg: float) -> dict[str, float]:
    """Return a copy of *action* with ``wrist_roll.pos`` shifted by *offset_deg*."""
    result = dict(action)
    if "wrist_roll.pos" in result:
        result["wrist_roll.pos"] = float(result["wrist_roll.pos"]) + offset_deg
    return result


def _permission_recovery_hint(port: str) -> str:
    return (
        "The device exists but the current user cannot open it.\n"
        "Fix the port permissions in another terminal, then retry here:\n"
        f"  sudo chmod 666 {port}\n"
        "Run 'lerobot-find-port' if you want to confirm the current device path.\n"
        "Or add your user to the serial-access group used by your distro."
    )


def diagnose_leader_port(port: str) -> LeaderPortDiagnostic:
    """Inspect *port* and return a structured diagnosis."""
    if not os.path.exists(port):
        return LeaderPortDiagnostic(
            kind="not_found",
            message=f"Serial device '{port}' was not found.",
            recovery_hint="Check the USB connection and run 'lerobot-find-port' to locate the arm.",
        )
    if not os.access(port, os.R_OK | os.W_OK):
        return LeaderPortDiagnostic(
            kind="permission_denied",
            message=f"Serial device '{port}' exists but is not readable and writable by this user.",
            recovery_hint=_permission_recovery_hint(port),
        )
    return LeaderPortDiagnostic(kind="ok", message=f"Serial device '{port}' looks accessible.")


def format_leader_connection_error(port: str, exc: Exception) -> str:
    """Return a user-facing connection error with recovery guidance."""
    details = str(exc).strip() or type(exc).__name__
    lower_details = details.lower()
    if "permission denied" in lower_details:
        diag = LeaderPortDiagnostic(
            kind="permission_denied",
            message=f"Serial device '{port}' rejected the connection with a permission error.",
            recovery_hint=_permission_recovery_hint(port),
        )
    elif "no such file" in lower_details or "could not connect on port" in lower_details:
        diag = LeaderPortDiagnostic(
            kind="not_found",
            message=f"Serial device '{port}' could not be opened.",
            recovery_hint=(
                "Check the USB connection and run 'lerobot-find-port' to locate the arm."
            ),
        )
    else:
        diag = diagnose_leader_port(port)

    message = f"Failed to connect on {port}: {details}"
    if diag.kind != "ok":
        message += f"\n{diag.message}"
    if diag.recovery_hint:
        message += f"\n{diag.recovery_hint}"
    return message


def get_leader(robot_type: str, port: str, leader_id: str) -> LeaderProtocol:
    """Create and return the appropriate ``SOLeader`` for *robot_type*."""
    if robot_type == "so100":
        from lerobot.teleoperators.so_leader.config_so_leader import SO100LeaderConfig
        from lerobot.teleoperators.so_leader.so_leader import SO100Leader

        return SO100Leader(SO100LeaderConfig(port=port, use_degrees=True, id=leader_id))

    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    return SO101Leader(SO101LeaderConfig(port=port, use_degrees=True, id=leader_id))


def check_robot_env_mismatch(env_id: str, robot_type: str) -> str | None:
    """Return a warning string if *env_id* encodes a robot that contradicts *robot_type*."""
    upper = env_id.upper()
    if robot_type == "so100" and "SO101" in upper:
        return f"Robot type is so100 but env '{env_id}' appears to target SO101."
    if robot_type == "so101" and "SO100" in upper:
        return f"Robot type is so101 but env '{env_id}' appears to target SO100."
    return None


def import_backend_for_env_id(env_id: str) -> None:
    """Import the simulator backend that matches the *env_id* prefix."""
    if env_id.startswith("MuJoCo"):
        importlib.import_module("so101_nexus.mujoco")
        return

    import gymnasium as gym

    try:
        gym.spec(env_id)
    except gym.error.Error as exc:
        raise ValueError(
            f"Unknown custom env_id {env_id!r}. Import its registration module with "
            "--env-module and pass it with --extra-env-id."
        ) from exc


class SOLeaderSynchronizer:
    """Move a calibrated SO leader to follower readback at a bounded target speed.

    Parameters
    ----------
    bus
        Connected LeRobot motor bus with degree body joints and percent gripper.
    wrist_roll_offset_deg
        Offset added to leader readings by teleoperation, inverted for alignment.
    """

    def __init__(self, bus: Any, *, wrist_roll_offset_deg: float) -> None:
        self.bus = bus
        self.wrist_roll_offset_deg = wrist_roll_offset_deg
        self._target: dict[str, float] | None = None
        self._initial: dict[str, float] = {}
        self._started = 0.0
        self._duration = 0.0
        self._holding = False
        self._lock = threading.RLock()

    @property
    def holding(self) -> bool:
        """Whether the last pose-hold command completed successfully."""
        return self._holding

    def start(self, target: dict[str, float]) -> None:
        """Hold the measured pose, then begin interpolated alignment."""
        with self._lock:
            self.cancel()
            target = apply_wrist_roll_offset_deg(target, -self.wrist_roll_offset_deg)
            target = {key.removesuffix(".pos"): value for key, value in target.items()}
            if not all(math.isfinite(value) for value in target.values()):
                raise ValueError("Synchronization targets must be finite")
            self._target = target
            self._duration = max(
                abs(value - self._initial[name]) / (40.0 if name == "gripper" else 20.0)
                for name, value in target.items()
            )
            self._started = time.monotonic()

    def tick(self) -> bool:
        """Advance alignment and keep torque enabled after convergence."""
        with self._lock:
            if self._target is None:
                return True
            try:
                elapsed = time.monotonic() - self._started
                if elapsed > self._duration + 3.0:
                    raise TimeoutError(
                        "Leader synchronization timed out; check alignment and calibration"
                    )
                fraction = min(1.0, elapsed / self._duration) if self._duration else 1.0
                self.bus.sync_write(
                    "Goal_Position",
                    {
                        name: self._initial[name] + fraction * (value - self._initial[name])
                        for name, value in self._target.items()
                    },
                )
                positions = self.bus.sync_read("Present_Position")
                if fraction == 1.0 and all(
                    abs(positions[name] - value) <= 2.0 for name, value in self._target.items()
                ):
                    self._target = None
                    return True
                return False
            except Exception as exc:
                try:
                    self.cancel()
                except Exception as hold_exc:
                    raise RuntimeError(
                        f"{exc}; leader pose hold failed: {hold_exc}. Support the physical arm now."
                    ) from exc
                raise

    def cancel(self) -> None:
        """Stop alignment and hold measured joint positions without releasing torque."""
        with self._lock:
            self._target = None
            self._holding = False
            positions = self.bus.sync_read("Present_Position")
            if set(positions) != set(SO101_JOINT_NAMES) or not all(
                math.isfinite(value) for value in positions.values()
            ):
                raise ValueError("Leader pose hold requires finite readings for every joint")
            self.bus.sync_write("Goal_Position", positions)
            self.bus.enable_torque()
            self._initial = positions
            self._holding = True

    def release(self) -> None:
        """Disable torque only after the operator explicitly supports the leader."""
        with self._lock:
            self._target = None
            self._holding = False
            self.bus.disable_torque()
