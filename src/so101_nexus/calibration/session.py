"""Own a single local arm across read-only alignment and explicit excitation."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.excitation import (
    ExcitationConfig,
    RecordedTrajectory,
    collect_excitation,
    generate_excitation,
    preflight_excitation,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from so101_nexus.profiles import RobotProfile


class CalibrationSession:
    """Serialize hardware access for one operator, including browser reconnects.

    Monitoring never changes torque. Collection owns the port until its worker
    exits. Browser heartbeat loss stops new commands while preserving servo torque.
    """

    def __init__(self, *, monitor_factory: Callable[..., Any] | None = None) -> None:
        """Create a disconnected session without opening hardware."""
        self.lock = threading.RLock()
        self.monitor_factory = monitor_factory
        self.monitor: Any = None
        self.profile: RobotProfile | None = None
        self.measured: np.ndarray | None = None
        self.confirmed: set[int] = set()
        self.gripper_confirmed = False
        self.config: ExcitationConfig | None = None
        self.verified = False
        self.port = ""
        self.status = "Disconnected. Discover a follower to begin. No motor writes."
        self.phase = "disconnected"
        self.worker: threading.Thread | None = None
        self.cancel = threading.Event()
        self.motion = threading.Event()
        self.last_seen = time.monotonic()
        self.recordings_dir: Path | None = None
        self.profile_path: Path | None = None
        self.artifacts: list[str] = []
        self.torque_possible = False
        self.progress = ""
        self._pending_bus: Any = None
        self._pending_lease: Any = None
        self._fitting = False

    def _idle(self) -> None:
        if self._pending_bus is not None:
            raise RuntimeError("Serial cleanup is incomplete. Retry disconnect before continuing.")
        if self._fitting:
            raise RuntimeError("Wait for fitting to finish before changing the session.")
        if self.worker is not None and self.worker.is_alive():
            raise RuntimeError("Stop the active worker before changing the session.")

    def _aligned(self) -> RobotProfile:
        if self.profile is None or len(self.confirmed) != 5:
            raise ValueError("Physically confirm all five body joints first.")
        if not self.gripper_confirmed:
            raise ValueError("Confirm the separate gripper endpoint mapping first.")
        return self.profile

    def _update_profile(self, profile: RobotProfile) -> None:
        if profile != self.profile:
            self.profile = profile
            self.verified = False
            if self.profile_path is not None:
                self.artifacts = [path for path in self.artifacts if path != str(self.profile_path)]
            self.profile_path = None

    def connect(self, port: str, calibration_dir: str) -> None:
        """Match calibration through a read-only bus and invalidate old alignment."""
        from so101_nexus.calibration.hardware import ReadOnlyFollower

        with self.lock:
            self._idle()
            self._close_monitor()
            self.profile = None
            self.confirmed.clear()
            self.gripper_confirmed = False
            self.verified = False
            self.config = None
            self.profile_path = None
            self.recordings_dir = None
            self.artifacts = []
            self.progress = ""
            monitor = (self.monitor_factory or ReadOnlyFollower)(port, Path(calibration_dir))
            self.monitor = monitor
            try:
                profile = monitor.connect()
                measured = monitor.read()
            except BaseException as error:
                self.phase = "disconnected"
                try:
                    self._close_monitor()
                except Exception as cleanup_error:
                    error.add_note(f"Monitor cleanup failed: {cleanup_error}")
                raise
            self.monitor, self.profile, self.measured = monitor, profile, measured
            self.port = port
            self.phase = "monitoring"
            self.torque_possible = any(monitor.torque.values())
            self.status = (
                f"Read-only connection. Calibration: {monitor.calibration_path}. "
                f"Motor torque: {monitor.torque}. Body alignment is provisional."
            )

    def _close_monitor(self) -> None:
        self.measured = None
        self.verified = False
        if self.monitor is not None:
            try:
                self.monitor.close()
            except Exception as error:
                self.phase = "fault"
                self.status = (
                    f"Serial cleanup failed: {error}. Port remains owned; retry disconnect."
                )
                raise
            self.monitor = None

    def poll(self) -> None:
        """Refresh read-only encoders and the browser heartbeat."""
        self.last_seen = time.monotonic()
        with self.lock:
            if self.monitor is None:
                return
            try:
                self.measured = self.monitor.read()
            except Exception as error:
                try:
                    self._close_monitor()
                finally:
                    self.phase = "disconnected"
                    self.verified = False
                    self.confirmed.clear()
                    self.gripper_confirmed = False
                    self.status = (
                        f"Encoder connection lost: {error}. Reconnect and verify alignment."
                    )

    def preview_joint_offset(self, index: int, delta_deg: float) -> None:
        """Adjust a provisional body-joint mapping without touching hardware.

        Parameters
        ----------
        index : int
            Body-joint index, from shoulder pan at zero to wrist roll at four.
        delta_deg : float
            Relative change to the displayed hardware-to-model offset in degrees.
        """
        with self.lock:
            self._idle()
            if self.monitor is None or self.profile is None or self.phase != "monitoring":
                raise RuntimeError("Connect read-only before previewing a joint offset.")
            if (
                type(index) is not int
                or not 0 <= index < 5
                or isinstance(delta_deg, bool)
                or not isinstance(delta_deg, (int, float))
                or not np.isfinite(delta_deg)
            ):
                raise ValueError("Choose a body joint and a finite offset change in degrees.")
            offsets = list(self.profile.joint_offsets_deg)
            offsets[index] += delta_deg
            self._update_profile(replace(self.profile, joint_offsets_deg=tuple(offsets)))
            self.confirmed.discard(index)
            self.verified = False
            self.status = (
                f"Provisional {SO101_JOINT_NAMES[index]} offset changed by {delta_deg:g} deg. "
                "Hardware did not move. Physically place and capture a known reference to confirm."
            )

    def capture_joint(self, index: int, model_deg: float, *, confirmed: bool) -> None:
        """Capture one physically understood joint without assuming its offset."""
        with self.lock:
            self._idle()
            if self.monitor is None or self.profile is None:
                raise RuntimeError("Connect read-only before capturing a joint.")
            if not 0 <= index < 5 or not np.isfinite(model_deg):
                raise ValueError("Choose a body joint and a finite reference angle in degrees.")
            if not confirmed:
                raise ValueError("Confirm physical placement against the ghost or an angle gauge.")
            self.measured = self.monitor.read()
            if any(self.monitor.torque.values()):
                raise ValueError(
                    "Reference capture requires torque off. Support, then release motors."
                )
            offsets = list(self.profile.joint_offsets_deg)
            offsets[index] = float(model_deg - self.measured[index])
            self._update_profile(replace(self.profile, joint_offsets_deg=tuple(offsets)))
            self.confirmed.add(index)
            self.verified = False
            self.status = (
                f"Captured {SO101_JOINT_NAMES[index]} at {model_deg:g} model deg. "
                f"{len(self.confirmed)}/5 body joints confirmed by operator. "
                "No automatic accuracy claim."
            )

    def set_gripper(self, lower_deg: float, upper_deg: float, *, confirmed: bool) -> None:
        """Confirm externally understood model angles at hardware 0 and 100 percent."""
        with self.lock:
            self._idle()
            if self.profile is None:
                raise RuntimeError("Connect before setting the gripper mapping.")
            if not confirmed:
                raise ValueError(
                    "Confirm the gripper endpoint angles independently of body alignment."
                )
            self._update_profile(replace(self.profile, gripper_limits_deg=(lower_deg, upper_deg)))
            self.gripper_confirmed = True
            self.verified = False
            self.status = (
                "Gripper endpoint mapping confirmed. "
                "Hardware percent remains distinct from model deg."
            )

    def save_profile(self, path: str | Path) -> Path:
        """Save a confirmed profile without overwriting previous work."""
        with self.lock:
            self._idle()
            profile = self._aligned()
            path = Path(path).expanduser().resolve()
            if path.exists():
                raise FileExistsError(f"Profile already exists: {path}")
            profile.save(path)
            self.profile_path = path
            self.artifacts = [str(path)]
            self.status = f"Saved {path}. Physical alignment remains operator verified."
            return path

    def load_profile(self, path: str | Path) -> None:
        """Restore operator-trusted alignment only against matching live calibration."""
        from so101_nexus.profiles import RobotProfile

        with self.lock:
            self._idle()
            self.verified = False
            self.confirmed.clear()
            self.gripper_confirmed = False
            if self.monitor is None:
                raise RuntimeError("Connect read-only before loading a trusted profile.")
            self.measured = self.monitor.read()
            candidate = RobotProfile.load(Path(path).expanduser())
            if candidate.calibration != self.monitor.profile.calibration:
                raise ValueError(
                    "Saved profile calibration differs from this follower's calibration."
                )
            bus = getattr(self.monitor, "bus", None)
            if bus is not None and not bus.is_calibrated:
                raise ValueError(
                    "Live motor calibration changed. Reconnect and identify calibration again."
                )
            if candidate.servo_gains != self.monitor.profile.servo_gains:
                raise ValueError("Saved profile servo gains differ from this follower's readback.")
            self.profile = candidate
            self.profile_path = Path(path).expanduser().resolve()
            self.artifacts = [str(self.profile_path)]
            self.confirmed = set(range(5))
            self.gripper_confirmed = True
            self.status = (
                "Trusted profile loaded against matching calibration. "
                "Alignment remains operator verified."
            )

    def set_period(self, period_s: float) -> None:
        """Set the command period in seconds and revoke obsolete simulation preflight."""
        with self.lock:
            self._idle()
            if self.profile is None:
                raise RuntimeError("Connect before setting the command period.")
            try:
                profile = replace(self.profile, control_period_s=period_s)
            except ValueError:
                self.verified = False
                raise
            self._update_profile(profile)

    def use_current_center(self) -> tuple[float, ...]:
        """Read an aligned live pose as a candidate excitation center in model degrees."""
        with self.lock:
            self._idle()
            if self.monitor is None:
                raise RuntimeError("Connect read-only before reading the excitation center.")
            profile = self._aligned()
            self.measured = self.monitor.read()
            return tuple(float(value) for value in profile.to_model_deg(self.measured))

    def set_plan(self, config: ExcitationConfig) -> None:
        """Invalidate preflight whenever the exact collection settings change."""
        with self.lock:
            self._idle()
            if config != self.config:
                self.config = config
                self.verified = False
                self.status = "Motion plan changed. Preview and run simulation preflight again."

    def preflight(self) -> None:
        """Verify the same deterministic trajectory configuration used by collection."""
        with self.lock:
            self._idle()
            self.verified = False
            profile = self._aligned()
            if self.config is None:
                raise ValueError("Preview a motion plan first.")
            preflight_excitation(generate_excitation(self.config, profile), profile, self.config)
            self.verified = True
            self.status = (
                "Simulation preflight passed for all three episodes. "
                "Support the arm at the center before enabling hold."
            )

    def should_stop(self) -> bool:
        """Stop commands after explicit cancellation or loss of browser heartbeat."""
        return self.cancel.is_set() or time.monotonic() - self.last_seen > 5

    def _make_robot(self, profile: RobotProfile, directory: Path) -> Any:
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        profile.export_lerobot(directory / f"{profile.robot_id}.json")
        return SO101Follower(
            SO101FollowerConfig(
                port=self.port,
                id=profile.robot_id,
                calibration_dir=directory,
                use_degrees=True,
                disable_torque_on_disconnect=False,
                position_p_coefficient=profile.servo_gains[0],
                position_i_coefficient=profile.servo_gains[1],
                position_d_coefficient=profile.servo_gains[2],
            )
        )

    def arm(self, output_dir: str | Path, *, supported: bool) -> None:
        """Transfer the supported arm from torque-free monitoring to active holding."""
        from so101_nexus.calibration.hardware import PortLease

        with self.lock:
            self._idle()
            profile = self._aligned()
            if not self.verified or self.config is None:
                raise ValueError("Run simulation preflight for the current settings first.")
            if not supported:
                raise ValueError(
                    "Confirm the arm is supported at the previewed center with workspace clear."
                )
            if self.monitor is None:
                raise RuntimeError(
                    "Reconnect read-only to verify current hardware before enabling hold."
                )
            self.measured = self.monitor.read()
            if any(self.monitor.torque.values()):
                raise ValueError("Support the arm and release motors before enabling a new hold.")
            destination = Path(output_dir).expanduser().resolve()
            if destination.exists():
                raise FileExistsError(f"Recordings directory already exists: {destination}")
            config = self.config
            self._close_monitor()
            lease = PortLease(self.port)
            lease.acquire()
            self.cancel.clear()
            self.motion.clear()
            self.last_seen = time.monotonic()
            self.phase = "arming"
            self.progress = ""
            self.status = (
                "Enabling hold at the measured center. Keep supporting until Holding appears."
            )
            self.recordings_dir = destination
            self.torque_possible = True

            self.worker = threading.Thread(
                target=self._collect,
                args=(profile, config, destination, lease),
                name="calibration-collection",
                daemon=True,
            )
            try:
                self.worker.start()
            except BaseException:
                lease.release()
                self.worker = None
                self.phase = "stopped"
                self.verified = False
                self.torque_possible = False
                self.status = (
                    "Collection worker could not start. Reconnect read-only before retrying."
                )
                raise

    def _wait_for_motion(self, robot: Any) -> None:
        with self.lock:
            self.phase = "holding"
            self.status = (
                "Holding at the center. Remove your hands, then confirm and start excitation. "
                "Hold times out after 60 seconds."
            )
        deadline = time.monotonic() + 60
        next_read = time.monotonic()
        while not self.motion.wait(0.05):
            if self.should_stop() or time.monotonic() > deadline:
                raise InterruptedError("Holding cancelled or timed out")
            if time.monotonic() >= next_read and callable(getattr(robot, "get_observation", None)):
                robot.get_observation()
                next_read = time.monotonic() + 0.2
        if self.should_stop():
            raise InterruptedError("Holding cancelled")
        with self.lock:
            self.phase = "running"
            self.status = (
                "Collecting excitation. Stop ends commands and preserves torque at the last target."
            )

    def _progress(self, name: str, completed: int, total: int) -> None:
        self.progress = f"{name}: {completed}/{total} samples"

    def _track_observation(self, robot: Any) -> None:
        read = getattr(robot, "get_observation", None)
        if not callable(read):
            return

        def observe() -> Any:
            observation = read()
            measured = np.asarray([observation[f"{name}.pos"] for name in SO101_JOINT_NAMES])
            if not np.isfinite(measured).all():
                raise ValueError("Hardware observation must contain six finite position readings.")
            # Publishing one array reference avoids blocking the timed collector on UI rendering.
            self.measured = measured
            return observation

        # Preserve the follower's identity so the collector retains its hardware preparation checks.
        robot.get_observation = observe

    def _collect(
        self, profile: RobotProfile, config: ExcitationConfig, destination: Path, lease: Any
    ) -> None:
        robot = None
        message = "Collection interrupted."
        try:
            with TemporaryDirectory(prefix="nexus-calibration-") as directory:
                robot = self._make_robot(profile, Path(directory))
                self._track_observation(robot)
                collect_excitation(
                    robot,
                    profile,
                    config,
                    output_dir=destination,
                    on_ready=lambda: self._wait_for_motion(robot),
                    should_stop=self.should_stop,
                    on_progress=self._progress,
                )
            message = f"Collection complete: {destination}."
        except Exception as error:
            message = f"Collection stopped: {error}."
        finally:
            with self.lock:
                bus = getattr(robot, "bus", None)
                if bus is not None and bus.is_connected:
                    self._pending_bus, self._pending_lease = bus, lease
                    message = "Serial cleanup failed. Port remains owned; retry disconnect."
                else:
                    lease.release()
                self.phase = "stopped"
                self.verified = False
                self.measured = None
                self.status = (
                    f"{message} Torque may remain enabled at the last target. "
                    "Support before release."
                )
                self.artifacts = ([str(self.profile_path)] if self.profile_path else []) + [
                    str(path) for path in sorted(destination.glob("*.npz"))
                ]

    def _close_pending(self) -> None:
        if self._pending_bus is not None:
            self._pending_bus.disconnect(disable_torque=False)
            self._pending_bus = None
            self._pending_lease.release()
            self._pending_lease = None

    def start_motion(self, *, hands_clear: bool) -> None:
        """Start the verified excitation only after a separate hands-clear action."""
        with self.lock:
            if self.phase != "holding" or self.should_stop():
                raise RuntimeError("Wait for active holding before starting excitation.")
            if not hands_clear:
                raise ValueError("Confirm hands and workspace are clear before starting motion.")
            self.motion.set()

    def stop(self) -> None:
        """Stop future commands while retaining holding torque at the last target."""
        self.cancel.set()
        self.verified = False
        active = self.worker is not None and self.worker.is_alive()
        self.status = (
            "Stop requested. Waiting for worker cleanup; torque is preserved."
            if active
            else "Stopped. No active collection worker; torque was not changed."
        )

    def disconnect(self) -> None:
        """Stop commands and close the port without releasing the motors."""
        self.stop()
        worker = self.worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2)
            if worker.is_alive():
                raise RuntimeError(
                    "Hardware worker is still stopping. Port remains owned; retry disconnect."
                )
        with self.lock:
            self._close_monitor()
            self._close_pending()
            self.phase = "disconnected"
            self.progress = ""
            self.verified = False
            self.status = (
                "Disconnected. Torque was not changed; support the arm before motor release."
            )

    def release(self, *, supported: bool) -> None:
        """Explicitly disable torque only after the operator supports the arm."""
        from lerobot.motors import SerialMotorsBus

        from so101_nexus.calibration.hardware import PortLease
        from so101_nexus.lerobot_adapter.normalization import build_so101_motors

        if not supported:
            raise ValueError(
                "Support the arm before releasing motors; it can fall when torque turns off."
            )
        with self.lock:
            self._idle()
            self._close_monitor()
            if not self.port:
                raise ValueError("Select and connect a follower first.")
            lease = PortLease(self.port).acquire()
            bus = None
            try:
                bus = SerialMotorsBus(port=self.port, motors=build_so101_motors(use_degrees=True))
                bus.connect(handshake=False)
                bus.disable_torque()
            finally:
                try:
                    if bus is not None:
                        bus.disconnect(disable_torque=False)
                finally:
                    if bus is not None and bus.is_connected:
                        self._pending_bus, self._pending_lease = bus, lease
                    else:
                        lease.release()
            self.torque_possible = False
            self.verified = False
            self.phase = "disconnected"
            self.status = (
                "Motors released, torque off. Keep the arm supported. "
                "Reconnect read-only to continue."
            )

    def fit(self, recordings_dir: str, output_dir: str, max_iters: int) -> tuple[list, list, str]:
        """Fit a candidate and expose held-out evidence without accepting dynamics."""
        from so101_nexus.sysid import fit_robot_profile

        with self.lock:
            self._idle()
            profile = self._aligned()
            paths = sorted(Path(recordings_dir).expanduser().glob("*.npz"))
            if not paths:
                raise ValueError("No completed recordings found. Collect all three episodes first.")
            destination = Path(output_dir).expanduser().resolve()
            if destination.exists():
                raise FileExistsError(f"Fit directory already exists: {destination}")
            previous_phase = self.phase
            self.phase = "fitting"
            self._fitting = True
            self.status = "Fitting candidate dynamics. Read-only measurements remain available."
        try:
            result = fit_robot_profile(
                profile,
                [RecordedTrajectory.load(path) for path in paths],
                output_dir=destination,
                max_iters=int(max_iters),
            )
            errors = [
                [episode.name, episode.split, joint, baseline, candidate]
                for episode in result.episodes
                for joint, baseline, candidate in zip(
                    SO101_JOINT_NAMES,
                    episode.baseline_rmse_deg,
                    episode.identified_rmse_deg,
                    strict=True,
                )
            ]
            diagnostics = [
                [
                    item.name,
                    item.identified,
                    item.at_bound,
                    item.poorly_identified,
                    item.ci95_half_width,
                ]
                for item in result.diagnostics
            ]
            summary = (
                f"Held-out improvement: {result.validation_improved}. "
                "Candidate saved, not accepted. "
                f"Review errors, uncertainty, and correlations in {destination / 'metrics.json'}."
            )
            with self.lock:
                self.artifacts += [
                    str(destination / name) for name in ("candidate_profile.json", "metrics.json")
                ]
                self.status = summary
            return errors, diagnostics, summary
        finally:
            with self.lock:
                self._fitting = False
                if self.phase == "fitting":
                    self.phase = previous_phase
