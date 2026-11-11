"""Generate, verify, and record repeatable SO-101 identification trajectories."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from copy import copy
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES, SO101_TCP_SITE_NAME, PhysicsConfig

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import mujoco

    from so101_nexus.profiles import RobotProfile


def _positive(value: float, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def _joints(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain six finite joint angles in degrees")
    return result


@dataclass(frozen=True)
class ExcitationConfig:
    """Explicit workspace pose and limits for automatic identification motion.

    Parameters
    ----------
    center_qpos_deg : tuple of float
        Verified, raised model pose at which the operator places the arm first.
    amplitudes_deg : tuple of float
        Maximum per-joint excursions, including the gripper angle.
    duration_s : float
        Duration of each of two training episodes and one held-out episode.
    ramp_s : float
        Smooth amplitude-envelope ramp at each end of an episode.
    table_z_m : float
        Height of the modeled horizontal work surface relative to the arm base.
    min_clearance_m : float
        Minimum tool-center height above the work surface.
    """

    center_qpos_deg: tuple[float, ...]
    amplitudes_deg: tuple[float, ...] = (8, 8, 8, 8, 10, 15)
    duration_s: float = 20.0
    ramp_s: float = 2.0
    max_velocity_deg_s: float = 45.0
    max_acceleration_deg_s2: float = 180.0
    joint_margin_deg: float = 3.0
    table_z_m: float = 0.0
    min_clearance_m: float = 0.02
    max_tracking_error_deg: float = 10.0
    max_start_error_deg: float = 3.0
    max_command_error_deg: float = 1.0
    max_read_duration_s: float = 0.1
    max_overrun_s: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        """Reject ambiguous poses and invalid motion or timing limits."""
        center = _joints(self.center_qpos_deg, "center_qpos_deg")
        amplitudes = _joints(self.amplitudes_deg, "amplitudes_deg")
        if np.any(amplitudes <= 0):
            raise ValueError("amplitudes_deg must be positive for every joint")
        object.__setattr__(self, "center_qpos_deg", tuple(float(value) for value in center))
        object.__setattr__(self, "amplitudes_deg", tuple(float(value) for value in amplitudes))
        for name in (
            "duration_s",
            "ramp_s",
            "max_velocity_deg_s",
            "max_acceleration_deg_s2",
            "joint_margin_deg",
            "min_clearance_m",
            "max_tracking_error_deg",
            "max_start_error_deg",
            "max_command_error_deg",
            "max_read_duration_s",
            "max_overrun_s",
        ):
            _positive(getattr(self, name), name)
        if self.ramp_s * 2 >= self.duration_s:
            raise ValueError("ramp_s must be less than half of duration_s")
        if isinstance(self.table_z_m, bool) or not math.isfinite(self.table_z_m):
            raise ValueError("table_z_m must be finite")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


def _array(value: Any, name: str, *, columns: int | None = None) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    shape_valid = (
        result.ndim == 1 if columns is None else result.ndim == 2 and result.shape[1] == columns
    )
    if not shape_valid or len(result) < 2 or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain at least two finite samples with the expected shape")
    if columns is None and (result[0] < 0 or np.any(np.diff(result) <= 0)):
        raise ValueError(f"{name} must be nonnegative and strictly increasing")
    result.flags.writeable = False
    return result


@dataclass(frozen=True)
class ExcitationTrajectory:
    """One planned episode of absolute model joint commands in degrees."""

    name: str
    split: Literal["train", "validation"]
    timestamps_s: np.ndarray
    commands_deg: np.ndarray

    def __post_init__(self) -> None:
        """Validate the episode before it can reach hardware."""
        if not self.name or self.split not in ("train", "validation"):
            raise ValueError("trajectory must have a name and train or validation split")
        object.__setattr__(self, "timestamps_s", _array(self.timestamps_s, "timestamps_s"))
        object.__setattr__(
            self, "commands_deg", _array(self.commands_deg, "commands_deg", columns=6)
        )
        if len(self.commands_deg) != len(self.timestamps_s):
            raise ValueError("commands_deg and timestamps_s must have equal lengths")


def profile_digest(profile: RobotProfile) -> str:
    """Hash the recording coordinate contract independently of fitted dynamics."""
    metadata = profile.to_dict()
    metadata.pop("dynamics", None)
    return hashlib.sha256(
        json.dumps(metadata, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class RecordedTrajectory:
    """Hardware observations and actual sent commands in model degrees.

    Command timestamps mark completion of a host write, not motor receipt.
    Measurement timestamps are the midpoint of the host read interval.
    """

    name: str
    split: Literal["train", "validation"]
    measurement_timestamps_s: np.ndarray
    command_timestamps_s: np.ndarray
    measured_qpos_deg: np.ndarray
    commanded_qpos_deg: np.ndarray
    profile_sha256: str
    read_start_timestamps_s: np.ndarray | None = None
    read_end_timestamps_s: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Reject malformed recordings before fitting or serialization."""
        if not self.name or self.split not in ("train", "validation"):
            raise ValueError("recording must have a name and train or validation split")
        if len(self.profile_sha256) != 64 or any(
            c not in "0123456789abcdef" for c in self.profile_sha256
        ):
            raise ValueError("profile_sha256 must be a lowercase SHA-256 digest")
        for name in ("measurement_timestamps_s", "command_timestamps_s"):
            object.__setattr__(self, name, _array(getattr(self, name), name))
        for name in ("measured_qpos_deg", "commanded_qpos_deg"):
            object.__setattr__(self, name, _array(getattr(self, name), name, columns=6))
        if len(self.measured_qpos_deg) != len(self.measurement_timestamps_s):
            raise ValueError("measurement arrays must have equal lengths")
        if len(self.commanded_qpos_deg) != len(self.command_timestamps_s):
            raise ValueError("command arrays must have equal lengths")
        starts, ends = self.read_start_timestamps_s, self.read_end_timestamps_s
        if (starts is None) != (ends is None):
            raise ValueError("both read timestamp bounds must be supplied together")
        if starts is not None and ends is not None:
            starts, ends = (
                _array(starts, "read_start_timestamps_s"),
                _array(ends, "read_end_timestamps_s"),
            )
            if starts.shape != self.measurement_timestamps_s.shape or ends.shape != starts.shape:
                raise ValueError("read timestamps must match measurements")
            if np.any(starts > self.measurement_timestamps_s) or np.any(
                ends < self.measurement_timestamps_s
            ):
                raise ValueError("read timestamps must bracket each measurement")
            object.__setattr__(self, "read_start_timestamps_s", starts)
            object.__setattr__(self, "read_end_timestamps_s", ends)

    def validate_profile(self, profile: RobotProfile) -> None:
        """Require the same robot, calibration, model, and command clock."""
        if self.profile_sha256 != profile_digest(profile):
            raise ValueError("recording profile does not match the supplied robot profile")

    def save(self, path: str | Path) -> None:
        """Atomically save arrays and JSON metadata without pickle."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            {
                "schema_version": 1,
                "name": self.name,
                "split": self.split,
                "profile_sha256": self.profile_sha256,
            }
        )
        arrays = {
            name: getattr(self, name)
            for name in (
                "measurement_timestamps_s",
                "command_timestamps_s",
                "measured_qpos_deg",
                "commanded_qpos_deg",
                "read_start_timestamps_s",
                "read_end_timestamps_s",
            )
            if getattr(self, name) is not None
        }
        descriptor, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".npz")
        try:
            with os.fdopen(descriptor, "wb") as stream:
                np.savez_compressed(stream, metadata=metadata, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> RecordedTrajectory:
        """Load a validated recording without executing serialized objects."""
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if not isinstance(metadata, dict) or type(metadata.get("schema_version")) is not int:
                raise ValueError("recording metadata requires integer schema_version")
            if metadata.pop("schema_version") != 1:
                raise ValueError("unsupported recording schema_version")
            try:
                return cls(
                    **metadata,
                    **{name: archive[name] for name in archive.files if name != "metadata"},
                )
            except TypeError as exc:
                raise ValueError(f"invalid recording fields: {exc}") from exc


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value**3 * (10 + value * (-15 + 6 * value))


def generate_excitation(
    config: ExcitationConfig, profile: RobotProfile
) -> tuple[ExcitationTrajectory, ...]:
    """Generate bounded multisines, joint chirps, and an independent held-out episode."""
    dt = profile.control_period_s
    count = round(config.duration_s / dt)
    if count < 12 or not math.isclose(count * dt, config.duration_s, abs_tol=1e-9):
        raise ValueError("duration_s must contain at least 12 whole control periods")
    times = np.arange(count + 1, dtype=np.float64) * dt
    envelope = _smoothstep(np.clip(np.minimum(times, times[-1] - times) / config.ramp_s, 0, 1))
    rng = np.random.default_rng(config.seed)
    center, amplitudes = np.array(config.center_qpos_deg), np.array(config.amplitudes_deg)
    episodes = []
    for index, name in enumerate(("multisine", "joint_chirps", "heldout")):
        if index == 1:
            local = times[:, None] - np.arange(6) * config.duration_s / 6
            segment = config.duration_s / 6
            window = _smoothstep(np.clip(np.minimum(local, segment - local) / (segment / 4), 0, 1))
            waves = window * np.sin(2 * np.pi * (0.15 * local + 0.45 * local**2 / segment))
        else:
            frequencies = rng.uniform(0.15, 1.05, (3, 6))
            phases = rng.uniform(0, 2 * np.pi, (3, 6))
            waves = np.sin(2 * np.pi * times[:, None, None] * frequencies + phases).mean(axis=1)
            if index == 2:
                waves = 0.8 * waves + 0.2 * np.sin(np.pi * times[:, None] / config.duration_s)
        excursion = waves * envelope[:, None] * amplitudes
        velocity = np.max(np.abs(np.diff(excursion, axis=0))) / dt
        acceleration = np.max(np.abs(np.diff(excursion, n=2, axis=0))) / dt**2
        scale = min(
            1.0,
            config.max_velocity_deg_s / max(velocity, 1e-12),
            config.max_acceleration_deg_s2 / max(acceleration, 1e-12),
        )
        episodes.append(
            ExcitationTrajectory(
                name, "validation" if index == 2 else "train", times, center + excursion * scale
            )
        )
    return tuple(episodes)


def _preflight_model(
    profile: RobotProfile, config: ExcitationConfig, model: mujoco.MjModel | None
) -> mujoco.MjModel:
    import mujoco

    from so101_nexus import get_so101_mujoco_model_path
    from so101_nexus.physics import apply_physics_config
    from so101_nexus.profiles import apply_robot_profile

    if model is None:
        spec = mujoco.MjSpec.from_file(str(get_so101_mujoco_model_path()))
        spec.worldbody.add_geom(
            name="sysid_table",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[1, 1, 0.01],
            pos=[0, 0, config.table_z_m],
        )
        model = spec.compile()
    else:
        model = copy(model)
    apply_robot_profile(model, profile)
    apply_physics_config(
        model, PhysicsConfig(control_period_s=profile.control_period_s), backend="mujoco"
    )
    return model


def _check_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: ExcitationConfig,
    name: str,
    robot_geoms: np.ndarray,
) -> None:
    import mujoco

    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise ValueError(f"{name}: simulation became nonfinite")
    for joint in SO101_JOINT_NAMES:
        item = model.joint(joint)
        if item.limited[0]:
            angle = float(data.qpos[item.qposadr[0]])
            margin = math.radians(config.joint_margin_deg)
            if not item.range[0] + margin <= angle <= item.range[1] - margin:
                raise ValueError(f"{name}: joint limit margin violated for {joint}")
    if (
        data.site_xpos[model.site(SO101_TCP_SITE_NAME).id, 2]
        < config.table_z_m + config.min_clearance_m
    ):
        raise ValueError(f"{name}: tool clearance violated")
    for contact in data.contact:
        if contact.dist <= 0 and np.any(robot_geoms[contact.geom]):
            names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) for i in contact.geom
            ]
            raise ValueError(f"{name}: collision between {names}")


def preflight_excitation(
    trajectories: Sequence[ExcitationTrajectory],
    profile: RobotProfile,
    config: ExcitationConfig,
    *,
    model: mujoco.MjModel | None = None,
) -> None:
    """Reject modeled collisions, joint limits, and low tool clearance before connection.

    The default scene contains the robot and a horizontal table. Supply the actual
    workspace model to include obstacles. Verification cannot certify hardware safety.
    """
    import mujoco

    if not trajectories:
        raise ValueError("at least one excitation trajectory is required")
    model = _preflight_model(profile, config, model)
    qadr = np.array([model.joint(name).qposadr[0] for name in SO101_JOINT_NAMES])
    actids = np.array([model.actuator(name).id for name in SO101_JOINT_NAMES])
    ctrl_range = model.actuator_ctrlrange[actids]
    robot_bodies = np.zeros(model.nbody, dtype=bool)
    base_id = model.body("base").id
    robot_bodies[base_id] = True
    for body in range(base_id + 1, model.nbody):
        robot_bodies[body] = robot_bodies[model.body_parentid[body]]
    robot_geoms = robot_bodies[model.geom_bodyid]
    substeps = round(profile.control_period_s / model.opt.timestep)
    if substeps < 1 or not math.isclose(
        substeps * model.opt.timestep, profile.control_period_s, abs_tol=1e-9
    ):
        raise ValueError("model timestep must divide profile control period")
    for trajectory in trajectories:
        if not np.allclose(
            np.diff(trajectory.timestamps_s), profile.control_period_s, rtol=0, atol=1e-9
        ):
            raise ValueError("trajectory timestamps must match profile control period")
        _check_commands(trajectory, profile, config)
        targets = np.deg2rad(trajectory.commands_deg)
        if np.any(targets < ctrl_range[:, 0]) or np.any(targets > ctrl_range[:, 1]):
            raise ValueError(f"{trajectory.name}: actuator control limit violated")
        data = mujoco.MjData(model)
        previous = targets[0]
        for target in targets:
            for fraction in np.arange(1, substeps + 1) / substeps:
                data.qpos[qadr] = previous + fraction * (target - previous)
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                _check_state(model, data, config, trajectory.name, robot_geoms)
            previous = target
        data = mujoco.MjData(model)
        data.qpos[qadr] = targets[0]
        mujoco.mj_forward(model, data)
        for target in targets:
            data.ctrl[actids] = target
            for _ in range(substeps):
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                _check_state(model, data, config, trajectory.name, robot_geoms)


def _check_commands(
    trajectory: ExcitationTrajectory, profile: RobotProfile, config: ExcitationConfig
) -> None:
    lower, upper = profile.model_joint_limits_deg()
    if np.any(trajectory.commands_deg < lower) or np.any(trajectory.commands_deg > upper):
        raise ValueError(f"{trajectory.name}: physical calibration limit violated")
    velocity = np.diff(trajectory.commands_deg, axis=0) / profile.control_period_s
    acceleration = np.diff(velocity, axis=0) / profile.control_period_s
    if np.max(np.abs(velocity)) > config.max_velocity_deg_s + 1e-9 or (
        acceleration.size and np.max(np.abs(acceleration)) > config.max_acceleration_deg_s2 + 1e-9
    ):
        raise ValueError(f"{trajectory.name}: velocity or acceleration limit violated")


def _robot_calibration(robot: Any) -> dict[str, Any]:
    return {
        name: asdict(value) if is_dataclass(value) else dict(value)
        for name, value in robot.calibration.items()
    }


def _read_position(
    robot: Any,
    profile: RobotProfile,
    clock: Callable[[], float],
    origin: float,
    max_duration: float,
) -> tuple[np.ndarray, float, float]:
    before = clock() - origin
    obs = robot.get_observation()
    after = clock() - origin
    if after < before or after - before > max_duration:
        raise RuntimeError("hardware read exceeded the allowed duration or clock moved backward")
    values = _joints([obs[f"{name}.pos"] for name in SO101_JOINT_NAMES], "hardware observation")
    return np.asarray(profile.to_model_deg(values)), before, after


def _collect_episode(
    robot: Any,
    profile: RobotProfile,
    config: ExcitationConfig,
    trajectory: ExcitationTrajectory,
    clock: Callable[[], float],
    sleep: Callable[[float], None],
) -> RecordedTrajectory:
    origin = clock()
    measured, commands, starts, ends, writes = [], [], [], [], []
    previous = trajectory.commands_deg[0]
    for index, (planned_time, target) in enumerate(
        zip(trajectory.timestamps_s, trajectory.commands_deg, strict=True)
    ):
        remaining = origin + planned_time - clock()
        if remaining > 0:
            sleep(remaining)
        if clock() - origin - planned_time > config.max_overrun_s:
            raise RuntimeError("hardware loop exceeded the allowed overrun")
        position, before, after = _read_position(
            robot, profile, clock, origin, config.max_read_duration_s
        )
        if clock() - origin - planned_time > config.max_overrun_s:
            raise RuntimeError("hardware read caused an excessive loop overrun")
        threshold = config.max_tracking_error_deg if index else config.max_start_error_deg
        if np.max(np.abs(position - previous)) > threshold:
            raise RuntimeError("hardware tracking error exceeded the allowed limit")
        action = profile.from_model_deg(target)
        sent = robot.send_action(
            {
                f"{name}.pos": float(value)
                for name, value in zip(SO101_JOINT_NAMES, action, strict=True)
            }
        )
        write = clock() - origin
        if write - planned_time > config.max_overrun_s:
            raise RuntimeError("hardware write exceeded the allowed overrun")
        actual = _joints([sent[f"{name}.pos"] for name in SO101_JOINT_NAMES], "sent action")
        previous = np.asarray(profile.to_model_deg(actual))
        if np.max(np.abs(previous - target)) > config.max_command_error_deg:
            raise RuntimeError(
                "sent command differs from planned excitation beyond the allowed limit"
            )
        measured.append(position)
        commands.append(previous)
        starts.append(before)
        ends.append(after)
        writes.append(write)
    return RecordedTrajectory(
        trajectory.name,
        trajectory.split,
        (np.array(starts) + ends) / 2,
        np.array(writes),
        np.array(measured),
        np.array(commands),
        profile_digest(profile),
        np.array(starts),
        np.array(ends),
    )


def _validate_robot(robot: Any, profile: RobotProfile) -> None:
    if getattr(robot, "is_connected", False):
        raise ValueError("automatic collection requires a disconnected robot")
    if getattr(robot.config, "use_degrees", None) is not True:
        raise ValueError("automatic collection requires LeRobot use_degrees=True")
    if getattr(robot.config, "cameras", {}) or getattr(robot, "cameras", {}):
        raise ValueError("automatic collection requires cameras disabled for consistent timing")
    if getattr(robot.config, "id", None) != profile.robot_id:
        raise ValueError("robot id does not match profile robot_id")
    servo_gains = tuple(
        getattr(robot.config, f"position_{coefficient}_coefficient", None)
        for coefficient in ("p", "i", "d")
    )
    if servo_gains != profile.servo_gains:
        raise ValueError("robot servo gains do not match profile")
    if _robot_calibration(robot) != {
        name: dict(value) for name, value in profile.calibration.items()
    }:
        raise ValueError("robot calibration does not match profile")


def _disconnect_robot(robot: Any) -> None:
    if robot.is_connected:
        robot.disconnect()
    elif (bus := getattr(robot, "bus", None)) is not None and bus.is_connected:
        bus.disconnect(disable_torque=getattr(robot.config, "disable_torque_on_disconnect", True))


def _prepare_follower_connection(
    robot: Any, profile: RobotProfile, config: ExcitationConfig
) -> None:
    try:
        from lerobot.robots.so_follower.so_follower import SOFollower
    except ModuleNotFoundError:
        return
    if not isinstance(robot, SOFollower):
        return

    from so101_nexus.lerobot_adapter.normalization import normalize_ticks

    bus = robot.bus
    try:
        bus.connect()
        torque = bus.sync_read("Torque_Enable", normalize=False)
        if set(torque) != set(SO101_JOINT_NAMES) or any(torque.values()):
            raise RuntimeError("automatic collection requires every motor torque off before start")
        if not bus.is_calibrated:
            raise RuntimeError("hardware calibration does not match the loaded calibration")
        raw = bus.sync_read("Present_Position", normalize=False)
        normalized = normalize_ticks(raw, motors=bus.motors, calibration=robot.calibration)
        position = profile.to_model_deg(
            _joints([normalized[name] for name in SO101_JOINT_NAMES], "hardware start position")
        )
        if (
            np.max(np.abs(np.asarray(position) - config.center_qpos_deg))
            > config.max_start_error_deg
        ):
            raise RuntimeError(
                "hardware start position differs from the verified excitation center"
            )
        # SOFollower.configure enables torque, so replace any stale servo target first.
        bus.sync_write("Goal_Position", raw, normalize=False)
    except BaseException as error:
        try:
            bus.disconnect(disable_torque=False)
        except Exception as cleanup_error:
            error.add_note(f"Preparation cleanup failed: {cleanup_error}")
        raise
    bus.disconnect(disable_torque=False)


def collect_excitation(
    robot: Any,
    profile: RobotProfile,
    config: ExcitationConfig,
    *,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    model: mujoco.MjModel | None = None,
    output_dir: str | Path | None = None,
) -> tuple[RecordedTrajectory, ...]:
    """Preflight every episode, then automatically collect through a LeRobot follower.

    Place the physical arm at ``center_qpos_deg`` before starting. The function
    owns connection and disconnection, never requests calibration, and stops sending
    commands on a fault. Cameras must be disabled for consistent sample timing.
    Hardware disconnect behavior belongs to the follower driver.
    An optional new ``output_dir`` preserves each completed episode immediately.
    Standard LeRobot SO followers must start torque-free; their current position
    seeds the servo goal before connection enables torque. Custom adapters own
    their connection behavior.
    """
    _validate_robot(robot, profile)
    trajectories = generate_excitation(config, profile)
    if model is None:
        preflight_excitation(trajectories, profile, config)
    else:
        preflight_excitation(trajectories, profile, config, model=model)
    destination = Path(output_dir) if output_dir is not None else None
    if destination is not None:
        destination.mkdir(parents=True, exist_ok=False)
    _prepare_follower_connection(robot, profile, config)
    try:
        robot.connect(calibrate=False)
        if not robot.is_calibrated:
            raise RuntimeError("hardware calibration does not match the loaded calibration")
        recordings = []
        for index, trajectory in enumerate(trajectories):
            recording = _collect_episode(robot, profile, config, trajectory, clock, sleep)
            if destination is not None:
                recording.save(destination / f"{index:02d}_{trajectory.name}.npz")
            recordings.append(recording)
    except BaseException as error:
        try:
            _disconnect_robot(robot)
        except Exception as cleanup_error:
            error.add_note(f"Hardware cleanup failed: {cleanup_error}")
        raise
    _disconnect_robot(robot)
    return tuple(recordings)
