"""Portable SO101 encoder calibration, coordinate alignment, and joint dynamics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES

if TYPE_CHECKING:
    import mujoco
    import torch

_CALIBRATION_FIELDS = frozenset({"id", "drive_mode", "homing_offset", "range_min", "range_max"})


def _finite(value: object, name: str, *, minimum: float = 0, positive: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (positive and value == minimum)
    ):
        bound = ">" if positive else ">="
        raise ValueError(f"{name} must be finite and {bound} {minimum}, got {value!r}")


def _angles(values: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or len(values) != length:
        raise ValueError(f"{name} must have exactly {length} degree values")
    for value in values:
        _finite(value, name, minimum=-math.inf)
    return tuple(float(value) for value in values)


def _servo_gains(values: object) -> tuple[int, int, int]:
    if (
        not isinstance(values, (tuple, list))
        or len(values) != 3
        or any(type(value) is not int or not 0 <= value <= 255 for value in values)
    ):
        raise ValueError("servo_gains must contain three integer P/I/D coefficients in [0, 255]")
    return values[0], values[1], values[2]


@dataclass(frozen=True)
class JointDynamics:
    """Effective hinge dynamics in SI units.

    ``armature`` is kg m², ``frictionloss`` and ``force_limit`` are N m,
    ``kp`` is N m/rad, and ``kd`` and ``joint_damping`` are N m s/rad.
    Actuator derivative gain and passive joint damping remain distinct.
    """

    armature: float
    frictionloss: float
    kp: float
    kd: float
    joint_damping: float = 0.0
    force_limit: float = 2.94

    def __post_init__(self) -> None:
        """Reject nonphysical values before they can reach a model."""
        for name, value in asdict(self).items():
            _finite(value, name, positive=name in ("kp", "force_limit"))


def _calibration_snapshot(
    calibration: Mapping[str, Mapping[str, int]],
) -> Mapping[str, Mapping[str, int]]:
    if not isinstance(calibration, Mapping) or set(calibration) != set(SO101_JOINT_NAMES):
        raise ValueError("calibration must contain exactly the six SO101 motor names")
    result = {}
    for expected_id, name in enumerate(SO101_JOINT_NAMES, 1):
        motor = calibration[name]
        if not isinstance(motor, Mapping) or set(motor) != _CALIBRATION_FIELDS:
            raise ValueError(f"calibration for {name} must contain {_CALIBRATION_FIELDS}")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in motor.values()):
            raise ValueError(f"calibration for {name} must contain integer fields")
        if motor["id"] != expected_id or motor["drive_mode"] not in (0, 1):
            raise ValueError(f"calibration for {name} has an invalid id or drive_mode")
        if not 0 <= motor["range_min"] < motor["range_max"] <= 4095:
            raise ValueError(f"calibration for {name} requires 0 <= range_min < range_max <= 4095")
        result[name] = MappingProxyType(dict(motor))
    return MappingProxyType(result)


@dataclass(frozen=True)
class RobotProfile:
    """Saved calibration and effective dynamics for one physical SO101 arm.

    The first five hardware values are LeRobot degrees; the sixth is gripper
    percent. ``joint_offsets_deg`` aligns body joints with model coordinates.
    ``gripper_limits_deg`` gives model angles at hardware 0 and 100 percent.
    A nonempty ``model_sha256`` pins the vendored robot assets, excluding task
    scenery. Dynamics describe the control loop at ``control_period_s`` with
    ``servo_gains`` as hardware P/I/D register coefficients, distinct from
    simulated actuator gains in SI units.
    """

    robot_id: str
    calibration: Mapping[str, Mapping[str, int]]
    joint_offsets_deg: tuple[float, ...] = (0, 0, 0, 0, 0)
    gripper_limits_deg: tuple[float, float] = (-10, 100)
    dynamics: Mapping[str, JointDynamics] = field(default_factory=dict)
    model_sha256: str = ""
    control_period_s: float = 0.02
    servo_gains: tuple[int, int, int] = (16, 0, 32)
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Validate and snapshot inputs so a saved profile stays reproducible."""
        if (
            not isinstance(self.robot_id, str)
            or not self.robot_id.strip()
            or self.robot_id in (".", "..")
            or any(char in self.robot_id for char in "/\\\0:")
        ):
            raise ValueError("robot_id must be a nonempty filename component")
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError(f"Unsupported profile schema_version: {self.schema_version!r}")
        _finite(self.control_period_s, "control_period_s", positive=True)
        offsets = _angles(self.joint_offsets_deg, 5, "joint_offsets_deg")
        limits = _angles(self.gripper_limits_deg, 2, "gripper_limits_deg")
        if limits[0] >= limits[1]:
            raise ValueError("gripper_limits_deg must be increasing")
        if not isinstance(self.model_sha256, str) or (
            self.model_sha256
            and (
                len(self.model_sha256) != 64
                or any(c not in "0123456789abcdef" for c in self.model_sha256)
            )
        ):
            raise ValueError("model_sha256 must be empty or a lowercase SHA256 hex digest")
        if not isinstance(self.dynamics, Mapping) or not set(self.dynamics) <= set(
            SO101_JOINT_NAMES
        ):
            raise ValueError("dynamics must be keyed by SO101 joint names")
        if any(not isinstance(value, JointDynamics) for value in self.dynamics.values()):
            raise TypeError("dynamics values must be JointDynamics")
        object.__setattr__(self, "calibration", _calibration_snapshot(self.calibration))
        object.__setattr__(self, "servo_gains", _servo_gains(self.servo_gains))
        object.__setattr__(self, "joint_offsets_deg", offsets)
        object.__setattr__(self, "gripper_limits_deg", (limits[0], limits[1]))
        object.__setattr__(self, "dynamics", MappingProxyType(dict(self.dynamics)))

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON-compatible data with explicit schema version."""
        return {
            "schema_version": self.schema_version,
            "robot_id": self.robot_id,
            "calibration": {name: dict(motor) for name, motor in self.calibration.items()},
            "joint_offsets_deg": list(self.joint_offsets_deg),
            "gripper_limits_deg": list(self.gripper_limits_deg),
            "dynamics": {name: asdict(value) for name, value in self.dynamics.items()},
            "model_sha256": self.model_sha256,
            "control_period_s": self.control_period_s,
            "servo_gains": list(self.servo_gains),
        }

    def __deepcopy__(self, memo: dict[int, object]) -> RobotProfile:
        """Share this immutable snapshot when environment configurations are copied."""
        return self

    def with_reference_pose(
        self,
        *,
        hardware_body_deg: tuple[float, ...],
        model_body_deg: tuple[float, ...],
    ) -> RobotProfile:
        """Align five body joints using corresponding measured and model poses.

        ``hardware_body_deg`` contains LeRobot degree readings at a known pose.
        ``model_body_deg`` contains that physical pose in model coordinates.
        The gripper retains its separately measured endpoint mapping.
        """
        hardware = _angles(hardware_body_deg, 5, "hardware_body_deg")
        model = _angles(model_body_deg, 5, "model_body_deg")
        return replace(
            self,
            joint_offsets_deg=tuple(
                target - measured for measured, target in zip(hardware, model, strict=True)
            ),
        )

    def model_joint_limits_deg(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return calibrated joint travel in model degrees, including alignment."""
        half_spans = tuple(
            (self.calibration[name]["range_max"] - self.calibration[name]["range_min"]) * 180 / 4095
            for name in SO101_JOINT_NAMES[:-1]
        )
        lower = tuple(
            offset - span for offset, span in zip(self.joint_offsets_deg, half_spans, strict=True)
        )
        upper = tuple(
            offset + span for offset, span in zip(self.joint_offsets_deg, half_spans, strict=True)
        )
        return (*lower, self.gripper_limits_deg[0]), (*upper, self.gripper_limits_deg[1])

    def save(self, path: str | Path) -> None:
        """Atomically save the profile without changing hardware calibration."""
        _write_json(path, self.to_dict())

    def export_lerobot(self, path: str | Path) -> None:
        """Atomically export the original calibration in standard LeRobot JSON."""
        _write_json(path, {name: dict(motor) for name, motor in self.calibration.items()})

    @classmethod
    def load(cls, path: str | Path) -> RobotProfile:
        """Load a strict versioned profile, rejecting unknown fields."""
        data = _load_json(path)
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("A profile with schema_version=1 is required")
        try:
            data["dynamics"] = {
                name: JointDynamics(**value) for name, value in data.get("dynamics", {}).items()
            }
            return cls(**data)
        except (TypeError, AttributeError) as exc:
            raise ValueError(f"Invalid robot profile: {exc}") from exc

    @classmethod
    def from_lerobot(
        cls,
        path: str | Path,
        *,
        robot_id: str,
        joint_offsets_deg: tuple[float, ...] = (0, 0, 0, 0, 0),
        gripper_limits_deg: tuple[float, float] = (-10, 100),
        servo_gains: tuple[int, int, int] = (16, 0, 32),
    ) -> RobotProfile:
        """Import a LeRobot motor-calibration JSON without modifying it."""
        return cls(
            robot_id=robot_id,
            calibration=_load_json(path),
            joint_offsets_deg=joint_offsets_deg,
            gripper_limits_deg=gripper_limits_deg,
            model_sha256=model_fingerprint(),
            servo_gains=servo_gains,
        )

    @overload
    def to_model_deg(self, values: np.ndarray) -> np.ndarray: ...

    @overload
    def to_model_deg(self, values: torch.Tensor) -> torch.Tensor: ...

    def to_model_deg(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Map hardware body degrees and gripper percent to model joint degrees."""
        return self._transform(values, inverse=False)

    @overload
    def from_model_deg(self, values: np.ndarray) -> np.ndarray: ...

    @overload
    def from_model_deg(self, values: torch.Tensor) -> torch.Tensor: ...

    def from_model_deg(self, values: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Map model joint degrees to hardware body degrees and gripper percent."""
        return self._transform(values, inverse=True)

    def _transform(
        self, values: np.ndarray | torch.Tensor, *, inverse: bool
    ) -> np.ndarray | torch.Tensor:
        if isinstance(values, np.ndarray):
            if values.dtype.kind not in "iuf":
                raise ValueError("Joint values must have a real numeric dtype")
            result = np.array(values, dtype=np.result_type(values.dtype, np.float32), copy=True)
            finite = bool(np.isfinite(result).all())
        else:
            import torch

            if not isinstance(values, torch.Tensor):
                raise TypeError("Joint values must be a NumPy array or torch tensor")
            if values.is_complex() or values.dtype == torch.bool:
                raise ValueError("Joint values must have a real numeric dtype")
            result = values.clone() if values.is_floating_point() else values.to(torch.float32)
            finite = bool(torch.isfinite(result).all())
        if result.ndim < 1 or result.shape[-1] != 6 or not finite:
            raise ValueError("Joint values must be finite with shape (..., 6)")
        low, high = self.gripper_limits_deg
        direction = -1 if inverse else 1
        if isinstance(result, np.ndarray):
            result[..., :5] += direction * np.asarray(self.joint_offsets_deg, dtype=result.dtype)
        else:
            result[..., :5] += direction * result.new_tensor(self.joint_offsets_deg)
        result[..., 5] = (
            ((result[..., 5] - low) * 100 / (high - low))
            if inverse
            else (low + result[..., 5] * (high - low) / 100)
        )
        return result


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_json(path: str | Path) -> Any:
    def unique_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON field: {key}")
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique_fields)


@cache
def model_fingerprint() -> str:
    """Hash the vendored robot XML and mesh assets independently of task scenery."""
    directory = Path(__file__).parent / "assets" / "SO101_menagerie"
    digest = hashlib.sha256()
    paths = [directory / "so101.xml", *sorted((directory / "assets").glob("*.stl"))]
    for path in paths:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def apply_robot_profile(model: mujoco.MjModel, profile: RobotProfile) -> None:
    """Apply calibrated travel and validated dynamics before CPU or Warp execution."""
    import mujoco

    if profile.model_sha256 and profile.model_sha256 != model_fingerprint():
        raise ValueError("Robot profile model fingerprint does not match the vendored robot")
    indices = {name: _profile_indices(model, name) for name in SO101_JOINT_NAMES}
    lower, upper = profile.model_joint_limits_deg()
    limits = {}
    for i, name in enumerate(SO101_JOINT_NAMES):
        joint_id, _, actuator_id = indices[name]
        joint_range = (
            model.jnt_range[joint_id] if model.jnt_limited[joint_id] else (-math.inf, math.inf)
        )
        ctrl_range = (
            model.actuator_ctrlrange[actuator_id]
            if model.actuator_ctrllimited[actuator_id]
            else (-math.inf, math.inf)
        )
        low = max(math.radians(lower[i]), joint_range[0], ctrl_range[0])
        high = min(math.radians(upper[i]), joint_range[1], ctrl_range[1])
        if low >= high:
            raise ValueError(f"Robot profile limits for {name!r} do not overlap model limits")
        limits[name] = (low, high)
    for name, (joint_id, _, actuator_id) in indices.items():
        model.jnt_range[joint_id] = limits[name]
        model.jnt_limited[joint_id] = True
        model.actuator_ctrlrange[actuator_id] = limits[name]
        model.actuator_ctrllimited[actuator_id] = True
    for name, dynamics in profile.dynamics.items():
        _, dof, actuator = indices[name]
        model.dof_armature[dof] = dynamics.armature
        model.dof_frictionloss[dof] = dynamics.frictionloss
        model.dof_damping[dof] = dynamics.joint_damping
        model.actuator_gainprm[actuator, 0] = dynamics.kp
        model.actuator_biasprm[actuator, 1] = -dynamics.kp
        model.actuator_biasprm[actuator, 2] = -dynamics.kd
        model.actuator_forcelimited[actuator] = True
        model.actuator_forcerange[actuator] = (-dynamics.force_limit, dynamics.force_limit)
    mujoco.mj_setConst(model, mujoco.MjData(model))


def _profile_indices(model: mujoco.MjModel, name: str) -> tuple[int, int, int]:
    import mujoco

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if joint_id < 0 or actuator_id < 0:
        raise ValueError(f"Robot profile requires joint and actuator {name!r}")
    if (
        model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE
        or model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT
        or model.actuator_trnid[actuator_id, 0] != joint_id
        or model.actuator_dyntype[actuator_id] != mujoco.mjtDyn.mjDYN_NONE
        or model.actuator_gaintype[actuator_id] != mujoco.mjtGain.mjGAIN_FIXED
        or model.actuator_biastype[actuator_id] != mujoco.mjtBias.mjBIAS_AFFINE
        or not np.array_equal(model.actuator_gear[actuator_id], (1, 0, 0, 0, 0, 0))
        or model.actuator_biasprm[actuator_id, 0] != 0
    ):
        raise ValueError(f"Robot profile requires a direct hinge position actuator for {name!r}")
    return joint_id, int(model.jnt_dofadr[joint_id]), actuator_id
