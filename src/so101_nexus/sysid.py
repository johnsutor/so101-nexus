"""Fit robot dynamics from timestamped trajectories with MuJoCo's sysid toolbox."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES, PhysicsConfig
from so101_nexus.profiles import JointDynamics, RobotProfile, apply_robot_profile

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import ModuleType

    import mujoco

    from so101_nexus.excitation import RecordedTrajectory

_BOUNDS = {
    "armature": (0.0, 0.5),
    "frictionloss": (0.0, 2.0),
    "kp": (1.0, 2500.0),
    "kd": (0.0, 100.0),
}
_DEFAULT_PARAMETERS = tuple(f"{name}.{attr}" for name in SO101_JOINT_NAMES for attr in ("kp", "kd"))


@dataclass(frozen=True)
class EpisodeFit:
    """Per-joint prediction errors in degrees for one recorded episode."""

    name: str
    split: Literal["train", "validation"]
    baseline_rmse_deg: tuple[float, ...]
    identified_rmse_deg: tuple[float, ...]


@dataclass(frozen=True)
class ParameterDiagnostic:
    """Local fit diagnostics; confidence intervals assume independent residuals."""

    name: str
    nominal: float
    identified: float
    bounds: tuple[float, float]
    frozen: bool
    ci95_half_width: float | None
    at_bound: bool
    poorly_identified: bool


@dataclass(frozen=True)
class FitResult:
    """Candidate profile and held-out evidence, without automatic acceptance."""

    candidate_profile: RobotProfile
    episodes: tuple[EpisodeFit, ...]
    diagnostics: tuple[ParameterDiagnostic, ...]
    correlations: tuple[tuple[str, str, float], ...]
    validation_improved: bool


def _toolbox() -> ModuleType:
    toolbox = import_module("mujoco.sysid")
    required = ("Parameter", "ParameterDict", "ModelSequences", "build_residual_fn", "optimize")
    if any(not hasattr(toolbox, name) for name in required):
        raise ImportError("System identification requires the sysid extra: uv sync --extra sysid")
    return toolbox


def _validate_parameters(
    parameters: Sequence[str], bounds: Mapping[str, tuple[float, float]], freeze: Sequence[str]
) -> None:
    supported = {f"{joint}.{attr}" for joint in SO101_JOINT_NAMES for attr in _BOUNDS}
    if not parameters or len(set(parameters)) != len(parameters) or set(parameters) - supported:
        raise ValueError(
            "parameters must be unique joint.parameter names using "
            "armature, frictionloss, kp, or kd"
        )
    if set(bounds) - set(parameters) or set(freeze) - set(parameters):
        raise ValueError("bounds and freeze must reference selected parameters")
    if len(set(freeze)) != len(freeze) or not set(parameters) - set(freeze):
        raise ValueError("At least one selected parameter must remain unfrozen")
    for name, limits in bounds.items():
        if (
            not isinstance(limits, (tuple, list))
            or len(limits) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in limits
            )
            or limits[0] < 0
            or limits[0] >= limits[1]
            or (name.rsplit(".", 1)[1] == "kp" and limits[0] <= 0)
        ):
            raise ValueError(f"Invalid bounds for {name!r}")


def _validate_trajectories(
    profile: RobotProfile, trajectories: Sequence[RecordedTrajectory]
) -> None:
    if {episode.split for episode in trajectories} != {"train", "validation"}:
        raise ValueError("Both training and independent validation episodes are required")
    if len({episode.name for episode in trajectories}) != len(trajectories):
        raise ValueError("Trajectory names must be unique across training and validation")
    for episode in trajectories:
        episode.validate_profile(profile)
        measurements, commands = episode.measurement_timestamps_s, episode.command_timestamps_s
        if len(measurements) < 3:
            raise ValueError("Each trajectory requires at least three measurements")
        if not np.any((commands >= measurements[0]) & (commands < measurements[-1])):
            raise ValueError("Each trajectory requires a command followed by a measurement")
    training = [episode for episode in trajectories if episode.split == "train"]
    validation = [episode for episode in trajectories if episode.split == "validation"]
    for held_out in validation:
        if any(
            np.array_equal(held_out.measured_qpos_deg, fitted.measured_qpos_deg)
            and np.array_equal(held_out.commanded_qpos_deg, fitted.commanded_qpos_deg)
            for fitted in training
        ):
            raise ValueError("Validation data must be independent of training recordings")


def _make_spec(profile: RobotProfile) -> mujoco.MjSpec:
    import mujoco

    from so101_nexus import get_so101_mujoco_model_path

    spec = mujoco.MjSpec.from_file(str(get_so101_mujoco_model_path()))
    spec.option.timestep = PhysicsConfig(control_period_s=profile.control_period_s).timestep_s
    model = spec.compile()
    apply_robot_profile(model, profile)
    for name in SO101_JOINT_NAMES:
        joint, actuator = model.joint(name), model.actuator(name)
        spec.joint(name).armature = model.dof_armature[joint.dofadr[0]]
        spec.joint(name).frictionloss = model.dof_frictionloss[joint.dofadr[0]]
        spec.joint(name).damping[0] = model.dof_damping[joint.dofadr[0]]
        spec.joint(name).range[:] = joint.range
        spec.joint(name).limited = bool(joint.limited[0])
        spec.actuator(name).gainprm[:] = actuator.gainprm
        spec.actuator(name).biasprm[:] = actuator.biasprm
        spec.actuator(name).forcerange[:] = actuator.forcerange
        spec.actuator(name).forcelimited = bool(actuator.forcelimited[0])
        spec.actuator(name).ctrlrange[:] = actuator.ctrlrange
        spec.actuator(name).ctrllimited = bool(actuator.ctrllimited[0])
        spec.add_sensor(
            name=f"sysid_{name}",
            type=mujoco.mjtSensor.mjSENS_JOINTPOS,
            objtype=mujoco.mjtObj.mjOBJ_JOINT,
            objname=name,
        )
    return spec


def _nominal_dynamics(model: mujoco.MjModel) -> dict[str, JointDynamics]:
    return {
        name: JointDynamics(
            armature=float(model.dof_armature[model.joint(name).dofadr[0]]),
            frictionloss=float(model.dof_frictionloss[model.joint(name).dofadr[0]]),
            kp=float(model.actuator(name).gainprm[0]),
            kd=float(-model.actuator(name).biasprm[2]),
            joint_damping=float(model.dof_damping[model.joint(name).dofadr[0]]),
            force_limit=float(model.actuator(name).forcerange[1]),
        )
        for name in SO101_JOINT_NAMES
    }


def _parameters(
    toolbox: Any,
    nominal: Mapping[str, JointDynamics],
    selected: Sequence[str],
    bounds: Mapping[str, tuple[float, float]],
    freeze: Sequence[str],
) -> Any:
    result = toolbox.ParameterDict()
    for name in selected:
        joint, attr = name.rsplit(".", 1)
        lower, upper = bounds.get(name, _BOUNDS[attr])
        value = getattr(nominal[joint], attr)
        if not lower <= value <= upper:
            raise ValueError(f"Nominal {name}={value} is outside its bounds")

        def modifier(spec: Any, parameter: Any, joint: str = joint, attr: str = attr) -> None:
            value = float(parameter.value[0])
            if attr in ("kp", "kd"):
                apply_gain = toolbox.apply_pgain if attr == "kp" else toolbox.apply_dgain
                apply_gain(spec, joint, value)
            else:
                setattr(spec.joint(joint), attr, value)

        result.add(toolbox.Parameter(name, value, lower, upper, name in freeze, modifier))
    return result


def _model_sequences(toolbox: Any, spec: Any, episode: RecordedTrajectory) -> Any:
    model = spec.compile()
    times = episode.measurement_timestamps_s - episode.measurement_timestamps_s[0]
    control_times = episode.command_timestamps_s - episode.measurement_timestamps_s[0]
    qpos_rad = np.radians(episode.measured_qpos_deg)
    controls_rad = np.radians(episode.commanded_qpos_deg)
    grid = np.arange(math.ceil(times[-1] / model.opt.timestep) + 1) * model.opt.timestep
    indices = np.searchsorted(control_times, grid, side="right") - 1
    controls = controls_rad[np.maximum(indices, 0)].copy()
    controls[indices < 0] = qpos_rad[0]
    velocity_rad_s = np.gradient(qpos_rad, times, axis=0)[0]
    initial = toolbox.create_initial_state(
        model,
        qpos_rad[0],
        velocity_rad_s,
        qpos_names=[f"{name}_qpos" for name in SO101_JOINT_NAMES],
        qvel_names=[f"{name}_qvel" for name in SO101_JOINT_NAMES],
    )
    return toolbox.ModelSequences(
        name=episode.name,
        spec=spec,
        sequence_name=episode.name,
        initial_state=initial,
        control=toolbox.TimeSeries.from_control_names(
            grid, controls, model, names=SO101_JOINT_NAMES
        ),
        sensordata=toolbox.TimeSeries.from_names(
            times, qpos_rad, model, names=[f"sysid_{name}" for name in SO101_JOINT_NAMES]
        ),
        allow_missing_sensors=True,
    )


def _position_residual(
    toolbox: Any,
    _params: Any,
    _predicted: Any,
    measured: Any,
    model: Any,
    _return_all: bool,
    *,
    state: np.ndarray,
) -> tuple[np.ndarray, Any, Any]:
    # Rollout sensors are evaluated before integration; state positions carry the
    # matching post-integration timestamps needed for hardware read alignment.
    times = measured.times[1:]
    addresses = np.array([model.joint(name).qposadr[0] for name in SO101_JOINT_NAMES])
    prediction = np.column_stack(
        [np.interp(times, state[:, 0], state[:, 1 + address]) for address in addresses]
    )
    predicted_ts = toolbox.TimeSeries(times, prediction, measured.signal_mapping)
    measured_ts = toolbox.TimeSeries(times, measured.data[1:], measured.signal_mapping)
    return (prediction - measured_ts.data).ravel(), predicted_ts, measured_ts


def _residual(toolbox: Any, sequences: Sequence[Any]) -> Any:
    def modify(*args: Any, **kwargs: Any) -> Any:
        return _position_residual(toolbox, *args, **kwargs)

    return toolbox.build_residual_fn(
        models_sequences=sequences, resample_true=False, modify_residual=modify, n_threads=1
    )


def _rmse(residual: Any, parameters: Any) -> tuple[float, ...]:
    values = parameters.copy()
    _, predicted, measured = residual(values.as_vector(), values, return_pred_all=True)
    errors = np.degrees(predicted[0][0].data - measured[0][0].data)
    return tuple(float(v) for v in np.sqrt(np.mean(errors**2, axis=0)))


def _diagnostics(
    toolbox: Any, initial: Any, identified: Any, optimizer_result: Any, residual: Any
) -> tuple[tuple[ParameterDiagnostic, ...], tuple[tuple[str, str, float], ...]]:
    values = identified.copy()
    errors, _, _ = residual(values.as_vector(), values)
    covariance, intervals = toolbox.calculate_intervals(errors, optimizer_result.jac)
    names = identified.get_non_frozen_parameter_names()
    widths = dict(zip(names, intervals, strict=True))
    result = []
    for name, parameter in identified.items():
        value = float(parameter.value[0])
        lower, upper = float(parameter.min_value[0]), float(parameter.max_value[0])
        width = float(widths.get(name, math.inf))
        at_bound = min(value - lower, upper - value) <= 0.001 * (upper - lower)
        result.append(
            ParameterDiagnostic(
                name=name,
                nominal=float(initial[name].nominal[0]),
                identified=value,
                bounds=(lower, upper),
                frozen=bool(parameter.frozen),
                ci95_half_width=width if math.isfinite(width) else None,
                at_bound=at_bound,
                poorly_identified=not parameter.frozen
                and (at_bound or not math.isfinite(width) or width > 0.5 * abs(value)),
            )
        )
    pairs = []
    if covariance.shape == (len(names), len(names)):
        sd = np.sqrt(np.clip(np.diag(covariance), 0, None))
        for i, j in zip(*np.triu_indices(len(names), k=1), strict=True):
            if sd[i] > 0 and sd[j] > 0:
                correlation = covariance[i, j] / (sd[i] * sd[j])
                if math.isfinite(correlation) and abs(correlation) >= 0.8:
                    pairs.append((names[i], names[j], float(correlation)))
    return tuple(result), tuple(pairs)


def fit_robot_profile(
    profile: RobotProfile,
    trajectories: Sequence[RecordedTrajectory],
    *,
    parameters: Sequence[str] = _DEFAULT_PARAMETERS,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    freeze: Sequence[str] = (),
    max_iters: int = 100,
    output_dir: str | Path | None = None,
    html_report: bool = False,
) -> FitResult:
    """Fit selected dynamics and evaluate independent validation episodes.

    Parameters
    ----------
    profile : RobotProfile
        Hardware calibration, model alignment, and nominal dynamics.
    trajectories : sequence of RecordedTrajectory
        Model-coordinate recordings containing training and validation episodes.
    parameters : sequence of str
        Selected ``joint.parameter`` names. Defaults to all actuator PD gains.
    bounds : mapping, optional
        Per-parameter lower and upper bounds in SI units.
    freeze : sequence of str
        Selected parameters held at their nominal values.
    max_iters : int
        Maximum optimizer iterations.
    output_dir : str or Path, optional
        Destination for ``candidate_profile.json`` and ``metrics.json``.
    html_report : bool
        Also generate the upstream HTML report; requires ``output_dir``.

    Returns
    -------
    FitResult
        Candidate dynamics, per-episode errors, and local fit diagnostics.
        ``validation_improved`` describes prediction error, not hardware safety.
    """
    bounds = {} if bounds is None else bounds
    _validate_parameters(parameters, bounds, freeze)
    if isinstance(max_iters, bool) or not isinstance(max_iters, int) or max_iters < 1:
        raise ValueError("max_iters must be a positive integer")
    if html_report and output_dir is None:
        raise ValueError("html_report requires output_dir")
    _validate_trajectories(profile, trajectories)
    toolbox = _toolbox()
    spec = _make_spec(profile)
    nominal = _nominal_dynamics(spec.compile())
    initial = _parameters(toolbox, nominal, parameters, bounds, freeze)
    sequences = [_model_sequences(toolbox, spec, episode) for episode in trajectories]
    train = [
        sequence
        for sequence, episode in zip(sequences, trajectories, strict=True)
        if episode.split == "train"
    ]
    residual = _residual(toolbox, train)
    # SI parameter sensitivities differ enough for raw-scale solver damping to stall gains.
    identified, optimizer_result = toolbox.optimize(
        initial_params=initial,
        residual_fn=residual,
        optimizer="mujoco",
        max_iters=max_iters,
        verbose=False,
        x_scale="jac",
    )
    candidate = dict(nominal)
    for name, parameter in identified.items():
        joint, attr = name.rsplit(".", 1)
        candidate[joint] = replace(candidate[joint], **{attr: float(parameter.value[0])})
    episodes = tuple(
        EpisodeFit(
            episode.name,
            episode.split,
            _rmse(_residual(toolbox, [sequence]), initial),
            _rmse(_residual(toolbox, [sequence]), identified),
        )
        for episode, sequence in zip(trajectories, sequences, strict=True)
    )
    diagnostics, correlations = _diagnostics(
        toolbox, initial, identified, optimizer_result, residual
    )
    validation = [episode for episode in episodes if episode.split == "validation"]
    improvement = all(
        np.mean(np.square(episode.identified_rmse_deg))
        < np.mean(np.square(episode.baseline_rmse_deg))
        for episode in validation
    )
    result = FitResult(
        replace(profile, dynamics=candidate), episodes, diagnostics, correlations, improvement
    )
    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        result.candidate_profile.save(directory / "candidate_profile.json")
        metrics = {
            "episodes": [asdict(e) for e in episodes],
            "diagnostics": [asdict(d) for d in diagnostics],
            "correlations": correlations,
            "validation_improved": improvement,
        }
        (directory / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n"
        )
        if html_report:
            toolbox.default_report(
                models_sequences=train,
                initial_params=initial,
                opt_params=identified,
                residual_fn=residual,
                opt_result=optimizer_result,
                save_path=directory,
                generate_videos=False,
                title="SO101-Nexus identification",
            )
    return result
