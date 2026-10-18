"""LeRobot robot adapter for SO101-Nexus simulator follower environments."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from so101_nexus.config import (
    ABSOLUTE_CONTROL_MODES,
    DELTA_CONTROL_MODES,
    EE_CONTROL_MODES,
    SO101_JOINT_NAMES,
    SO101_TCP_FRAME_NAME,
)
from so101_nexus.lerobot_adapter.normalization import (
    GRIPPER_NAME,
    GripperLimitsRad,
    _control_bounds,
    action_for_env,
    build_so101_motors,
    leader_action_to_sim_qpos,
    motor_ticks_to_sim_rad,
    normalize_ticks,
    read_gripper_limits_rad,
    read_privileged_state,
    read_sim_qpos,
    sim_rad_to_motor_ticks,
    unnormalize_values,
)
from so101_nexus.lerobot_adapter.sim_camera import SimCamera
from so101_nexus.lerobot_adapter.sim_camera_config import SimCameraConfig
from so101_nexus.lerobot_adapter.sim_follower_config import SimSOFollowerConfig
from so101_nexus.teleop.leader import import_backend_for_env_id

if TYPE_CHECKING:
    from lerobot.model.kinematics import RobotKinematics
    from lerobot.processor import RobotAction, RobotObservation

logger = logging.getLogger(__name__)

# LeRobot's end-effector action features (see
# lerobot.robots.so_follower.robot_kinematic_processor). The position triple is in
# world metres and the orientation triple is an absolute world rotation vector.
# The gripper stays a normalized motor value, so it keeps the units of the
# joint-space path.
EE_POSE_KEYS = ("ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz")
EE_GRIPPER_KEY = "ee.gripper_pos"
EE_ACTION_KEYS = (*EE_POSE_KEYS, EE_GRIPPER_KEY)

DEFAULT_CONTROL_MODE = "pd_joint_pos"


def _array_to_list(value: Any) -> Any:
    """Convert NumPy, torch, or Warp array-like state to JSON-compatible lists."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    elif hasattr(value, "numpy") and callable(value.numpy):
        value = value.numpy()
    return np.asarray(value).tolist()


def _read_gripper_limits_rad(env: object, *, ee_control: bool) -> GripperLimitsRad:
    """Read simulator gripper limits for the active action layout."""
    if not ee_control:
        return read_gripper_limits_rad(env)
    bounds = _control_bounds(env)
    if bounds is None:
        raise TypeError("Simulator env does not expose gripper control limits.")
    low, high = bounds
    # The end-effector action is [x, y, z, wx, wy, wz, gripper], so the gripper
    # target is last rather than at its joint-space motor index.
    return float(low[-1]), float(high[-1])


def _clip_to_action_space(env: gym.Env, action: np.ndarray) -> np.ndarray:
    """Clip an action vector to the env action space when it exposes bounds."""
    space = getattr(env.unwrapped, "action_space", None)
    low = getattr(space, "low", None)
    high = getattr(space, "high", None)
    if low is None or high is None:
        return action
    return np.clip(action, low, high)


@dataclass(frozen=True)
class StepInfo:
    """Last termination metadata returned by a simulator ``env.step`` call."""

    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0


def _coerce_termination_flag(value: object) -> bool:
    """Coerce scalar or batched termination flags to a Python bool."""
    if hasattr(value, "detach") and callable(value.detach):
        tensor_like = cast("Any", value)
        value = tensor_like.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.shape == ():
        return bool(arr.item())
    return bool(arr.any())


class SimSOFollower(Robot):
    """LeRobot-compatible follower backed by a SO101-Nexus Gymnasium simulator."""

    config_class = SimSOFollowerConfig
    name = "sim_so_follower"

    def __init__(self, config: SimSOFollowerConfig) -> None:
        super().__init__(config)
        self.config = config
        self._control_mode = str(config.env_kwargs.get("control_mode", DEFAULT_CONTROL_MODE))
        if self._control_mode in DELTA_CONTROL_MODES:
            raise ValueError(
                f"SimSOFollower sends absolute targets, but env_kwargs requested "
                f"control_mode={self._control_mode!r}, whose action is a normalized "
                "[-1, 1] delta. LeRobot joint and end-effector action features are "
                f"absolute, so build the env with one of {list(ABSOLUTE_CONTROL_MODES)}."
            )
        self._is_ee_control = self._control_mode in EE_CONTROL_MODES
        self.motors = build_so101_motors(use_degrees=config.use_degrees)
        self.cameras: dict[str, SimCamera] = {}
        for name, camera_config in config.cameras.items():
            if not isinstance(camera_config, SimCameraConfig):
                raise TypeError(
                    f"SimSOFollower camera {name!r} must use SimCameraConfig, got "
                    f"{type(camera_config).__name__}."
                )
            self.cameras[name] = SimCamera(camera_config)

        self._env: gym.Env | None = None
        self._gripper_limits_rad: GripperLimitsRad | None = None
        self._last_step_info: StepInfo | None = None
        self._pending_leader_init_action: dict[str, Any] | None = None
        self._camera_frames: dict[str, np.ndarray] = {}

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

    @property
    def _ee_ft(self) -> dict[str, type]:
        return dict.fromkeys(EE_ACTION_KEYS, float)

    @property
    def _cameras_ft(self) -> dict[str, tuple[int | None, int | None, int]]:
        return {
            name: (self.config.cameras[name].height, self.config.cameras[name].width, 3)
            for name in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int | None, int | None, int]]:
        """Return LeRobot dataset features produced by simulator observations."""
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Return LeRobot action features accepted by the simulator follower.

        End-effector control modes take LeRobot's ``ee.*`` pose features in place of
        per-motor targets. The control mode is fixed by ``config.env_kwargs`` at
        construction, so the feature set never changes for a given follower.
        """
        return self._ee_ft if self._is_ee_control else self._motors_ft

    @property
    def is_connected(self) -> bool:
        """Return whether the simulator env and all configured cameras are connected."""
        return self._env is not None and all(
            camera.is_connected for camera in self.cameras.values()
        )

    @property
    def is_calibrated(self) -> bool:
        """Return whether all expected SO101 motor calibrations are loaded."""
        return set(self.calibration) == set(self.motors) and all(
            self.calibration[name].id == self.motors[name].id for name in self.motors
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Create the simulator env and bind configured simulator cameras."""
        if not self.is_calibrated:
            raise RuntimeError(
                f"Missing or invalid calibration file for {self}: {self.calibration_fpath}"
            )

        import_backend_for_env_id(self.config.env_id)
        make_kwargs: dict[str, Any] = {
            "render_mode": "rgb_array",
            "control_mode": DEFAULT_CONTROL_MODE,
        }
        make_kwargs.update(self.config.env_kwargs)

        try:
            self._env = gym.make(self.config.env_id, **make_kwargs)
            self._env.reset(seed=self.config.seed)
            self._gripper_limits_rad = _read_gripper_limits_rad(
                self._env, ee_control=self._is_ee_control
            )
            if self._pending_leader_init_action is not None:
                self._reset_env(seed=self.config.seed)
            self._last_step_info = None

            for camera in self.cameras.values():
                camera.bind_env(self._env)
                camera.connect(warmup=False)

            self.configure()
            logger.info("%s connected.", self)
        except Exception:
            self.disconnect()
            raise

    def calibrate(self) -> None:
        """Raise because simulator followers require an explicit calibration file."""
        raise RuntimeError(
            "SimSOFollower uses an existing LeRobot SO101 calibration file. "
            f"Create or copy one to {self.calibration_fpath} before connecting."
        )

    def configure(self) -> None:
        """No-op hook for LeRobot's robot interface."""

    def setup_motors(self) -> None:
        """Skip physical motor setup for the simulator adapter."""

    def set_initial_leader_action(self, action: dict[str, Any] | None) -> None:
        """Set a leader action to seed ``env.reset(options={'init_qpos': ...})``."""
        self._pending_leader_init_action = None if action is None else dict(action)

    @check_if_not_connected
    def get_rest_action(self, current_action: dict[str, float]) -> dict[str, float]:
        """Solve the configured resting tool pose in leader motor coordinates.

        Parameters
        ----------
        current_action
            Measured leader body degrees and gripper percent, without a wrist offset.
        """
        env = self._env
        if env is None:
            raise RuntimeError("follower environment is not connected")
        robot_config = getattr(getattr(env.unwrapped, "config", None), "robot", None)
        bounds = _control_bounds(env)
        if robot_config is None or bounds is None:
            raise ValueError(
                "Policy episode return requires a configured resting posture and joint bounds"
            )
        limits = self._require_gripper_limits()
        current_qpos_rad = leader_action_to_sim_qpos(
            current_action,
            motors=self.motors,
            calibration=self.calibration,
            gripper_limits_rad=limits,
        )
        rest_deg = np.array(robot_config.rest_qpos_deg, dtype=np.float64, copy=True)
        low_deg, high_deg = (np.degrees(bound) for bound in bounds)
        if rest_deg.shape != (6,) or not np.isfinite(rest_deg).all():
            raise ValueError("Resting pose requires six finite joint angles in degrees")
        rest_deg[-1] = np.clip(rest_deg[-1], low_deg[-1], high_deg[-1])
        target_deg = self._solve_rest_joints_deg(
            np.degrees(current_qpos_rad), rest_deg, low_deg, high_deg
        )
        target_qpos_rad = np.radians(target_deg)
        ticks = sim_rad_to_motor_ticks(
            target_qpos_rad, calibration=self.calibration, gripper_limits_rad=limits
        )
        values = normalize_ticks(ticks, motors=self.motors, calibration=self.calibration)
        return {f"{name}.pos": value for name, value in values.items()}

    @cached_property
    def _rest_kinematics(self) -> RobotKinematics:
        from lerobot.model.kinematics import RobotKinematics

        from so101_nexus import get_so101_urdf_path

        return RobotKinematics(
            str(get_so101_urdf_path()),
            target_frame_name=SO101_TCP_FRAME_NAME,
            joint_names=list(SO101_JOINT_NAMES[:-1]),
        )

    def _solve_rest_joints_deg(
        self,
        current_deg: np.ndarray,
        rest_deg: np.ndarray,
        low_deg: np.ndarray,
        high_deg: np.ndarray,
    ) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        current_deg = np.asarray(current_deg, dtype=np.float64)
        if current_deg.shape != (6,) or not np.isfinite(current_deg).all():
            raise ValueError("Rest IK requires six finite current joint angles in degrees")
        if np.any(rest_deg < low_deg) or np.any(rest_deg > high_deg):
            raise ValueError("Configured resting posture is outside joint bounds")
        kinematics = self._rest_kinematics
        desired = kinematics.forward_kinematics(rest_deg).copy()
        # The configured posture selects the resting branch after a local minimum.
        for seed in (current_deg, rest_deg):
            # LeRobot's first solver iteration uses cached forward kinematics.
            kinematics.forward_kinematics(seed)
            target_deg = kinematics.inverse_kinematics(
                seed, desired, orientation_weight=1.0, max_iters=100
            )
            target_deg[-1] = rest_deg[-1]
            if (
                not np.isfinite(target_deg).all()
                or np.any(target_deg < low_deg)
                or np.any(target_deg > high_deg)
            ):
                continue
            actual = kinematics.forward_kinematics(target_deg)
            angle_rad = Rotation.from_matrix(actual[:3, :3] @ desired[:3, :3].T).magnitude()
            if np.linalg.norm(actual[:3, 3] - desired[:3, 3]) <= 0.001 and angle_rad <= 0.01:
                return target_deg
        raise ValueError(
            "Resting pose IK did not converge within joint bounds; leader return canceled"
        )

    @check_if_not_connected
    def reset(self, seed: int | None = None) -> None:
        """Reset the connected simulator, applying any pending leader pose.

        Parameters
        ----------
        seed : int or None
            Episode seed, or None to continue the simulator's seeded RNG stream.
        """
        self._reset_env(seed=seed)

    def _reset_env(self, *, seed: int | None) -> None:
        if self._env is None:
            raise RuntimeError("SimSOFollower is not connected to an environment")
        options = None
        if self._pending_leader_init_action is not None:
            options = {
                "init_qpos": leader_action_to_sim_qpos(
                    self._pending_leader_init_action,
                    motors=self.motors,
                    calibration=self.calibration,
                    gripper_limits_rad=self._require_gripper_limits(),
                )
            }
        self._camera_frames.clear()
        self._last_step_info = None
        self._env.reset(seed=seed, options=options)
        self._pending_leader_init_action = None

    def _cache_camera_frames(self, obs: object) -> None:
        self._camera_frames = {
            name: camera._frame_from_observation(obs).copy()
            for name, camera in self.cameras.items()
            if camera.config.source != "render" and camera._extract_from_obs(obs) is not None
        }

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """Read simulator qpos and camera frames in LeRobot observation format.

        The first read after step consumes its camera snapshot. Other reads
        render fresh frames, including after reset when renderer state can still
        settle. State channels always read the current physics state. Advance the
        privately owned simulator through this adapter so its camera snapshots
        stay aligned with the recording state.
        """
        if self._env is None:
            raise RuntimeError("SimSOFollower is not connected to an environment")
        gripper_limits_rad = self._require_gripper_limits()

        start = time.perf_counter()
        qpos_rad = read_sim_qpos(self._env)
        ticks = sim_rad_to_motor_ticks(
            qpos_rad,
            calibration=self.calibration,
            gripper_limits_rad=gripper_limits_rad,
        )
        motor_values = normalize_ticks(ticks, motors=self.motors, calibration=self.calibration)
        obs_dict: RobotObservation = {
            f"{motor}.pos": value for motor, value in motor_values.items()
        }
        logger.debug("%s read sim state: %.1fms", self, (time.perf_counter() - start) * 1e3)

        privileged_state = read_privileged_state(self._env)
        if privileged_state is not None:
            obs_dict["environment_state"] = privileged_state

        camera_frames, self._camera_frames = self._camera_frames, {}
        get_obs = getattr(self._env.unwrapped, "_get_obs", None)
        camera_obs = (
            get_obs()
            if callable(get_obs)
            and any(
                camera.config.source != "render" and name not in camera_frames
                for name, camera in self.cameras.items()
            )
            else None
        )
        for camera_name, camera in self.cameras.items():
            start = time.perf_counter()
            obs_dict[camera_name] = (
                camera_frames[camera_name]
                if camera_name in camera_frames
                else (
                    camera._frame_from_observation(camera_obs)
                    if camera_obs is not None and camera.config.source != "render"
                    else camera.read_latest()
                ).copy()
            )
            logger.debug(
                "%s read %s: %.1fms",
                self,
                camera_name,
                (time.perf_counter() - start) * 1e3,
            )

        return obs_dict

    def last_step_info(self) -> StepInfo | None:
        """Return metadata captured by the most recent ``send_action`` call."""
        return self._last_step_info

    def initial_state_provenance(self) -> dict[str, Any]:
        """Return native simulator coordinates that identify the connected reset.

        Hinge coordinates use radians, translations use meters, and camera FOV
        uses degrees. These snapshot fields preserve the backend's replay format.
        """
        if self._env is None:
            raise RuntimeError("SimSOFollower is not connected to an environment")
        unwrapped = self._env.unwrapped
        data = getattr(unwrapped, "data", None)
        model = getattr(unwrapped, "model", None)
        if model is None:
            model = getattr(unwrapped, "mjm", None)
        fields: dict[str, Any] = {}
        for owner, names in (
            (data, ("time", "qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat")),
            (
                model,
                (
                    "body_pos",
                    "site_pos",
                    "cam_pos",
                    "cam_quat",
                    "cam_fovy",
                    "geom_rgba",
                    "mat_rgba",
                ),
            ),
        ):
            if owner is None:
                continue
            for name in names:
                value = getattr(owner, name, None)
                if value is not None:
                    fields[name] = _array_to_list(value)
        return fields

    def rendering_metadata(self) -> dict[str, Any]:
        """Return rendering provenance for the currently allocated camera contexts."""
        if self._env is None:
            raise RuntimeError("SimSOFollower is not connected to an environment")
        return getattr(self._env.unwrapped, "rendering_metadata", {})

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send a normalized LeRobot joint or end-effector target to the simulator."""
        env = self._env
        if env is None:
            raise RuntimeError("SimSOFollower is not connected to an environment")
        gripper_limits_rad = self._require_gripper_limits()
        if self._is_ee_control:
            return self._send_ee_action(env, action, gripper_limits_rad)

        goal_pos = {
            key.removesuffix(".pos"): float(value)
            for key, value in action.items()
            if key.endswith(".pos")
        }
        unknown_motors = set(goal_pos) - set(self.motors)
        if unknown_motors:
            raise KeyError(f"Unknown SO101 motor action keys: {sorted(unknown_motors)}")

        if self.config.max_relative_target is not None:
            present_pos = self._read_present_motor_values(gripper_limits_rad)
            goal_present_pos = {
                motor: (goal, present_pos[motor]) for motor, goal in goal_pos.items()
            }
            goal_pos = ensure_safe_goal_position(
                goal_present_pos,
                self.config.max_relative_target,
            )

        ticks = unnormalize_values(goal_pos, motors=self.motors, calibration=self.calibration)
        target_qpos = motor_ticks_to_sim_rad(
            ticks,
            calibration=self.calibration,
            gripper_limits_rad=gripper_limits_rad,
        )
        sent_qpos = action_for_env(env, target_qpos)
        self._step_env(env, sent_qpos)

        sent_ticks = sim_rad_to_motor_ticks(
            sent_qpos,
            calibration=self.calibration,
            gripper_limits_rad=gripper_limits_rad,
        )
        sent_values = normalize_ticks(
            {motor: sent_ticks[motor] for motor in goal_pos},
            motors=self.motors,
            calibration=self.calibration,
        )
        return {f"{motor}.pos": value for motor, value in sent_values.items()}

    def _step_env(self, env: gym.Env, action: np.ndarray) -> None:
        self._camera_frames.clear()
        obs, reward, terminated, truncated, info = env.step(action)
        self._cache_camera_frames(obs)
        self._last_step_info = StepInfo(
            terminated=_coerce_termination_flag(terminated),
            truncated=_coerce_termination_flag(truncated),
            info=dict(info) if isinstance(info, dict) else {},
            reward=float(reward),
        )

    def _send_ee_action(
        self,
        env: gym.Env,
        action: RobotAction,
        gripper_limits_rad: GripperLimitsRad,
    ) -> RobotAction:
        values = {key: float(value) for key, value in action.items() if key.startswith("ee.")}
        unexpected = sorted(set(values) - set(EE_ACTION_KEYS))
        missing = [key for key in EE_ACTION_KEYS if key not in values]
        if missing or unexpected:
            raise KeyError(
                "End-effector actions must provide exactly "
                f"{list(EE_ACTION_KEYS)}: missing {missing}, unexpected {unexpected}."
            )

        gripper_value = values[EE_GRIPPER_KEY]
        if self.config.max_relative_target is not None:
            # Only the gripper is a motor-space target. The pose components are metres
            # and radians, so the env action space bounds them instead.
            cap = self.config.max_relative_target
            present = self._read_present_motor_values(gripper_limits_rad)[GRIPPER_NAME]
            gripper_value = ensure_safe_goal_position(
                {GRIPPER_NAME: (gripper_value, present)},
                cap[GRIPPER_NAME] if isinstance(cap, dict) else cap,
            )[GRIPPER_NAME]

        ee_action = np.array(
            [
                *(values[key] for key in EE_POSE_KEYS),
                self._gripper_value_to_sim_rad(env, gripper_value, gripper_limits_rad),
            ],
            dtype=np.float64,
        )
        sent_action = _clip_to_action_space(env, ee_action)
        self._step_env(env, sent_action)

        sent: RobotAction = dict(
            zip(EE_POSE_KEYS, sent_action[:-1].tolist(), strict=True),
        )
        sent[EE_GRIPPER_KEY] = gripper_value
        return sent

    def _gripper_value_to_sim_rad(
        self,
        env: gym.Env,
        value: float,
        gripper_limits_rad: GripperLimitsRad,
    ) -> float:
        """Convert a normalized LeRobot gripper value to simulator radians."""
        ticks = sim_rad_to_motor_ticks(
            read_sim_qpos(env),
            calibration=self.calibration,
            gripper_limits_rad=gripper_limits_rad,
        )
        ticks[GRIPPER_NAME] = unnormalize_values(
            {GRIPPER_NAME: value},
            motors=self.motors,
            calibration=self.calibration,
        )[GRIPPER_NAME]
        return float(
            motor_ticks_to_sim_rad(
                ticks,
                calibration=self.calibration,
                gripper_limits_rad=gripper_limits_rad,
            )[-1]
        )

    def disconnect(self) -> None:
        """Close the simulator env and disconnect simulator cameras."""
        env = self._env
        for camera in self.cameras.values():
            camera.disconnect()
        self._env = None
        self._gripper_limits_rad = None
        self._last_step_info = None
        self._pending_leader_init_action = None
        self._camera_frames.clear()
        if env is not None:
            env.close()
        logger.info("%s disconnected.", self)

    def _require_gripper_limits(self) -> GripperLimitsRad:
        if self._gripper_limits_rad is None:
            raise RuntimeError("SimSOFollower is missing simulator gripper limits.")
        return self._gripper_limits_rad

    def _read_present_motor_values(
        self,
        gripper_limits_rad: GripperLimitsRad,
    ) -> dict[str, float]:
        if self._env is None:
            raise RuntimeError("SimSOFollower is not connected to an environment")
        qpos_rad = read_sim_qpos(self._env)
        ticks = sim_rad_to_motor_ticks(
            qpos_rad,
            calibration=self.calibration,
            gripper_limits_rad=gripper_limits_rad,
        )
        return normalize_ticks(ticks, motors=self.motors, calibration=self.calibration)
