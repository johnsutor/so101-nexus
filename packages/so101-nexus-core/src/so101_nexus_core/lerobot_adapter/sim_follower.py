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

from so101_nexus_core.lerobot_adapter.normalization import (
    GripperLimitsRad,
    action_for_env,
    build_so101_motors,
    leader_action_to_sim_qpos,
    motor_ticks_to_sim_rad,
    normalize_ticks,
    read_gripper_limits_rad,
    read_sim_qpos,
    sim_rad_to_motor_ticks,
    unnormalize_values,
)
from so101_nexus_core.lerobot_adapter.sim_camera import SimCamera
from so101_nexus_core.lerobot_adapter.sim_camera_config import SimCameraConfig
from so101_nexus_core.lerobot_adapter.sim_follower_config import SimSOFollowerConfig
from so101_nexus_core.teleop.leader import import_backend_for_env_id

if TYPE_CHECKING:
    from lerobot.processor import RobotAction, RobotObservation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepInfo:
    """Last termination metadata returned by a simulator ``env.step`` call."""

    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


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

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.motors}

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
        """Return LeRobot action features accepted by the simulator follower."""
        return self._motors_ft

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
            "control_mode": "pd_joint_pos",
        }
        make_kwargs.update(self.config.env_kwargs)

        try:
            self._env = gym.make(self.config.env_id, **make_kwargs)
            self._env.reset()
            self._gripper_limits_rad = read_gripper_limits_rad(self._env)
            if self._pending_leader_init_action is not None:
                # The first reset exposes gripper limits; the second reset
                # lets the env settle with the arm held at the leader pose.
                init_qpos = leader_action_to_sim_qpos(
                    self._pending_leader_init_action,
                    motors=self.motors,
                    calibration=self.calibration,
                    gripper_limits_rad=self._gripper_limits_rad,
                )
                self._env.reset(options={"init_qpos": init_qpos})
                self._pending_leader_init_action = None
            self._last_step_info = None

            for camera in self.cameras.values():
                camera.bind_env(self._env)
                camera.connect()

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
    def get_observation(self) -> RobotObservation:
        """Read simulator qpos and camera frames in LeRobot observation format."""
        assert self._env is not None
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

        for camera_name, camera in self.cameras.items():
            start = time.perf_counter()
            obs_dict[camera_name] = camera.read_latest()
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

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Send a normalized LeRobot joint target to the simulator."""
        assert self._env is not None
        gripper_limits_rad = self._require_gripper_limits()

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
        sent_qpos = action_for_env(self._env, target_qpos)
        _obs, _reward, terminated, truncated, info = self._env.step(sent_qpos)
        self._last_step_info = StepInfo(
            terminated=_coerce_termination_flag(terminated),
            truncated=_coerce_termination_flag(truncated),
            info=dict(info) if isinstance(info, dict) else {},
        )

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

    def disconnect(self) -> None:
        """Close the simulator env and disconnect simulator cameras."""
        env = self._env
        for camera in self.cameras.values():
            camera.disconnect()
        self._env = None
        self._gripper_limits_rad = None
        self._last_step_info = None
        self._pending_leader_init_action = None
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
        assert self._env is not None
        qpos_rad = read_sim_qpos(self._env)
        ticks = sim_rad_to_motor_ticks(
            qpos_rad,
            calibration=self.calibration,
            gripper_limits_rad=gripper_limits_rad,
        )
        return normalize_ticks(ticks, motors=self.motors, calibration=self.calibration)
