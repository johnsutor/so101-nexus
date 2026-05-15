"""LeRobot robot adapter for SO101-Nexus simulator follower environments."""

from __future__ import annotations

import logging
import time
from functools import cached_property
from typing import Any

import gymnasium as gym
import numpy as np
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.robots.utils import ensure_safe_goal_position
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from so101_nexus_core.lerobot_adapter.normalization import (
    MOTOR_NAMES,
    GripperLimitsRad,
    action_for_env,
    build_so101_motors,
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

logger = logging.getLogger(__name__)


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
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self._env is not None and all(camera.is_connected for camera in self.cameras.values())

    @property
    def is_calibrated(self) -> bool:
        return set(self.calibration) == set(self.motors) and all(
            self.calibration[name].id == self.motors[name].id for name in self.motors
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        if not self.is_calibrated:
            if calibrate:
                self.calibrate()
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

            for camera in self.cameras.values():
                camera.bind_env(self._env)
                camera.connect()

            self.configure()
            logger.info("%s connected.", self)
        except Exception:
            self.disconnect()
            raise

    def calibrate(self) -> None:
        raise RuntimeError(
            "SimSOFollower uses an existing LeRobot SO101 calibration file. "
            f"Create or copy one to {self.calibration_fpath} before connecting."
        )

    def configure(self) -> None:
        """No-op hook for LeRobot's robot interface."""

    def setup_motors(self) -> None:
        """Simulator adapter has no physical motors to configure."""

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
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

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
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
        self._env.step(sent_qpos)

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

    @check_if_not_connected
    def disconnect(self) -> None:
        env = self._env
        for camera in self.cameras.values():
            camera.disconnect()
        self._env = None
        self._gripper_limits_rad = None
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
