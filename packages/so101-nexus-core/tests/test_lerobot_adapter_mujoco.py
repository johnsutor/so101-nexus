"""MuJoCo smoke tests for the LeRobot simulator follower adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import draccus
import pytest

from so101_nexus_core.config import SO101_JOINT_NAMES

if TYPE_CHECKING:
    from pathlib import Path

pytest.importorskip("gymnasium")
lerobot_motors = pytest.importorskip("lerobot.motors")
pytest.importorskip("mujoco")
pytest.importorskip("so101_nexus_mujoco")
MotorCalibration = lerobot_motors.MotorCalibration


def _write_calibration(calibration_dir: Path, robot_id: str = "sim_test") -> None:
    calibration = {
        name: MotorCalibration(
            id=i,
            drive_mode=0,
            homing_offset=0,
            range_min=1000,
            range_max=3000,
        )
        for i, name in enumerate(SO101_JOINT_NAMES, start=1)
    }
    calibration_dir.mkdir(parents=True, exist_ok=True)
    with open(calibration_dir / f"{robot_id}.json", "w") as f, draccus.config_type("json"):
        draccus.dump(calibration, f, indent=4)


def _neutral_action() -> dict[str, float]:
    return {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}


def test_sim_follower_connects_to_mujoco_reach(tmp_path: Path) -> None:
    from so101_nexus_core.lerobot_adapter import SimSOFollower, SimSOFollowerConfig

    _write_calibration(tmp_path)
    robot = SimSOFollower(
        SimSOFollowerConfig(
            id="sim_test",
            calibration_dir=tmp_path,
            env_id="MuJoCoReach-v1",
            env_kwargs={"robot_init_qpos_noise": 0.0},
        )
    )
    try:
        robot.connect()
        obs = robot.get_observation()
        sent = robot.send_action(_neutral_action())
        obs_after = robot.get_observation()

        assert set(sent) == {f"{name}.pos" for name in SO101_JOINT_NAMES}
        assert all(isinstance(obs[f"{name}.pos"], float) for name in SO101_JOINT_NAMES)
        assert all(isinstance(obs_after[f"{name}.pos"], float) for name in SO101_JOINT_NAMES)
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_mujoco_env_clipping_is_reflected_in_returned_action(tmp_path: Path) -> None:
    from so101_nexus_core.lerobot_adapter import SimSOFollower, SimSOFollowerConfig

    _write_calibration(tmp_path)
    robot = SimSOFollower(
        SimSOFollowerConfig(
            id="sim_test",
            calibration_dir=tmp_path,
            env_id="MuJoCoReach-v1",
            env_kwargs={"robot_init_qpos_noise": 0.0},
        )
    )
    try:
        robot.connect()
        edge_action = _neutral_action()
        edge_action["shoulder_pan.pos"] = 180.0

        sent = robot.send_action(edge_action)
        env = robot._env.unwrapped
        first_ctrl = env.data.ctrl[env._actuator_ids].copy()
        sent_again = robot.send_action(sent)
        second_ctrl = env.data.ctrl[env._actuator_ids].copy()

        assert sent["shoulder_pan.pos"] < edge_action["shoulder_pan.pos"]
        assert sent_again == pytest.approx(sent, abs=0.2)
        assert second_ctrl == pytest.approx(first_ctrl, abs=0.002)
    finally:
        if robot.is_connected:
            robot.disconnect()
