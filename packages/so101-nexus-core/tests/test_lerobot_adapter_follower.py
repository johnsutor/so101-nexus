"""Tests for the LeRobot simulator follower robot adapter."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from so101_nexus_core.config import SO101_JOINT_NAMES

if TYPE_CHECKING:
    from pathlib import Path

gym = pytest.importorskip("gymnasium")
lerobot_motors = pytest.importorskip("lerobot.motors")
MotorCalibration = lerobot_motors.MotorCalibration


def _calibration() -> dict[str, MotorCalibration]:
    return {
        name: MotorCalibration(
            id=i,
            drive_mode=0,
            homing_offset=0,
            range_min=1000,
            range_max=3000,
        )
        for i, name in enumerate(SO101_JOINT_NAMES, start=1)
    }


def _write_calibration(calibration_dir: Path, robot_id: str = "sim_test") -> None:
    import draccus

    calibration_dir.mkdir(parents=True, exist_ok=True)
    with open(calibration_dir / f"{robot_id}.json", "w") as f, draccus.config_type("json"):
        draccus.dump(_calibration(), f, indent=4)


class _AbsoluteJointEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}
    last_init_kwargs: dict[str, Any] = {}

    def __init__(
        self,
        render_mode: str | None = None,
        control_mode: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        type(self).last_init_kwargs = {
            "render_mode": render_mode,
            "control_mode": control_mode,
            **kwargs,
        }
        self.render_mode = render_mode
        self.control_mode = control_mode
        self.qpos = np.zeros(6, dtype=np.float64)
        self.step_calls = 0
        self.closed = False
        self._ctrl_low = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -0.2], dtype=np.float64)
        self._ctrl_high = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 1.2], dtype=np.float64)
        self.action_space = gym.spaces.Box(
            low=self._ctrl_low.astype(np.float32),
            high=self._ctrl_high.astype(np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64),
                "wrist_camera": gym.spaces.Box(low=0, high=255, shape=(6, 8, 3), dtype=np.uint8),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if options and "init_qpos" in options:
            self.qpos = np.asarray(options["init_qpos"], dtype=np.float64)
        else:
            self.qpos = np.zeros(6, dtype=np.float64)
        return self._get_obs(), {}

    def step(self, action):
        self.step_calls += 1
        self.qpos = np.asarray(action, dtype=np.float64)
        return self._get_obs(), 0.0, False, False, {}

    def _get_current_qpos(self) -> np.ndarray:
        return self.qpos.copy()

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {
            "state": self.qpos.copy(),
            "wrist_camera": np.full((6, 8, 3), 127, dtype=np.uint8),
        }

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_env_id() -> str:
    env_id = "SO101NexusAdapterFollowerFake-v0"
    gym.register(id=env_id, entry_point=_AbsoluteJointEnv)
    try:
        yield env_id
    finally:
        gym.envs.registration.registry.pop(env_id, None)


def _make_config(tmp_path: Path, env_id: str, **kwargs: Any):
    from so101_nexus_core.lerobot_adapter import SimCameraConfig, SimSOFollowerConfig

    _write_calibration(tmp_path)
    return SimSOFollowerConfig(
        id="sim_test",
        calibration_dir=tmp_path,
        env_id=env_id,
        cameras={"wrist": SimCameraConfig(source="wrist_camera", width=8, height=6, fps=30)},
        **kwargs,
    )


def test_make_robot_from_config_returns_sim_follower(tmp_path: Path, fake_env_id: str) -> None:
    from lerobot.robots import make_robot_from_config

    from so101_nexus_core.lerobot_adapter import SimSOFollower

    robot = make_robot_from_config(_make_config(tmp_path, fake_env_id))

    assert isinstance(robot, SimSOFollower)
    assert robot.calibration_dir == tmp_path


def test_features_include_motors_and_cameras(tmp_path: Path, fake_env_id: str) -> None:
    from so101_nexus_core.lerobot_adapter import SimSOFollower

    robot = SimSOFollower(_make_config(tmp_path, fake_env_id))

    assert robot.action_features == {f"{name}.pos": float for name in SO101_JOINT_NAMES}
    assert robot.observation_features["shoulder_pan.pos"] is float
    assert robot.observation_features["wrist"] == (6, 8, 3)


def test_connect_refuses_missing_calibration(tmp_path: Path, fake_env_id: str) -> None:
    from so101_nexus_core.lerobot_adapter import SimSOFollowerConfig
    from so101_nexus_core.lerobot_adapter.sim_follower import SimSOFollower

    robot = SimSOFollower(
        SimSOFollowerConfig(id="missing", calibration_dir=tmp_path, env_id=fake_env_id)
    )

    with pytest.raises(RuntimeError, match="calibration"):
        robot.connect()


def test_connect_builds_env_and_binds_cameras(tmp_path: Path, fake_env_id: str) -> None:
    from so101_nexus_core.lerobot_adapter import SimSOFollower

    robot = SimSOFollower(
        _make_config(tmp_path, fake_env_id, env_kwargs={"custom_option": "kept"})
    )
    try:
        robot.connect()

        assert robot.is_connected
        assert _AbsoluteJointEnv.last_init_kwargs["render_mode"] == "rgb_array"
        assert _AbsoluteJointEnv.last_init_kwargs["control_mode"] == "pd_joint_pos"
        assert _AbsoluteJointEnv.last_init_kwargs["custom_option"] == "kept"
        assert robot.cameras["wrist"].is_connected
    finally:
        robot.disconnect()


def test_get_observation_reads_normalized_qpos_and_camera(
    tmp_path: Path, fake_env_id: str
) -> None:
    from so101_nexus_core.lerobot_adapter import SimSOFollower

    robot = SimSOFollower(_make_config(tmp_path, fake_env_id))
    try:
        robot.connect()
        robot._env.unwrapped.qpos[0] = math.pi / 2

        obs = robot.get_observation()

        assert obs["shoulder_pan.pos"] == pytest.approx(89.9, abs=0.2)
        assert obs["gripper.pos"] == pytest.approx(14.3, abs=0.2)
        assert obs["wrist"].shape == (6, 8, 3)
    finally:
        robot.disconnect()


def test_send_action_steps_env_with_unnormalized_sim_qpos(
    tmp_path: Path, fake_env_id: str
) -> None:
    from so101_nexus_core.lerobot_adapter import SimSOFollower

    robot = SimSOFollower(_make_config(tmp_path, fake_env_id))
    try:
        robot.connect()
        action = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}
        action["shoulder_pan.pos"] = 90.0
        action["gripper.pos"] = 50.0

        sent = robot.send_action(action)

        env = robot._env.unwrapped
        assert env.step_calls == 1
        assert env.qpos[0] == pytest.approx(math.pi / 2, abs=0.004)
        assert env.qpos[-1] == pytest.approx(0.5, abs=0.002)
        assert sent["shoulder_pan.pos"] == pytest.approx(90.0, abs=0.2)
    finally:
        robot.disconnect()


def test_max_relative_target_clamps_from_current_sim_position(
    tmp_path: Path, fake_env_id: str
) -> None:
    from so101_nexus_core.lerobot_adapter import SimSOFollower

    robot = SimSOFollower(_make_config(tmp_path, fake_env_id, max_relative_target=10.0))
    try:
        robot.connect()
        action = {f"{name}.pos": 0.0 for name in SO101_JOINT_NAMES}
        action["shoulder_pan.pos"] = 90.0

        sent = robot.send_action(action)

        assert sent["shoulder_pan.pos"] == pytest.approx(10.0, abs=0.2)
        assert robot._env.unwrapped.qpos[0] == pytest.approx(math.radians(10), abs=0.004)
    finally:
        robot.disconnect()


def test_disconnect_closes_env_and_cameras(tmp_path: Path, fake_env_id: str) -> None:
    from so101_nexus_core.lerobot_adapter import SimSOFollower

    robot = SimSOFollower(_make_config(tmp_path, fake_env_id))
    robot.connect()
    env = robot._env.unwrapped

    robot.disconnect()

    assert env.closed
    assert not robot.is_connected
