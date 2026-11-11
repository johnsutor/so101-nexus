"""Repeatable excitation and timestamped collection fail before unsafe commands."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.excitation import (
    ExcitationConfig,
    ExcitationTrajectory,
    RecordedTrajectory,
    collect_excitation,
    generate_excitation,
    preflight_excitation,
    profile_digest,
)
from so101_nexus.profiles import RobotProfile


@pytest.fixture
def profile():
    return RobotProfile(
        robot_id="test_arm",
        calibration={
            name: {
                "id": i,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095,
            }
            for i, name in enumerate(SO101_JOINT_NAMES, 1)
        },
    )


@pytest.fixture
def config():
    return ExcitationConfig(center_qpos_deg=(0, -15, 30, 45, 0, 50), duration_s=1, ramp_s=0.2)


def test_excitation_is_repeatable_bounded_and_includes_heldout(profile, config):
    trajectories = generate_excitation(config, profile)
    repeated = generate_excitation(config, profile)
    assert [t.split for t in trajectories] == ["train", "train", "validation"]
    for trajectory, other in zip(trajectories, repeated, strict=True):
        np.testing.assert_array_equal(trajectory.commands_deg, other.commands_deg)
        np.testing.assert_allclose(trajectory.commands_deg[[0, -1]], [config.center_qpos_deg] * 2)
        velocity = np.diff(trajectory.commands_deg, axis=0) / profile.control_period_s
        acceleration = np.diff(velocity, axis=0) / profile.control_period_s
        assert np.max(np.abs(velocity)) <= config.max_velocity_deg_s + 1e-9
        assert np.max(np.abs(acceleration)) <= config.max_acceleration_deg_s2 + 1e-9
        assert np.all(np.ptp(trajectory.commands_deg, axis=0) > 0)
    assert not np.array_equal(trajectories[0].commands_deg, trajectories[-1].commands_deg)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration_s": 0},
        {"ramp_s": -1},
        {"max_velocity_deg_s": float("nan")},
        {"center_qpos_deg": (0,) * 5},
        {"amplitudes_deg": (0,) * 6},
        {"seed": True},
    ],
)
def test_invalid_excitation_rejected(kwargs):
    with pytest.raises(ValueError, match="must"):
        ExcitationConfig(**{"center_qpos_deg": (0,) * 6, **kwargs})


def test_recording_roundtrip_and_profile_identity(tmp_path, profile):
    times = np.arange(4) * 0.02
    recording = RecordedTrajectory(
        name="heldout",
        split="validation",
        measurement_timestamps_s=times,
        command_timestamps_s=times + 0.001,
        measured_qpos_deg=np.zeros((4, 6)),
        commanded_qpos_deg=np.ones((4, 6)),
        profile_sha256=profile_digest(profile),
    )
    path = tmp_path / "episode.npz"
    recording.save(path)
    loaded = RecordedTrajectory.load(path)
    np.testing.assert_array_equal(loaded.measured_qpos_deg, recording.measured_qpos_deg)
    loaded.validate_profile(profile)
    other = replace(profile, robot_id="other_arm")
    with pytest.raises(ValueError, match="profile"):
        loaded.validate_profile(other)


def test_recording_rejects_nonmonotonic_time(profile):
    with pytest.raises(ValueError, match="increasing"):
        RecordedTrajectory(
            "invalid",
            "train",
            np.array([0, 0]),
            np.array([0, 0.02]),
            np.zeros((2, 6)),
            np.zeros((2, 6)),
            profile_digest(profile),
        )


def test_preflight_rejects_joint_limit_before_hardware(profile, config):
    unsafe = ExcitationConfig(center_qpos_deg=(180, -15, 30, 45, 0, 50), duration_s=1, ramp_s=0.2)
    with pytest.raises(ValueError, match="limit"):
        preflight_excitation(generate_excitation(unsafe, profile), profile, unsafe)


def test_preflight_checks_both_valid_paths(profile, config):
    preflight_excitation(generate_excitation(config, profile), profile, config)


def test_preflight_checks_geometry_after_final_integration(monkeypatch, profile, config):
    import mujoco

    trajectory = generate_excitation(config, profile)[0]
    final_step = len(trajectory.timestamps_s) * 8
    calls = 0
    original = mujoco.mj_step

    def step(model, data):
        nonlocal calls
        original(model, data)
        calls += 1
        if calls == final_step:
            data.qpos[:6] = np.radians([0, -30, 90, 45, 0, 50])

    monkeypatch.setattr(mujoco, "mj_step", step)
    with pytest.raises(ValueError, match="clearance"):
        preflight_excitation([trajectory], profile, config)


def test_preflight_rejects_physical_limits_before_connect(profile, config):
    calibration = {name: dict(motor) for name, motor in profile.calibration.items()}
    calibration["shoulder_lift"].update(range_min=2000, range_max=2100)
    limited = replace(profile, calibration=calibration)
    robot = FakeRobot(limited, config.center_qpos_deg)
    with pytest.raises(ValueError, match="physical calibration limit"):
        collect_excitation(robot, limited, config)
    assert not robot.connected
    assert not robot.commands


def test_preflight_rejects_unbounded_custom_commands(profile, config):
    commands = np.tile(config.center_qpos_deg, (3, 1)).astype(float)
    commands[1, 0] += 5
    trajectory = ExcitationTrajectory("fast", "train", np.array([0, 0.02, 0.04]), commands)
    with pytest.raises(ValueError, match="velocity or acceleration"):
        preflight_excitation([trajectory], profile, config)


def test_preflight_rejects_tool_clearance(profile, config):
    high_table = replace(config, min_clearance_m=1)
    with pytest.raises(ValueError, match="clearance"):
        preflight_excitation(generate_excitation(config, profile), profile, high_table)


def test_preflight_ignores_scene_contacts_and_preserves_supplied_model(profile, config):
    import mujoco

    from so101_nexus import get_so101_mujoco_model_path

    spec = mujoco.MjSpec.from_file(str(get_so101_mujoco_model_path()))
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01])
    box = spec.worldbody.add_body(pos=[1, 1, 0.005])
    box.add_freejoint()
    box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01])
    model = spec.compile()
    model.opt.timestep = 0.005
    preflight_excitation(generate_excitation(config, profile), profile, config, model=model)
    assert model.opt.timestep == 0.005


def test_recording_schema_rejects_bool(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, metadata=json.dumps({"schema_version": True}))
    with pytest.raises(ValueError, match="integer schema_version"):
        RecordedTrajectory.load(path)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        self.now += 0.0001
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeRobot:
    def __init__(self, profile, center):
        self.config = SimpleNamespace(
            use_degrees=True,
            id=profile.robot_id,
            position_p_coefficient=16,
            position_i_coefficient=0,
            position_d_coefficient=32,
            cameras={},
        )
        self.calibration = profile.calibration
        self.position = np.asarray(profile.from_model_deg(np.array(center)))
        self.connected = False
        self.commands = []
        self.fail_read = False
        self.is_calibrated = True

    @property
    def is_connected(self):
        return self.connected

    def connect(self, calibrate=True):
        assert calibrate is False
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        if self.fail_read and self.commands:
            raise RuntimeError("bus failure")
        return {
            f"{name}.pos": value
            for name, value in zip(SO101_JOINT_NAMES, self.position, strict=True)
        }

    def send_action(self, action):
        self.commands.append(action)
        self.position = np.array([action[f"{name}.pos"] for name in SO101_JOINT_NAMES])
        return action


def test_collector_preflights_everything_before_connect(monkeypatch, profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)

    def reject(*args):
        assert not robot.connected
        raise ValueError("preflight failed")

    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", reject)
    with pytest.raises(ValueError, match="preflight"):
        collect_excitation(robot, profile, config)
    assert not robot.commands


def test_collector_records_model_degrees_and_actual_timestamps(monkeypatch, profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)
    clock = FakeClock()
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    recordings = collect_excitation(robot, profile, config, clock=clock, sleep=clock.sleep)
    assert len(recordings) == 3
    assert not robot.connected
    for recording in recordings:
        recording.validate_profile(profile)
        assert np.all(np.diff(recording.measurement_timestamps_s) > 0)
        assert np.all(recording.command_timestamps_s > recording.read_end_timestamps_s)
        np.testing.assert_allclose(recording.measured_qpos_deg[0], config.center_qpos_deg)


def test_collector_disconnects_and_stops_after_bus_failure(monkeypatch, profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)
    robot.fail_read = True
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="bus failure"):
        collect_excitation(robot, profile, config, clock=clock, sleep=clock.sleep)
    assert not robot.connected
    assert len(robot.commands) == 1


@pytest.mark.parametrize("field,value", [("use_degrees", False), ("id", "other")])
def test_collector_rejects_robot_coordinate_mismatch(profile, config, field, value):
    robot = FakeRobot(profile, config.center_qpos_deg)
    setattr(robot.config, field, value)
    with pytest.raises(ValueError, match=r"degrees|robot_id"):
        collect_excitation(robot, profile, config)
    assert not robot.commands


def test_collector_cleans_up_failed_connect(monkeypatch, profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)

    def failed_connect(**kwargs):
        robot.connected = True
        raise RuntimeError("connect failed")

    monkeypatch.setattr(robot, "connect", failed_connect)
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    with pytest.raises(RuntimeError, match="connect failed"):
        collect_excitation(robot, profile, config)
    assert not robot.connected


def test_collector_checks_hardware_calibration_before_first_command(monkeypatch, profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)
    robot.is_calibrated = False
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    with pytest.raises(RuntimeError, match="hardware calibration"):
        collect_excitation(robot, profile, config)
    assert not robot.connected
    assert not robot.commands


def test_collector_rejects_servo_gain_mismatch_before_connect(profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)
    robot.config.position_p_coefficient = 32
    with pytest.raises(ValueError, match="servo gains"):
        collect_excitation(robot, profile, config)
    assert not robot.connected
    assert not robot.commands


@pytest.mark.parametrize(
    "fault", ["start", "tracking", "read_duration", "write_duration", "clipping"]
)
def test_collector_aborts_without_future_commands(monkeypatch, profile, config, fault):
    robot = FakeRobot(profile, config.center_qpos_deg)
    clock = FakeClock()
    if fault == "start":
        robot.position[0] += 20
    original_read, original_send = robot.get_observation, robot.send_action

    def read():
        if fault == "read_duration":
            clock.now += 0.2
        result = original_read()
        if fault == "tracking" and robot.commands:
            result["shoulder_pan.pos"] += 20
        return result

    def send(action):
        result = original_send(action)
        if fault == "write_duration":
            clock.now += 0.2
        if fault == "clipping":
            result = {**result, "shoulder_pan.pos": result["shoulder_pan.pos"] + 20}
        return result

    monkeypatch.setattr(robot, "get_observation", read)
    monkeypatch.setattr(robot, "send_action", send)
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    with pytest.raises(RuntimeError, match=r"tracking|duration|overrun|differs"):
        collect_excitation(robot, profile, config, clock=clock, sleep=clock.sleep)
    assert not robot.connected
    assert len(robot.commands) <= 1


def test_collector_persists_completed_episode_before_later_fault(
    monkeypatch, profile, config, tmp_path
):
    from so101_nexus import excitation

    robot = FakeRobot(profile, config.center_qpos_deg)
    clock = FakeClock()
    original = excitation._collect_episode

    def collect(*args):
        if args[3].name == "joint_chirps":
            raise RuntimeError("second episode failed")
        return original(*args)

    monkeypatch.setattr(excitation, "preflight_excitation", lambda *args: None)
    monkeypatch.setattr(excitation, "_collect_episode", collect)
    output = tmp_path / "logs"
    with pytest.raises(RuntimeError, match="second episode failed"):
        collect_excitation(
            robot, profile, config, clock=clock, sleep=clock.sleep, output_dir=output
        )
    saved = list(output.glob("*.npz"))
    assert len(saved) == 1
    assert RecordedTrajectory.load(saved[0]).name == "multisine"
    assert not robot.connected


def test_collector_rejects_existing_output_before_connect(monkeypatch, profile, config, tmp_path):
    robot = FakeRobot(profile, config.center_qpos_deg)
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    with pytest.raises(FileExistsError):
        collect_excitation(robot, profile, config, output_dir=tmp_path)
    assert not robot.connected


def test_collector_checks_deadline_after_read_before_command(monkeypatch, profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)
    clock = FakeClock()
    original = robot.get_observation

    def slow_read():
        clock.now += 0.08
        return original()

    monkeypatch.setattr(robot, "get_observation", slow_read)
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    with pytest.raises(RuntimeError, match="overrun"):
        collect_excitation(robot, profile, config, clock=clock, sleep=clock.sleep)
    assert not robot.commands
    assert not robot.connected


def test_collector_rejects_camera_capture_before_connect(profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)
    robot.config.cameras = {"wrist": object()}
    with pytest.raises(ValueError, match="cameras"):
        collect_excitation(robot, profile, config)
    assert not robot.connected


def test_failed_connect_does_not_disconnect_unconnected_robot(monkeypatch, profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)

    def fail(**kwargs):
        raise OSError("original connection failure")

    def decorated_disconnect():
        raise RuntimeError("robot is not connected")

    monkeypatch.setattr(robot, "connect", fail)
    monkeypatch.setattr(robot, "disconnect", decorated_disconnect)
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    with pytest.raises(OSError, match="original connection failure"):
        collect_excitation(robot, profile, config)


def test_partial_bus_connection_is_closed_without_masking_original(monkeypatch, profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)
    robot.config.disable_torque_on_disconnect = False
    bus = SimpleNamespace(is_connected=False, disabled_torque=None)
    robot.bus = bus

    def fail(**kwargs):
        bus.is_connected = True
        raise OSError("partial connection failure")

    def disconnect(disable_torque=True):
        bus.disabled_torque = disable_torque
        bus.is_connected = False

    def decorated_disconnect():
        raise RuntimeError("robot is not connected")

    bus.disconnect = disconnect
    monkeypatch.setattr(robot, "connect", fail)
    monkeypatch.setattr(robot, "disconnect", decorated_disconnect)
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    with pytest.raises(OSError, match="partial connection failure"):
        collect_excitation(robot, profile, config)
    assert not bus.is_connected
    assert bus.disabled_torque is False


def test_cleanup_error_is_attached_to_original_fault(monkeypatch, profile, config):
    robot = FakeRobot(profile, config.center_qpos_deg)
    robot.fail_read = True

    def fail_disconnect():
        raise OSError("disconnect failure")

    monkeypatch.setattr(robot, "disconnect", fail_disconnect)
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    clock = FakeClock()
    with pytest.raises(RuntimeError, match="bus failure") as caught:
        collect_excitation(robot, profile, config, clock=clock, sleep=clock.sleep)
    assert any("disconnect failure" in note for note in caught.value.__notes__)


class PreparationBus:
    def __init__(self, robot, profile, config):
        from lerobot.motors import MotorCalibration

        from so101_nexus.lerobot_adapter.normalization import build_so101_motors, unnormalize_values

        self.motors = build_so101_motors(use_degrees=True)
        robot.calibration = {
            name: MotorCalibration(**dict(value)) for name, value in profile.calibration.items()
        }
        values = profile.from_model_deg(np.array(config.center_qpos_deg))
        self.raw_positions = unnormalize_values(
            dict(zip(SO101_JOINT_NAMES, values, strict=True)),
            motors=self.motors,
            calibration=robot.calibration,
        )
        self.is_connected = False
        self.is_calibrated = True
        self.torque = dict.fromkeys(SO101_JOINT_NAMES, 0)
        self.events = []
        self.seeded_positions = None

    def connect(self):
        self.events.append("read_only_connect")
        self.is_connected = True

    def sync_read(self, name, *, normalize=True):
        assert normalize is False
        self.events.append(f"read_{name}")
        return self.torque if name == "Torque_Enable" else self.raw_positions

    def sync_write(self, name, values, *, normalize=True):
        assert name == "Goal_Position"
        assert normalize is False
        assert not any(self.torque.values())
        self.events.append("seed_goal")
        self.seeded_positions = dict(values)

    def disconnect(self, disable_torque=True):
        assert disable_torque is False
        self.events.append("close_without_torque_change")
        self.is_connected = False


def _prepared_robot(monkeypatch, profile, config):
    pytest.importorskip("lerobot")
    robot = FakeRobot(profile, config.center_qpos_deg)
    robot.bus = PreparationBus(robot, profile, config)
    monkeypatch.setattr("lerobot.robots.so_follower.so_follower.SOFollower", FakeRobot)
    return robot


def test_follower_seeds_current_raw_goal_before_enabling_torque(monkeypatch, profile, config):
    robot = _prepared_robot(monkeypatch, profile, config)
    original = robot.connect

    def connect(**kwargs):
        assert robot.bus.seeded_positions == robot.bus.raw_positions
        assert not robot.bus.is_connected
        robot.bus.events.append("regular_connect_enables_torque")
        original(**kwargs)

    monkeypatch.setattr(robot, "connect", connect)
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    clock = FakeClock()
    collect_excitation(robot, profile, config, clock=clock, sleep=clock.sleep)
    assert robot.bus.events == [
        "read_only_connect",
        "read_Torque_Enable",
        "read_Present_Position",
        "seed_goal",
        "close_without_torque_change",
        "regular_connect_enables_torque",
    ]


@pytest.mark.parametrize("fault", ["torque", "calibration", "start"])
def test_follower_preparation_rejects_before_any_write(monkeypatch, profile, config, fault):
    robot = _prepared_robot(monkeypatch, profile, config)
    if fault == "torque":
        robot.bus.torque["shoulder_pan"] = 1
    elif fault == "calibration":
        robot.bus.is_calibrated = False
    else:
        robot.bus.raw_positions["shoulder_pan"] += 500
    monkeypatch.setattr("so101_nexus.excitation.preflight_excitation", lambda *args: None)
    with pytest.raises(RuntimeError, match=r"torque|calibration|start"):
        collect_excitation(robot, profile, config)
    assert not robot.bus.is_connected
    assert not robot.connected
    assert robot.bus.seeded_positions is None
    assert not robot.commands


def test_excitation_config_snapshots_mutable_joint_inputs():
    center = np.array((0, -15, 30, 45, 0, 50), dtype=float)
    amplitudes = np.ones(6)
    config = ExcitationConfig(center_qpos_deg=center, amplitudes_deg=amplitudes)
    center[:] = 1000
    amplitudes[:] = 1000
    assert config.center_qpos_deg == (0, -15, 30, 45, 0, 50)
    assert config.amplitudes_deg == (1, 1, 1, 1, 1, 1)
