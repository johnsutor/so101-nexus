"""Robot profiles reach both physics backends and the LeRobot coordinate boundary."""

from dataclasses import asdict

import numpy as np
import pytest

from so101_nexus import MoveConfig, RobotConfig


def _profile():
    from so101_nexus import SO101_JOINT_NAMES
    from so101_nexus.profiles import JointDynamics, RobotProfile

    return RobotProfile(
        robot_id="profile_test",
        calibration={
            name: {
                "id": i,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 1000,
                "range_max": 3000,
            }
            for i, name in enumerate(SO101_JOINT_NAMES, 1)
        },
        joint_offsets_deg=(5, -3, 2, 0, 1),
        gripper_limits_deg=(-5, 65),
        dynamics={"shoulder_pan": JointDynamics(0.1, 0.03, 250, 20)},
    )


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
def test_profile_reaches_physics_before_device_transfer(env_factory, backend):
    profile = _profile()
    env = env_factory(
        backend=backend, task="Move", config=MoveConfig(robot=RobotConfig(profile=profile))
    ).unwrapped
    model = env.model if backend == "mujoco" else env.mjm
    joint = model.joint("shoulder_pan")
    actuator = model.actuator("shoulder_pan")
    assert model.dof_armature[joint.dofadr[0]] == 0.1
    assert actuator.gainprm[0] == 250
    assert actuator.biasprm[1] == -250
    assert actuator.biasprm[2] == -20
    if backend == "warp":
        assert np.asarray(env.model.actuator_gainprm.numpy())[0, actuator.id, 0] == 250


def test_robot_config_rejects_untyped_profile():
    with pytest.raises(TypeError, match="RobotProfile"):
        RobotConfig(profile={})


def test_profile_rejects_control_clock_mismatch(env_factory):
    from dataclasses import replace

    profile = replace(_profile(), control_period_s=0.04)
    with pytest.raises(ValueError, match="control_period"):
        env_factory(task="Move", config=MoveConfig(robot=RobotConfig(profile=profile)))


def test_follower_profile_coordinates_and_persistence(tmp_path):
    from so101_nexus import SO101_JOINT_NAMES
    from so101_nexus.lerobot_adapter import SimSOFollower, SimSOFollowerConfig

    profile = _profile()
    path = tmp_path / "robot.json"
    profile.save(path)
    robot = SimSOFollower(
        SimSOFollowerConfig(
            id=profile.robot_id,
            calibration_dir=tmp_path / "unused",
            env_id="MuJoCoMove-v1",
            profile_path=path,
            cameras={},
        )
    )
    assert {name: asdict(cal) for name, cal in robot.calibration.items()} == profile.calibration
    robot.connect()
    try:
        action = dict(
            zip((f"{n}.pos" for n in SO101_JOINT_NAMES), (0, 0, 0, 0, 0, 50), strict=True)
        )
        sent = robot.send_action(action)
        env = robot._env.unwrapped
        np.testing.assert_allclose(
            np.degrees(env.data.ctrl[env._actuator_ids]), (5, -3, 2, 0, 1, 30), atol=1e-6
        )
        assert sent == pytest.approx(action)
        sent = robot.send_action({**action, "gripper.pos": 110})
        assert sent["gripper.pos"] == 100
        assert np.degrees(env.data.ctrl[env._actuator_ids][-1]) == pytest.approx(65)
        sent = robot.send_action({**action, "shoulder_pan.pos": 100})
        hardware_limit_deg = 1000 * 360 / 4095
        assert sent["shoulder_pan.pos"] == pytest.approx(hardware_limit_deg)
        assert np.degrees(env.data.ctrl[env._actuator_ids][0]) == pytest.approx(
            hardware_limit_deg + 5
        )
        obs = robot.get_observation()
        expected = profile.from_model_deg(np.degrees(env._get_current_qpos()))
        expected[-1] = np.clip(expected[-1], 0, 100)
        np.testing.assert_allclose([obs[f"{n}.pos"] for n in SO101_JOINT_NAMES], expected)
        robot.set_initial_leader_action(action)
        robot.reset(seed=3)
        assert robot.get_observation()["shoulder_pan.pos"] == pytest.approx(0, abs=0.5)
    finally:
        robot.disconnect()


def test_follower_rejects_profile_with_percentage_body_units(tmp_path):
    from so101_nexus.lerobot_adapter import SimSOFollower, SimSOFollowerConfig

    path = tmp_path / "robot.json"
    _profile().save(path)
    with pytest.raises(ValueError, match="use_degrees"):
        SimSOFollower(
            SimSOFollowerConfig(
                id="profile_test", env_id="MuJoCoMove-v1", profile_path=path, use_degrees=False
            )
        )
