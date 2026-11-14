"""Robot profiles reach both physics backends and the LeRobot coordinate boundary."""

from dataclasses import asdict, replace

import numpy as np
import pytest

from so101_nexus import MoveConfig, PhysicsConfig, RobotConfig


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


def test_training_checkpoint_preserves_robot_profile(tmp_path):
    import torch

    from examples._warp_training import Agent, ObsNormalizer, _save

    config = MoveConfig(robot=RobotConfig(profile=_profile()))
    path = tmp_path / "checkpoint.pt"
    _save(Agent(6, 6, 8), ObsNormalizer(6, "cpu"), path, 0, 0, env_config=config)
    restored = torch.load(path, weights_only=False)["env_config"]
    assert restored.robot.profile.to_dict() == config.robot.profile.to_dict()


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
def test_profile_rejects_explicit_control_clock_mismatch(env_factory, backend):
    profile = replace(_profile(), control_period_s=0.04)
    with pytest.raises(ValueError, match="control_period"):
        env_factory(
            backend,
            task="Move",
            config=MoveConfig(robot=RobotConfig(profile=profile), physics=PhysicsConfig()),
        )


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
@pytest.mark.parametrize("control_period_s", [0.02, 0.04, 1 / 30])
def test_profile_clock_drives_implicit_physics(env_factory, backend, control_period_s):
    profile = replace(_profile(), control_period_s=control_period_s)
    config = MoveConfig(robot=RobotConfig(profile=profile), reset_settle_frames=0)
    env = env_factory(backend, task="Move", config=config).unwrapped
    env.reset(seed=8)
    time_before = float(env.data.time) if backend == "mujoco" else env._time.clone()
    env.step(np.zeros(env.action_space.shape))
    time_after = float(env.data.time) if backend == "mujoco" else env._time
    np.testing.assert_allclose(time_after - time_before, control_period_s, atol=1e-8)
    assert env.control_dt == pytest.approx(control_period_s)
    assert env.metadata["render_fps"] == pytest.approx(1 / control_period_s)
    model = env.model if backend == "mujoco" else env.mjm
    assert model.opt.timestep <= PhysicsConfig().timestep_s
    lower, upper = profile.model_joint_limits_deg()
    action_space = env.action_space if backend == "mujoco" else env.single_action_space
    np.testing.assert_allclose(action_space.low, np.radians(lower), atol=1e-7)
    np.testing.assert_allclose(action_space.high, np.radians(upper), atol=1e-7)


@pytest.mark.parametrize("control_period_s", [0.02, 1 / 30])
def test_follower_profile_coordinates_and_persistence(tmp_path, control_period_s):
    from so101_nexus import SO101_JOINT_NAMES
    from so101_nexus.lerobot_adapter import SimSOFollower, SimSOFollowerConfig

    profile = replace(_profile(), control_period_s=control_period_s)
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
        assert env.control_dt == pytest.approx(control_period_s)
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
