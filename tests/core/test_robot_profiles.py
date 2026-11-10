"""Portable profiles preserve calibration, alignment, and joint dynamics."""

import json
from copy import deepcopy
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.profiles import (
    JointDynamics,
    RobotProfile,
    apply_robot_profile,
    model_fingerprint,
)


def calibration():
    return {
        name: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 1000, "range_max": 3000}
        for i, name in enumerate(SO101_JOINT_NAMES, 1)
    }


@pytest.fixture
def profile():
    return RobotProfile(
        robot_id="bench-arm",
        calibration=calibration(),
        joint_offsets_deg=(10, -5, 0, 2, -3),
        dynamics={"shoulder_pan": JointDynamics(0.1, 0.02, 250, 20)},
    )


def test_profile_round_trip_is_portable_and_does_not_mutate_source(tmp_path, profile):
    path = tmp_path / "nested" / "arm.json"
    profile.save(path)
    restored = RobotProfile.load(path)
    assert restored == profile
    assert restored.model_sha256 == ""
    assert json.loads(path.read_text())["calibration"] == calibration()
    assert list(path.parent.iterdir()) == [path]
    with pytest.raises(TypeError):
        restored.calibration["shoulder_pan"]["range_min"] = 0


def test_lerobot_import_preserves_the_original_file(tmp_path):
    path = tmp_path / "follower.json"
    content = json.dumps(calibration())
    path.write_text(content)
    profile = RobotProfile.from_lerobot(path, robot_id="arm", joint_offsets_deg=(1, 2, 3, 4, 5))
    assert profile.calibration == calibration()
    assert profile.joint_offsets_deg == (1, 2, 3, 4, 5)
    assert profile.model_sha256 == model_fingerprint()
    assert path.read_text() == content


def test_custom_hardware_servo_gains_round_trip(tmp_path):
    path = tmp_path / "follower.json"
    path.write_text(json.dumps(calibration()))
    profile = RobotProfile.from_lerobot(path, robot_id="arm", servo_gains=(20, 1, 40))
    saved = tmp_path / "profile.json"
    profile.save(saved)
    assert RobotProfile.load(saved).servo_gains == (20, 1, 40)
    assert RobotProfile("arm", calibration()).servo_gains == (16, 0, 32)


def test_exported_calibration_loads_as_standard_lerobot_data(tmp_path, profile):
    path = tmp_path / "new-machine" / f"{profile.robot_id}.json"
    profile.export_lerobot(path)
    assert json.loads(path.read_text()) == calibration()
    assert (
        RobotProfile.from_lerobot(path, robot_id=profile.robot_id).calibration
        == profile.calibration
    )


@pytest.mark.parametrize(
    "robot_id", [".", "..", "../arm", "/arm", "arm/other", "arm\\other", "arm\0"]
)
def test_robot_identity_is_a_single_filename_component(profile, robot_id):
    with pytest.raises(ValueError, match="robot_id"):
        replace(profile, robot_id=robot_id)


@pytest.mark.parametrize(
    "gains", [(16, 32), (True, 0, 32), (16, 0.0, 32), (-1, 0, 32), (256, 0, 32)]
)
def test_invalid_servo_gains_rejected(profile, gains):
    with pytest.raises(ValueError, match="servo_gains"):
        replace(profile, servo_gains=gains)


def test_zero_and_maximum_servo_register_values_are_preserved(profile):
    assert replace(profile, servo_gains=[0, 255, 255]).servo_gains == (0, 255, 255)


@pytest.mark.parametrize(
    "changes",
    [
        {"robot_id": ""},
        {"robot_id": 1},
        {"schema_version": 2},
        {"schema_version": True},
        {"joint_offsets_deg": (0, 0)},
        {"joint_offsets_deg": (0, 0, 0, 0, float("nan"))},
        {"gripper_limits_deg": (1, 1)},
        {"gripper_limits_deg": (10, -10)},
        {"model_sha256": "abc"},
        {"control_period_s": 0},
        {"control_period_s": True},
        {"calibration": {}},
        {"dynamics": {"unknown": JointDynamics(0.1, 0.02, 250, 20)}},
        {"dynamics": {"shoulder_pan": {"kp": 1}}},
    ],
)
def test_invalid_profile_is_rejected(profile, changes):
    with pytest.raises((TypeError, ValueError)):
        replace(profile, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"id": 2},
        {"id": True},
        {"drive_mode": 2},
        {"range_min": -1},
        {"range_max": 4096},
        {"range_max": 1000},
        {"homing_offset": 1.5},
        {"extra": 1},
    ],
)
def test_invalid_motor_calibration_is_rejected(changes):
    motors = calibration()
    motors["shoulder_pan"].update(changes)
    with pytest.raises((TypeError, ValueError), match="shoulder_pan"):
        RobotProfile("arm", motors)


@pytest.mark.parametrize(
    "field", ["armature", "frictionloss", "kp", "kd", "joint_damping", "force_limit"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, True])
def test_invalid_joint_dynamics_is_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        replace(JointDynamics(0.1, 0.02, 250, 20), **{field: value})


@pytest.mark.parametrize("field", ["kp", "force_limit"])
def test_positive_dynamics_parameters_reject_zero(field):
    with pytest.raises(ValueError, match=field):
        replace(JointDynamics(0, 0, 250, 0), **{field: 0})


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version": 9}',
        '{"schema_version": 1, "extra": 2}',
        "[]",
        '{"robot_id":"a", "robot_id":"b"}',
    ],
)
def test_invalid_json_profile_is_rejected(tmp_path, payload):
    path = tmp_path / "arm.json"
    path.write_text(payload)
    with pytest.raises(ValueError, match=r"schema_version|Invalid robot profile|Duplicate JSON"):
        RobotProfile.load(path)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("tensor", [False, True])
def test_alignment_maps_physical_reference_and_gripper_endpoints(profile, batched, tensor):
    values = np.array([[0, 1, 2, 3, 4, 0], [0, 1, 2, 3, 4, 100]], dtype=np.float32)
    if not batched:
        values = values[0]
    original = values.copy()
    if tensor:
        torch = pytest.importorskip("torch")
        values = torch.from_numpy(values)
    transformed = profile.to_model_deg(values)
    recovered = profile.from_model_deg(transformed)
    expected = original.copy()
    expected[..., :5] += [10, -5, 0, 2, -3]
    expected[..., 5] = -10 + original[..., 5] * 1.1
    np.testing.assert_allclose(transformed, expected)
    np.testing.assert_allclose(recovered, original, atol=1e-5)
    np.testing.assert_array_equal(values, original)
    if tensor:
        assert transformed.dtype == values.dtype
        assert transformed.device == values.device


@pytest.mark.parametrize("values", [np.zeros(5), np.zeros((6, 1)), np.full(6, np.nan)])
def test_alignment_rejects_invalid_values(profile, values):
    with pytest.raises(ValueError, match="finite with shape"):
        profile.to_model_deg(values)


def test_profile_takes_defensive_copies():
    motors = calibration()
    dynamics = {"shoulder_pan": JointDynamics(0.1, 0.02, 250, 20)}
    profile = RobotProfile("arm", motors, dynamics=dynamics)
    motors["shoulder_pan"]["range_min"] = 0
    dynamics.clear()
    assert profile.calibration["shoulder_pan"]["range_min"] == 1000
    assert "shoulder_pan" in profile.dynamics


def test_immutable_profile_can_be_deep_copied(profile):
    assert deepcopy(profile) is profile


def test_known_reference_pose_establishes_alignment_without_changing_source(profile):
    aligned = profile.with_reference_pose(
        hardware_body_deg=(2, -86, 95, 5, -3),
        model_body_deg=(0, -90, 90, 0, 0),
    )
    actual = aligned.to_model_deg(np.array([2, -86, 95, 5, -3, 0]))
    np.testing.assert_allclose(actual, [0, -90, 90, 0, 0, -10])
    assert aligned.joint_offsets_deg == (-2, -4, -5, -5, 3)
    assert profile.joint_offsets_deg == (10, -5, 0, 2, -3)
    assert aligned.calibration == profile.calibration


def test_model_joint_limits_derive_from_calibrated_travel_and_alignment(profile):
    lower, upper = profile.model_joint_limits_deg()
    half_span_deg = 1000 * 360 / 4095
    np.testing.assert_allclose(lower[:5], np.array(profile.joint_offsets_deg) - half_span_deg)
    np.testing.assert_allclose(upper[:5], np.array(profile.joint_offsets_deg) + half_span_deg)
    assert (lower[-1], upper[-1]) == profile.gripper_limits_deg


@pytest.mark.parametrize("hardware", [(0, 0), (0, 0, 0, 0, float("nan"))])
def test_reference_alignment_rejects_invalid_poses(profile, hardware):
    with pytest.raises(ValueError, match="hardware_body_deg"):
        profile.with_reference_pose(hardware_body_deg=hardware, model_body_deg=(0,) * 5)


@pytest.mark.parametrize("method", ["save", "export_lerobot"])
def test_save_failure_preserves_previous_profile_and_removes_temporary_file(
    tmp_path, profile, monkeypatch, method
):
    from pathlib import Path

    path = tmp_path / "arm.json"
    path.write_text("previous profile")

    def fail_replace(self, target):
        raise OSError("disk failure")

    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        getattr(profile, method)(path)
    assert path.read_text() == "previous profile"
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("tensor", [False, True])
@pytest.mark.parametrize("dtype", ["complex64", "bool"])
def test_alignment_rejects_nonreal_coordinates(profile, tensor, dtype):
    values = np.zeros(6, dtype=dtype)
    if tensor:
        torch = pytest.importorskip("torch")
        values = torch.from_numpy(values)
    with pytest.raises(ValueError, match="real numeric dtype"):
        profile.to_model_deg(values)


def test_dynamics_reaches_compiled_robot(env_factory, profile):
    env = env_factory(task="Move").unwrapped
    apply_robot_profile(env.model, profile)
    joint = env.model.joint("shoulder_pan")
    dof = joint.dofadr[0]
    actuator = env.model.actuator("shoulder_pan")
    assert env.model.dof_armature[dof] == 0.1
    assert env.model.dof_frictionloss[dof] == 0.02
    assert env.model.dof_damping[dof] == 0
    assert actuator.gainprm[0] == 250
    assert actuator.biasprm[1] == -250
    assert actuator.biasprm[2] == -20
    np.testing.assert_array_equal(actuator.forcerange, [-2.94, 2.94])


def test_armature_refreshes_derived_inverse_inertia(env_factory, profile):
    env = env_factory(task="Move").unwrapped
    dof = env.model.joint("shoulder_pan").dofadr[0]
    before = float(env.model.dof_invweight0[dof])
    apply_robot_profile(env.model, profile)
    assert env.model.dof_invweight0[dof] < before


def test_profile_intersects_joint_and_actuator_limits_without_dynamics(env_factory, profile):
    env = env_factory(task="Move").unwrapped
    model = env.model
    shoulder = model.joint("shoulder_pan")
    actuator = model.actuator("shoulder_pan")
    model.jnt_range[shoulder.id] = np.radians([-60, 100])
    model.actuator_ctrlrange[actuator.id] = np.radians([-70, 90])
    narrowed = replace(profile, dynamics={}, joint_offsets_deg=(20, 0, 0, 0, 0))
    apply_robot_profile(model, narrowed)
    np.testing.assert_allclose(np.degrees(shoulder.range), [-60, 90])
    np.testing.assert_allclose(np.degrees(actuator.ctrlrange), [-60, 90])
    assert shoulder.limited[0]
    assert actuator.ctrllimited[0]
    wrist = model.joint("wrist_roll")
    np.testing.assert_allclose(np.degrees(wrist.range), [-1000 * 360 / 4095, 1000 * 360 / 4095])


def test_disjoint_hardware_limits_are_rejected_before_any_model_writes(env_factory, profile):
    env = env_factory(task="Move").unwrapped
    model = env.model
    fields = ("jnt_range", "actuator_ctrlrange", "dof_armature")
    before = {name: getattr(model, name).copy() for name in fields}
    invalid = replace(profile, joint_offsets_deg=(0, 0, 0, 0, 1000))
    with pytest.raises(ValueError, match=r"wrist_roll.*overlap"):
        apply_robot_profile(model, invalid)
    for name in fields:
        np.testing.assert_array_equal(getattr(model, name), before[name])


@pytest.mark.parametrize("field,value", [("gear", 2), ("biasprm", 1)])
def test_nonstandard_actuator_transmission_is_rejected_before_mutation(
    env_factory, profile, field, value
):
    model = env_factory(task="Move").unwrapped.model
    getattr(model.actuator("wrist_roll"), field)[0] = value
    before = model.jnt_range.copy()
    with pytest.raises(ValueError, match="wrist_roll"):
        apply_robot_profile(model, profile)
    np.testing.assert_array_equal(model.jnt_range, before)


def test_model_mismatch_is_rejected_before_mutation(env_factory, profile):
    env = env_factory(task="Move").unwrapped
    before = env.model.dof_armature.copy()
    with pytest.raises(ValueError, match="fingerprint"):
        apply_robot_profile(env.model, replace(profile, model_sha256="0" * 64))
    np.testing.assert_array_equal(before, env.model.dof_armature)


def test_invalid_actuator_is_rejected_before_mutation(env_factory, profile):
    env = env_factory(task="Move").unwrapped
    before = env.model.dof_armature.copy()
    env.model.actuator("wrist_roll").biastype[0] = mujoco.mjtBias.mjBIAS_NONE
    with pytest.raises(ValueError, match="wrist_roll"):
        apply_robot_profile(env.model, profile)
    np.testing.assert_array_equal(before, env.model.dof_armature)


def test_model_fingerprint_is_repeatable():
    assert model_fingerprint() == model_fingerprint()
    assert len(model_fingerprint()) == 64


def test_missing_model_elements_are_rejected_without_writes(profile):
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    with pytest.raises(ValueError, match="requires joint and actuator"):
        apply_robot_profile(model, profile)
