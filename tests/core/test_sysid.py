"""System identification fits training data and evaluates independent episodes."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from so101_nexus import SO101_JOINT_NAMES
from so101_nexus.sysid import fit_robot_profile


@pytest.fixture
def profile():
    from so101_nexus.profiles import RobotProfile

    calibration = {
        name: {
            "id": index,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 0,
            "range_max": 4095,
        }
        for index, name in enumerate(SO101_JOINT_NAMES, start=1)
    }
    return RobotProfile(robot_id="test_arm", calibration=calibration)


def test_sysid_dependencies_have_actionable_error(monkeypatch):
    from so101_nexus import sysid

    monkeypatch.setattr(sysid, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(ImportError, match="sysid"):
        sysid._toolbox()


def test_fit_requires_independent_validation(profile):
    with pytest.raises(ValueError, match=r"training.*validation"):
        fit_robot_profile(profile, [])


def test_fit_rejects_unknown_parameters_before_optional_import(profile):
    with pytest.raises(ValueError, match="parameter"):
        fit_robot_profile(profile, [], parameters=("shoulder_pan.mass",))


def test_fit_rejects_invalid_bounds_before_optional_import(profile):
    with pytest.raises(ValueError, match="bounds"):
        fit_robot_profile(profile, [], bounds={"shoulder_pan.kp": (2, 1)})


def _synthetic_episode(profile, name, split, gain, phase):
    import mujoco

    from so101_nexus import get_so101_mujoco_model_path
    from so101_nexus.excitation import RecordedTrajectory, profile_digest
    from so101_nexus.profiles import apply_robot_profile

    model = mujoco.MjModel.from_xml_path(str(get_so101_mujoco_model_path()))
    apply_robot_profile(model, profile)
    model.opt.timestep = 0.0025
    model.actuator("shoulder_pan").gainprm[0] = gain
    model.actuator("shoulder_pan").biasprm[1] = -gain
    data = mujoco.MjData(model)
    initial = np.radians([0, -30, 45, 0, 0, 45])
    data.qpos[:] = initial
    data.ctrl[:] = initial
    mujoco.mj_forward(model, data)
    times = np.arange(101) * 0.02
    commands = np.broadcast_to(initial, (len(times), 6)).copy()
    commands[:, 0] = np.radians(8) * np.sin(2 * np.pi * 0.7 * times + phase)
    commands[:5] = initial
    measured = []
    for command in commands:
        measured.append(data.qpos.copy())
        data.ctrl[:] = command
        for _ in range(8):
            mujoco.mj_step(model, data)
    return RecordedTrajectory(
        name=name,
        split=split,
        measurement_timestamps_s=times,
        command_timestamps_s=times,
        measured_qpos_deg=np.degrees(measured),
        commanded_qpos_deg=np.degrees(commands),
        profile_sha256=profile_digest(profile),
    )


def test_identification_improves_held_out_prediction(profile, tmp_path):
    from so101_nexus import sysid

    try:
        sysid._toolbox()
    except ImportError:
        pytest.skip("Install the sysid extra to run the optimizer integration test")
    train = _synthetic_episode(profile, "train", "train", 400, 0)
    validation = _synthetic_episode(profile, "validation", "validation", 400, 0.8)
    result = fit_robot_profile(
        profile,
        [train, validation],
        parameters=("shoulder_pan.kp", "shoulder_pan.kd"),
        bounds={"shoulder_pan.kp": (100, 1500)},
        freeze=("shoulder_pan.kd",),
        max_iters=35,
        output_dir=tmp_path,
    )
    held_out = next(episode for episode in result.episodes if episode.split == "validation")
    assert held_out.identified_rmse_deg[0] < held_out.baseline_rmse_deg[0] * 0.1
    assert result.validation_improved
    assert result.candidate_profile.dynamics["shoulder_pan"].kp == pytest.approx(400, rel=0.03)
    assert result.candidate_profile.dynamics["shoulder_pan"].kd == pytest.approx(2.731)
    assert result.candidate_profile.dynamics["shoulder_pan"].joint_damping == pytest.approx(0.6)
    frozen = next(
        diagnostic for diagnostic in result.diagnostics if diagnostic.name.endswith(".kd")
    )
    assert frozen.frozen
    assert not frozen.poorly_identified
    assert profile.dynamics == {}
    assert (tmp_path / "candidate_profile.json").is_file()
    assert (tmp_path / "metrics.json").is_file()


def test_fit_rejects_profile_mismatch_and_duplicate_episode_names(profile):
    train = _synthetic_episode(profile, "train", "train", 400, 0)
    validation = replace(train, split="validation")
    with pytest.raises(ValueError, match="unique"):
        fit_robot_profile(profile, [train, validation])
    with pytest.raises(ValueError, match="profile"):
        fit_robot_profile(
            profile, [train, replace(validation, name="held_out", profile_sha256="0" * 64)]
        )


@pytest.mark.parametrize("origin_shift", [0, 1000])
def test_fit_rejects_renamed_or_time_shifted_training_copy(profile, origin_shift, monkeypatch):
    from so101_nexus import sysid

    def unexpected_toolbox():
        raise AssertionError("Invalid recordings must be rejected before optional imports")

    monkeypatch.setattr(sysid, "_toolbox", unexpected_toolbox)
    train = _synthetic_episode(profile, "train", "train", 400, 0)
    validation = replace(
        train,
        name="validation",
        split="validation",
        measurement_timestamps_s=train.measurement_timestamps_s + origin_shift,
        command_timestamps_s=train.command_timestamps_s + origin_shift,
    )
    with pytest.raises(ValueError, match="independent"):
        fit_robot_profile(profile, [train, validation])


@pytest.mark.parametrize("control_offset", [-10, 10, 2])
def test_fit_rejects_unobserved_commands_before_optional_import(
    profile, control_offset, monkeypatch
):
    from so101_nexus import sysid

    def unexpected_toolbox():
        raise AssertionError("Invalid recordings must be rejected before optional imports")

    monkeypatch.setattr(sysid, "_toolbox", unexpected_toolbox)
    train = _synthetic_episode(profile, "train", "train", 400, 0)
    validation = _synthetic_episode(profile, "validation", "validation", 400, 0.8)
    train = replace(
        train,
        measurement_timestamps_s=train.measurement_timestamps_s + 20,
        command_timestamps_s=train.command_timestamps_s + 20 + control_offset,
    )
    with pytest.raises(ValueError, match=r"command.*measurement"):
        fit_robot_profile(profile, [train, validation])


def test_identification_model_preserves_runtime_joint_and_actuator_limits(profile):
    from so101_nexus import sysid

    calibration = {name: dict(motor) for name, motor in profile.calibration.items()}
    calibration["shoulder_pan"].update(range_min=1000, range_max=3000)
    limited = replace(profile, calibration=calibration, joint_offsets_deg=(5, 0, 0, 0, 0))
    model = sysid._make_spec(limited).compile()
    expected = np.radians(np.array([-1000, 1000]) * 360 / 4095 + 5)
    np.testing.assert_allclose(model.joint("shoulder_pan").range, expected)
    np.testing.assert_allclose(model.actuator("shoulder_pan").ctrlrange, expected)
    assert model.joint("shoulder_pan").limited[0]
    assert model.actuator("shoulder_pan").ctrllimited[0]


def test_controls_hold_until_actual_write_time(profile):
    from so101_nexus import sysid

    try:
        toolbox = sysid._toolbox()
    except ImportError:
        pytest.skip("Install the sysid extra to run toolbox integration tests")
    episode = _synthetic_episode(profile, "delayed", "train", 400, 0)
    commands = episode.commanded_qpos_deg.copy()
    commands[0, 0] = 30
    delayed = replace(
        episode,
        measurement_timestamps_s=episode.measurement_timestamps_s + 10,
        command_timestamps_s=episode.command_timestamps_s + 10.005,
        commanded_qpos_deg=commands,
    )
    sequences = sysid._model_sequences(toolbox, sysid._make_spec(profile), delayed)
    control = sequences.control[0]
    assert control.times[0] == 0
    assert control.data[1, 0] == 0
    assert control.data[3, 0] == pytest.approx(np.radians(30))
    assert control.data[9, 0] == pytest.approx(np.radians(30))
    assert control.data[11, 0] == 0


def test_worse_validation_stays_an_unaccepted_candidate(profile):
    from so101_nexus import sysid

    try:
        sysid._toolbox()
    except ImportError:
        pytest.skip("Install the sysid extra to run toolbox integration tests")
    train = _synthetic_episode(profile, "train", "train", 400, 0)
    validation = _synthetic_episode(profile, "validation", "validation", 998.22, 0.8)
    result = fit_robot_profile(
        profile,
        [train, validation],
        parameters=("shoulder_pan.kp",),
        bounds={"shoulder_pan.kp": (100, 1500)},
        max_iters=35,
    )
    assert not result.validation_improved
    assert result.episodes[1].identified_rmse_deg[0] > result.episodes[1].baseline_rmse_deg[0]
    assert profile.dynamics == {}
