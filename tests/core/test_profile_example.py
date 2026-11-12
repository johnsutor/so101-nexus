"""The hardware example imports portable calibration without relying on a local cache."""

import json
import runpy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from so101_nexus import SO101_JOINT_NAMES, RobotProfile


@pytest.fixture
def example():
    return runpy.run_path(str(Path(__file__).parents[2] / "examples" / "calibrate_robot.py"))


@pytest.fixture
def profile_path(tmp_path):
    from so101_nexus.lerobot_adapter.synthetic_calibration import build_synthetic_calibration

    path = tmp_path / "profile.json"
    RobotProfile(
        "portable-arm",
        {name: asdict(value) for name, value in build_synthetic_calibration().items()},
    ).save(path)
    return path


def test_create_aligns_reference_and_preserves_existing_output(example, tmp_path):
    calibration = {
        name: {
            "id": index,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": 0,
            "range_max": 4095,
        }
        for index, name in enumerate(SO101_JOINT_NAMES, 1)
    }
    source, destination = tmp_path / "calibration.json", tmp_path / "profile.json"
    source.write_text(json.dumps(calibration))
    kwargs = {
        "calibration": source,
        "robot_id": "portable-arm",
        "output": destination,
        "hardware_body_deg": (1, 2, 3, 4, 5),
        "model_body_deg": (0, 0, 0, 0, 0),
        "servo_gains": (20, 0, 30),
        "control_period_s": 1 / 30,
    }
    example["create"](**kwargs)
    profile = RobotProfile.load(destination)
    assert profile.joint_offsets_deg == (-1, -2, -3, -4, -5)
    assert profile.servo_gains == (20, 0, 30)
    assert profile.control_period_s == 1 / 30
    assert profile.calibration == calibration
    previous = destination.read_bytes()
    with pytest.raises(FileExistsError):
        example["create"](**kwargs)
    assert destination.read_bytes() == previous


def test_collect_uses_bundled_calibration_and_pid(example, monkeypatch, tmp_path):
    from so101_nexus.lerobot_adapter.synthetic_calibration import build_synthetic_calibration

    profile = RobotProfile(
        "portable-arm",
        {name: asdict(value) for name, value in build_synthetic_calibration().items()},
        servo_gains=(20, 0, 30),
    )
    path = tmp_path / "profile.json"
    profile.save(path)
    seen = []

    def collect(robot, loaded, config, *, output_dir):
        assert loaded == profile
        assert {
            name: asdict(value) for name, value in robot.calibration.items()
        } == profile.calibration
        assert robot.bus.calibration == robot.calibration
        assert robot.config.position_p_coefficient == 20
        assert robot.config.position_i_coefficient == 0
        assert robot.config.position_d_coefficient == 30
        assert output_dir == tmp_path / "logs"
        seen.append(robot.calibration_fpath)

    monkeypatch.setitem(example["collect"].__globals__, "collect_excitation", collect)
    example["collect"](path, "/unused", tmp_path / "logs", (0, -15, 30, 45, 0, 50))
    assert len(seen) == 1
    assert not seen[0].exists()


@pytest.mark.parametrize("period", [0.02, 1 / 30, 0.03])
def test_check_verifies_short_excitation_without_hardware(example, profile_path, capsys, period):
    replace(RobotProfile.load(profile_path), control_period_s=period).save(profile_path)
    example["check"](profile_path, (0, -15, 30, 45, 0, 50), duration_s=1)
    output = capsys.readouterr().out
    assert "3 trajectories" in output
    assert f"{1 / period:g} Hz" in output
    assert f"{round(1 / period) * period:g} s each" in output
    assert list(profile_path.parent.iterdir()) == [profile_path]


def test_check_rejects_invalid_pose(example, profile_path):
    with pytest.raises(ValueError, match="limit"):
        example["check"](profile_path, (180, -15, 30, 45, 0, 50), duration_s=1)


@pytest.mark.parametrize("exists", [False, True])
def test_fit_explains_missing_recordings_before_optimization(example, profile_path, exists):
    recordings = profile_path.parent / "recordings"
    if exists:
        recordings.mkdir()
    with pytest.raises((FileNotFoundError, ValueError), match=r"recordings|\.npz"):
        example["fit"](profile_path, recordings, profile_path.parent / "fit")


@pytest.mark.parametrize("period", [0, float("nan")])
def test_create_rejects_invalid_command_period(example, profile_path, period):
    calibration = profile_path.parent / "calibration.json"
    RobotProfile.load(profile_path).export_lerobot(calibration)
    output = profile_path.parent / "invalid.json"
    with pytest.raises(ValueError, match="control_period_s"):
        example["create"](
            calibration,
            "portable-arm",
            output,
            hardware_body_deg=(0, 0, 0, 0, 0),
            model_body_deg=(0, 0, 0, 0, 0),
            control_period_s=period,
        )
    assert not output.exists()
