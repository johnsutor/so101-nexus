"""The hardware example imports portable calibration without relying on a local cache."""

import json
import runpy
from dataclasses import asdict
from pathlib import Path

import pytest

from so101_nexus import SO101_JOINT_NAMES, RobotProfile


@pytest.fixture
def example():
    return runpy.run_path(str(Path(__file__).parents[2] / "examples" / "calibrate_robot.py"))


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
    }
    example["create"](**kwargs)
    profile = RobotProfile.load(destination)
    assert profile.joint_offsets_deg == (-1, -2, -3, -4, -5)
    assert profile.servo_gains == (20, 0, 30)
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
