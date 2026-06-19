"""Tests for explicit synthetic LeRobot calibration helpers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.lerobot_adapter.normalization import TICKS_PER_RADIAN

pytest.importorskip("lerobot")


_NEW_CALIB_CTRL_RANGES_RAD = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
}


def test_build_synthetic_calibration_returns_all_so101_motors() -> None:
    from so101_nexus.lerobot_adapter.synthetic_calibration import (
        build_synthetic_calibration,
    )

    calibration = build_synthetic_calibration()

    assert tuple(calibration) == SO101_JOINT_NAMES
    assert [calibration[name].id for name in SO101_JOINT_NAMES] == [1, 2, 3, 4, 5, 6]


def test_body_ranges_are_centered_and_cover_known_so101_limits() -> None:
    from so101_nexus.lerobot_adapter.synthetic_calibration import (
        build_synthetic_calibration,
    )

    calibration = build_synthetic_calibration()

    for name, (lower, upper) in _NEW_CALIB_CTRL_RANGES_RAD.items():
        cal = calibration[name]
        assert cal.range_min + cal.range_max == 4096
        assert cal.drive_mode == 0
        assert cal.homing_offset == 0

        synthetic_lower = (cal.range_min - 2048) / TICKS_PER_RADIAN
        synthetic_upper = (cal.range_max - 2048) / TICKS_PER_RADIAN
        assert synthetic_lower <= lower + math.radians(0.1)
        assert synthetic_upper >= upper - math.radians(0.1)


def test_gripper_synthetic_range_uses_full_motor_resolution() -> None:
    from so101_nexus.lerobot_adapter.synthetic_calibration import (
        build_synthetic_calibration,
    )

    gripper = build_synthetic_calibration()["gripper"]

    assert gripper.range_min == 0
    assert gripper.range_max == 4095


def test_write_synthetic_calibration_is_loadable_by_lerobot_robot(tmp_path: Path) -> None:
    from so101_nexus.lerobot_adapter import SimSOFollower, SimSOFollowerConfig
    from so101_nexus.lerobot_adapter.synthetic_calibration import (
        build_synthetic_calibration,
        write_synthetic_calibration,
    )

    fpath = write_synthetic_calibration(tmp_path, "synthetic_sim")
    robot = SimSOFollower(
        SimSOFollowerConfig(
            id="synthetic_sim",
            calibration_dir=tmp_path,
            env_id="MuJoCoReach-v1",
        )
    )

    assert fpath == tmp_path / "synthetic_sim.json"
    assert robot.calibration == build_synthetic_calibration()
