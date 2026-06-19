"""Explicit synthetic LeRobot calibration files for simulator-only recordings."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any, cast

import draccus
from lerobot.motors import MotorCalibration

from so101_nexus import get_so101_simulation_dir
from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.lerobot_adapter.normalization import (
    GRIPPER_NAME,
    MOTOR_RESOLUTION,
    TICKS_PER_RADIAN,
)

if TYPE_CHECKING:
    from pathlib import Path

_SYNTHETIC_MID_TICK = MOTOR_RESOLUTION // 2
_SO101_MJCF = "so101_new_calib.xml"


def build_synthetic_calibration() -> dict[str, MotorCalibration]:
    """Build simulator-only SO101 calibration data.

    The motor-id assignment matches live SO follower hardware:
    ``1=shoulder_pan`` through ``6=gripper``. The returned data uses the same
    LeRobot ``MotorCalibration`` schema as a physical follower calibration, but
    it is not a substitute for a real robot's measured calibration.
    """
    ctrl_ranges = _read_body_ctrl_ranges_rad()
    calibration: dict[str, MotorCalibration] = {}
    for motor_id, name in enumerate(SO101_JOINT_NAMES, start=1):
        if name == GRIPPER_NAME:
            range_min = 0
            range_max = MOTOR_RESOLUTION - 1
        else:
            lower, upper = ctrl_ranges[name]
            half_span = math.ceil(max(abs(lower), abs(upper)) * TICKS_PER_RADIAN)
            range_min = max(0, _SYNTHETIC_MID_TICK - half_span)
            range_max = min(MOTOR_RESOLUTION - 1, _SYNTHETIC_MID_TICK + half_span)
        calibration[name] = MotorCalibration(
            id=motor_id,
            drive_mode=0,
            homing_offset=0,
            range_min=range_min,
            range_max=range_max,
        )
    return calibration


def write_synthetic_calibration(calibration_dir: Path, robot_id: str) -> Path:
    """Write a synthetic calibration JSON file loadable by LeRobot robots."""
    calibration_dir.mkdir(parents=True, exist_ok=True)
    fpath = calibration_dir / f"{robot_id}.json"
    with open(fpath, "w") as f, draccus.config_type("json"):
        cast("Any", draccus.dump)(build_synthetic_calibration(), f, indent=4)
    return fpath


def _read_body_ctrl_ranges_rad() -> dict[str, tuple[float, float]]:
    xml_path = get_so101_simulation_dir() / _SO101_MJCF
    root = ET.parse(xml_path).getroot()
    ranges: dict[str, tuple[float, float]] = {}
    for actuator in root.findall("./actuator/position"):
        name = actuator.get("name")
        if name is None or name == GRIPPER_NAME:
            continue
        raw_ctrlrange = actuator.get("ctrlrange")
        if raw_ctrlrange is None:
            continue
        lower, upper = (float(part) for part in raw_ctrlrange.split())
        ranges[name] = (lower, upper)

    expected = set(SO101_JOINT_NAMES[:-1])
    if set(ranges) != expected:
        raise RuntimeError(
            f"Could not read all SO101 body ctrlranges from {xml_path}: "
            f"expected {sorted(expected)}, got {sorted(ranges)}."
        )
    return {name: ranges[name] for name in SO101_JOINT_NAMES[:-1]}
