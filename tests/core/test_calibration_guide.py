"""The profile guide advances only after explicit, valid physical-reference actions."""

from pathlib import Path

import numpy as np
import pytest

from so101_nexus import SO101_JOINT_NAMES, RobotProfile
from so101_nexus.calibration.session import CalibrationSession


class Monitor:
    def __init__(self, profile):
        self.profile = profile
        self.torque = dict.fromkeys(SO101_JOINT_NAMES, 0)
        self.calibration_path = Path("fake.json")
        self.fail_read = False
        self.closed = False

    def connect(self):
        return self.profile

    def read(self):
        if self.fail_read:
            raise OSError("encoder read failed")
        return np.array([10, 20, 30, 40, 50, 60.0])

    def close(self):
        self.closed = True


@pytest.fixture
def guide():
    from so101_nexus.calibration.guide import CalibrationGuide

    profile = RobotProfile(
        "guide-arm",
        {
            name: {"id": i, "drive_mode": 0, "homing_offset": 0, "range_min": 0, "range_max": 4095}
            for i, name in enumerate(SO101_JOINT_NAMES, 1)
        },
    )
    return CalibrationGuide(CalibrationSession(monitor_factory=lambda *_: Monitor(profile)))


def test_guide_connects_readonly_and_captures_body_joints_in_order(guide):
    assert guide.stage == 0
    assert guide.session.monitor is None
    guide.connect("/dev/fake", "/unused")
    assert guide.stage == 1
    assert not guide.session.confirmed
    assert guide.reference_deg == [0, 0, 0, 0, 0, 45]
    for index in range(5):
        assert guide.joint_index == index
        assert guide.focused is (index >= 3)
        guide.capture_next(0)
    assert guide.stage == 2
    assert guide.session.confirmed == set(range(5))
    assert guide.session.profile.joint_offsets_deg == (-10, -20, -30, -40, -50)


def test_guide_requires_separate_gripper_confirmation_before_save(guide, tmp_path):
    guide.connect("/dev/fake", "/unused")
    with pytest.raises(ValueError, match="stage"):
        guide.save_profile(tmp_path / "early.json", 0.02)
    for _ in range(5):
        guide.capture_next(0)
    with pytest.raises(ValueError, match="Confirm"):
        guide.confirm_gripper(-5, 80, confirmed=False)
    assert guide.stage == 2
    guide.confirm_gripper(-5, 80, confirmed=True)
    assert guide.stage == 3
    saved = guide.save_profile(tmp_path / "arm.json", 1 / 30)
    profile = RobotProfile.load(saved)
    assert profile.gripper_limits_deg == (-5, 80)
    assert profile.control_period_s == 1 / 30
    with pytest.raises(FileExistsError):
        guide.save_profile(saved, 1 / 30)


@pytest.mark.parametrize("fault", ["read", "torque"])
def test_capture_failure_does_not_advance_or_confirm(guide, fault):
    guide.connect("/dev/fake", "/unused")
    monitor = guide.session.monitor
    if fault == "read":
        monitor.fail_read = True
    else:
        monitor.torque["wrist_roll"] = 1
    with pytest.raises((ValueError, OSError), match=r"torque|read failed"):
        guide.capture_next(0)
    assert guide.stage == 1
    assert guide.joint_index == 0
    assert not guide.session.confirmed


def test_reference_edits_and_rotation_preview_require_physical_capture(guide):
    guide.connect("/dev/fake", "/unused")
    guide.session.confirmed.add(0)
    guide.session.verified = True
    guide.set_reference(25)
    assert guide.reference_deg[0] == 25
    assert not guide.session.confirmed
    assert not guide.session.verified
    guide.preview_rotation(90)
    assert guide.session.profile.joint_offsets_deg[0] == 90
    assert guide.reference_deg[0] == 25
    assert not guide.session.confirmed
    with pytest.raises(ValueError, match="stage"):
        guide.save_profile("/unused/arm.json", 0.02)
    guide.capture_next(25)
    assert guide.session.profile.joint_offsets_deg[0] == 15
    assert guide.session.confirmed == {0}


def test_invalid_reference_and_wrong_stage_do_not_change_guide(guide):
    with pytest.raises(ValueError, match="stage"):
        guide.capture_next(0)
    guide.connect("/dev/fake", "/unused")
    previous = guide.reference_deg.copy()
    with pytest.raises(ValueError, match="finite"):
        guide.set_reference(float("nan"))
    assert guide.reference_deg == previous
    with pytest.raises(ValueError, match="stage"):
        guide.confirm_gripper(-10, 100, confirmed=True)
    assert guide.stage == 1


def test_previous_joint_returns_to_final_body_joint_from_gripper(guide):
    guide.connect("/dev/fake", "/unused")
    guide.previous_joint()
    assert guide.joint_index == 0
    for _ in range(5):
        guide.capture_next(0)
    guide.previous_joint()
    assert guide.stage == 1
    assert guide.joint_index == 4
    assert guide.focused
    guide.previous_joint()
    assert guide.joint_index == 3


def test_failed_reconnection_resets_the_stage_without_capturing(guide):
    guide.connect("/dev/fake", "/unused")
    guide.capture_next(0)

    def unavailable(*_):
        raise OSError("port unavailable")

    guide.session.monitor_factory = unavailable
    with pytest.raises(OSError, match="port unavailable"):
        guide.connect("/dev/fake", "/unused")
    assert guide.stage == 0
    assert not guide.session.confirmed


def test_loading_trusted_profile_never_uses_current_pose_as_reference(guide, tmp_path):
    guide.connect("/dev/fake", "/unused")
    path = tmp_path / "trusted.json"
    guide.session.profile.save(path)
    guide.load_profile(path)
    assert guide.stage == 3
    assert guide.reference_deg == [0, 0, 0, 0, 0, 45]
    assert guide.session.confirmed == set(range(5))
    assert guide.session.gripper_confirmed


def test_save_step_returns_to_whole_arm_after_gripper_confirmation(guide):
    guide.connect("/dev/fake", "/unused")
    for _ in range(5):
        guide.capture_next(0)
    assert guide.focused
    guide.view = "side"
    guide.confirm_gripper(-10, 100, confirmed=True)
    assert guide.stage == 3
    assert guide.view == "front"
    assert not guide.focused


def test_trusted_profile_load_returns_to_whole_arm(guide, tmp_path):
    guide.connect("/dev/fake", "/unused")
    path = tmp_path / "trusted.json"
    guide.session.profile.save(path)
    guide.focused = True
    guide.view = "side"
    guide.load_profile(path)
    assert guide.stage == 3
    assert guide.view == "front"
    assert not guide.focused


@pytest.mark.parametrize("value", [None, "0", True, float("nan")])
def test_invalid_reference_values_do_not_revoke_or_change_alignment(guide, value):
    guide.connect("/dev/fake", "/unused")
    guide.session.confirmed.add(0)
    guide.session.verified = True
    previous = guide.reference_deg.copy()
    with pytest.raises(ValueError, match="finite"):
        guide.set_reference(value)
    assert guide.reference_deg == previous
    assert guide.session.confirmed == {0}
    assert guide.session.verified


def test_capture_rejects_visible_angle_until_the_ghost_reference_matches(guide, monkeypatch):
    from unittest.mock import Mock

    guide.connect("/dev/fake", "/unused")
    guide.set_reference(1)
    read = Mock(wraps=guide.session.monitor.read)
    monkeypatch.setattr(guide.session.monitor, "read", read)
    with pytest.raises(ValueError, match="Reference view is updating"):
        guide.capture_next(15)
    read.assert_not_called()
    assert guide.joint_index == 0
    assert guide.stage == 1
    assert guide.reference_deg[0] == 1
    assert not guide.session.confirmed
    guide.set_reference(15)
    guide.capture_next(15)
    read.assert_called_once_with()
    assert guide.session.profile.joint_offsets_deg[0] == 5
    assert guide.joint_index == 1


@pytest.mark.parametrize("value", [None, "0", True, float("nan")])
def test_capture_rejects_invalid_visible_angles_before_encoder_reads(guide, monkeypatch, value):
    from unittest.mock import Mock

    guide.connect("/dev/fake", "/unused")
    read = Mock(side_effect=AssertionError("Invalid references must not read hardware"))
    monkeypatch.setattr(guide.session.monitor, "read", read)
    with pytest.raises(ValueError, match="Reference view is updating"):
        guide.capture_next(value)
    read.assert_not_called()
    assert guide.joint_index == 0
    assert not guide.session.confirmed


def test_saved_profile_resumes_completion_after_readonly_match(guide, tmp_path):
    from dataclasses import replace

    path = tmp_path / "confirmed.json"
    profile = guide.session.monitor_factory().profile
    saved = replace(profile, joint_offsets_deg=(1, 2, 3, 4, 5))
    saved.save(path)
    guide.saved_profile = path
    guide.connect("/dev/fake", "/unused")
    assert guide.stage == 3
    assert guide.session.profile == saved
    assert guide.session.confirmed == set(range(5))
    assert guide.session.gripper_confirmed
    guide.recalibrate()
    assert guide.stage == 1
    assert guide.joint_index == 0
    assert not guide.session.confirmed
    assert not guide.session.gripper_confirmed
    assert guide.session.profile_path is None
    assert path.exists()


def test_saved_profile_for_different_robot_does_not_skip_alignment(guide, tmp_path):
    from dataclasses import replace

    path = tmp_path / "other.json"
    profile = guide.session.monitor_factory().profile
    calibration = {name: dict(values) for name, values in profile.calibration.items()}
    calibration[SO101_JOINT_NAMES[0]]["homing_offset"] = 10
    replace(profile, calibration=calibration).save(path)
    guide.saved_profile = path
    guide.connect("/dev/fake", "/unused")
    assert guide.stage == 1
    assert not guide.session.confirmed
    assert guide.session.profile_path is None
    assert "differs" in guide.session.status


def test_busy_reconnect_keeps_the_current_guide_stage_and_monitor(guide):
    guide.connect("/dev/fake", "/unused")
    guide.stage = 3
    monitor = guide.session.monitor
    guide.session._preflighting = True
    with pytest.raises(RuntimeError, match="motion check"):
        guide.connect("/dev/fake", "/unused")
    assert guide.stage == 3
    assert guide.session.monitor is monitor
    assert not monitor.closed
