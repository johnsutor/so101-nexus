"""Tests for composable observation components."""

import math

import pytest

from so101_nexus_core.observations import (
    EndEffectorPose,
    GazeDirection,
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    Observation,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
)


class TestObservationBase:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            Observation()

    def test_subclass_requires_name_and_size(self):
        class Bad(Observation):
            pass

        with pytest.raises(TypeError):
            Bad()


class TestStateComponents:
    @pytest.mark.parametrize(
        "cls,expected_name,expected_size",
        [
            (JointPositions, "joint_positions", 6),
            (EndEffectorPose, "end_effector_pose", 7),
            (TargetOffset, "target_offset", 3),
            (GazeDirection, "gaze_direction", 3),
            (GraspState, "grasp_state", 1),
            (ObjectPose, "object_pose", 7),
            (ObjectOffset, "object_offset", 3),
            (TargetPosition, "target_position", 3),
        ],
    )
    def test_name_and_size(self, cls, expected_name, expected_size):
        comp = cls()
        assert comp.name == expected_name
        assert comp.size == expected_size

    def test_repr_includes_name(self):
        comp = JointPositions()
        assert "JointPositions" in repr(comp)


class TestCameraComponents:
    def test_wrist_camera_defaults(self):
        cam = WristCamera()
        assert cam.name == "wrist_camera"
        assert cam.width == 640
        assert cam.height == 480
        assert cam.fov_deg_range == (60.0, 90.0)
        assert cam.pitch_deg_range == (-34.4, 0.0)
        assert cam.pos_x_noise == 0.005
        assert cam.pos_y_center == 0.04
        assert cam.pos_y_noise == 0.01
        assert cam.pos_z_center == -0.04
        assert cam.pos_z_noise == 0.01

    def test_wrist_camera_custom_params(self):
        cam = WristCamera(
            width=320,
            height=240,
            fov_deg_range=(50.0, 80.0),
            pitch_deg_range=(-20.0, 10.0),
            pos_x_noise=0.01,
            pos_y_center=0.05,
            pos_y_noise=0.02,
            pos_z_center=-0.03,
            pos_z_noise=0.005,
        )
        assert cam.width == 320
        assert cam.height == 240
        assert cam.fov_deg_range == (50.0, 80.0)
        assert cam.pitch_deg_range == (-20.0, 10.0)
        assert cam.pos_x_noise == 0.01
        assert cam.pos_y_center == 0.05
        assert cam.pos_y_noise == 0.02
        assert cam.pos_z_center == -0.03
        assert cam.pos_z_noise == 0.005

    def test_wrist_camera_fov_rad_range(self):
        cam = WristCamera(fov_deg_range=(60.0, 90.0))
        lo, hi = cam.fov_rad_range
        assert math.isclose(lo, math.radians(60.0), rel_tol=1e-9)
        assert math.isclose(hi, math.radians(90.0), rel_tol=1e-9)

    def test_wrist_camera_pitch_rad_range(self):
        cam = WristCamera(pitch_deg_range=(-34.4, 0.0))
        lo, hi = cam.pitch_rad_range
        assert math.isclose(lo, math.radians(-34.4), rel_tol=1e-9)
        assert math.isclose(hi, math.radians(0.0), abs_tol=1e-12)

    def test_overhead_camera_defaults(self):
        cam = OverheadCamera()
        assert cam.name == "overhead_camera"
        assert cam.width == 640
        assert cam.height == 480
        assert cam.fov_deg == 45.0

    def test_overhead_camera_custom_fov_deg(self):
        cam = OverheadCamera(fov_deg=60.0)
        assert cam.fov_deg == 60.0

    def test_camera_invalid_resolution(self):
        with pytest.raises(ValueError, match="Camera dimensions must be > 0"):
            WristCamera(width=0, height=480)
        with pytest.raises(ValueError, match="Camera dimensions must be > 0"):
            OverheadCamera(width=640, height=-1)

    def test_camera_is_observation_with_zero_size(self):
        assert isinstance(WristCamera(), Observation)
        assert isinstance(OverheadCamera(), Observation)
        assert WristCamera().size == 0
        assert OverheadCamera().size == 0

    def test_repr_includes_resolution(self):
        cam = WristCamera(width=320, height=240)
        r = repr(cam)
        assert "320" in r
        assert "240" in r

        cam2 = OverheadCamera(width=800, height=600)
        r2 = repr(cam2)
        assert "800" in r2
        assert "600" in r2
