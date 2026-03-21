"""Tests for composable observation components."""

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
        assert cam.width == 224
        assert cam.height == 224

    def test_wrist_camera_custom_resolution(self):
        cam = WristCamera(width=128, height=128)
        assert cam.width == 128
        assert cam.height == 128

    def test_overhead_camera_defaults(self):
        cam = OverheadCamera()
        assert cam.name == "overhead_camera"
        assert cam.width == 224
        assert cam.height == 224

    def test_camera_invalid_resolution(self):
        with pytest.raises(ValueError, match="Camera dimensions must be > 0"):
            WristCamera(width=0, height=224)
        with pytest.raises(ValueError, match="Camera dimensions must be > 0"):
            OverheadCamera(width=224, height=-1)

    def test_camera_is_observation(self):
        assert isinstance(WristCamera(), Observation)
        assert isinstance(OverheadCamera(), Observation)
