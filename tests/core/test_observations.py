"""Tests for composable observation components."""

import math

import pytest

from so101_nexus.observations import (
    EndEffectorPose,
    GazeDirection,
    GazeState,
    GraspState,
    GripperContactForce,
    JointEfforts,
    JointPositions,
    JointVelocities,
    ObjectOffset,
    ObjectPose,
    ObjectVelocity,
    Observation,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
    observations_from_feature_names,
    privileged_state_feature_names,
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
            (JointVelocities, "joint_velocities", 6),
            (JointEfforts, "joint_efforts", 6),
            (GripperContactForce, "gripper_contact_force", 3),
            (EndEffectorPose, "end_effector_pose", 7),
            (TargetOffset, "target_offset", 3),
            (GazeDirection, "gaze_direction", 3),
            (GazeState, "gaze_state", 1),
            (GraspState, "grasp_state", 1),
            (ObjectPose, "object_pose", 7),
            (ObjectVelocity, "object_velocity", 6),
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


class TestPrivilegedStateFeatureNames:
    def test_none_returns_empty(self):
        assert privileged_state_feature_names(None) == []

    def test_camera_only_components_contribute_no_names(self):
        # Camera components report size == 0, so no scalar dims are named.
        assert privileged_state_feature_names([WristCamera(), OverheadCamera()]) == []

    def test_mixed_components_named_in_order_with_per_dim_suffixes(self):
        names = privileged_state_feature_names(
            [EndEffectorPose(), GraspState(), ObjectPose(), ObjectOffset()]
        )
        assert names == [
            "end_effector_pose_0",
            "end_effector_pose_1",
            "end_effector_pose_2",
            "end_effector_pose_3",
            "end_effector_pose_4",
            "end_effector_pose_5",
            "end_effector_pose_6",
            "grasp_state_0",
            "object_pose_0",
            "object_pose_1",
            "object_pose_2",
            "object_pose_3",
            "object_pose_4",
            "object_pose_5",
            "object_pose_6",
            "object_offset_0",
            "object_offset_1",
            "object_offset_2",
        ]

    def test_cameras_interleaved_with_state_are_skipped(self):
        # A camera between two state components must not shift or add names.
        assert privileged_state_feature_names([GraspState(), WristCamera(), ObjectOffset()]) == [
            "grasp_state_0",
            "object_offset_0",
            "object_offset_1",
            "object_offset_2",
        ]


class TestObservationsFromFeatureNames:
    def test_round_trips_every_default_layout(self):
        """A recording's declared schema rebuilds the layout that produced it."""
        from so101_nexus.config import (
            LookAtConfig,
            MoveConfig,
            PickAndPlaceConfig,
            PickConfig,
            StackCubeConfig,
            TouchConfig,
        )

        for config in (
            PickConfig(),
            TouchConfig(),
            PickAndPlaceConfig(),
            StackCubeConfig(),
            LookAtConfig(),
            MoveConfig(),
        ):
            names = privileged_state_feature_names(config.observations)
            rebuilt = observations_from_feature_names(names)
            assert privileged_state_feature_names(rebuilt) == names
            assert [type(c) for c in rebuilt] == [type(c) for c in config.observations]

    def test_empty_names_give_an_empty_layout(self):
        assert observations_from_feature_names([]) == []

    def test_unknown_component_is_rejected(self):
        with pytest.raises(ValueError, match="does not start an observation component"):
            observations_from_feature_names(["wrist_pressure_0"])

    def test_truncated_component_is_rejected(self):
        # A dataset that dropped columns must fail loudly, not silently realign
        # the remaining ones onto the wrong components.
        with pytest.raises(ValueError, match="object_pose"):
            observations_from_feature_names(["object_pose_0", "object_pose_1"])

    def test_reordered_dimensions_are_rejected(self):
        with pytest.raises(ValueError, match="object_offset"):
            observations_from_feature_names(
                ["object_offset_0", "object_offset_2", "object_offset_1"]
            )

    def test_camera_components_are_not_resolvable(self):
        # Cameras contribute no state dimensions, so no name can name one.
        with pytest.raises(ValueError, match="does not start an observation component"):
            observations_from_feature_names(["wrist_camera_0"])
