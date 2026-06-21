import numpy as np
import pytest

from so101_nexus.config import (
    EXTENDED_POSE,
    POSES,
    REST_POSE,
    SO101_JOINT_NAMES,
    EnvironmentConfig,
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    Pose,
    RenderConfig,
    RewardConfig,
    RobotConfig,
    _normalize_objects,
)
from so101_nexus.constants import COLOR_MAP, sample_color
from so101_nexus.objects import CubeObject
from so101_nexus.observations import OverheadCamera, WristCamera


class TestConfigInheritance:
    def test_pick_config_inherits_base_defaults(self):
        cfg = PickConfig(goal_thresh=0.05)
        assert cfg.goal_thresh == 0.05
        assert cfg.spawn_half_size == 0.05

    def test_pick_and_place_inherits_base_defaults(self):
        cfg = PickAndPlaceConfig()
        assert cfg.goal_thresh == 0.025
        assert cfg.spawn_half_size == 0.05

    def test_pick_config_goal_thresh(self):
        cfg = PickConfig()
        assert cfg.goal_thresh == 0.025

    def test_custom_render(self):
        cfg = PickConfig(render=RenderConfig(width=128, height=128))
        assert cfg.render.width == 128
        assert cfg.render.height == 128

    def test_custom_reward(self):
        cfg = PickConfig(reward=RewardConfig(action_delta_penalty=0.01))
        assert cfg.reward.action_delta_penalty == 0.01

    def test_configs_are_mutable(self):
        cfg = PickConfig()
        cfg.goal_thresh = 0.05
        assert cfg.goal_thresh == 0.05


class TestConfigConsistency:
    def test_spawn_defaults_shape(self):
        cfg = EnvironmentConfig()
        assert len(cfg.spawn_center) == 2
        assert cfg.spawn_half_size > 0

    def test_render_defaults_valid(self):
        cfg = RenderConfig()
        assert cfg.width > 0
        assert cfg.height > 0

    def test_max_episode_steps_field_removed(self):
        # max_episode_steps was removed as a dead config knob; episode length is
        # owned by the gym registration.
        cfg = EnvironmentConfig()
        assert not hasattr(cfg, "max_episode_steps")

    def test_default_reset_settle_frames_is_5(self):
        cfg = EnvironmentConfig()
        assert cfg.reset_settle_frames == 5

    def test_reset_settle_frames_accepts_zero(self):
        cfg = EnvironmentConfig(reset_settle_frames=0)
        assert cfg.reset_settle_frames == 0

    @pytest.mark.parametrize("value", [-1, -5])
    def test_reset_settle_frames_rejects_negative_values(self, value):
        with pytest.raises(ValueError, match="reset_settle_frames"):
            EnvironmentConfig(reset_settle_frames=value)

    @pytest.mark.parametrize("value", [1.5, "5"])
    def test_reset_settle_frames_rejects_non_integer_values(self, value):
        with pytest.raises(ValueError, match="reset_settle_frames"):
            EnvironmentConfig(reset_settle_frames=value)  # type: ignore[arg-type]


class TestApplyPenalties:
    def test_default_config_is_identity(self):
        cfg = RewardConfig()
        base = 0.73
        result = cfg.apply_penalties(base, action_delta_norm=2.0, energy_norm=3.0)
        assert result == pytest.approx(base)

    def test_nonzero_energy_penalty_lowers_reward(self):
        cfg = RewardConfig(energy_penalty=0.1)
        base = 1.0
        result = cfg.apply_penalties(base, energy_norm=2.0)
        assert result == pytest.approx(1.0 - 0.1 * 2.0)
        assert result < base

    def test_nonzero_action_delta_penalty_lowers_reward(self):
        cfg = RewardConfig(action_delta_penalty=0.2)
        base = 1.0
        result = cfg.apply_penalties(base, action_delta_norm=0.5)
        assert result == pytest.approx(1.0 - 0.2 * 0.5)
        assert result < base

    def test_compute_matches_apply_penalties(self):
        cfg = RewardConfig(action_delta_penalty=0.05, energy_penalty=0.1)
        computed = cfg.compute(
            reach_progress=0.5,
            is_grasped=False,
            task_progress=0.0,
            is_complete=False,
            action_delta_norm=1.5,
            energy_norm=2.5,
        )
        base = cfg.compute(
            reach_progress=0.5, is_grasped=False, task_progress=0.0, is_complete=False
        )
        expected = cfg.apply_penalties(base, action_delta_norm=1.5, energy_norm=2.5)
        assert computed == pytest.approx(expected)

    def test_apply_penalties_works_on_numpy_arrays(self):
        cfg = RewardConfig(action_delta_penalty=0.1, energy_penalty=0.2)
        base = np.array([1.0, 0.5])
        delta = np.array([1.0, 2.0])
        energy = np.array([0.5, 1.0])
        result = cfg.apply_penalties(base, action_delta_norm=delta, energy_norm=energy)
        expected = base - 0.1 * delta - 0.2 * energy
        np.testing.assert_allclose(result, expected)

    def test_apply_penalties_works_on_torch_tensors(self):
        torch = pytest.importorskip("torch")
        cfg = RewardConfig(action_delta_penalty=0.1, energy_penalty=0.2)
        base = torch.tensor([1.0, 0.5])
        delta = torch.tensor([1.0, 2.0])
        energy = torch.tensor([0.5, 1.0])
        result = cfg.apply_penalties(base, action_delta_norm=delta, energy_norm=energy)
        expected = base - 0.1 * delta - 0.2 * energy
        assert torch.allclose(result, expected)


class TestPickAndPlaceInvariants:
    def test_separation_covers_cube_diameter(self):
        cfg = PickAndPlaceConfig()
        assert cfg.min_cube_target_separation >= 2.0 * cfg.cube_half_size


class TestJointInvariants:
    def test_joint_and_rest_lengths_match(self):
        cfg = RobotConfig()
        assert len(SO101_JOINT_NAMES) == len(cfg.rest_qpos_deg)

    def test_joint_names_unique(self):
        assert len(set(SO101_JOINT_NAMES)) == len(SO101_JOINT_NAMES)

    def test_rest_qpos_finite(self):
        cfg = RobotConfig()
        assert np.isfinite(np.array(cfg.rest_qpos_deg)).all()

    def test_rest_qpos_deg_rad_consistent(self):
        cfg = RobotConfig()
        assert np.array(cfg.rest_qpos_rad) == pytest.approx(np.radians(np.array(cfg.rest_qpos_deg)))


class TestColorMaps:
    def test_rgba_entries_are_valid(self):
        for rgba in COLOR_MAP.values():
            assert len(rgba) == 4
            assert all(0.0 <= c <= 1.0 for c in rgba)

    def test_gray_in_color_map(self):
        assert "gray" in COLOR_MAP
        assert COLOR_MAP["gray"] == [0.5, 0.5, 0.5, 1.0]


class TestSampleColor:
    def test_single_color_returns_rgba(self):
        assert sample_color("red") == COLOR_MAP["red"]

    def test_list_returns_valid_rgba(self):
        rng = np.random.default_rng(42)
        result = sample_color(["red", "blue"], rng)
        assert result in [COLOR_MAP["red"], COLOR_MAP["blue"]]

    def test_single_element_list(self):
        assert sample_color(["green"]) == COLOR_MAP["green"]

    def test_seeded_rng_is_reproducible(self):
        colors = ["red", "green", "blue", "yellow"]
        a = sample_color(colors, np.random.default_rng(7))
        b = sample_color(colors, np.random.default_rng(7))
        assert a == b

    def test_sample_color_name_matches_rgba(self):
        from so101_nexus.constants import sample_color_name

        colors = ["red", "green", "blue"]
        name = sample_color_name(colors, np.random.default_rng(11))
        rgba = sample_color(colors, np.random.default_rng(11))
        assert name in colors
        assert COLOR_MAP[name] == rgba


class TestRewardWeights:
    def test_weights_sum_to_one(self):
        reward = RewardConfig()
        total = reward.reaching + reward.grasping + reward.task_objective + reward.completion_bonus
        assert total == pytest.approx(1.0)

    def test_weights_positive(self):
        reward = RewardConfig()
        assert reward.reaching > 0
        assert reward.grasping > 0
        assert reward.task_objective > 0
        assert reward.completion_bonus > 0


class TestRewardCompute:
    def test_zero_and_one_corners(self):
        r = RewardConfig()
        assert r.compute(0.0, False, 0.0, False) == pytest.approx(0.0)
        assert r.compute(1.0, True, 1.0, True) == pytest.approx(1.0)

    def test_reward_range(self):
        r = RewardConfig()
        for rp in [0.0, 0.5, 1.0]:
            for ig in [False, True]:
                for tp in [0.0, 0.5, 1.0]:
                    for ic in [False, True]:
                        reward = r.compute(rp, ig, tp, ic)
                        assert 0.0 <= reward <= 1.0

    def test_individual_component_isolation(self):
        r = RewardConfig()
        assert r.compute(1.0, False, 0.0, False) == pytest.approx(r.reaching)
        assert r.compute(0.0, True, 0.0, False) == pytest.approx(r.grasping)
        assert r.compute(0.0, False, 1.0, False) == pytest.approx(r.task_objective)
        assert r.compute(0.0, False, 0.0, True) == pytest.approx(r.completion_bonus)

    def test_action_delta_penalty_reduces_reward(self):
        r = RewardConfig(action_delta_penalty=0.1)
        base = r.compute(1.0, True, 1.0, True, action_delta_norm=0.0)
        penalized = r.compute(1.0, True, 1.0, True, action_delta_norm=1.0)
        assert penalized < base
        assert base - penalized == pytest.approx(0.1)

    def test_default_action_delta_is_zero(self):
        r = RewardConfig()
        assert r.action_delta_penalty == 0.0
        base = r.compute(1.0, True, 1.0, True)
        with_delta = r.compute(1.0, True, 1.0, True, action_delta_norm=5.0)
        assert base == pytest.approx(with_delta)

    def test_reward_config_tanh_shaping_scale_default(self):
        cfg = RewardConfig()
        assert cfg.tanh_shaping_scale == pytest.approx(5.0)


class TestEnergyPenalty:
    def test_energy_penalty_zero_no_op(self):
        r = RewardConfig(energy_penalty=0.0)
        base = r.compute(1.0, True, 1.0, True, energy_norm=0.0)
        with_energy = r.compute(1.0, True, 1.0, True, energy_norm=5.0)
        assert base == pytest.approx(with_energy)

    def test_energy_penalty_nonzero_reduces_reward(self):
        r = RewardConfig(energy_penalty=0.05)
        base = r.compute(1.0, True, 1.0, True, energy_norm=0.0)
        penalized = r.compute(1.0, True, 1.0, True, energy_norm=2.0)
        assert penalized < base
        assert base - penalized == pytest.approx(0.05 * 2.0)

    def test_energy_penalty_default_is_zero(self):
        r = RewardConfig()
        assert r.energy_penalty == 0.0

    def test_energy_and_action_delta_penalties_additive(self):
        r = RewardConfig(action_delta_penalty=0.1, energy_penalty=0.05)
        base = r.compute(1.0, True, 1.0, True, action_delta_norm=0.0, energy_norm=0.0)
        penalized = r.compute(1.0, True, 1.0, True, action_delta_norm=1.0, energy_norm=2.0)
        expected_reduction = 0.1 * 1.0 + 0.05 * 2.0
        assert base - penalized == pytest.approx(expected_reduction)


class TestPickConfig:
    def test_default_single_cube(self):
        cfg = PickConfig()
        assert len(cfg.objects) == 1
        assert cfg.n_distractors == 0

    def test_multi_object_with_distractors(self):
        objs = [CubeObject() for _ in range(4)]
        cfg = PickConfig(objects=objs, n_distractors=2)
        assert len(cfg.objects) == 4
        assert cfg.n_distractors == 2

    def test_invalid_pool_size_raises(self):
        with pytest.raises(ValueError, match="objects pool must have at least"):
            PickConfig(objects=[CubeObject()], n_distractors=2)

    def test_negative_distractors_raises(self):
        with pytest.raises(ValueError, match="n_distractors must be >= 0"):
            PickConfig(n_distractors=-1)

    def test_single_scene_object_wrapped_in_list(self):
        obj = CubeObject()
        cfg = PickConfig(objects=obj)
        assert isinstance(cfg.objects, list)
        assert len(cfg.objects) == 1


class TestObjectNormalization:
    def test_normalize_objects_uses_default_when_none(self):
        default = CubeObject(color="blue")

        result = _normalize_objects(None, default)

        assert result == [default]

    def test_normalize_objects_wraps_single_object(self):
        obj = CubeObject(color="green")

        result = _normalize_objects(obj, CubeObject())

        assert result == [obj]

    def test_normalize_objects_copies_iterable_and_rejects_empty(self):
        objects = [CubeObject(color="red")]

        result = _normalize_objects(objects, CubeObject())

        assert result == objects
        assert result is not objects
        with pytest.raises(ValueError, match="objects must not be empty"):
            _normalize_objects([], CubeObject())


class TestTaskDescriptions:
    def test_move_config_task_description(self):
        cfg = MoveConfig(direction="up", target_distance=0.10)
        assert cfg.task_description == "Move the end-effector up by 0.10 m."

    def test_move_config_task_description_other_direction(self):
        cfg = MoveConfig(direction="forward", target_distance=0.05)
        assert cfg.task_description == "Move the end-effector forward by 0.05 m."

    def test_look_at_config_task_description(self):
        from so101_nexus.objects import CubeObject

        obj = CubeObject(color="red")
        cfg = LookAtConfig(objects=[obj])
        assert cfg.task_description == f"Look at the {obj!r}."

    def test_pick_and_place_config_task_description_str_colors(self):
        cfg = PickAndPlaceConfig(cube_colors="red", target_colors="blue")
        assert cfg.task_description == "Pick up the small red cube and place it on the blue circle"

    def test_pick_and_place_config_task_description_list_colors(self):
        cfg = PickAndPlaceConfig(cube_colors=["red", "green"], target_colors=["blue"])
        assert cfg.task_description == "Pick up the small red cube and place it on the blue circle"

    def test_describe_pick_target_cube(self):
        from so101_nexus.config import describe_pick_target
        from so101_nexus.objects import CubeObject

        obj = CubeObject(color="green")
        assert describe_pick_target(obj) == f"Pick up the {obj!r}."

    def test_describe_pick_target_ycb(self):
        from so101_nexus.config import describe_pick_target
        from so101_nexus.objects import YCBObject

        obj = YCBObject(model_id="009_gelatin_box")
        assert describe_pick_target(obj) == f"Pick up the {obj!r}."


def test_robot_config_grasp_force_threshold_default():
    cfg = RobotConfig()
    assert cfg.grasp_force_threshold == pytest.approx(0.5)


def test_robot_config_static_vel_threshold_default():
    cfg = RobotConfig()
    assert cfg.static_vel_threshold == pytest.approx(0.2)


class TestPose:
    def test_fixed_joints_stay_fixed(self):
        pose = Pose(
            name="test",
            shoulder_pan_deg=10.0,
            shoulder_lift_deg=-90.0,
            elbow_flex_deg=90.0,
            wrist_flex_deg=37.0,
            wrist_roll_deg=0.0,
            gripper_deg=-10.0,
        )
        rng = np.random.default_rng(42)
        result = pose.sample(rng)
        assert result == (10.0, -90.0, 90.0, 37.0, 0.0, -10.0)

    def test_free_joints_within_range(self):
        pose = Pose(
            name="test",
            shoulder_pan_deg=(-110.0, 110.0),
            shoulder_lift_deg=-90.0,
            elbow_flex_deg=90.0,
            wrist_flex_deg=37.0,
            wrist_roll_deg=(-157.0, 163.0),
            gripper_deg=(-10.0, 100.0),
        )
        rng = np.random.default_rng(42)
        for _ in range(50):
            result = pose.sample(rng)
            assert -110.0 <= result[0] <= 110.0
            assert result[1] == -90.0
            assert result[2] == 90.0
            assert result[3] == 37.0
            assert -157.0 <= result[4] <= 163.0
            assert -10.0 <= result[5] <= 100.0

    def test_sample_rad_converts_to_radians(self):
        pose = Pose(
            name="test",
            shoulder_pan_deg=0.0,
            shoulder_lift_deg=-90.0,
            elbow_flex_deg=90.0,
            wrist_flex_deg=0.0,
            wrist_roll_deg=0.0,
            gripper_deg=0.0,
        )
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        d = pose.sample(rng1)
        r = pose.sample_rad(rng2)
        assert r == pytest.approx(tuple(np.radians(v) for v in d))

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="min must be <= max"):
            Pose(
                name="bad",
                shoulder_pan_deg=(110.0, -110.0),
                shoulder_lift_deg=0.0,
                elbow_flex_deg=0.0,
                wrist_flex_deg=0.0,
                wrist_roll_deg=0.0,
                gripper_deg=0.0,
            )

    def test_name_stored(self):
        pose = Pose(
            name="mypose",
            shoulder_pan_deg=0.0,
            shoulder_lift_deg=0.0,
            elbow_flex_deg=0.0,
            wrist_flex_deg=0.0,
            wrist_roll_deg=0.0,
            gripper_deg=0.0,
        )
        assert pose.name == "mypose"


class TestBuiltinPoses:
    def test_rest_pose_in_registry(self):
        assert "rest" in POSES
        assert POSES["rest"] is REST_POSE

    def test_extended_pose_in_registry(self):
        assert "extended" in POSES
        assert POSES["extended"] is EXTENDED_POSE

    def test_rest_pose_fixed_joints_match_legacy_defaults(self):
        rng = np.random.default_rng(0)
        sample = REST_POSE.sample(rng)
        assert sample[1] == pytest.approx(-90.0)
        assert sample[2] == pytest.approx(90.0)
        assert sample[3] == pytest.approx(37.8152144786)

    def test_extended_pose_has_different_fixed_joints(self):
        rng = np.random.default_rng(0)
        rest = REST_POSE.sample(rng)
        rng2 = np.random.default_rng(0)
        ext = EXTENDED_POSE.sample(rng2)
        assert ext[1] != rest[1]
        assert ext[2] != rest[2]

    def test_all_poses_sample_six_joints(self):
        rng = np.random.default_rng(42)
        for name, pose in POSES.items():
            result = pose.sample(rng)
            assert len(result) == 6, f"Pose {name!r} returned {len(result)} joints"


class TestRobotConfigInitPose:
    def test_init_pose_default_is_none(self):
        cfg = RobotConfig()
        assert cfg.init_pose is None

    def test_init_pose_accepts_string(self):
        cfg = RobotConfig(init_pose="rest")
        assert cfg.init_pose == "rest"

    def test_init_pose_accepts_pose_object(self):
        pose = Pose(
            name="custom",
            shoulder_pan_deg=0.0,
            shoulder_lift_deg=0.0,
            elbow_flex_deg=0.0,
            wrist_flex_deg=0.0,
            wrist_roll_deg=0.0,
            gripper_deg=0.0,
        )
        cfg = RobotConfig(init_pose=pose)
        assert cfg.init_pose is pose

    def test_init_pose_invalid_string_raises(self):
        with pytest.raises(ValueError, match="Unknown pose name"):
            RobotConfig(init_pose="nonexistent")

    def test_resolve_pose_returns_pose_for_string(self):
        cfg = RobotConfig(init_pose="rest")
        assert cfg.resolve_pose() is POSES["rest"]

    def test_resolve_pose_returns_pose_for_object(self):
        pose = Pose(
            name="custom",
            shoulder_pan_deg=0.0,
            shoulder_lift_deg=0.0,
            elbow_flex_deg=0.0,
            wrist_flex_deg=0.0,
            wrist_roll_deg=0.0,
            gripper_deg=0.0,
        )
        cfg = RobotConfig(init_pose=pose)
        assert cfg.resolve_pose() is pose

    def test_resolve_pose_returns_none_when_not_set(self):
        cfg = RobotConfig()
        assert cfg.resolve_pose() is None


class TestNewConfigFields:
    def test_environment_config_has_robot_colors(self):
        cfg = EnvironmentConfig()
        assert cfg.robot_colors == "yellow"

    def test_environment_config_has_robot_init_qpos_noise(self):
        cfg = EnvironmentConfig()
        assert cfg.robot_init_qpos_noise == 0.02

    def test_pick_and_place_has_cube_colors(self):
        cfg = PickAndPlaceConfig()
        assert cfg.cube_colors == "red"

    def test_pick_and_place_has_target_colors(self):
        cfg = PickAndPlaceConfig()
        assert cfg.target_colors == "blue"

    def test_robot_colors_custom(self):
        cfg = EnvironmentConfig(robot_colors="red")
        assert cfg.robot_colors == "red"

    def test_ground_colors_default(self):
        cfg = EnvironmentConfig()
        assert cfg.ground_colors == "gray"

    def test_robot_colors_list(self):
        cfg = EnvironmentConfig(robot_colors=["red", "blue"])
        assert cfg.robot_colors == ["red", "blue"]


class TestConfigValidation:
    def test_invalid_cube_colors_pick_and_place(self):
        with pytest.raises(ValueError, match="cube_colors"):
            PickAndPlaceConfig(cube_colors="neon")

    def test_invalid_target_colors(self):
        with pytest.raises(ValueError, match="target_colors"):
            PickAndPlaceConfig(target_colors="neon")

    def test_same_cube_and_target_color_warns(self):
        with pytest.warns(UserWarning, match="overlap"):
            PickAndPlaceConfig(cube_colors="red", target_colors="red")

    def test_invalid_ground_color(self):
        with pytest.raises(ValueError, match="ground_colors"):
            EnvironmentConfig(ground_colors="magenta")

    def test_invalid_robot_color(self):
        with pytest.raises(ValueError, match="robot_colors"):
            EnvironmentConfig(robot_colors="magenta")

    def test_spawn_min_radius_negative_raises(self):
        with pytest.raises(ValueError, match="spawn_min_radius"):
            EnvironmentConfig(spawn_min_radius=-0.01)

    def test_spawn_max_radius_le_min_raises(self):
        with pytest.raises(ValueError, match="spawn_max_radius"):
            EnvironmentConfig(spawn_min_radius=0.3, spawn_max_radius=0.1)

    def test_spawn_max_radius_equal_min_raises(self):
        with pytest.raises(ValueError, match="spawn_max_radius"):
            EnvironmentConfig(spawn_min_radius=0.2, spawn_max_radius=0.2)

    def test_spawn_angle_negative_raises(self):
        with pytest.raises(ValueError, match="spawn_angle_half_range_deg"):
            EnvironmentConfig(spawn_angle_half_range_deg=-1.0)

    def test_spawn_angle_over_180_raises(self):
        with pytest.raises(ValueError, match="spawn_angle_half_range_deg"):
            EnvironmentConfig(spawn_angle_half_range_deg=181.0)

    def test_spawn_angle_zero_ok(self):
        cfg = EnvironmentConfig(spawn_angle_half_range_deg=0.0)
        assert cfg.spawn_angle_half_range_deg == 0.0

    def test_spawn_angle_180_ok(self):
        cfg = EnvironmentConfig(spawn_angle_half_range_deg=180.0)
        assert cfg.spawn_angle_half_range_deg == 180.0

    def test_valid_spawn_radius_ok(self):
        cfg = EnvironmentConfig(spawn_min_radius=0.05, spawn_max_radius=0.30)
        assert cfg.spawn_min_radius == 0.05
        assert cfg.spawn_max_radius == 0.30

    def test_robot_config_rest_qpos_wrong_length(self):
        with pytest.raises(ValueError, match="rest_qpos_deg must have exactly 6"):
            RobotConfig(rest_qpos_deg=(0.0, 0.0, 0.0))

    def test_pick_and_place_negative_target_disc_radius(self):
        with pytest.raises(ValueError, match="target_disc_radius must be > 0"):
            PickAndPlaceConfig(target_disc_radius=-0.01)

    def test_pick_and_place_negative_min_separation(self):
        with pytest.raises(ValueError, match="min_cube_target_separation must be >= 0"):
            PickAndPlaceConfig(min_cube_target_separation=-0.01)

    def test_pick_config_negative_min_object_separation(self):
        with pytest.raises(ValueError, match="min_object_separation must be >= 0"):
            PickConfig(min_object_separation=-0.01)

    def test_obs_mode_visual_requires_camera_component(self):
        with pytest.raises(ValueError, match=r"obs_mode.*visual.*requires.*camera"):
            EnvironmentConfig(obs_mode="visual")

    def test_obs_mode_visual_with_wrist_camera_ok(self):
        cfg = EnvironmentConfig(
            obs_mode="visual",
            observations=[WristCamera(width=64, height=48)],
        )
        assert cfg.obs_mode == "visual"

    def test_obs_mode_visual_with_overhead_camera_ok(self):
        cfg = EnvironmentConfig(
            obs_mode="visual",
            observations=[OverheadCamera(width=64, height=48)],
        )
        assert cfg.obs_mode == "visual"


def test_pose_bounds_rad_matches_specs():
    p = Pose(
        name="t",
        shoulder_pan_deg=10.0,
        shoulder_lift_deg=(-20.0, 30.0),
        elbow_flex_deg=0.0,
        wrist_flex_deg=0.0,
        wrist_roll_deg=0.0,
        gripper_deg=(0.0, 90.0),
    )
    lo, hi = p.bounds_rad()
    np.testing.assert_allclose(lo, np.radians([10.0, -20.0, 0.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(hi, np.radians([10.0, 30.0, 0.0, 0.0, 0.0, 90.0]))
    assert lo[0] == hi[0]  # fixed joint


def test_reward_config_compute_tensor_matches_scalar():
    torch = pytest.importorskip("torch")
    rc = RewardConfig(action_delta_penalty=0.1, energy_penalty=0.2)
    rp = [0.5, 1.0]
    tg = [0.2, 0.8]
    grasped = [False, True]
    complete = [False, True]
    adn = [0.3, 0.4]
    en = [1.0, 2.0]
    scalar = [
        rc.compute(
            reach_progress=rp[i],
            is_grasped=grasped[i],
            task_progress=tg[i],
            is_complete=complete[i],
            action_delta_norm=adn[i],
            energy_norm=en[i],
        )
        for i in range(2)
    ]
    out = rc.compute(
        reach_progress=torch.tensor(rp),
        is_grasped=torch.tensor(grasped),
        task_progress=torch.tensor(tg),
        is_complete=torch.tensor(complete),
        action_delta_norm=torch.tensor(adn),
        energy_norm=torch.tensor(en),
    )
    np.testing.assert_allclose(out.numpy(), scalar, rtol=1e-6)
