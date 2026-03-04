import numpy as np
import pytest

from so101_nexus_core.types import (
    CUBE_COLOR_MAP,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    DEFAULT_CUBE_HALF_SIZE,
    DEFAULT_CUBE_MASS,
    DEFAULT_CUBE_SPAWN_HALF_SIZE,
    DEFAULT_GOAL_THRESH,
    DEFAULT_LIFT_THRESHOLD,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_MAX_GOAL_HEIGHT,
    DEFAULT_MIN_CUBE_TARGET_SEPARATION,
    DEFAULT_TARGET_DISC_RADIUS,
    DEFAULT_WRIST_CAM_FOV_DEG_RANGE,
    DEFAULT_WRIST_CAM_FOV_RAD_RANGE,
    DEFAULT_WRIST_CAM_FOV_RANGE,
    REWARD_WEIGHT_COMPLETION_BONUS,
    REWARD_WEIGHT_GRASPING,
    REWARD_WEIGHT_REACHING,
    REWARD_WEIGHT_TASK_OBJECTIVE,
    SO101_JOINT_NAMES,
    SO101_REST_QPOS,
    TARGET_COLOR_MAP,
    YCB_ENV_NAME_MAP,
    ColorName,
    CubeColorName,
    TargetColorName,
    compute_normalized_reward,
)


class TestColorNameUnification:
    def test_color_name_exists(self):
        assert ColorName is not None

    def test_cube_color_name_is_alias(self):
        assert CubeColorName is ColorName

    def test_target_color_name_is_alias(self):
        assert TargetColorName is ColorName


class TestYCBEnvNameMap:
    def test_ycb_env_name_map_exists(self):
        assert isinstance(YCB_ENV_NAME_MAP, dict)

    def test_ycb_env_name_map_has_expected_entries(self):
        assert YCB_ENV_NAME_MAP["058_golf_ball"] == "GolfBall"
        assert YCB_ENV_NAME_MAP["011_banana"] == "Banana"
        assert len(YCB_ENV_NAME_MAP) == 10


class TestDeadCodeRemoval:
    def test_default_ycb_spawn_half_size_removed(self):
        import so101_nexus_core.types as types_mod

        assert not hasattr(types_mod, "DEFAULT_YCB_SPAWN_HALF_SIZE")


class TestSharedConstants:
    def test_default_cube_half_size(self):
        assert DEFAULT_CUBE_HALF_SIZE == 0.0125

    def test_default_cube_mass(self):
        assert DEFAULT_CUBE_MASS == 0.01

    def test_default_goal_thresh(self):
        assert DEFAULT_GOAL_THRESH == 0.025

    def test_default_lift_threshold(self):
        assert DEFAULT_LIFT_THRESHOLD == 0.05

    def test_default_max_goal_height(self):
        assert DEFAULT_MAX_GOAL_HEIGHT == 0.08

    def test_default_cube_spawn_half_size(self):
        assert DEFAULT_CUBE_SPAWN_HALF_SIZE == 0.05

    def test_default_max_episode_steps(self):
        assert DEFAULT_MAX_EPISODE_STEPS == 256

    def test_default_camera_dimensions(self):
        assert DEFAULT_CAMERA_WIDTH == 224
        assert DEFAULT_CAMERA_HEIGHT == 224

    def test_wrist_cam_fov_range(self):
        lo, hi = DEFAULT_WRIST_CAM_FOV_RANGE
        assert lo == 60.0
        assert hi == 90.0

    def test_wrist_cam_fov_deg_range(self):
        lo, hi = DEFAULT_WRIST_CAM_FOV_DEG_RANGE
        assert lo == 60.0
        assert hi == 90.0

    def test_wrist_cam_fov_rad_range(self):
        lo, hi = DEFAULT_WRIST_CAM_FOV_RAD_RANGE
        assert lo == pytest.approx(np.pi / 3)
        assert hi == pytest.approx(np.pi / 2)

    def test_fov_ranges_consistent(self):
        deg_lo, deg_hi = DEFAULT_WRIST_CAM_FOV_DEG_RANGE
        rad_lo, rad_hi = DEFAULT_WRIST_CAM_FOV_RAD_RANGE
        assert rad_lo == pytest.approx(np.radians(deg_lo))
        assert rad_hi == pytest.approx(np.radians(deg_hi))

    def test_so101_rest_qpos_length(self):
        assert len(SO101_REST_QPOS) == 6

    def test_so101_rest_qpos_values(self):
        expected = [0.0, -1.5708, 1.5708, 0.66, 0.0, -1.1]
        for actual, exp in zip(SO101_REST_QPOS, expected):
            assert actual == pytest.approx(exp)

    def test_so101_joint_names(self):
        assert SO101_JOINT_NAMES == [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
            "gripper",
        ]


class TestPickAndPlaceConstants:
    def test_target_color_name_type_exists(self):
        assert TargetColorName is not None

    def test_target_color_map_matches_cube_color_map(self):
        assert TARGET_COLOR_MAP == CUBE_COLOR_MAP

    def test_default_target_disc_radius(self):
        assert DEFAULT_TARGET_DISC_RADIUS == 0.05

    def test_default_min_cube_target_separation(self):
        assert DEFAULT_MIN_CUBE_TARGET_SEPARATION == 0.0375

    def test_separation_is_three_times_cube_half_size(self):
        assert DEFAULT_MIN_CUBE_TARGET_SEPARATION == pytest.approx(3 * DEFAULT_CUBE_HALF_SIZE)


class TestRewardWeights:
    def test_weights_sum_to_one(self):
        total = (
            REWARD_WEIGHT_REACHING
            + REWARD_WEIGHT_GRASPING
            + REWARD_WEIGHT_TASK_OBJECTIVE
            + REWARD_WEIGHT_COMPLETION_BONUS
        )
        assert total == pytest.approx(1.0)

    def test_individual_weight_values(self):
        assert REWARD_WEIGHT_REACHING == 0.25
        assert REWARD_WEIGHT_GRASPING == 0.25
        assert REWARD_WEIGHT_TASK_OBJECTIVE == 0.40
        assert REWARD_WEIGHT_COMPLETION_BONUS == 0.10

    def test_all_weights_positive(self):
        assert REWARD_WEIGHT_REACHING > 0
        assert REWARD_WEIGHT_GRASPING > 0
        assert REWARD_WEIGHT_TASK_OBJECTIVE > 0
        assert REWARD_WEIGHT_COMPLETION_BONUS > 0


class TestComputeNormalizedReward:
    def test_all_zero_returns_zero(self):
        reward = compute_normalized_reward(
            reach_progress=0.0,
            is_grasped=False,
            task_progress=0.0,
            is_complete=False,
        )
        assert reward == pytest.approx(0.0)

    def test_all_max_returns_one(self):
        reward = compute_normalized_reward(
            reach_progress=1.0,
            is_grasped=True,
            task_progress=1.0,
            is_complete=True,
        )
        assert reward == pytest.approx(1.0)

    def test_only_reaching(self):
        reward = compute_normalized_reward(
            reach_progress=1.0,
            is_grasped=False,
            task_progress=0.0,
            is_complete=False,
        )
        assert reward == pytest.approx(REWARD_WEIGHT_REACHING)

    def test_only_grasped(self):
        reward = compute_normalized_reward(
            reach_progress=0.0,
            is_grasped=True,
            task_progress=0.0,
            is_complete=False,
        )
        assert reward == pytest.approx(REWARD_WEIGHT_GRASPING)

    def test_only_task_progress(self):
        reward = compute_normalized_reward(
            reach_progress=0.0,
            is_grasped=False,
            task_progress=1.0,
            is_complete=False,
        )
        assert reward == pytest.approx(REWARD_WEIGHT_TASK_OBJECTIVE)

    def test_only_complete(self):
        reward = compute_normalized_reward(
            reach_progress=0.0,
            is_grasped=False,
            task_progress=0.0,
            is_complete=True,
        )
        assert reward == pytest.approx(REWARD_WEIGHT_COMPLETION_BONUS)

    def test_partial_reaching(self):
        reward = compute_normalized_reward(
            reach_progress=0.5,
            is_grasped=False,
            task_progress=0.0,
            is_complete=False,
        )
        assert reward == pytest.approx(0.5 * REWARD_WEIGHT_REACHING)

    def test_partial_task_progress(self):
        reward = compute_normalized_reward(
            reach_progress=0.0,
            is_grasped=False,
            task_progress=0.5,
            is_complete=False,
        )
        assert reward == pytest.approx(0.5 * REWARD_WEIGHT_TASK_OBJECTIVE)

    def test_reach_plus_grasp_equals_half(self):
        reward = compute_normalized_reward(
            reach_progress=1.0,
            is_grasped=True,
            task_progress=0.0,
            is_complete=False,
        )
        assert reward == pytest.approx(0.5)

    def test_reward_always_in_zero_one_range(self):
        import itertools

        for rp, ig, tp, ic in itertools.product(
            [0.0, 0.5, 1.0], [False, True], [0.0, 0.5, 1.0], [False, True]
        ):
            reward = compute_normalized_reward(rp, ig, tp, ic)
            assert 0.0 <= reward <= 1.0, f"Out of range: {reward} for ({rp}, {ig}, {tp}, {ic})"

    def test_return_type_is_float(self):
        reward = compute_normalized_reward(0.5, True, 0.5, False)
        assert isinstance(reward, float)
