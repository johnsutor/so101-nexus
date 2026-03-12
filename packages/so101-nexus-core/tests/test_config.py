import numpy as np
import pytest

from so101_nexus_core.config import (
    COLOR_MAP,
    SO101_JOINT_NAMES,
    YCB_ENV_NAME_MAP,
    YCB_OBJECTS,
    CameraConfig,
    EnvironmentConfig,
    PickAndPlaceConfig,
    PickCubeConfig,
    PickYCBConfig,
    RewardConfig,
    RobotConfig,
    sample_color,
)


class TestConfigInheritance:
    def test_pick_cube_inherits_base_defaults(self):
        cfg = PickCubeConfig(cube_half_size=0.02, max_episode_steps=512)
        assert cfg.cube_half_size == 0.02
        assert cfg.max_episode_steps == 512
        assert cfg.goal_thresh == 0.025

    def test_pick_and_place_inherits_base_defaults(self):
        cfg = PickAndPlaceConfig()
        assert cfg.goal_thresh == 0.025
        assert cfg.spawn_half_size == 0.05

    def test_pick_ycb_inherits_base_defaults(self):
        cfg = PickYCBConfig()
        assert cfg.goal_thresh == 0.025

    def test_custom_camera(self):
        cfg = PickCubeConfig(camera=CameraConfig(width=128, height=128))
        assert cfg.camera.width == 128
        assert cfg.camera.height == 128

    def test_custom_reward(self):
        cfg = PickCubeConfig(reward=RewardConfig(action_delta_penalty=0.01))
        assert cfg.reward.action_delta_penalty == 0.01


class TestConfigConsistency:
    def test_spawn_defaults_shape(self):
        cfg = EnvironmentConfig()
        assert len(cfg.spawn_center) == 2
        assert cfg.spawn_half_size > 0

    def test_camera_defaults_valid(self):
        cfg = CameraConfig()
        assert cfg.width > 0
        assert cfg.height > 0

    def test_fov_deg_rad_consistent(self):
        cfg = CameraConfig()
        deg_lo, deg_hi = cfg.wrist_fov_deg_range
        rad_lo, rad_hi = cfg.wrist_fov_rad_range
        assert rad_lo == pytest.approx(np.radians(deg_lo))
        assert rad_hi == pytest.approx(np.radians(deg_hi))

    def test_default_max_episode_steps_is_1024(self):
        cfg = EnvironmentConfig()
        assert cfg.max_episode_steps == 1024


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


class TestYCBMappings:
    def test_ycb_env_map_has_same_keys_as_objects(self):
        assert set(YCB_ENV_NAME_MAP.keys()) == set(YCB_OBJECTS.keys())

    def test_ycb_labels_are_non_empty(self):
        assert all(v for v in YCB_OBJECTS.values())
        assert all(v for v in YCB_ENV_NAME_MAP.values())


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


class TestCameraConfigWristParams:
    def test_camera_config_wrist_pitch_defaults(self):
        cam = CameraConfig()
        lo, hi = cam.wrist_pitch_deg_range
        assert lo == pytest.approx(-34.4, abs=0.1)
        assert hi == pytest.approx(0.0)

    def test_camera_config_wrist_pitch_rad_range(self):
        cam = CameraConfig()
        lo_rad, hi_rad = cam.wrist_pitch_rad_range
        assert lo_rad == pytest.approx(np.radians(-34.4), abs=0.001)
        assert hi_rad == pytest.approx(0.0)

    def test_camera_config_wrist_cam_pos_defaults(self):
        cam = CameraConfig()
        assert cam.wrist_cam_pos_x_noise == pytest.approx(0.005)
        assert cam.wrist_cam_pos_y_center == pytest.approx(0.04)
        assert cam.wrist_cam_pos_y_noise == pytest.approx(0.01)
        assert cam.wrist_cam_pos_z_center == pytest.approx(-0.04)
        assert cam.wrist_cam_pos_z_noise == pytest.approx(0.01)


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
