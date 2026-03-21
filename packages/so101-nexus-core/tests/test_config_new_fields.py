import pytest

from so101_nexus_core.config import (
    EnvironmentConfig,
    PickAndPlaceConfig,
    PickConfig,
    RobotConfig,
)


class TestNewConfigFields:
    def test_environment_config_has_camera_mode(self):
        cfg = EnvironmentConfig()
        assert cfg.camera_mode == "fixed"

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

    def test_camera_mode_custom(self):
        cfg = EnvironmentConfig(camera_mode="wrist")
        assert cfg.camera_mode == "wrist"

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

    def test_invalid_camera_mode(self):
        with pytest.raises(ValueError, match="camera_mode"):
            EnvironmentConfig(camera_mode="overhead")

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
