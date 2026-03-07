import pytest

from so101_nexus_core.config import (
    EnvironmentConfig,
    PickAndPlaceConfig,
    PickCubeConfig,
    PickYCBConfig,
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

    def test_pick_cube_has_cube_colors(self):
        cfg = PickCubeConfig()
        assert cfg.cube_colors == "red"

    def test_pick_cube_custom_color(self):
        cfg = PickCubeConfig(cube_colors="green")
        assert cfg.cube_colors == "green"

    def test_pick_and_place_has_cube_colors(self):
        cfg = PickAndPlaceConfig()
        assert cfg.cube_colors == "red"

    def test_pick_and_place_has_target_colors(self):
        cfg = PickAndPlaceConfig()
        assert cfg.target_colors == "blue"

    def test_pick_ycb_has_model_id(self):
        cfg = PickYCBConfig()
        assert cfg.model_id == "058_golf_ball"

    def test_pick_ycb_custom_model(self):
        cfg = PickYCBConfig(model_id="011_banana")
        assert cfg.model_id == "011_banana"

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

    def test_pick_cube_list_colors(self):
        cfg = PickCubeConfig(cube_colors=["red", "blue"])
        assert cfg.cube_colors == ["red", "blue"]


class TestConfigValidation:
    def test_invalid_cube_colors_pick_cube(self):
        with pytest.raises(ValueError, match="cube_colors"):
            PickCubeConfig(cube_colors="neon")

    def test_invalid_cube_colors_pick_and_place(self):
        with pytest.raises(ValueError, match="cube_colors"):
            PickAndPlaceConfig(cube_colors="neon")

    def test_invalid_target_colors(self):
        with pytest.raises(ValueError, match="target_colors"):
            PickAndPlaceConfig(target_colors="neon")

    def test_same_cube_and_target_color_warns(self):
        with pytest.warns(UserWarning, match="overlap"):
            PickAndPlaceConfig(cube_colors="red", target_colors="red")

    def test_invalid_model_id(self):
        with pytest.raises(ValueError, match="model_id"):
            PickYCBConfig(model_id="invalid_object")

    def test_invalid_camera_mode(self):
        with pytest.raises(ValueError, match="camera_mode"):
            EnvironmentConfig(camera_mode="overhead")

    def test_invalid_cube_half_size_too_small(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickCubeConfig(cube_half_size=0.001)

    def test_invalid_cube_half_size_too_large(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickCubeConfig(cube_half_size=0.1)

    def test_invalid_color_in_list(self):
        with pytest.raises(ValueError, match="cube_colors"):
            PickCubeConfig(cube_colors=["red", "neon"])

    def test_invalid_ground_color(self):
        with pytest.raises(ValueError, match="ground_colors"):
            EnvironmentConfig(ground_colors="magenta")

    def test_invalid_robot_color(self):
        with pytest.raises(ValueError, match="robot_colors"):
            EnvironmentConfig(robot_colors="magenta")
