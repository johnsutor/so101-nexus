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

    def test_environment_config_has_robot_color(self):
        cfg = EnvironmentConfig()
        assert cfg.robot_color is None

    def test_environment_config_has_robot_init_qpos_noise(self):
        cfg = EnvironmentConfig()
        assert cfg.robot_init_qpos_noise == 0.02

    def test_pick_cube_has_cube_color(self):
        cfg = PickCubeConfig()
        assert cfg.cube_color == "red"

    def test_pick_cube_custom_color(self):
        cfg = PickCubeConfig(cube_color="green")
        assert cfg.cube_color == "green"

    def test_pick_and_place_has_cube_color(self):
        cfg = PickAndPlaceConfig()
        assert cfg.cube_color == "red"

    def test_pick_and_place_has_target_color(self):
        cfg = PickAndPlaceConfig()
        assert cfg.target_color == "blue"

    def test_pick_ycb_has_model_id(self):
        cfg = PickYCBConfig()
        assert cfg.model_id == "058_golf_ball"

    def test_pick_ycb_custom_model(self):
        cfg = PickYCBConfig(model_id="011_banana")
        assert cfg.model_id == "011_banana"

    def test_camera_mode_custom(self):
        cfg = EnvironmentConfig(camera_mode="wrist")
        assert cfg.camera_mode == "wrist"

    def test_robot_color_custom(self):
        cfg = EnvironmentConfig(robot_color=(1.0, 0.0, 0.0, 1.0))
        assert cfg.robot_color == (1.0, 0.0, 0.0, 1.0)


class TestConfigValidation:
    def test_invalid_cube_color_pick_cube(self):
        with pytest.raises(ValueError, match="cube_color"):
            PickCubeConfig(cube_color="neon")

    def test_invalid_cube_color_pick_and_place(self):
        with pytest.raises(ValueError, match="cube_color"):
            PickAndPlaceConfig(cube_color="neon")

    def test_invalid_target_color(self):
        with pytest.raises(ValueError, match="target_color"):
            PickAndPlaceConfig(target_color="neon")

    def test_same_cube_and_target_color(self):
        with pytest.raises(ValueError, match="must differ"):
            PickAndPlaceConfig(cube_color="red", target_color="red")

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
