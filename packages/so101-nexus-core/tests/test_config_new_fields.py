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
