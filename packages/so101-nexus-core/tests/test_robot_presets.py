from __future__ import annotations

import numpy as np
import pytest

from so101_nexus_core.config import (
    ROBOT_CAMERA_PRESETS,
    EnvironmentConfig,
    PickAndPlaceConfig,
    PickConfig,
)
from so101_nexus_core.robot_presets import build_maniskill_robot_configs


@pytest.mark.parametrize(
    "config, expects_cube_half_size, expects_max_goal_height",
    [
        (EnvironmentConfig(), False, False),
        (PickAndPlaceConfig(), True, False),
        (PickConfig(), False, True),
    ],
)
def test_build_maniskill_robot_configs_shape(
    config: EnvironmentConfig,
    expects_cube_half_size: bool,
    expects_max_goal_height: bool,
):
    configs = build_maniskill_robot_configs(config=config)
    assert set(configs.keys()) == {"so100", "so101"}

    so100 = configs["so100"]
    assert so100["goal_thresh"] == config.goal_thresh
    assert so100["cube_spawn_half_size"] == config.spawn_half_size
    assert so100["cube_spawn_center"] == config.spawn_center
    assert so100["wrist_cam_fov_range"][0] < so100["wrist_cam_fov_range"][1]
    assert ("cube_half_size" in so100) is expects_cube_half_size
    assert ("max_goal_height" in so100) is expects_max_goal_height


def test_build_maniskill_robot_configs_converts_euler_deg_to_rad():
    configs = build_maniskill_robot_configs(config=EnvironmentConfig())
    so100 = configs["so100"]
    preset = ROBOT_CAMERA_PRESETS["so100"]
    assert np.array(so100["wrist_cam_euler_center"]) == pytest.approx(
        np.radians(np.array(preset.wrist_cam_euler_center_deg))
    )
    assert np.array(so100["wrist_cam_euler_noise"]) == pytest.approx(
        np.radians(np.array(preset.wrist_cam_euler_noise_deg))
    )
