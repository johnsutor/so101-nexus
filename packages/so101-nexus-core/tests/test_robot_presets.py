from __future__ import annotations

import pytest

from so101_nexus_core.config import (
    EnvironmentConfig,
    PickAndPlaceConfig,
    PickCubeConfig,
    PickYCBConfig,
)
from so101_nexus_core.robot_presets import build_maniskill_robot_configs


@pytest.mark.parametrize(
    "config, expects_cube_half_size, expects_max_goal_height",
    [
        (EnvironmentConfig(), False, False),
        (PickAndPlaceConfig(), True, False),
        (PickCubeConfig(), True, True),
        (PickYCBConfig(), False, True),
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
