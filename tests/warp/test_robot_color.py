"""Robot material colors in the shared Warp scene model."""

import numpy as np
import pytest

pytest.importorskip("mujoco_warp")

from so101_nexus import TouchConfig
from so101_nexus.constants import COLOR_MAP


@pytest.mark.parametrize("colors", ["black", ["blue", "red"]])
def test_warp_robot_materials_use_configured_color(env_factory, colors):
    env = env_factory(backend="warp", config=TouchConfig(robot_colors=colors))
    model = env.unwrapped.mjm
    expected = COLOR_MAP[colors if isinstance(colors, str) else colors[0]]
    np.testing.assert_allclose(model.material("upper_arm_so101_v1_material").rgba, expected)
    np.testing.assert_allclose(model.material("sts3215_03a_v1_material").rgba, [0.1, 0.1, 0.1, 1])
