"""Pure config-validation tests for PickAndPlaceConfig.

These don't require MuJoCo to run — they exercise argument validation and
shared-constants propagation on ``PickAndPlaceConfig`` / ``PickAndPlaceEnv``.
Kept separate from ``test_envs.py`` so the fast-config tests don't run under
xvfb in CI.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus_mujoco  # noqa: F401
from so101_nexus_core.config import PickAndPlaceConfig
from so101_nexus_mujoco.pick_and_place import PickAndPlaceEnv

_CFG = PickAndPlaceConfig()


class TestConstructionValidation:
    def test_invalid_cube_colors(self):
        with pytest.raises(ValueError, match="cube_colors"):
            PickAndPlaceConfig(cube_colors="neon")

    def test_invalid_target_colors(self):
        with pytest.raises(ValueError, match="target_colors"):
            PickAndPlaceConfig(target_colors="neon")

    def test_same_cube_and_target_color_warns(self):
        with pytest.warns(UserWarning, match="overlap"):
            PickAndPlaceConfig(cube_colors="red", target_colors="red")

    def test_invalid_cube_half_size(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickAndPlaceConfig(cube_half_size=0.001)


class TestSharedConstants:
    def test_default_cube_half_size_matches_core(self):
        env = PickAndPlaceEnv()
        assert env.cube_half_size == _CFG.cube_half_size
        env.close()

    def test_disc_radius_matches_core(self):
        env = PickAndPlaceEnv()
        assert env.target_disc_radius == _CFG.target_disc_radius
        env.close()


class TestGoalThreshConfig:
    def test_goal_thresh_from_config(self):
        env = PickAndPlaceEnv()
        assert env.config.goal_thresh == _CFG.goal_thresh
        env.close()


class TestRobotInitQposNoise:
    def test_noise_param_exists(self):
        env = PickAndPlaceEnv(robot_init_qpos_noise=0.05)
        assert env.robot_init_qpos_noise == 0.05
        env.close()

    def test_noise_produces_different_qpos(self):
        import numpy as np

        env = PickAndPlaceEnv(robot_init_qpos_noise=0.02)
        qpos_list = []
        for seed in range(5):
            env.reset(seed=seed)
            qpos_list.append(env._get_current_qpos().copy())
        env.close()
        all_same = all(np.allclose(qpos_list[0], q) for q in qpos_list[1:])
        assert not all_same
