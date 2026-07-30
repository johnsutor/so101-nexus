"""Pure config-validation tests for StackCubeConfig.

These don't require MuJoCo to run - they exercise argument validation and
shared-constants propagation on ``StackCubeConfig`` / ``StackCubeEnv``.
Kept separate from ``test_envs.py`` so the fast-config tests don't run under
xvfb in CI (mirrors ``test_pick_and_place_config.py``).
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus.mujoco  # noqa: F401
from so101_nexus.config import StackCubeConfig
from so101_nexus.mujoco.stack_cube import StackCubeEnv
from so101_nexus.objects import CubeObject

_CFG = StackCubeConfig()


class TestConstructionValidation:
    def test_invalid_cube_a_colors(self):
        with pytest.raises(ValueError, match="cube_a_colors"):
            StackCubeConfig(cube_a_colors="neon")

    def test_invalid_cube_b_colors(self):
        with pytest.raises(ValueError, match="cube_b_colors"):
            StackCubeConfig(cube_b_colors="neon")

    def test_same_cube_a_and_cube_b_color_warns(self):
        with pytest.warns(UserWarning, match="overlap"):
            StackCubeConfig(cube_a_colors="red", cube_b_colors="red")

    def test_empty_cube_a_colors(self):
        with pytest.raises(ValueError, match="cube_a_colors"):
            StackCubeConfig(cube_a_colors=[])

    def test_empty_cube_b_colors(self):
        with pytest.raises(ValueError, match="cube_b_colors"):
            StackCubeConfig(cube_b_colors=[])

    def test_disjoint_list_colors_do_not_warn(self, recwarn):
        StackCubeConfig(cube_a_colors=["red", "orange"], cube_b_colors=["blue", "green"])
        assert len(recwarn) == 0

    def test_overlapping_list_colors_warn(self):
        with pytest.warns(UserWarning, match="overlap"):
            StackCubeConfig(cube_a_colors=["red", "blue"], cube_b_colors=["blue", "green"])

    def test_invalid_cube_half_size(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            StackCubeConfig(cube_half_size=0.001)

    def test_invalid_cube_mass(self):
        with pytest.raises(ValueError, match="cube_mass"):
            StackCubeConfig(cube_mass=0.0)

    def test_invalid_min_cube_separation(self):
        with pytest.raises(ValueError, match="min_cube_separation"):
            StackCubeConfig(min_cube_separation=-0.01)

    def test_invalid_stack_alignment_margin(self):
        with pytest.raises(ValueError, match="stack_alignment_margin"):
            StackCubeConfig(stack_alignment_margin=-0.01)

    def test_invalid_cube_static_lin_threshold(self):
        with pytest.raises(ValueError, match="cube_static_lin_threshold"):
            StackCubeConfig(cube_static_lin_threshold=-0.01)

    def test_invalid_cube_static_ang_threshold(self):
        with pytest.raises(ValueError, match="cube_static_ang_threshold"):
            StackCubeConfig(cube_static_ang_threshold=-0.5)

    def test_cube_static_thresholds_match_maniskill_defaults(self):
        cfg = StackCubeConfig()
        assert cfg.cube_static_lin_threshold == 0.01
        assert cfg.cube_static_ang_threshold == 0.5


class TestDefaults:
    def test_default_colors_are_distinct(self):
        assert _CFG.cube_a_colors != _CFG.cube_b_colors

    def test_default_cube_a_color_is_red(self):
        assert _CFG.cube_a_colors == "red"

    def test_default_cube_b_color_is_blue(self):
        assert _CFG.cube_b_colors == "blue"

    def test_default_has_no_distractors(self):
        assert _CFG.n_distractors == 0

    def test_default_distractor_pool_avoids_target_colors(self):
        colors = {obj.color for obj in _CFG.distractors}
        assert colors.isdisjoint({_CFG.cube_a_colors, _CFG.cube_b_colors})

    def test_default_distractor_pool_matches_configured_cube_geometry(self):
        cfg = StackCubeConfig(cube_half_size=0.02, cube_mass=0.05)
        assert [(o.half_size, o.mass) for o in cfg.distractors] == [(0.02, 0.05)] * 3


class TestDistractorValidation:
    def test_negative_distractors_raises(self):
        with pytest.raises(ValueError, match="n_distractors must be >= 0"):
            StackCubeConfig(n_distractors=-1)

    def test_more_distractors_than_pool_raises(self):
        with pytest.raises(ValueError, match="distractors pool must have at least"):
            StackCubeConfig(distractors=[CubeObject(color="green")], n_distractors=2)

    def test_empty_distractor_pool_raises(self):
        with pytest.raises(ValueError, match="distractors must not be empty"):
            StackCubeConfig(distractors=[])

    def test_single_distractor_wrapped_in_list(self):
        cfg = StackCubeConfig(distractors=CubeObject(color="green"), n_distractors=1)
        assert len(cfg.distractors) == 1
        assert cfg.distractors[0].color == "green"

    def test_distractor_color_matching_target_warns(self):
        with pytest.warns(UserWarning, match="ambiguous"):
            StackCubeConfig(distractors=[CubeObject(color="red")], n_distractors=1)

    def test_unused_distractor_pool_does_not_warn(self, recwarn):
        StackCubeConfig(distractors=[CubeObject(color="red")], n_distractors=0)
        assert len(recwarn) == 0


class TestSharedConstants:
    def test_default_cube_half_size_matches_core(self):
        env = StackCubeEnv()
        assert env.cube_half_size == _CFG.cube_half_size
        env.close()


class TestGoalThreshConfig:
    def test_stack_alignment_margin_from_config(self):
        env = StackCubeEnv()
        assert env.config.stack_alignment_margin == _CFG.stack_alignment_margin
        env.close()


class TestRobotInitQposNoise:
    def test_noise_param_exists(self):
        env = StackCubeEnv(robot_init_qpos_noise=0.05)
        assert env.robot_init_qpos_noise == 0.05
        env.close()

    def test_noise_produces_different_qpos(self):
        import numpy as np

        env = StackCubeEnv(robot_init_qpos_noise=0.02)
        qpos_list = []
        for seed in range(5):
            env.reset(seed=seed)
            qpos_list.append(env._get_current_qpos().copy())
        env.close()
        all_same = all(np.allclose(qpos_list[0], q) for q in qpos_list[1:])
        assert not all_same
