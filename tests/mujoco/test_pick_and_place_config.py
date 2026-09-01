"""Pure config-validation tests for PickAndPlaceConfig.

These don't require MuJoCo to run - they exercise argument validation and
shared-constants propagation on ``PickAndPlaceConfig`` / ``PickAndPlaceEnv``.
Kept separate from ``test_envs.py`` so the fast-config tests don't run under
xvfb in CI.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import so101_nexus.mujoco  # noqa: F401
from so101_nexus.config import PickAndPlaceConfig
from so101_nexus.mujoco.pick_and_place import PickAndPlaceEnv
from so101_nexus.objects import CubeObject, YCBObject

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

    def test_combined_cube_size_inputs_raise(self):
        with pytest.raises(ValueError, match="either cube_half_size or cube_side_length_mm"):
            PickAndPlaceConfig(cube_half_size=0.0125, cube_side_length_mm=25.4)

    def test_negative_object_static_lin_threshold(self):
        with pytest.raises(ValueError, match="object_static_lin_threshold"):
            PickAndPlaceConfig(object_static_lin_threshold=-0.01)

    def test_negative_object_static_ang_threshold(self):
        with pytest.raises(ValueError, match="object_static_ang_threshold"):
            PickAndPlaceConfig(object_static_ang_threshold=-0.5)

    def test_negative_min_object_separation(self):
        with pytest.raises(ValueError, match="min_object_separation"):
            PickAndPlaceConfig(min_object_separation=-0.01)


class TestDistractorDefaults:
    def test_default_has_no_distractors(self):
        assert _CFG.n_distractors == 0

    def test_default_distractor_pool_avoids_carried_and_target_colors(self):
        colors = {obj.color for obj in _CFG.distractors}
        assert colors.isdisjoint({_CFG.cube_colors, _CFG.target_colors})

    def test_default_distractor_pool_matches_configured_cube_geometry(self):
        cfg = PickAndPlaceConfig(cube_half_size=0.02, cube_mass=0.05)
        assert [(o.half_size, o.mass) for o in cfg.distractors] == [(0.02, 0.05)] * 3


class TestDistractorValidation:
    def test_negative_distractors_raises(self):
        with pytest.raises(ValueError, match="n_distractors must be >= 0"):
            PickAndPlaceConfig(n_distractors=-1)

    def test_more_distractors_than_pool_raises(self):
        with pytest.raises(ValueError, match="distractors pool must have at least"):
            PickAndPlaceConfig(distractors=[CubeObject(color="green")], n_distractors=2)

    def test_empty_distractor_pool_raises(self):
        with pytest.raises(ValueError, match="distractors must not be empty"):
            PickAndPlaceConfig(distractors=[])

    def test_single_distractor_wrapped_in_list(self):
        cfg = PickAndPlaceConfig(distractors=CubeObject(color="green"), n_distractors=1)
        assert len(cfg.distractors) == 1
        assert cfg.distractors[0].color == "green"

    def test_distractor_color_matching_carried_cube_warns(self):
        with pytest.warns(UserWarning, match="ambiguous"):
            PickAndPlaceConfig(distractors=[CubeObject(color="red")], n_distractors=1)

    def test_distractor_color_matching_explicit_pool_warns(self):
        with pytest.warns(UserWarning, match="ambiguous"):
            PickAndPlaceConfig(
                objects=[CubeObject(color="green")],
                distractors=[CubeObject(color="green")],
                n_distractors=1,
            )

    def test_distractor_color_matching_disc_does_not_warn(self, recwarn):
        # The instruction names the goal as a circle, so a blue distractor cube
        # beside the blue disc is unambiguous.
        PickAndPlaceConfig(distractors=[CubeObject(color="blue")], n_distractors=1)
        assert len(recwarn) == 0

    def test_unused_distractor_pool_does_not_warn(self, recwarn):
        PickAndPlaceConfig(distractors=[CubeObject(color="red")], n_distractors=0)
        assert len(recwarn) == 0

    def test_non_cube_distractors_never_warn(self, recwarn):
        PickAndPlaceConfig(distractors=[YCBObject("009_gelatin_box")], n_distractors=1)
        assert len(recwarn) == 0


class TestSharedConstants:
    def test_default_cube_half_size_matches_core(self):
        env = PickAndPlaceEnv()
        assert env.cube_half_size == _CFG.cube_half_size
        env.close()

    def test_cube_side_length_reaches_environment(self):
        config = PickAndPlaceConfig(cube_side_length_mm=25.4)
        env = PickAndPlaceEnv(config=config)
        try:
            env.reset(seed=0)
            slot = env._slots[env._target_slot_idx]
            assert env.cube_half_size == pytest.approx(0.0127)
            assert [obj.half_size for obj in config.object_pool()] == pytest.approx([0.0127])
            assert env.model.geom_size[slot.geom_id] == pytest.approx([0.0127] * 3)
        finally:
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
