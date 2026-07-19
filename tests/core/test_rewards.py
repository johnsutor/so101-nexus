import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus.rewards import (
    cube_stack_offset_ok,
    cube_static_ok,
    orientation_progress,
    reach_progress,
    simple_reward,
)

finite_float = st.floats(allow_nan=False, allow_infinity=False)
positive_scale = st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False)
unit_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


class TestReachProgress:
    @given(
        distance=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        scale=positive_scale,
    )
    @settings(max_examples=200)
    def test_matches_tanh_formula_for_nonnegative_distance(self, distance, scale):
        assert reach_progress(distance, scale=scale) == pytest.approx(
            1.0 - math.tanh(scale * distance)
        )

    @given(
        distance=st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        scale=positive_scale,
    )
    @settings(max_examples=200)
    def test_nonpositive_distance_clamped_to_zero(self, distance, scale):
        assert reach_progress(distance, scale=scale) == pytest.approx(1.0)

    @given(distance=finite_float, scale=positive_scale)
    @settings(max_examples=100)
    def test_scalar_path_returns_float(self, distance, scale):
        assert isinstance(reach_progress(distance, scale=scale), float)


class TestOrientationProgress:
    @given(cos_similarity=finite_float)
    @settings(max_examples=200)
    def test_matches_clamped_linear_formula(self, cos_similarity):
        expected = (max(-1.0, min(1.0, cos_similarity)) + 1.0) / 2.0
        assert orientation_progress(cos_similarity) == pytest.approx(expected)


class TestSimpleReward:
    """Reach/orient/move reward: shaped in [0, 1-bonus], success clamps to 1.0."""

    @given(progress=unit_float, completion_bonus=unit_float, success=st.booleans())
    @settings(max_examples=200)
    def test_matches_clamped_completion_formula(self, progress, completion_bonus, success):
        reward = simple_reward(
            progress=progress, completion_bonus=completion_bonus, success=success
        )
        shaped = (1.0 - completion_bonus) * progress
        expected = shaped + (1.0 - shaped) * success
        assert reward == pytest.approx(expected)

    @given(progress=unit_float, completion_bonus=unit_float)
    @settings(max_examples=200)
    def test_success_yields_full_budget_and_dominates(self, progress, completion_bonus):
        """Success clamps to the full budget (global max); completion_bonus is the margin."""
        won = simple_reward(progress=progress, completion_bonus=completion_bonus, success=True)
        lost = simple_reward(progress=progress, completion_bonus=completion_bonus, success=False)
        assert won == pytest.approx(1.0)
        assert lost <= 1.0 - completion_bonus + 1e-9
        assert won >= lost - 1e-9


class TestCubeStaticOk:
    """Mirrors ManiSkill StackCubeEnv.evaluate's is_cubeA_static check."""

    def test_below_both_thresholds_is_static(self):
        assert cube_static_ok(0.0, 0.0, lin_threshold=0.01, ang_threshold=0.5)
        assert cube_static_ok(0.009, 0.49, lin_threshold=0.01, ang_threshold=0.5)

    def test_at_threshold_is_static(self):
        assert cube_static_ok(0.01, 0.5, lin_threshold=0.01, ang_threshold=0.5)

    def test_above_lin_threshold_is_not_static(self):
        assert not cube_static_ok(0.011, 0.0, lin_threshold=0.01, ang_threshold=0.5)

    def test_above_ang_threshold_is_not_static(self):
        assert not cube_static_ok(0.0, 0.51, lin_threshold=0.01, ang_threshold=0.5)

    @given(
        lin=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        ang=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
        lin_thresh=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        ang_thresh=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    )
    def test_matches_threshold_formula(self, lin, ang, lin_thresh, ang_thresh):
        expected = (lin <= lin_thresh) and (ang <= ang_thresh)
        assert (
            bool(cube_static_ok(lin, ang, lin_threshold=lin_thresh, ang_threshold=ang_thresh))
            == expected
        )


class TestCubeStackOffsetOk:
    """Mirrors ManiSkill StackCubeEnv.evaluate's xy_flag / z_flag check."""

    def test_exact_stack_pose_is_ok(self):
        # Cube A centred directly above cube B by exactly 2 * half_size.
        assert cube_stack_offset_ok(0.0, 0.0, 0.025, cube_half_size=0.0125, margin=0.005)

    def test_within_margin_is_ok(self):
        half = 0.0125
        margin = 0.005
        xy_edge = math.sqrt(2) * half + margin - 1e-4
        assert cube_stack_offset_ok(
            xy_edge, 0.0, 2 * half + margin - 1e-4, cube_half_size=half, margin=margin
        )

    def test_xy_offset_beyond_margin_fails(self):
        half = 0.0125
        margin = 0.005
        xy_edge = math.sqrt(2) * half + margin + 1e-3
        assert not cube_stack_offset_ok(xy_edge, 0.0, 2 * half, cube_half_size=half, margin=margin)

    def test_z_offset_beyond_margin_fails(self):
        half = 0.0125
        margin = 0.005
        # cube A sitting on the floor beside cube B, not stacked on top.
        assert not cube_stack_offset_ok(0.0, 0.0, 0.0, cube_half_size=half, margin=margin)

    def test_too_high_fails(self):
        half = 0.0125
        margin = 0.005
        assert not cube_stack_offset_ok(0.0, 0.0, 4 * half, cube_half_size=half, margin=margin)

    @given(
        dx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        dy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        dz=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
        half=st.floats(min_value=0.005, max_value=0.05, allow_nan=False),
        margin=st.floats(min_value=0.0, max_value=0.02, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_matches_reference_xy_and_z_flags(self, dx, dy, dz, half, margin):
        expected = (math.hypot(dx, dy) <= math.sqrt(2) * half + margin) and (
            abs(dz - 2 * half) <= margin
        )
        assert (
            bool(cube_stack_offset_ok(dx, dy, dz, cube_half_size=half, margin=margin)) == expected
        )
