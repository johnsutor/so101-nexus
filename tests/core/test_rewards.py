import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus.rewards import orientation_progress, reach_progress, simple_reward

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
