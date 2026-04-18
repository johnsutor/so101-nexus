"""Property-based tests for so101_nexus_core.rewards."""

from __future__ import annotations

import math

from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus_core.rewards import orientation_progress, reach_progress, simple_reward

finite_float = st.floats(allow_nan=False, allow_infinity=False)


@given(
    distance=finite_float,
    scale=st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_reach_progress_bounded_in_unit_interval(distance, scale):
    value = reach_progress(distance, scale=scale)
    assert 0.0 <= value <= 1.0
    assert math.isfinite(value)


@given(scale=st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_reach_progress_one_at_zero_distance(scale):
    assert reach_progress(0.0, scale=scale) == 1.0


@given(
    d1=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    d2=st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
    scale=st.floats(min_value=1e-3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_reach_progress_monotonic(d1, d2, scale):
    """Strictly smaller distance never produces a smaller reward."""
    if d1 <= d2:
        assert reach_progress(d1, scale=scale) >= reach_progress(d2, scale=scale) - 1e-12


@given(cos=st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=200)
def test_orientation_progress_bounded(cos):
    value = orientation_progress(cos)
    assert 0.0 <= value <= 1.0


@given(
    progress=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    bonus=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    success=st.booleans(),
)
@settings(max_examples=200)
def test_simple_reward_bounded(progress, bonus, success):
    reward = simple_reward(progress=progress, completion_bonus=bonus, success=success)
    assert 0.0 <= reward <= 1.0
    assert math.isfinite(reward)


@given(
    progress=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    bonus=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_simple_reward_success_dominates(progress, bonus):
    """Success must never reduce the reward."""
    won = simple_reward(progress=progress, completion_bonus=bonus, success=True)
    lost = simple_reward(progress=progress, completion_bonus=bonus, success=False)
    assert won >= lost - 1e-12
