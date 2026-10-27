import math

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from so101_nexus.rewards import (
    cube_stack_offset_ok,
    object_static_ok,
    orientation_progress,
    reach_progress,
    simple_reward,
)

finite_float = st.floats(allow_nan=False, allow_infinity=False)
positive_scale = st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False)
unit_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(d1=finite_float, d2=finite_float, scale=positive_scale)
@example(d1=-0.5, d2=0.0, scale=5.0)
@example(d1=0.05, d2=0.3, scale=5.0)
@settings(max_examples=200, derandomize=True)
def test_reach_progress_formula_bounds_and_monotonicity(d1, d2, scale):
    near, far = sorted((d1, d2))
    values = [reach_progress(distance, scale=scale) for distance in (near, far)]
    for distance, value in zip((near, far), values, strict=True):
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0
        assert value == pytest.approx(1.0 - math.tanh(scale * max(0.0, distance)))
        if distance <= 0.0:
            assert value == 1.0
    assert values[0] >= values[1]


@given(cos_similarity=finite_float)
@settings(max_examples=200, derandomize=True)
def test_orientation_progress_matches_clamped_linear_formula(cos_similarity):
    expected = (max(-1.0, min(1.0, cos_similarity)) + 1.0) / 2.0
    value = orientation_progress(cos_similarity)
    assert value == pytest.approx(expected)
    assert 0.0 <= value <= 1.0


@given(progress=unit_float, completion_bonus=unit_float)
@example(progress=0.4, completion_bonus=0.1)
@settings(max_examples=200, derandomize=True)
def test_simple_reward_formula_and_success_dominance(progress, completion_bonus):
    lost = simple_reward(progress=progress, completion_bonus=completion_bonus, success=False)
    won = simple_reward(progress=progress, completion_bonus=completion_bonus, success=True)
    assert lost == pytest.approx((1.0 - completion_bonus) * progress)
    assert won == pytest.approx(1.0)
    for reward in (lost, won):
        assert isinstance(reward, float)
        assert math.isfinite(reward)
        assert 0.0 <= reward <= 1.0
    assert lost <= 1.0 - completion_bonus
    assert won >= lost


@given(
    lin=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    ang=st.floats(min_value=0.0, max_value=10.0, allow_nan=False),
    lin_thresh=unit_float,
    ang_thresh=unit_float,
)
@example(lin=0.0, ang=0.0, lin_thresh=0.01, ang_thresh=0.5)
@example(lin=0.009, ang=0.49, lin_thresh=0.01, ang_thresh=0.5)
@example(lin=0.01, ang=0.5, lin_thresh=0.01, ang_thresh=0.5)
@example(lin=0.011, ang=0.0, lin_thresh=0.01, ang_thresh=0.5)
@example(lin=0.0, ang=0.51, lin_thresh=0.01, ang_thresh=0.5)
@settings(derandomize=True)
def test_object_static_ok_thresholds(lin, ang, lin_thresh, ang_thresh):
    actual = object_static_ok(lin, ang, lin_threshold=lin_thresh, ang_threshold=ang_thresh)
    assert isinstance(actual, bool)
    assert actual == ((lin <= lin_thresh) and (ang <= ang_thresh))


@given(
    dx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    dy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    dz=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False),
    half=st.floats(min_value=0.005, max_value=0.05, allow_nan=False),
    margin=st.floats(min_value=0.0, max_value=0.02, allow_nan=False),
)
@example(dx=0.0, dy=0.0, dz=0.025, half=0.0125, margin=0.005)
@example(dx=math.sqrt(2) * 0.0125 + 0.005 - 1e-4, dy=0.0, dz=0.0299, half=0.0125, margin=0.005)
@example(dx=math.sqrt(2) * 0.0125 + 0.006, dy=0.0, dz=0.025, half=0.0125, margin=0.005)
@example(dx=0.0, dy=0.0, dz=0.0, half=0.0125, margin=0.005)
@example(dx=0.0, dy=0.0, dz=0.05, half=0.0125, margin=0.005)
@example(dx=0.0, dy=math.sqrt(2) * 0.0125 + 0.005, dz=0.025, half=0.0125, margin=0.005)
@example(dx=0.0, dy=0.0, dz=0.625, half=0.25, margin=0.125)
@example(dx=0.0, dy=0.0, dz=0.375, half=0.25, margin=0.125)
@settings(max_examples=200, derandomize=True)
def test_cube_stack_offset_ok_geometry_and_boundaries(dx, dy, dz, half, margin):
    expected = (math.hypot(dx, dy) <= math.sqrt(2) * half + margin) and (
        abs(dz - 2 * half) <= margin
    )
    actual = cube_stack_offset_ok(dx, dy, dz, cube_half_size=half, margin=margin)
    assert isinstance(actual, bool)
    assert actual == expected
