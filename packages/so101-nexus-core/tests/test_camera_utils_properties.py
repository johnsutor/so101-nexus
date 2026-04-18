"""Property-based tests for camera parameter computation."""

from __future__ import annotations

import math

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from so101_nexus_core.camera_utils import (
    compute_angled_camera_params,
    compute_overhead_camera_params,
    compute_overhead_eye_target,
)

finite_small = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(
    cx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    cy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    radius=finite_small,
    margin=finite_small,
)
@settings(max_examples=200)
def test_overhead_distance_positive_and_finite(cx, cy, radius, margin):
    params = compute_overhead_camera_params(
        spawn_center=(cx, cy), spawn_max_radius=radius, margin=margin
    )
    d = params["distance"]
    assert math.isfinite(d)
    assert d > 0.0
    assert params["elevation"] == -90
    assert params["azimuth"] == 0


@given(
    cx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    cy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    radius=finite_small,
    margin=finite_small,
)
@settings(max_examples=200)
def test_overhead_eye_target_axes_consistent(cx, cy, radius, margin):
    eye, target = compute_overhead_eye_target(
        spawn_center=(cx, cy), spawn_max_radius=radius, margin=margin
    )
    # Eye is directly above target.
    assert eye[0] == target[0]
    assert eye[1] == target[1]
    assert eye[2] > target[2]
    assert target[2] == 0.0


@given(
    cx=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    cy=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    radius=finite_small,
    margin=finite_small,
    elev=st.floats(min_value=-89.0, max_value=0.0, allow_nan=False, allow_infinity=False),
    az=st.floats(min_value=0.0, max_value=360.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200)
def test_angled_camera_distance_bigger_than_overhead(cx, cy, radius, margin, elev, az):
    overhead = compute_overhead_camera_params(
        spawn_center=(cx, cy), spawn_max_radius=radius, margin=margin
    )
    angled = compute_angled_camera_params(
        spawn_center=(cx, cy),
        spawn_max_radius=radius,
        margin=margin,
        elevation=elev,
        azimuth=az,
    )
    assert angled["distance"] > overhead["distance"]
    assert np.array_equal(angled["lookat"], overhead["lookat"])
