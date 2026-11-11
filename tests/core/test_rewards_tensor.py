"""Scalar/NumPy/torch parity for the tensor-agnostic reward formulas."""

import math

import numpy as np
import pytest

from so101_nexus.rewards import (
    cube_stack_offset_ok,
    lift_progress,
    object_static_ok,
    orientation_progress,
    pick_return_reach_potential,
    pick_return_success,
    pick_return_task_potential,
    reach_progress,
    simple_reward,
)


@pytest.fixture(params=["numpy", "torch_float32", "torch_float64"])
def array(request):
    if request.param == "numpy":
        return np.asarray
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, request.param.removeprefix("torch_"))

    def make(values):
        tensor = torch.from_numpy(np.asarray(values))
        return tensor.to(dtype) if tensor.is_floating_point() else tensor

    return make


def test_object_static_ok_batch_matches_scalar(array):
    lin = [0.0, 0.005, 0.02, 0.0]
    ang = [0.0, 0.6, 0.0, 0.49]
    expected = [
        object_static_ok(lin_v, ang_v, lin_threshold=0.01, ang_threshold=0.5)
        for lin_v, ang_v in zip(lin, ang, strict=True)
    ]
    actual = object_static_ok(array(lin), array(ang), lin_threshold=0.01, ang_threshold=0.5)
    np.testing.assert_array_equal(actual, expected)
    assert isinstance(object_static_ok(0.0, 0.0, lin_threshold=0.01, ang_threshold=0.5), bool)


def test_reach_progress_batch_matches_scalar_and_preserves_dtype(array):
    distances = array([-0.5, 0.0, 0.05, 0.3, 1.7])
    actual = reach_progress(distances, scale=5.0)
    expected = [reach_progress(float(distance), scale=5.0) for distance in distances]
    assert isinstance(actual, type(distances))
    assert actual.dtype == distances.dtype
    tolerance = 1e-12 if np.asarray(actual).dtype == np.float64 else 1e-6
    np.testing.assert_allclose(
        actual, expected, rtol=tolerance, atol=0.0 if tolerance == 1e-12 else 1e-7
    )
    assert actual[0] == pytest.approx(1.0)


def test_simple_reward_batch_with_bool_success(array):
    progress = array([0.5, 1.0])
    actual = simple_reward(progress=progress, completion_bonus=0.1, success=array([False, True]))
    assert isinstance(actual, type(progress))
    np.testing.assert_allclose(actual, [0.45, 1.0], rtol=1e-6)


def test_orientation_progress_batch_matches_scalar(array):
    values = [-2.0, -1.0, -0.3, 0.0, 0.5, 1.0, 2.0]
    expected = [orientation_progress(value) for value in values]
    actual = orientation_progress(array(values))
    tolerance = 1e-12 if np.asarray(actual).dtype == np.float64 else 1e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance)
    assert isinstance(orientation_progress(0.3), float)


def test_lift_progress_batch_matches_scalar(array):
    heights = [-0.02, 0.0, 0.03, 0.1]
    grasped = [True, True, False, True]
    expected = [
        lift_progress(h, scale=5.0, grasped=g) for h, g in zip(heights, grasped, strict=True)
    ]
    actual = lift_progress(array(heights), scale=5.0, grasped=array(grasped))
    tolerance = 1e-12 if np.asarray(actual).dtype == np.float64 else 1e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance)
    assert isinstance(lift_progress(0.05, scale=5.0, grasped=True), float)
    assert lift_progress(0.05, scale=5.0, grasped=False) == 0.0


def test_cube_stack_offset_ok_batch_matches_scalar(array):
    half, margin = 0.0125, 0.005
    dx = [0.0, 0.02, 0.0, 0.05]
    dy = [0.0, 0.0, 0.0, 0.0]
    dz = [2 * half, 2 * half, 0.0, 2 * half]
    expected = [
        cube_stack_offset_ok(x, y, z, cube_half_size=half, margin=margin)
        for x, y, z in zip(dx, dy, dz, strict=True)
    ]
    actual = cube_stack_offset_ok(
        array(dx), array(dy), array(dz), cube_half_size=half, margin=margin
    )
    np.testing.assert_array_equal(actual, expected)
    assert isinstance(
        cube_stack_offset_ok(0.0, 0.0, 2 * half, cube_half_size=half, margin=margin), bool
    )


@pytest.mark.parametrize("backend", ["scalar", "numpy", "torch"])
def test_potential_state_preserves_progress_regression_and_inputs(backend):
    from so101_nexus.rewards import _PotentialState

    current = (0.75, 0.25, 0.5)
    previous = (0.25, 0.75, 0.5)
    if backend == "numpy":
        current = tuple(np.array([value], dtype=np.float32) for value in current)
        previous = tuple(np.array([value], dtype=np.float32) for value in previous)
    elif backend == "torch":
        torch = pytest.importorskip("torch")
        current = tuple(torch.tensor([value], dtype=torch.float32) for value in current)
        previous = tuple(torch.tensor([value], dtype=torch.float32) for value in previous)
    state = _PotentialState(*previous)
    for actual, expected, now, before in zip(
        state._update_potentials(*current), (0.5, -0.5, 0.0), current, previous, strict=True
    ):
        assert (actual if backend == "scalar" else actual.item()) == expected
        if backend == "scalar":
            assert isinstance(actual, float)
        else:
            assert actual.dtype == now.dtype
            assert actual.shape == now.shape
            assert now.item() + before.item() == 1.0
    assert all(
        (delta if backend == "scalar" else delta.item()) == 0.0
        for delta in state._update_potentials(*current)
    )


@pytest.fixture(params=["scalar", "numpy", "torch"])
def value(request):
    def make(scalar):
        if request.param == "numpy":
            return np.array([scalar])
        if request.param == "torch":
            torch = pytest.importorskip("torch")
            return torch.tensor([scalar], dtype=torch.float64)
        return scalar

    return make


@pytest.mark.parametrize("grasped", [0.0, 0.5, 0.6], ids=["ungrasped", "threshold", "grasped"])
def test_pick_return_reach_potential_holds_credit_during_grasp(value, grasped):
    expected = 1.0 if grasped > 0.5 else 1.0 - math.tanh(1.0)
    actual = pick_return_reach_potential(value(0.2), value(grasped), scale=5.0)
    assert (actual.item() if hasattr(actual, "item") else actual) == pytest.approx(expected)


@pytest.mark.parametrize(
    "height,error,speed,grasped,expected",
    [
        pytest.param(
            0.025,
            90.0,
            1.0,
            1.0,
            math.tanh(0.025 / 0.08) / math.tanh(0.05 / 0.08) / 3.0,
            id="lifting",
        ),
        pytest.param(0.05, 5.0, 0.0, 1.0, 1.0 / 3.0, id="clearance-boundary"),
        pytest.param(0.06, 5.0, 0.1, 1.0, (3.0 - math.tanh(1.5)) / 3.0, id="settling"),
        pytest.param(0.06, 185.0, 0.0, 1.0, (2.0 - math.tanh(5.0)) / 3.0, id="returning"),
        pytest.param(0.06, 0.0, 0.0, 0.5, 0.0, id="grasp-boundary"),
    ],
)
def test_pick_return_task_potential_stages(value, height, error, speed, grasped, expected):
    actual = pick_return_task_potential(
        value(height),
        value(error),
        value(speed),
        value(grasped),
        lift_threshold=0.05,
        return_threshold_deg=5.0,
        scale=5.0,
        velocity_scale=15.0,
    )
    assert (actual.item() if hasattr(actual, "item") else actual) == pytest.approx(expected)


@pytest.mark.parametrize(
    "height,error,grasped,static,expected",
    [
        pytest.param(0.06, 5.0, 1.0, True, True, id="success"),
        pytest.param(0.05, 0.0, 1.0, True, False, id="lift-boundary"),
        pytest.param(0.06, 5.01, 1.0, True, False, id="outside-rest-tolerance"),
        pytest.param(0.06, 0.0, 0.5, True, False, id="grasp-boundary"),
        pytest.param(0.06, 0.0, 1.0, False, False, id="moving"),
    ],
)
def test_pick_return_success_requires_lifted_grasp_at_rest(
    value, height, error, grasped, static, expected
):
    actual = pick_return_success(
        value(height),
        value(error),
        value(grasped),
        value(static) > 0.5,
        lift_threshold=0.05,
        return_threshold_deg=5.0,
    )
    assert bool(actual) == expected
