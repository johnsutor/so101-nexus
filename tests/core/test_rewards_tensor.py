"""Scalar/numpy/torch parity for the tensor-agnostic reach reward formulas."""

import numpy as np
import pytest

from so101_nexus.rewards import reach_progress, simple_reward


def test_reach_progress_numpy_batch_matches_scalar():
    distances = np.array([-0.5, 0.0, 0.05, 0.3, 1.7])
    batched = reach_progress(distances, scale=5.0)
    scalar = np.array([reach_progress(float(d), scale=5.0) for d in distances])
    assert isinstance(batched, np.ndarray)
    np.testing.assert_allclose(batched, scalar, rtol=1e-12, atol=0.0)
    assert batched[0] == pytest.approx(1.0)  # negative clamped


def test_reach_progress_torch_matches_scalar_and_preserves_dtype():
    torch = pytest.importorskip("torch")
    distances = [0.0, 0.05, 0.3, 1.7]
    out = reach_progress(torch.tensor(distances, dtype=torch.float32), scale=5.0)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    scalar = [reach_progress(d, scale=5.0) for d in distances]
    np.testing.assert_allclose(out.numpy(), scalar, rtol=1e-6, atol=1e-7)


def test_simple_reward_torch_batch_with_bool_success():
    torch = pytest.importorskip("torch")
    out = simple_reward(
        progress=torch.tensor([0.5, 1.0]),
        completion_bonus=0.1,
        success=torch.tensor([False, True]),
    )
    assert isinstance(out, torch.Tensor)
    assert out[0].item() == pytest.approx(0.9 * 0.5)
    assert out[1].item() == pytest.approx(0.9 * 1.0 + 0.1)


def test_scalar_paths_still_return_plain_float():
    assert isinstance(reach_progress(0.5, scale=5.0), float)
    assert isinstance(simple_reward(progress=0.5, completion_bonus=0.1, success=True), float)
    assert reach_progress(0.0, scale=5.0) == pytest.approx(1.0)
    assert simple_reward(progress=0.4, completion_bonus=0.1, success=False) == pytest.approx(
        0.9 * 0.4
    )


def test_orientation_progress_numpy_and_torch_match_scalar():
    from so101_nexus.rewards import orientation_progress

    vals = [-2.0, -1.0, -0.3, 0.0, 0.5, 1.0, 2.0]
    scalar = [orientation_progress(v) for v in vals]
    np.testing.assert_allclose(orientation_progress(np.array(vals)), scalar, rtol=1e-12)
    assert isinstance(orientation_progress(0.3), float)
    torch = pytest.importorskip("torch")
    out = orientation_progress(torch.tensor(vals, dtype=torch.float64))
    np.testing.assert_allclose(out.numpy(), scalar, rtol=1e-12)


def test_lift_progress_numpy_and_torch_match_scalar():
    from so101_nexus.rewards import lift_progress

    heights = [-0.02, 0.0, 0.03, 0.1]
    grasped = [True, True, False, True]
    scalar = [lift_progress(h, scale=5.0, grasped=g) for h, g in zip(heights, grasped, strict=True)]
    out_np = lift_progress(np.array(heights), scale=5.0, grasped=np.array(grasped))
    np.testing.assert_allclose(out_np, scalar, rtol=1e-12)
    assert isinstance(lift_progress(0.05, scale=5.0, grasped=True), float)
    assert lift_progress(0.05, scale=5.0, grasped=False) == 0.0
    torch = pytest.importorskip("torch")
    out_t = lift_progress(
        torch.tensor(heights, dtype=torch.float64),
        scale=5.0,
        grasped=torch.tensor(grasped),
    )
    np.testing.assert_allclose(out_t.numpy(), scalar, rtol=1e-6)
