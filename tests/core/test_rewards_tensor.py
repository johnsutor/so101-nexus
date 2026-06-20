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
