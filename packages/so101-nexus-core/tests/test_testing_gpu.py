"""Unit tests for the shared GPU-availability skip helper (no GPU required)."""

from __future__ import annotations

import pytest

from so101_nexus_core.testing import skip_if_vectorized_runtime_unavailable

# Real messages observed from SAPIEN/ManiSkill when the vectorized (GPU PhysX)
# runtime is unavailable. Each must be treated as a skip, not a failure.
_GPU_MESSAGES = [
    'failed to find device "cuda"',  # CPU-only CI runner
    "GPU PhysX can only be enabled once before any other code involving PhysX",
    "CUDA error: an illegal memory access was encountered",
    "CUDA out of memory",
    "no CUDA-capable device is detected",
    "RuntimeError: Found no NVIDIA driver on your system",
]


@pytest.mark.parametrize("message", _GPU_MESSAGES)
def test_gpu_unavailable_messages_skip(message):
    with pytest.raises(pytest.skip.Exception):
        skip_if_vectorized_runtime_unavailable(RuntimeError(message))


@pytest.mark.parametrize(
    "message",
    [
        "KeyError: 'gripper_link'",  # a real bad-link-name bug must NOT be masked
        "AssertionError: gripper has no collision shapes",
        "ValueError: robot_uids must be one of ['so100', 'so101']",
    ],
)
def test_non_gpu_errors_propagate(message):
    err = RuntimeError(message)
    with pytest.raises(RuntimeError) as excinfo:
        skip_if_vectorized_runtime_unavailable(err)
    assert excinfo.value is err  # re-raised unchanged, not turned into a skip
