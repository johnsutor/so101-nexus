"""Helper for skipping tests when the vectorized (GPU PhysX) runtime is unavailable."""

from __future__ import annotations

# Lowercase substrings identifying a genuine GPU / vectorized-runtime
# availability failure (matched case-insensitively). Only these become skips;
# any other construction error must propagate so it fails the test instead of
# being silently masked. "failed to find device" covers SAPIEN's no-CUDA-device
# error on CPU-only CI runners (RuntimeError: failed to find device "cuda").
GPU_UNAVAILABLE_MARKERS = (
    "gpu physx can only be enabled once",
    "cuda",
    "out of memory",
    "no cuda-capable device",
    "found no nvidia",
    "failed to find device",
)


def skip_if_vectorized_runtime_unavailable(exc: Exception) -> None:
    """``pytest.skip`` on a known GPU/vectorized-runtime error; re-raise otherwise.

    Use inside an ``except Exception as exc:`` block around vectorized
    (``num_envs > 1``) env construction so genuine construction errors (a bad
    link name, a failed patch) surface as failures rather than being hidden as
    availability skips. Matching is case-insensitive.

    Parameters
    ----------
    exc : Exception
        The exception raised while constructing the vectorized environment.
    """
    import pytest

    message = str(exc).lower()
    if any(marker in message for marker in GPU_UNAVAILABLE_MARKERS):
        pytest.skip(f"Vectorized (GPU PhysX) runtime unavailable: {exc}")
    raise exc
