"""Helper for skipping tests when the vectorized (GPU PhysX) runtime is unavailable."""

from __future__ import annotations

# Substrings identifying a genuine GPU / vectorized-runtime availability failure.
# Only these become skips; any other construction error must propagate so it
# fails the test instead of being silently masked.
GPU_UNAVAILABLE_MARKERS = (
    "GPU PhysX can only be enabled once",
    "CUDA",
    "out of memory",
    "no CUDA-capable device",
    "Found no NVIDIA",
)


def skip_if_vectorized_runtime_unavailable(exc: Exception) -> None:
    """``pytest.skip`` on a known GPU/vectorized-runtime error; re-raise otherwise.

    Use inside an ``except Exception as exc:`` block around vectorized
    (``num_envs > 1``) env construction so genuine construction errors (a bad
    link name, a failed patch) surface as failures rather than being hidden as
    availability skips.

    Parameters
    ----------
    exc : Exception
        The exception raised while constructing the vectorized environment.
    """
    import pytest

    if any(marker in str(exc) for marker in GPU_UNAVAILABLE_MARKERS):
        pytest.skip(f"Vectorized (GPU PhysX) runtime unavailable: {exc}")
    raise exc
