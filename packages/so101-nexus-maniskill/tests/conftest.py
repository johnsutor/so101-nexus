"""Shared fixtures for ManiSkill SO101-Nexus tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator

import so101_nexus_maniskill  # noqa: F401 - registers envs
from so101_nexus_core.config import ReachConfig

_BASE_KWARGS = {"obs_mode": "state", "render_mode": None}

# Markers for the SAPIEN single-process GPU-PhysX limitation and GPU
# unavailability. Only these are treated as "vectorized runtime unavailable"
# skips; any other construction error (e.g. a bad link name or a failed patch)
# must propagate so it fails the test instead of being silently skipped.
_GPU_UNAVAILABLE_MARKERS = (
    "GPU PhysX can only be enabled once",
    "CUDA",
    "out of memory",
    "no CUDA-capable device",
    "Found no NVIDIA",
)


def _skip_if_gpu_unavailable(exc: Exception) -> None:
    """Skip on a known GPU/vectorized-runtime availability error; else re-raise."""
    message = str(exc)
    if any(marker in message for marker in _GPU_UNAVAILABLE_MARKERS):
        pytest.skip(f"ManiSkill vectorized runtime unavailable: {exc}")
    raise exc


@pytest.fixture
def so101_reach_env() -> Iterator:
    """A single-env SO101 Reach environment (num_envs=1)."""
    env = gym.make("ManiSkillReachSO101-v1", config=ReachConfig(), num_envs=1, **_BASE_KWARGS)
    try:
        yield env.unwrapped
    finally:
        env.close()


@pytest.fixture
def so101_reach_env_vec() -> Iterator:
    """A vectorized SO101 Reach environment (num_envs=2) for per-_objs patch coverage."""
    try:
        env = gym.make("ManiSkillReachSO101-v1", config=ReachConfig(), num_envs=2, **_BASE_KWARGS)
    except Exception as exc:  # narrowed: only GPU-availability errors become skips
        _skip_if_gpu_unavailable(exc)
    try:
        yield env.unwrapped
    finally:
        env.close()
