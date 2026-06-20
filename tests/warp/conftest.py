"""Fixtures for Warp backend tests. All tests skip cleanly without mujoco_warp."""

import pytest

pytest.importorskip("mujoco_warp")
pytest.importorskip("torch")


@pytest.fixture(scope="session")
def warp_cpu_device():
    """Warp CPU device: the documented deterministic execution target."""
    import warp as wp

    return wp.get_device("cpu")
