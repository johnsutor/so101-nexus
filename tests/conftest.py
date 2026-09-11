"""Headless MuJoCo rendering for the test suite."""

import os
from contextlib import ExitStack

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture
def env_factory():
    """Construct registered environments and close them after the test."""
    import gymnasium as gym

    with ExitStack() as stack:

        def make(backend="mujoco", task="Touch", **kwargs):
            pytest.importorskip(f"so101_nexus.{backend}")
            if backend == "warp":
                pytest.importorskip("mujoco_warp")
                pytest.importorskip("torch")
                env = gym.make_vec(f"Warp{task}-v1", num_envs=2, device="cpu", **kwargs)
            else:
                env = gym.make(f"MuJoCo{task}-v1", **kwargs)
            stack.callback(env.close)
            return env

        yield make
