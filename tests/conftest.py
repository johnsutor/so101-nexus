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

        def make(backend="mujoco", task="Touch", version=1, **kwargs):
            pytest.importorskip(f"so101_nexus.{backend}")
            if backend == "warp":
                pytest.importorskip("mujoco_warp")
                pytest.importorskip("torch")
                device = kwargs.pop("device", "cpu")
                env = gym.make_vec(f"Warp{task}-v{version}", num_envs=2, device=device, **kwargs)
            else:
                env = gym.make(f"MuJoCo{task}-v{version}", **kwargs)
            stack.callback(env.close)
            return env

        yield make
