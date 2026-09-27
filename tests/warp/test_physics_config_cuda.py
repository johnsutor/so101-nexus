"""Configured clocks apply to captured and direct CUDA stepping."""

import numpy as np
import pytest

from so101_nexus import MoveConfig, PhysicsConfig


@pytest.mark.parametrize("timestep", [0.0025, 0.001])
def test_configured_cuda_graph_and_direct_clocks(env_factory, timestep):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    env = env_factory(
        backend="warp",
        task="Move",
        device="cuda",
        config=MoveConfig(
            physics=PhysicsConfig(
                timestep_s=timestep,
                iterations=100,
                ls_iterations=100,
                impratio=1000,
                gripper_solimp=(0.99, 0.999, 0.001, 0.5, 2.0),
            ),
            reset_settle_frames=2,
            terminate_on_success=False,
        ),
    ).unwrapped
    assert round(0.02 / timestep) == env._N_SUBSTEPS
    np.testing.assert_allclose(env.model.opt.timestep.numpy(), timestep, rtol=1e-7)
    assert env._step_graph is not None
    env.reset(seed=321)
    np.testing.assert_allclose(env.data.time.numpy(), 0.04, atol=1e-7, rtol=0)
    for graph in (env._step_graph, None):
        env._step_graph = graph
        before = env.data.time.numpy().copy()
        env.step(env.ctrl.clone())
        np.testing.assert_allclose(env.data.time.numpy() - before, 0.02, atol=1e-7, rtol=0)
