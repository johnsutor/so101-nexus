import pytest

pytestmark = pytest.mark.warp


def test_make_vec_constructs_native_batched_env():
    import gymnasium as gym
    import torch

    import so101_nexus.warp  # noqa: F401
    from so101_nexus.config import ReachConfig
    from so101_nexus.observations import JointPositions, TargetOffset

    envs = gym.make_vec(
        "WarpReach-v1",
        num_envs=4,
        config=ReachConfig(observations=[JointPositions(), TargetOffset()]),
        device="cpu",
        vectorization_mode="vector_entry_point",
    )
    obs, _ = envs.reset(seed=0)
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (4, 9)
    envs.close()
