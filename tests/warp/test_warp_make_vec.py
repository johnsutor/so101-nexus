import pytest

pytestmark = pytest.mark.warp


def test_make_vec_constructs_native_batched_env():
    import gymnasium as gym
    import torch

    import so101_nexus.warp  # noqa: F401
    from so101_nexus.config import TouchConfig
    from so101_nexus.observations import JointPositions, ObjectOffset

    envs = gym.make_vec(
        "WarpTouch-v1",
        num_envs=4,
        config=TouchConfig(observations=[JointPositions(), ObjectOffset()]),
        device="cpu",
        vectorization_mode="vector_entry_point",
    )
    obs, _ = envs.reset(seed=0)
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (4, 9)
    envs.close()


def test_make_vec_forwards_max_episode_steps():
    import gymnasium as gym
    import torch

    import so101_nexus.warp  # noqa: F401
    from so101_nexus.config import TouchConfig

    default = gym.make_vec("WarpTouch-v1", num_envs=2, device="cpu")
    assert default.unwrapped.max_episode_steps == 512
    default.close()

    envs = gym.make_vec(
        "WarpTouch-v1",
        num_envs=2,
        config=TouchConfig(),
        device="cpu",
        max_episode_steps=2,
    )
    try:
        assert envs.unwrapped.max_episode_steps == 2
        envs.reset(seed=0)
        _, _, _, t1, _ = envs.step(torch.zeros((2, 6)))
        _, _, _, t2, _ = envs.step(torch.zeros((2, 6)))
        assert not bool(t1.any())
        assert bool(t2.all())
    finally:
        envs.close()
