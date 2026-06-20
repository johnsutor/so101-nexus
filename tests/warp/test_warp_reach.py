import pytest

pytestmark = pytest.mark.warp


def _make_env(num_envs=8, observations=None, seed=0):
    from so101_nexus.config import ReachConfig
    from so101_nexus.observations import JointPositions, TargetOffset
    from so101_nexus.warp.reach_env import WarpReachVectorEnv

    obs = observations if observations is not None else [JointPositions(), TargetOffset()]
    config = ReachConfig(observations=obs)
    return WarpReachVectorEnv(num_envs=num_envs, config=config, device="cpu", seed=seed)


def test_construction_spaces_and_obs_shape():
    import torch

    env = _make_env(num_envs=8)
    assert env.num_envs == 8
    assert env.single_action_space.shape == (6,)
    assert env.single_observation_space.shape == (9,)  # JointPositions(6) + TargetOffset(3)
    obs, info = env.reset(seed=0)
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (8, 9)
    assert obs.device.type == "cpu"
    assert torch.isfinite(obs).all()


def test_default_reach_obs_is_six_dim():
    from so101_nexus.observations import JointPositions

    env = _make_env(num_envs=4, observations=[JointPositions()])
    assert env.single_observation_space.shape == (6,)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 6)


def test_reset_is_seeded_deterministic():
    import torch

    a, _ = _make_env(num_envs=8, seed=123).reset(seed=123)
    b, _ = _make_env(num_envs=8, seed=123).reset(seed=123)
    assert torch.allclose(a, b)


def test_robot_init_qpos_noise_config_is_honored():
    from so101_nexus.config import ReachConfig
    from so101_nexus.observations import JointPositions
    from so101_nexus.warp.reach_env import WarpReachVectorEnv

    config = ReachConfig(observations=[JointPositions()], robot_init_qpos_noise=0.0)
    env = WarpReachVectorEnv(num_envs=4, config=config, device="cpu", seed=0)
    assert env.robot_init_qpos_noise == 0.0


def test_camera_observation_rejected():
    from so101_nexus.config import ReachConfig
    from so101_nexus.observations import JointPositions, WristCamera
    from so101_nexus.warp.reach_env import WarpReachVectorEnv

    config = ReachConfig(observations=[JointPositions(), WristCamera()])
    with pytest.raises(NotImplementedError):
        WarpReachVectorEnv(num_envs=2, config=config, device="cpu")
