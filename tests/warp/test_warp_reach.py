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


def test_step_shapes_and_finiteness():
    import torch

    env = _make_env(num_envs=8)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(torch.zeros((8, 6)))
    assert obs.shape == (8, 9)
    assert reward.shape == (8,)
    assert terminated.shape == (8,)
    assert truncated.shape == (8,)
    assert reward.dtype == torch.float32
    assert torch.isfinite(reward).all()
    assert terminated.dtype == torch.bool


def test_reward_matches_scalar_core_per_world():
    import numpy as np
    import torch

    from so101_nexus.rewards import reach_progress, simple_reward

    env = _make_env(num_envs=16, seed=7)
    env.reset(seed=7)
    _, reward, _, _, info = env.step(torch.zeros((16, 6)))
    dist = info["tcp_to_target_dist"].numpy()
    cb = env.config.reward.completion_bonus
    scale = env.config.reward.tanh_shaping_scale
    thr = env.config.success_threshold
    expected = np.array(
        [
            simple_reward(
                progress=reach_progress(float(d), scale=scale),
                completion_bonus=cb,
                success=bool(d < thr),
            )
            for d in dist
        ]
    )
    np.testing.assert_allclose(reward.numpy(), expected, rtol=1e-5, atol=1e-6)


def test_step_trajectory_is_seeded_deterministic():
    import torch

    a = _make_env(num_envs=8, seed=42)
    b = _make_env(num_envs=8, seed=42)
    a.reset(seed=42)
    b.reset(seed=42)
    action_rng = torch.Generator().manual_seed(0)
    for _ in range(10):
        action = torch.rand((8, 6), generator=action_rng) * 2.0 - 1.0
        oa, ra, ta, _, _ = a.step(action)
        ob, rb, tb, _, _ = b.step(action)
        assert torch.allclose(oa, ob)
        assert torch.allclose(ra, rb)
        assert torch.equal(ta, tb)


def test_truncation_autoresets_world():
    import torch

    env = _make_env(num_envs=4, seed=1)
    env.max_episode_steps = 3
    env.reset(seed=1)
    for _ in range(2):
        _, _, _, truncated, _ = env.step(torch.zeros((4, 6)))
        assert not truncated.any()
    targets_before = env._targets.clone()
    _, _, _, truncated, _ = env.step(torch.zeros((4, 6)))
    assert truncated.all()
    assert (env._elapsed == 0).all()  # reset counters
    assert not torch.allclose(env._targets, targets_before)  # resampled targets


def test_autoreset_masks_prev_action_across_episode_boundary():
    """A new episode's first action_delta_norm is zero for any first action."""
    import pytest
    import torch

    from so101_nexus.config import ReachConfig, RewardConfig
    from so101_nexus.observations import JointPositions, TargetOffset
    from so101_nexus.warp.reach_env import WarpReachVectorEnv

    config = ReachConfig(
        observations=[JointPositions(), TargetOffset()],
        reward=RewardConfig(action_delta_penalty=1.0),
    )
    env = WarpReachVectorEnv(num_envs=1, config=config, device="cpu", max_episode_steps=1, seed=0)
    env.reset(seed=0)
    env.step(torch.ones((1, 6)))  # truncated -> autoreset; next step starts a new episode

    _, reward, _, _, info = env.step(torch.ones((1, 6)))

    assert info["action_delta_norm"].item() == pytest.approx(0.0)
    assert reward.item() > 0.0


def test_unsupported_obs_component_rejected_at_construction():
    from so101_nexus.config import ReachConfig
    from so101_nexus.observations import GazeDirection, JointPositions
    from so101_nexus.warp.reach_env import WarpReachVectorEnv

    # GazeDirection is a look-at task-specific component, unsupported by reach.
    config = ReachConfig(observations=[JointPositions(), GazeDirection()])
    with pytest.raises(NotImplementedError, match="GazeDirection"):
        WarpReachVectorEnv(num_envs=2, config=config, device="cpu")
