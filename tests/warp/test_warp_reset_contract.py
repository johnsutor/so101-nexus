"""Reset-contract parity between public reset() and same-step autoreset.

The batched mujoco_warp step cannot settle a subset of worlds, so same-step
autoreset skips the robot settle that reset() applies. The reset *reference*
state (lift baselines, move targets) must still be captured settle-independently
so it is identical across both paths; otherwise targets and baselines would jump
across episode boundaries during vectorized training.
"""

import pytest

pytestmark = pytest.mark.warp


def _force_autoreset_all(env) -> None:
    """Truncate every world on the next step so all worlds same-step autoreset."""
    env._elapsed[:] = env.max_episode_steps
    env.step(env.action_space.sample())


def test_pick_lift_baseline_settle_independent_across_autoreset():
    import torch

    from so101_nexus.config import PickConfig
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    env = WarpPickLiftVectorEnv(
        num_envs=4, config=PickConfig(reset_settle_frames=5), device="cpu", seed=0
    )
    env.reset(seed=0)
    reset_baseline = env._initial_obj_z.clone()

    _force_autoreset_all(env)
    autoreset_baseline = env._initial_obj_z.clone()

    half = env._cube.half_size
    assert torch.allclose(reset_baseline, autoreset_baseline)
    assert torch.allclose(reset_baseline, torch.full((4,), half, dtype=reset_baseline.dtype))


def test_pick_and_place_baseline_settle_independent_across_autoreset():
    import torch

    from so101_nexus.config import PickAndPlaceConfig
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceVectorEnv

    env = WarpPickAndPlaceVectorEnv(
        num_envs=4, config=PickAndPlaceConfig(reset_settle_frames=5), device="cpu", seed=0
    )
    env.reset(seed=0)
    reset_baseline = env._initial_obj_z.clone()

    _force_autoreset_all(env)
    autoreset_baseline = env._initial_obj_z.clone()

    half = env.config.cube_half_size
    assert torch.allclose(reset_baseline, autoreset_baseline)
    assert torch.allclose(reset_baseline, torch.full((4,), half, dtype=reset_baseline.dtype))


def test_move_target_settle_independent():
    """The Move target is captured pre-settle, so it does not depend on settle frames."""
    import torch

    from so101_nexus.config import MoveConfig
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    def make(settle):
        cfg = MoveConfig(target_distance=0.1, reset_settle_frames=settle)
        return WarpMoveVectorEnv(num_envs=4, config=cfg, device="cpu", seed=0)

    init = {"init_qpos": [0.1, -0.2, 0.3, 0.0, 0.1, 0.0]}
    env0 = make(0)
    env5 = make(5)
    env0.reset(seed=0, options=init)
    env5.reset(seed=0, options=init)
    # Same reset pose, different settle: identical targets prove settle-independence,
    # so reset() and the no-settle autoreset path capture the same Move target.
    assert torch.allclose(env0._targets, env5._targets)


def test_move_target_distance_preserved_across_autoreset():
    """Same-step autoreset places the Move target at exactly target_distance (no settle)."""
    import torch

    from so101_nexus.config import MoveConfig
    from so101_nexus.warp.move_env import WarpMoveVectorEnv

    env = WarpMoveVectorEnv(
        num_envs=4,
        config=MoveConfig(target_distance=0.1, reset_settle_frames=5),
        device="cpu",
        seed=0,
    )
    env.reset(seed=0)
    _force_autoreset_all(env)
    dist = torch.linalg.norm(env._targets - env._tcp_pos(), dim=1)
    assert torch.allclose(dist, torch.full((4,), 0.1), atol=1e-4)
