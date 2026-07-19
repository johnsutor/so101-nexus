"""StackCube-specific Warp environment tests.

Generic construction/obs-shape coverage lives in ``test_warp_envs.py``'s
``_ENVS`` matrix; cross-backend reward parity lives in
``test_warp_cross_backend_reward.py``. This file covers behavior unique to
``WarpStackCubeVectorEnv``: spawn separation, per-world colour slot selection
(parity with the MuJoCo backend's per-episode resampling), off-world parking of
unselected slots, and the release-required success gate.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.warp


def _make_env(num_envs=8, seed=0, config=None):
    from so101_nexus.config import StackCubeConfig
    from so101_nexus.warp.stack_cube import WarpStackCubeVectorEnv

    return WarpStackCubeVectorEnv(
        num_envs=num_envs, config=config or StackCubeConfig(), device="cpu", seed=seed
    )


def test_stack_cube_reset_obs_finite():
    import torch

    env = _make_env(num_envs=6, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (6, 30)
    assert torch.isfinite(obs).all()
    assert "task_description" in info


def test_stack_cube_default_colors_red_on_blue():
    """Out of the box every world stacks the red cube on the blue cube."""
    env = _make_env(num_envs=4)
    env.reset(seed=0)
    assert set(env.cube_a_color_names) == {"red"}
    assert set(env.cube_b_color_names) == {"blue"}
    assert env.task_description == "Stack the red cube on the blue cube."


def test_stack_cube_colors_vary_per_world():
    """Each world samples its own colour pair at reset (slot-pool parity with
    the MuJoCo backend's per-episode ``geom_rgba`` resampling)."""
    from so101_nexus.config import StackCubeConfig

    config = StackCubeConfig(
        cube_a_colors=["red", "orange", "yellow"], cube_b_colors=["blue", "green"]
    )
    env = _make_env(num_envs=16, config=config)
    env.reset(seed=0)
    assert len(set(env.cube_a_color_names)) > 1
    assert len(set(env.cube_b_color_names)) > 1
    for a_name, b_name, desc in zip(
        env.cube_a_color_names, env.cube_b_color_names, env.task_descriptions, strict=True
    ):
        assert desc == f"Stack the {a_name} cube on the {b_name} cube."


def test_stack_cube_colors_resample_across_resets():
    from so101_nexus.config import StackCubeConfig

    config = StackCubeConfig(
        cube_a_colors=["red", "orange", "yellow"], cube_b_colors=["blue", "green"]
    )
    env = _make_env(num_envs=4, config=config)
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    for seed in range(10):
        env.reset(seed=seed)
        seen_a.update(env.cube_a_color_names)
        seen_b.update(env.cube_b_color_names)
    assert len(seen_a) > 1
    assert len(seen_b) > 1


def test_stack_cube_parked_slots_stay_off_world():
    """Unselected colour slots park beyond the spawn annulus (Warp contact bits
    are model-global, so inactive cubes are parked, not collision-disabled)."""
    import torch

    from so101_nexus.config import StackCubeConfig

    config = StackCubeConfig(
        cube_a_colors=["red", "orange", "yellow"], cube_b_colors=["blue", "green"]
    )
    env = _make_env(num_envs=8, config=config)
    env.reset(seed=0)
    cfg = env.config
    center = torch.tensor(cfg.spawn_center)
    for j in range(len(env._slot_qadr)):
        qa = int(env._slot_qadr[j])
        r = torch.linalg.norm(env.qpos[:, qa : qa + 2] - center, dim=1)
        is_selected = (env._a_qadr == qa) | (env._b_qadr == qa)
        if bool(is_selected.any()):
            assert bool((r[is_selected] <= cfg.spawn_max_radius + 1e-6).all())
            assert bool((r[is_selected] >= cfg.spawn_min_radius - 1e-6).all())
        if bool((~is_selected).any()):
            assert bool((r[~is_selected] > cfg.spawn_max_radius).all())


def test_stack_cube_min_separation_respected():
    """Cube centres clear ``min_cube_separation`` plus their combined bounding
    radii, mirroring the MuJoCo backend's spawn-separation invariant.
    """
    import numpy as np
    import torch

    from so101_nexus.config import StackCubeConfig

    cfg = StackCubeConfig()
    env = _make_env(num_envs=32, seed=0, config=cfg)
    env.reset(seed=0)
    a_xy = env._cube_a_pos()[:, :2]
    b_xy = env._cube_b_pos()[:, :2]
    dist = torch.linalg.norm(a_xy - b_xy, dim=1)
    min_expected = cfg.min_cube_separation + 2 * (np.sqrt(2) * cfg.cube_half_size)
    assert bool((dist >= min_expected - 1e-4).all())


def test_stack_cube_success_requires_release():
    """A cube A that is on-goal and static but still grasped is not a success."""
    import torch

    env = _make_env(num_envs=2, seed=0)
    env.reset(seed=0)
    half = env.cube_half_size

    b_pos = torch.tensor([[0.15, 0.0, half]] * 2)
    a_pos = torch.tensor([[0.15, 0.0, 3 * half]] * 2)  # exactly 2*half above cube B
    env._cube_a_pos = lambda: a_pos
    env._cube_a_pose7 = lambda: torch.cat([a_pos, torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)], dim=1)
    env._cube_b_pos = lambda: b_pos
    env._is_robot_static = lambda: torch.ones(2, dtype=torch.bool)
    env._is_grasping = lambda: torch.ones(2)

    zero = torch.zeros(2)
    _, success_grasped, info = env._compute_reward_terminated(zero, zero)
    assert bool(info["is_stacked"].all())
    assert not bool(success_grasped.any())

    env._is_grasping = lambda: torch.zeros(2)
    _, success_released, _ = env._compute_reward_terminated(zero, zero)
    assert bool(success_released.all())


def test_stack_cube_success_requires_cube_a_static():
    """A stacked, released cube A still moving through the tolerance band is
    not a success (ManiSkill's ``is_cubeA_static`` gate): it must settle first.
    """
    import torch

    env = _make_env(num_envs=2, seed=0)
    env.reset(seed=0)
    half = env.cube_half_size

    b_pos = torch.tensor([[0.15, 0.0, half]] * 2)
    a_pos = torch.tensor([[0.15, 0.0, 3 * half]] * 2)  # exactly 2*half above cube B
    env._cube_a_pos = lambda: a_pos
    env._cube_b_pos = lambda: b_pos
    env._is_robot_static = lambda: torch.ones(2, dtype=torch.bool)
    env._is_grasping = lambda: torch.zeros(2)

    zero = torch.zeros(2)
    env.qvel.zero_()  # both cubes settled
    _, success, info = env._compute_reward_terminated(zero, zero)
    assert bool(success.all())
    assert bool(info["is_cube_a_static"].all())

    # Cube A sliding at 0.05 m/s in world 0 only: that world is not a success.
    rows = torch.arange(2)
    env.qvel[rows, env._a_dadr] = torch.tensor([0.05, 0.0])
    _, success, info = env._compute_reward_terminated(zero, zero)
    assert not bool(success[0])
    assert bool(success[1])
    assert not bool(info["is_cube_a_static"][0])
    assert bool(info["is_cube_a_static"][1])

    # Rocking at 1.0 rad/s in world 0: still not a success.
    env.qvel.zero_()
    env.qvel[rows, env._a_dadr + 4] = torch.tensor([1.0, 0.0])
    _, success, _ = env._compute_reward_terminated(zero, zero)
    assert not bool(success[0])
    assert bool(success[1])


def test_stack_cube_reward_no_dwelling_when_hovering():
    """Regression coverage for the reward-hacking trap (see
    ``test_pick_and_place_reward_no_dwelling_when_hovering`` in
    ``test_warp_envs.py`` for the sibling formula): repeated
    ``_compute_reward_terminated`` calls at an unchanged "grasped, hovering
    above the goal" state pay ~0 further task_objective reward after the
    first (a potential-shaping delta, not the raw ``Phi_stack`` value).
    """
    import torch

    env = _make_env(num_envs=2, seed=0)
    env.reset(seed=0)
    env._is_grasping = lambda: torch.ones(2)  # force grasped, no real contact needed

    b_pos = env._cube_b_pos()
    half = env.cube_half_size
    rows = torch.arange(2, device=env.device)
    env.qpos[rows, env._a_qadr] = b_pos[:, 0]
    env.qpos[rows, env._a_qadr + 1] = b_pos[:, 1]
    env.qpos[rows, env._a_qadr + 2] = b_pos[:, 2] + 4 * half  # hovering above the goal

    zero = torch.zeros(2)
    reward1, _, _ = env._compute_reward_terminated(zero, zero)
    reward2, _, info2 = env._compute_reward_terminated(zero, zero)

    # The reward a fully-zero-delta formula would pay on the second call --
    # what the actual (delta-shaped) reward must match, since reach, grasp,
    # and task potential all held steady between the two identical hover
    # snapshots.
    expected_no_dwelling_credit = env.config.reward.compute(
        reach_progress=torch.zeros(2),
        is_grasped=torch.zeros(2),
        task_progress=torch.zeros(2),
        is_complete=info2["success"],
    )
    assert torch.allclose(reward2, expected_no_dwelling_credit, atol=1e-5)
    # The first hover snapshot still earned real, one-time credit for having
    # moved to that state (nonzero delta from the reset baseline).
    assert bool((reward1 > reward2 + 1e-3).all())
