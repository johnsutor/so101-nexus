"""Warp pick-and-place completion is measured on the object, not on the arm.

Backend mirror of ``tests/mujoco/test_place_success_predicate.py``; the two
backends' agreement on the predicate itself is covered by
``test_warp_cross_backend_reward.py``.
"""

import pytest

pytestmark = pytest.mark.warp

NUM_ENVS = 2


def _placed_env(**config_kwargs):
    """A batched env whose object reads as placed on the goal, everything else stubbed."""
    import torch

    import so101_nexus.warp  # noqa: F401
    from so101_nexus.config import PickAndPlaceConfig
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceVectorEnv

    env = WarpPickAndPlaceVectorEnv(
        num_envs=NUM_ENVS,
        config=PickAndPlaceConfig(**config_kwargs),
        device="cpu",
        seed=0,
    )
    env.reset(seed=0)
    env._obj_placement_state = lambda *a, **k: (
        torch.zeros(NUM_ENVS),
        torch.ones(NUM_ENVS, dtype=torch.bool),
    )
    return env


def _pin(env, *, grasped: float, obj_static: bool, robot_static: bool):
    import torch

    env._is_grasping = lambda: torch.full((NUM_ENVS,), grasped)
    env._is_obj_static = lambda: torch.full((NUM_ENVS,), obj_static, dtype=torch.bool)
    env._is_robot_static = lambda: torch.full((NUM_ENVS,), robot_static, dtype=torch.bool)


def _success(env):
    import torch

    zero = torch.zeros(NUM_ENVS)
    _, success, info = env._compute_reward_terminated(zero, zero)
    return success, info


@pytest.mark.parametrize(
    ("grasped", "obj_static", "expected"),
    [
        (0.0, True, True),  # placed, settled, released
        (1.0, True, False),  # still held: release is mandatory
        (0.0, False, False),  # released but still moving: not settled
        (1.0, False, False),
    ],
)
def test_success_requires_a_settled_and_released_object(grasped, obj_static, expected):
    env = _placed_env()
    try:
        _pin(env, grasped=grasped, obj_static=obj_static, robot_static=True)
        success, _ = _success(env)
        assert success.tolist() == [expected] * NUM_ENVS
    finally:
        env.close()


@pytest.mark.parametrize("robot_static", [True, False])
def test_arm_velocity_does_not_decide_success(robot_static):
    """A retreating arm over a settled, released object is a success, not a failure."""
    env = _placed_env()
    try:
        _pin(env, grasped=0.0, obj_static=True, robot_static=robot_static)
        success, info = _success(env)
        assert success.all()
        assert info["is_robot_static"].tolist() == [robot_static] * NUM_ENVS
        assert info["is_obj_static"].all()
    finally:
        env.close()


@pytest.mark.parametrize(
    ("field", "dof_offset"),
    [("object_static_lin_threshold", 0), ("object_static_ang_threshold", 3)],
)
def test_object_static_thresholds_are_live_knobs(field, dof_offset):
    """Both config fields must change what counts as settled at runtime, each
    against the velocity component it governs (linear DOFs 0-2, angular 3-5)."""
    import torch

    speed = 0.05
    loose = _placed_env(**{field: speed * 2.0})
    strict = _placed_env(**{field: speed / 2.0})
    try:
        for env in (loose, strict):
            env._is_grasping = lambda: torch.zeros(NUM_ENVS)
            dof = env._target_dadr
            for world in range(NUM_ENVS):
                base = int(dof[world])
                env.qvel[world, base : base + 6] = 0.0
                env.qvel[world, base + dof_offset] = speed

        assert loose._is_obj_static().all()
        assert not strict._is_obj_static().any()
        assert _success(loose)[0].all()
        assert not _success(strict)[0].any()
    finally:
        loose.close()
        strict.close()


def test_terminate_on_success_false_keeps_the_episode_running():
    """Success is still reported every step; only ``terminated`` is suppressed."""
    import torch

    env = _placed_env(terminate_on_success=False)
    try:
        _pin(env, grasped=0.0, obj_static=True, robot_static=True)
        actions = torch.zeros(NUM_ENVS, env.action_space.shape[-1])
        _, _, terminated, truncated, info = env.step(actions)

        assert info["success"].all()
        assert not terminated.any()
        assert not truncated.any()
    finally:
        env.close()


def test_terminate_on_success_true_terminates():
    import torch

    env = _placed_env()
    try:
        _pin(env, grasped=0.0, obj_static=True, robot_static=True)
        actions = torch.zeros(NUM_ENVS, env.action_space.shape[-1])
        _, _, terminated, _, info = env.step(actions)

        assert info["success"].all()
        assert terminated.all()
    finally:
        env.close()
