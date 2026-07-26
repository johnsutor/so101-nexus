"""Pick-and-place completion is measured on the object, not on the arm.

``success`` requires the object to be within the goal tolerance, settled, and
released. Arm joint velocity is reported in ``info`` but must never decide the
outcome: the intended terminal behaviour is release-and-retreat, and gating on
arm staticness scores the retreat itself as failure. See
docs/superpowers/plans/2026-07-26-place-success-predicate-and-terminate-flag.md.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym

import so101_nexus.mujoco  # noqa: F401
from so101_nexus.config import PickAndPlaceConfig


def _placed_env(**config_kwargs):
    """An env whose object reads as placed on the goal, with everything else stubbed."""
    env = gym.make("MuJoCoPickAndPlace-v1", config=PickAndPlaceConfig(**config_kwargs))
    inner = env.unwrapped
    inner.reset(seed=0)
    inner._obj_placement_state = lambda *a, **k: (0.0, True)
    return env, inner


def _pin(inner, *, grasped: float, obj_static: bool, robot_static: bool):
    inner._is_grasping = lambda: grasped
    inner._is_obj_static = lambda: obj_static
    inner._is_robot_static = lambda: robot_static


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
    env, inner = _placed_env()
    try:
        _pin(inner, grasped=grasped, obj_static=obj_static, robot_static=True)
        assert inner._get_info()["success"] is expected
    finally:
        env.close()


@pytest.mark.parametrize("robot_static", [True, False])
def test_arm_velocity_does_not_decide_success(robot_static):
    """A retreating arm over a settled, released object is a success, not a failure."""
    env, inner = _placed_env()
    try:
        _pin(inner, grasped=0.0, obj_static=True, robot_static=robot_static)
        info = inner._get_info()
        assert info["success"] is True
        assert info["is_robot_static"] is robot_static
    finally:
        env.close()


@pytest.mark.parametrize(
    ("field", "dof_offset"),
    [("object_static_lin_threshold", 0), ("object_static_ang_threshold", 3)],
)
def test_object_static_thresholds_are_live_knobs(field, dof_offset):
    """Both config fields must change what counts as settled at runtime, each
    against the velocity component it governs (linear DOFs 0-2, angular 3-5)."""
    speed = 0.05
    env, inner = _placed_env(**{field: speed * 2.0})
    strict_env, strict = _placed_env(**{field: speed / 2.0})
    try:
        for target in (inner, strict):
            target._is_grasping = lambda: 0.0
            addr = target._slots[target._target_slot_idx].dof_addr
            target.data.qvel[addr : addr + 6] = 0.0
            target.data.qvel[addr + dof_offset] = speed

        assert inner._is_obj_static() is True
        assert strict._is_obj_static() is False
        assert inner._get_info()["success"] is True
        assert strict._get_info()["success"] is False
    finally:
        env.close()
        strict_env.close()


def test_placement_and_success_are_plain_bools():
    """``is_obj_placed`` compares a numpy.float64, so without an explicit cast the
    predicate leaks a numpy.bool_ into ``info`` and breaks JSON encoding for any
    consumer logging raw rollout info."""
    import json

    env = gym.make("MuJoCoPickAndPlace-v1")
    inner = env.unwrapped
    try:
        env.reset(seed=0)
        # Object laterally over the goal but too high: the height term decides,
        # which is exactly the state that leaked a numpy.bool_.
        target = inner._get_target_pos()
        info = inner._get_info()
        assert type(info["is_obj_placed"]) is bool
        assert type(info["success"]) is bool
        json.dumps({k: v for k, v in info.items() if k in ("is_obj_placed", "success")})

        dist, placed = inner._obj_placement_state(
            np.array([target[0], target[1], inner._initial_obj_z + 1.0]), target
        )
        assert type(placed) is bool
        assert isinstance(dist, float)
    finally:
        env.close()


def test_info_reports_object_staticness_alongside_robot_staticness():
    env, inner = _placed_env()
    try:
        _pin(inner, grasped=0.0, obj_static=True, robot_static=False)
        info = inner._get_info()
        assert info["is_obj_static"] is True
        assert info["is_robot_static"] is False
    finally:
        env.close()


def test_terminate_on_success_false_keeps_the_episode_running():
    """Success is still reported every step; only ``terminated`` is suppressed, so
    an alternative predicate can be evaluated offline against a full rollout."""
    env = gym.make("MuJoCoPickAndPlace-v1", config=PickAndPlaceConfig(terminate_on_success=False))
    inner = env.unwrapped
    try:
        env.reset(seed=0)
        inner._obj_placement_state = lambda *a, **k: (0.0, True)
        _pin(inner, grasped=0.0, obj_static=True, robot_static=True)

        _, _, terminated, truncated, info = env.step(np.zeros(env.action_space.shape))
        assert info["success"] is True
        assert terminated is False
        assert truncated is False
    finally:
        env.close()


def test_terminate_on_success_true_terminates():
    env = gym.make("MuJoCoPickAndPlace-v1")
    inner = env.unwrapped
    try:
        env.reset(seed=0)
        inner._obj_placement_state = lambda *a, **k: (0.0, True)
        _pin(inner, grasped=0.0, obj_static=True, robot_static=True)

        _, _, terminated, _, info = env.step(np.zeros(env.action_space.shape))
        assert info["success"] is True
        assert terminated is True
    finally:
        env.close()
