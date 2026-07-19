"""StackCube-specific MuJoCo environment tests.

Cross-task matrix coverage (Gymnasium contract, observation components,
control modes, cameras, episode truncation) lives in ``test_envs.py`` via
``ENV_MATRIX``/``_CAMERA_ENVS``; this file covers behavior unique to
``StackCubeEnv``: the stacked/success predicate, the reused
``place_task_potential`` reward shaping, spawn separation, and task
descriptions.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import pytest

import so101_nexus.mujoco  # noqa: F401 - registers envs
from so101_nexus.config import REWARD_COMPONENT_KEYS, StackCubeConfig


def test_stack_cube_default_obs_shape():
    """StackCube default obs is a 30-dim flat vector, matching PickAndPlace."""
    env = gym.make("MuJoCoStackCube-v1")
    try:
        obs, _ = env.reset()
        assert obs.shape == (30,)
    finally:
        env.close()


def test_stack_cube_success_false_at_reset():
    env = gym.make("MuJoCoStackCube-v1")
    try:
        _, info = env.reset()
        assert not info["success"]
    finally:
        env.close()


def test_stack_cube_reward_in_bounds():
    """Per-step reward stays within the documented multi-phase bound (see
    ``test_envs.REWARD_RANGE_OVERRIDES``): reaching/grasping/task_objective are
    potential-based deltas that can swing to -0.75 in the worst case.
    """
    env = gym.make("MuJoCoStackCube-v1")
    try:
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert -0.75 <= float(reward) <= 1.0
    finally:
        env.close()


def test_stack_cube_task_description_mentions_both_colors():
    config = StackCubeConfig(cube_a_colors="red", cube_b_colors="blue")
    env = gym.make("MuJoCoStackCube-v1", config=config)
    try:
        desc = env.unwrapped.task_description  # type: ignore[attr-defined]
        assert isinstance(desc, str)
        assert "red" in desc
        assert "blue" in desc
    finally:
        env.close()


def test_stack_cube_task_description_starts_capital():
    env = gym.make("MuJoCoStackCube-v1")
    try:
        desc = env.unwrapped.task_description  # type: ignore[attr-defined]
        assert desc[0].isupper()
    finally:
        env.close()


def test_stack_cube_task_description_is_instance_attr():
    """Task description is an instance attribute, not a class attribute."""
    env = gym.make("MuJoCoStackCube-v1")
    try:
        assert "task_description" in env.unwrapped.__dict__  # type: ignore[attr-defined]
    finally:
        env.close()


def test_stack_cube_colors_resample_per_episode():
    """Cube colours are re-sampled every reset (unlike Warp's compile-time fix)."""
    config = StackCubeConfig(cube_a_colors=["red", "orange", "yellow"], cube_b_colors="blue")
    env = gym.make("MuJoCoStackCube-v1", config=config)
    try:
        seen = set()
        for seed in range(15):
            env.reset(seed=seed)
            seen.add(env.unwrapped.cube_a_color_name)  # type: ignore[attr-defined]
        assert len(seen) > 1
    finally:
        env.close()


def test_stack_cube_cubes_spawn_in_bounds():
    """Both cubes spawn within [spawn_min_radius, spawn_max_radius] of spawn_center."""
    cfg = StackCubeConfig()
    env = gym.make("MuJoCoStackCube-v1")
    try:
        for seed in range(10):
            env.reset(seed=seed)
            inner = env.unwrapped
            cx, cy = cfg.spawn_center
            for pose_fn in (inner._get_cube_a_pose, inner._get_cube_b_pose):  # type: ignore[attr-defined]
                xy = pose_fn()[:2]
                r = float(np.hypot(xy[0] - cx, xy[1] - cy))
                assert cfg.spawn_min_radius - 1e-6 <= r <= cfg.spawn_max_radius + 1e-6
    finally:
        env.close()


def test_stack_cube_min_separation_respected():
    """Cube centres clear ``min_cube_separation`` plus their combined bounding
    radii (the invariant ``sample_separated_positions`` enforces on a
    successful placement within its retry budget).
    """
    cfg = StackCubeConfig()
    env = gym.make("MuJoCoStackCube-v1")
    try:
        min_expected = cfg.min_cube_separation + 2 * (np.sqrt(2) * cfg.cube_half_size)
        for seed in range(15):
            env.reset(seed=seed)
            inner = env.unwrapped
            a_xy = inner._get_cube_a_pose()[:2]  # type: ignore[attr-defined]
            b_xy = inner._get_cube_b_pose()[:2]  # type: ignore[attr-defined]
            dist = float(np.linalg.norm(a_xy - b_xy))
            assert dist >= min_expected - 1e-6
    finally:
        env.close()


def test_stack_cube_reward_components_sum_to_reward():
    env = gym.make("MuJoCoStackCube-v1")
    try:
        env.reset(seed=0)
        action = env.action_space.sample()
        _, reward, _, _, info = env.step(action)
        components = info["reward_components"]
        assert set(components) == set(REWARD_COMPONENT_KEYS)
        assert sum(components.values()) == pytest.approx(float(reward), abs=1e-5)
    finally:
        env.close()


def test_stack_cube_is_stacked_true_when_cube_a_directly_above_cube_b():
    """``_stack_state`` reports ``is_stacked`` for a synthetic on-goal pose."""
    env = gym.make("MuJoCoStackCube-v1")
    try:
        inner = env.unwrapped
        inner.reset(seed=0)
        half = inner.cube_half_size  # type: ignore[attr-defined]
        b_pos = np.array([0.15, 0.0, half])
        a_pos = np.array([0.15, 0.0, 3 * half])  # 2*half above cube B's centre
        _, is_stacked = inner._stack_state(a_pos, b_pos)  # type: ignore[attr-defined]
        assert is_stacked
    finally:
        env.close()


def test_stack_cube_is_stacked_false_when_offset():
    env = gym.make("MuJoCoStackCube-v1")
    try:
        inner = env.unwrapped
        inner.reset(seed=0)
        half = inner.cube_half_size  # type: ignore[attr-defined]
        b_pos = np.array([0.15, 0.0, half])
        a_pos = np.array([0.30, 0.0, half])  # far away, on the floor
        _, is_stacked = inner._stack_state(a_pos, b_pos)  # type: ignore[attr-defined]
        assert not is_stacked
    finally:
        env.close()


def test_stack_cube_success_requires_release():
    """A cube A that is on-goal and static but still grasped is not a success.

    Mirrors ManiSkill's ``StackCubeEnv.evaluate``: the robot must let go.
    """
    env = gym.make("MuJoCoStackCube-v1")
    try:
        inner = env.unwrapped
        inner.reset(seed=0)
        half = inner.cube_half_size  # type: ignore[attr-defined]
        b_pos = np.array([0.15, 0.0, half])
        a_pos = np.array([0.15, 0.0, 3 * half])
        inner._get_cube_a_pose = lambda: np.array([*a_pos, 1.0, 0.0, 0.0, 0.0])  # type: ignore[method-assign]
        inner._get_cube_b_pose = lambda: np.array([*b_pos, 1.0, 0.0, 0.0, 0.0])  # type: ignore[method-assign]
        inner._is_robot_static = lambda: True  # type: ignore[method-assign]

        inner._is_grasping = lambda: 1.0  # type: ignore[method-assign]
        info_grasped = inner._get_info()
        assert info_grasped["is_stacked"]
        assert not info_grasped["success"]

        inner._is_grasping = lambda: 0.0  # type: ignore[method-assign]
        info_released = inner._get_info()
        assert info_released["success"]
    finally:
        env.close()


def _pin_stacked_ungrasped(inner):
    """Stub cube poses to a perfectly stacked pose, arm static, grasp released."""
    half = inner.cube_half_size
    b_pos = np.array([0.15, 0.0, half])
    a_pos = np.array([0.15, 0.0, 3 * half])
    inner._get_cube_a_pose = lambda: np.array([*a_pos, 1.0, 0.0, 0.0, 0.0])
    inner._get_cube_b_pose = lambda: np.array([*b_pos, 1.0, 0.0, 0.0, 0.0])
    inner._is_robot_static = lambda: True
    inner._is_grasping = lambda: 0.0


def test_stack_cube_success_requires_cube_a_static():
    """A stacked, released cube A still moving through the tolerance band is
    not a success (ManiSkill's ``is_cubeA_static`` gate): it must settle first.
    """
    env = gym.make("MuJoCoStackCube-v1")
    try:
        inner = env.unwrapped
        inner.reset(seed=0)
        _pin_stacked_ungrasped(inner)  # type: ignore[attr-defined]
        dof = inner._slot_a.dof_addr  # type: ignore[attr-defined]

        inner.data.qvel[dof : dof + 6] = 0.0
        assert inner._get_info()["success"]  # type: ignore[attr-defined]

        # Sliding linearly through the band above the 0.01 m/s threshold.
        inner.data.qvel[dof : dof + 6] = [0.05, 0.0, 0.0, 0.0, 0.0, 0.0]
        info = inner._get_info()  # type: ignore[attr-defined]
        assert info["is_stacked"]
        assert not info["is_cube_a_static"]
        assert not info["success"]

        # Rocking in place above the 0.5 rad/s angular threshold.
        inner.data.qvel[dof : dof + 6] = [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        info = inner._get_info()  # type: ignore[attr-defined]
        assert info["is_stacked"]
        assert not info["is_cube_a_static"]
        assert not info["success"]

        # Settled: success again.
        inner.data.qvel[dof : dof + 6] = 0.0
        assert inner._get_info()["success"]  # type: ignore[attr-defined]
    finally:
        env.close()


def test_stack_cube_cube_static_thresholds_are_live_knobs():
    """Tightening the configured thresholds flips the same slow-moving cube
    from static to not static, proving the config fields gate success."""
    env = gym.make("MuJoCoStackCube-v1")
    try:
        inner = env.unwrapped
        inner.reset(seed=0)
        _pin_stacked_ungrasped(inner)  # type: ignore[attr-defined]
        dof = inner._slot_a.dof_addr  # type: ignore[attr-defined]
        inner.data.qvel[dof : dof + 6] = [0.005, 0.0, 0.0, 0.0, 0.0, 0.0]

        assert inner._get_info()["success"]  # type: ignore[attr-defined]  # 5 mm/s < 10 mm/s default
        inner.config.cube_static_lin_threshold = 1e-3
        assert not inner._get_info()["success"]  # type: ignore[attr-defined]
    finally:
        env.close()


def test_stack_cube_hovering_earns_no_dwelling_task_objective_reward():
    """Hovering at a fixed task potential must score ~0 on the task_objective
    facet after the first step; mirrors the PickAndPlace regression test in
    ``test_envs.py`` for the same reward-hacking trap (see
    ``docs/superpowers/plans/2026-07-12-potential-based-task-progress-shaping.md``).
    """
    env = gym.make("MuJoCoStackCube-v1")
    try:
        inner = env.unwrapped
        inner.reset(seed=0)
        weight = inner.config.reward.task_objective

        def step_info(task_potential):
            return {
                "tcp_to_obj_dist": 0.0,
                "is_grasped": 1.0,
                "is_stacked": False,
                "success": False,
                "task_potential": task_potential,
            }

        inner._prev_task_potential = 0.0
        first = step_info(0.85)
        inner._compute_reward(first)
        assert first["reward_components"]["task_objective"] == pytest.approx(weight * 0.85)

        for _ in range(5):
            hover = step_info(0.85)
            inner._compute_reward(hover)
            assert hover["reward_components"]["task_objective"] == pytest.approx(0.0, abs=1e-9)
    finally:
        env.close()


def test_stack_cube_reward_nonnegative_along_ideal_trajectory():
    """Forward progress must never pay negative reward along the ideal
    reach-grasp-lift-carry-lower-release trajectory (mirrors the PickAndPlace
    regression test for the same monotone potential shape).
    """
    env = gym.make("MuJoCoStackCube-v1")
    try:
        inner = env.unwrapped
        inner.reset(seed=0)
        half = inner.cube_half_size  # type: ignore[attr-defined]
        b_pos = np.array([0.0, 0.0, half])

        def stage(*, tcp_to_obj, a_xy, height, arm_speed, grasped, stacked):
            inner.data.qvel[inner._arm_qvel_addrs] = arm_speed
            a_pos = np.array([a_xy, 0.0, height])
            info = {
                "tcp_to_obj_dist": tcp_to_obj,
                "is_grasped": float(grasped),
                "is_stacked": stacked,
                "success": False,
                "task_potential": inner._task_potential(a_pos, b_pos, float(grasped), stacked),
            }
            return inner._compute_reward(info), info

        trajectory = [
            ("reset", 0.25, 0.15, half, 0.0, False, False),
            ("approach", 0.02, 0.15, half, 0.5, False, False),
            ("grasp", 0.01, 0.15, half, 0.0, True, False),
            ("lift", 0.01, 0.15, 0.05, 0.3, True, False),
            ("carry", 0.01, 0.05, 0.05, 0.5, True, False),
            ("align", 0.01, 0.005, 3 * half, 0.2, True, True),
            ("release", 0.01, 0.0, 3 * half, 0.05, False, True),
        ]
        rewards: dict[str, float] = {}
        infos: dict[str, dict] = {}
        for i, (label, tcp, xy, h, speed, grasped, stacked) in enumerate(trajectory):
            reward, info = stage(
                tcp_to_obj=tcp, a_xy=xy, height=h, arm_speed=speed, grasped=grasped, stacked=stacked
            )
            if i > 0:  # the reset stage only seeds the _prev_* baselines
                rewards[label], infos[label] = reward, info

        negative = {label: r for label, r in rewards.items() if r < -1e-9}
        assert not negative, f"forward progress paid negative reward: {negative}"
        assert infos["release"]["reward_components"]["grasping"] == pytest.approx(0.0, abs=1e-9)
    finally:
        env.close()
