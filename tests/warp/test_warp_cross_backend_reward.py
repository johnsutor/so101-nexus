"""Cross-backend reward parity (mujoco vs warp).

The MuJoCo and Warp backends wire the same shared reward primitives
(``so101_nexus.rewards``, ``RewardConfig.compute``/``compute_components``)
into two independently hand-written env classes -- nothing enforces at import
time that the wiring (what gets passed to those primitives, and when
per-episode delta state is read/updated) stays in sync between them.

These tests drive both backends' reward-computation methods from matched
synthetic geometric states, bypassing the two physics engines entirely (which
are not expected to produce bit-identical trajectories for the same actions --
see ``test_warp_cross_backend_camera.py``'s docstring). Bypassing physics is
what makes an exact numeric match a meaningful, deterministic assertion here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.warp

NUM_ENVS = 2
REL_TOL = 1e-5


def _pick_lift_envs():
    import gymnasium as gym

    import so101_nexus.mujoco
    import so101_nexus.warp  # noqa: F401 - registers Warp*-v1
    from so101_nexus.config import PickConfig
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    m_env = gym.make("MuJoCoPickLift-v1")
    m = m_env.unwrapped
    m.reset(seed=0)
    w = WarpPickLiftVectorEnv(num_envs=NUM_ENVS, config=PickConfig(), device="cpu", seed=0)
    w.reset(seed=0)
    return m_env, m, w


def _set_pick_lift_state(m, w, *, tcp_to_obj_dist: float, is_grasped: float, lift_height: float):
    """Pin both backends to the same logical geometry (not physics state).

    A common ``_initial_obj_z`` baseline of 0 makes ``lift_height`` mean the
    same thing in both backends; overriding the pose/grasp accessors (rather
    than hand-supplying ``success``) lets each backend derive ``success``
    itself from the same ``lift_height``/``lift_threshold`` comparison, so a
    mismatch there is caught rather than papered over.
    """
    import torch

    m._initial_obj_z = 0.0
    w._initial_obj_z[:] = 0.0

    obj = [0.0, 0.0, lift_height]
    tcp = [tcp_to_obj_dist, 0.0, lift_height]
    m._get_target_pose = lambda: np.array([*obj, 1.0, 0.0, 0.0, 0.0])
    m._get_tcp_pose = lambda: np.array([*tcp, 1.0, 0.0, 0.0, 0.0])
    m._is_grasping = lambda: float(is_grasped)

    w._target_pos = lambda: torch.tensor([obj] * NUM_ENVS, dtype=torch.float32)
    w._tcp_pos = lambda: torch.tensor([tcp] * NUM_ENVS, dtype=torch.float32)
    w._is_grasping = lambda: torch.full((NUM_ENVS,), float(is_grasped))
    w._is_robot_static = lambda: torch.zeros(NUM_ENVS, dtype=torch.bool)


def _reset_prev_state(m, w, *, reach: float = 0.0, grasp: float = 0.0, task: float = 0.0):
    m._prev_reach_progress = reach
    m._prev_grasp_progress = grasp
    m._prev_task_potential = task
    w._prev_reach_progress[:] = reach
    w._prev_grasp_progress[:] = grasp
    w._prev_task_potential[:] = task


def test_pick_lift_reward_matches_mujoco_across_trajectory():
    """Reach, grasp, dwell, and lift-to-success all pay identical reward in
    both backends, driven from the same logical (not physical) trajectory.

    The dwell step (unchanged state, reward held constant at 0) and the
    success step (reward clamped to 1.0) are load-bearing: those are exactly
    the two behaviors docs/superpowers/plans/
    2026-07-16-pick-grasp-potential-shaping.md's fix touches, so a backend
    that drifted on either would be caught here, not just by each backend's
    own dwelling-regression test in isolation.
    """
    import torch

    m_env, m, w = _pick_lift_envs()
    try:
        _reset_prev_state(m, w)
        trajectory: list[tuple[float, float, float]] = [
            (0.30, 0.0, 0.0),  # reaching
            (0.15, 0.0, 0.0),  # reaching further
            (0.02, 0.0, 0.0),  # reached
            (0.02, 1.0, 0.0),  # grasp
            (0.02, 1.0, 0.0),  # dwell: grasped, not lifting -- must plateau
            (0.02, 1.0, 0.06),  # lift above the default 0.05 m threshold -> success
        ]
        zero = torch.zeros(NUM_ENVS)
        for tcp_to_obj_dist, is_grasped, lift_height in trajectory:
            _set_pick_lift_state(
                m,
                w,
                tcp_to_obj_dist=tcp_to_obj_dist,
                is_grasped=is_grasped,
                lift_height=lift_height,
            )
            info_m = m._get_info()
            r_m = m._compute_reward(info_m)
            r_w, success_w, _ = w._compute_reward_terminated(zero, zero)

            assert r_m == pytest.approx(float(r_w[0]), abs=REL_TOL), (
                tcp_to_obj_dist,
                is_grasped,
                lift_height,
            )
            assert bool(info_m["success"]) == bool(success_w[0])
    finally:
        m_env.close()


def _pick_and_place_envs():
    import gymnasium as gym

    import so101_nexus.mujoco
    import so101_nexus.warp  # noqa: F401
    from so101_nexus.config import PickAndPlaceConfig
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceVectorEnv

    m_env = gym.make("MuJoCoPickAndPlace-v1")
    m = m_env.unwrapped
    m.reset(seed=0)
    w = WarpPickAndPlaceVectorEnv(
        num_envs=NUM_ENVS, config=PickAndPlaceConfig(), device="cpu", seed=0
    )
    w.reset(seed=0)
    return m_env, m, w


@pytest.mark.parametrize(
    ("obj_xy", "height", "speed", "grasped", "placed"),
    [
        (0.02, 0.05, 1.0, 1.0, False),  # elevated, arm fast -> low potential
        (0.02, 0.05, 0.02, 1.0, False),  # elevated, arm nearly still -> partial credit
        (0.0, 0.0, 0.0, 1.0, True),  # placed and still -> full potential
        (0.3, 0.05, 0.5, 0.0, False),  # ungrasped, far -> gated to 0
    ],
)
def test_pick_and_place_task_potential_formula_matches_mujoco(
    obj_xy, height, speed, grasped, placed
):
    """``_task_potential`` (Phi_place) is independently implemented per
    backend (it composes shared ``rewards.reach_progress`` calls itself,
    unlike the fully-shared ``RewardConfig.compute``), so its geometric
    formula -- not just the reward wiring around it -- gets its own direct
    parity check across a spread of xy/height/speed/gate combinations.
    """
    import torch

    m_env, m, w = _pick_and_place_envs()
    try:
        m._initial_obj_z = 0.0
        w._initial_obj_z[:] = 0.0
        m.data.qvel[m._arm_qvel_addrs] = speed
        w.qvel[:, w._arm_dof_adr] = speed

        obj_pos_m = np.array([obj_xy, 0.0, height])
        target_pos_m = np.array([0.0, 0.0, 0.0])
        phi_m = m._task_potential(obj_pos_m, target_pos_m, grasped, placed)

        obj_pos_w = torch.tensor([[obj_xy, 0.0, height]] * NUM_ENVS)
        target_pos_w = torch.tensor([[0.0, 0.0, 0.0]] * NUM_ENVS)
        phi_w = w._task_potential(
            obj_pos_w,
            target_pos_w,
            torch.full((NUM_ENVS,), grasped),
            torch.full((NUM_ENVS,), placed),
        )

        assert phi_m == pytest.approx(float(phi_w[0]), abs=REL_TOL)
    finally:
        m_env.close()


def _set_pick_and_place_state(
    m, w, *, tcp_to_obj_dist: float, is_grasped: float, task_potential: float, success: bool
):
    """Stub out the geometric potential itself (covered separately above) and
    pin both backends to the same reach/grasp/placement/staticness inputs.
    """
    import torch

    obj = [tcp_to_obj_dist, 0.0, 0.0]
    m._get_object_pose = lambda: np.array([*obj, 1.0, 0.0, 0.0, 0.0])
    m._get_tcp_pose = lambda: np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    m._is_grasping = lambda: float(is_grasped)
    m._task_potential = lambda *a, **k: task_potential
    m._obj_placement_state = lambda *a, **k: (0.0, success)
    m._is_robot_static = lambda: success

    w._target_pos = lambda: torch.tensor([obj] * NUM_ENVS, dtype=torch.float32)
    w._tcp_pos = lambda: torch.zeros(NUM_ENVS, 3)
    w._target_disc_pos = lambda: torch.zeros(NUM_ENVS, 3)
    w._is_grasping = lambda: torch.full((NUM_ENVS,), float(is_grasped))
    w._task_potential = lambda *a, **k: torch.full((NUM_ENVS,), task_potential)
    w._obj_placement_state = lambda *a, **k: (
        torch.zeros(NUM_ENVS),
        torch.full((NUM_ENVS,), success),
    )
    w._is_robot_static = lambda: torch.full((NUM_ENVS,), success, dtype=torch.bool)


def test_pick_and_place_reward_matches_mujoco_across_trajectory():
    """Reach, grasp, carry (dwelling on task potential), and place-to-success
    all pay identical reward in both backends. Companion to the PickLift
    trajectory test above for the other multi-phase task.
    """
    import torch

    m_env, m, w = _pick_and_place_envs()
    try:
        _reset_prev_state(m, w)
        trajectory: list[tuple[float, float, float, bool]] = [
            (0.30, 0.0, 0.0, False),  # reaching
            (0.02, 0.0, 0.0, False),  # reached
            (0.02, 1.0, 0.0, False),  # grasp
            (0.02, 1.0, 0.35, False),  # carrying toward the goal: genuine progress
            (0.02, 1.0, 0.35, False),  # dwell: hovering at the same potential -- must plateau
            (0.02, 1.0, 1.0, True),  # placed and static -> success
        ]
        zero = torch.zeros(NUM_ENVS)
        for tcp_to_obj_dist, is_grasped, task_potential, success in trajectory:
            _set_pick_and_place_state(
                m,
                w,
                tcp_to_obj_dist=tcp_to_obj_dist,
                is_grasped=is_grasped,
                task_potential=task_potential,
                success=success,
            )
            info_m = {
                "tcp_to_obj_dist": tcp_to_obj_dist,
                "is_grasped": is_grasped,
                "task_potential": task_potential,
                "success": success,
            }
            r_m = m._compute_reward(info_m)
            r_w, success_w, _ = w._compute_reward_terminated(zero, zero)

            assert r_m == pytest.approx(float(r_w[0]), abs=REL_TOL), (
                tcp_to_obj_dist,
                is_grasped,
                task_potential,
                success,
            )
            assert success == bool(success_w[0])
    finally:
        m_env.close()
