"""Pure reward and progress functions shared across simulation backends.

``reach_progress``, ``orientation_progress``, ``lift_progress``, and
``simple_reward`` accept Python floats, NumPy arrays, and torch tensors,
dispatching by duck typing so the scalar MuJoCo backend can call them after
scalar extraction and the batched MuJoCo Warp backend can call the same formulas
on tensors. ``torch`` is never imported here; tensor support relies on operator
overloading.
"""

from __future__ import annotations

import math

import numpy as np


def reach_progress(distance, *, scale):
    """Tanh-shaped progress in [0, 1]: 1 at distance 0, decaying toward 0.

    Accepts a Python float, NumPy array, or torch tensor. Negative distances are
    clamped to 0. The scalar path returns a plain ``float`` and is bit-identical
    to the original implementation.

    Parameters
    ----------
    distance : float or numpy.ndarray or torch.Tensor
        Non-negative distance(s) to target.
    scale : float
        Positive shaping steepness (``RewardConfig.tanh_shaping_scale``).
    """
    if hasattr(distance, "tanh"):  # torch.Tensor
        return 1.0 - (scale * distance.clamp(min=0.0)).tanh()
    if isinstance(distance, np.ndarray):
        return 1.0 - np.tanh(scale * np.clip(distance, 0.0, None))
    return 1.0 - math.tanh(scale * max(0.0, distance))


def orientation_progress(cos_similarity):
    """Map cosine similarity in [-1, 1] to a reward in [0, 1].

    Accepts a Python float, NumPy array, or torch tensor (dispatch by duck
    typing, like ``reach_progress``). Values outside [-1, 1] are clamped. The
    scalar path returns a plain ``float``.

    Parameters
    ----------
    cos_similarity : float or numpy.ndarray or torch.Tensor
        Cosine similarity between current and target direction.
    """
    if hasattr(cos_similarity, "clamp"):  # torch.Tensor
        return (cos_similarity.clamp(-1.0, 1.0) + 1.0) / 2.0
    if isinstance(cos_similarity, np.ndarray):
        return (np.clip(cos_similarity, -1.0, 1.0) + 1.0) / 2.0
    return (max(-1.0, min(1.0, cos_similarity)) + 1.0) / 2.0


def lift_progress(height, *, scale, grasped):
    """Tanh-shaped lift progress in [0, 1], zero unless grasped.

    Formula: ``tanh(scale * max(height, 0)) * grasped``. Accepts a Python float,
    NumPy array, or torch tensor for ``height``; ``grasped`` is a matching
    bool/float scalar or batch (operator overloading promotes bool to float).
    The scalar path returns a plain ``float`` and is bit-identical to the inline
    lift shaping it replaces.

    Parameters
    ----------
    height : float or numpy.ndarray or torch.Tensor
        Lift height above the per-episode baseline (metres).
    scale : float
        Positive shaping steepness (``RewardConfig.tanh_shaping_scale``).
    grasped : bool or numpy.ndarray or torch.Tensor
        Whether the object is currently grasped; ungrasped contributes 0.
    """
    if hasattr(height, "tanh"):  # torch.Tensor
        return (scale * height.clamp(min=0.0)).tanh() * grasped
    if isinstance(height, np.ndarray):
        return np.tanh(scale * np.clip(height, 0.0, None)) * grasped
    return math.tanh(scale * max(0.0, height)) * float(grasped)


def simple_reward(*, progress, completion_bonus, success):
    """Reward for single-objective tasks (reach, move, look-at).

    Shaping fills ``[0, 1 - completion_bonus]`` via ``progress``; success lifts
    the reward to the full budget (1.0), so a successful terminal step is always
    the global maximum with ``completion_bonus`` as the guaranteed margin. This
    matches the terminal clamp in ``RewardConfig.compute``. Accepts scalar or
    batched ``progress`` / ``success``; ``success`` may be a Python bool, NumPy
    bool array, or torch bool tensor (operator overloading promotes bool to
    float). The scalar path returns a plain ``float``.

    Parameters
    ----------
    progress : float or numpy.ndarray or torch.Tensor
        Task progress in [0, 1].
    completion_bonus : float
        Fraction of reward budget reserved for completion, in [0, 1].
    success : bool or numpy.ndarray or torch.Tensor
        Whether the task is complete.
    """
    shaped = (1.0 - completion_bonus) * progress
    return shaped + (1.0 - shaped) * success


def potential_shaping(potential, prev_potential):
    """Potential-based shaping delta: ``potential(s') - potential(s)``.

    The ``gamma=1`` telescoping special case of Ng, Harada & Russell's
    potential-based reward shaping theorem ("Policy Invariance Under Reward
    Transformations," ICML 1999, Theorem 1): a shaping reward of the form
    ``F(s, s') = gamma * Phi(s') - Phi(s)`` is a necessary and sufficient
    condition for the shaped MDP's optimal policy to equal the unshaped one's.
    Summed over an episode this term telescopes to ``Phi(final) - Phi(initial)``,
    bounded regardless of episode length, so a policy that reaches a high
    potential and then stops moving earns ~0 further reward instead of the
    unbounded dwelling reward a raw ``Phi(s')`` term would pay out every step
    (see ``RewardConfig`` callers for the concrete exploit this closes).
    ``gamma=1`` rather than Devlin & Kudenko's fully general dynamic potential
    ("Dynamic Potential-Based Reward Shaping," AAMAS 2012, Eq. 4) because
    callers here do not know the training algorithm's discount factor; for
    ``gamma`` close to 1 (typical PPO/FPO settings, 0.95-0.99) this is a close
    approximation, and it is exact at ``gamma=1``.

    Accepts a Python float, NumPy array, or torch tensor for either argument
    (plain subtraction; no dispatch needed, unlike the other functions here).

    Parameters
    ----------
    potential : float or numpy.ndarray or torch.Tensor
        Potential function value at the current state, ``Phi(s')``.
    prev_potential : float or numpy.ndarray or torch.Tensor
        Potential function value at the previous state, ``Phi(s)``.
    """
    return potential - prev_potential


def _elementwise_max(a, b):
    """Elementwise ``max(a, b)`` across Python floats, NumPy arrays, and torch tensors."""
    diff = b - a
    if hasattr(diff, "clamp"):  # torch.Tensor
        return a + diff.clamp(min=0.0)
    if isinstance(diff, np.ndarray):
        return a + np.clip(diff, 0.0, None)
    return max(a, b)


def place_reach_potential(tcp_to_obj_dist, is_obj_placed, *, scale):
    """Reach potential for place tasks: ``max(reach_progress, is_obj_placed)``.

    Once the object rests on the goal the reach sub-goal is moot, so
    ``is_obj_placed`` holds the potential at 1.0: retreating the arm after
    delivery pays no negative ``potential_shaping`` delta, while knocking the
    object off the goal still does. Accepts Python scalars, NumPy arrays, or
    torch tensors (duck-typed like ``reach_progress``).

    Parameters
    ----------
    tcp_to_obj_dist : float or numpy.ndarray or torch.Tensor
        Distance from the TCP to the carried object (metres).
    is_obj_placed : bool or numpy.ndarray or torch.Tensor
        Whether the object currently rests on the goal.
    scale : float
        Positive shaping steepness (``RewardConfig.tanh_shaping_scale``).
    """
    return _elementwise_max(reach_progress(tcp_to_obj_dist, scale=scale), is_obj_placed * 1.0)


def place_grasp_potential(is_grasped, is_obj_placed):
    """Grasp potential for place tasks: 1.0 while the object is held or delivered.

    Raw ``is_grasped`` treats the mandatory release on the goal as a
    regression and pays a -1 ``potential_shaping`` delta at the finish; a
    sub-goal that a later phase must undo has to be held up by its successor
    condition. With the ``is_obj_placed`` hold, releasing on the goal pays 0
    while dropping the object mid-carry still pays the full negative delta.
    Accepts Python scalars, NumPy arrays, or torch tensors.

    Parameters
    ----------
    is_grasped : float or numpy.ndarray or torch.Tensor
        Whether the object is currently grasped (thresholded at 0.5).
    is_obj_placed : bool or numpy.ndarray or torch.Tensor
        Whether the object currently rests on the goal.
    """
    return ((is_grasped > 0.5) | is_obj_placed) * 1.0


def place_task_potential(
    obj_to_target_xy, height_gap, arm_speed, is_grasped, is_obj_placed, *, scale, velocity_scale
):
    """``Phi_place(s)``: staged transport-then-settle place progress in [0, 1].

    ``gate * 0.5 * (transport + is_obj_placed * still)`` where ``transport =
    reach_progress(max(obj_to_target_xy, height_gap))``, ``still =
    reach_progress(arm_speed, velocity_scale)``, and ``gate`` is
    grasped-or-placed. Monotone non-decreasing along the ideal
    grasp-lift-carry-lower-settle trajectory, the property a potential fed
    through ``potential_shaping`` needs for forward progress to pay a
    positive per-step delta:

    - The Chebyshev ``max`` distance makes the mandatory lift free while far
      from the goal (xy dominates), unlike a height-back-near-rest factor
      (which pays negative on lift-off) or a plain 3D norm (which still dips
      slightly), and hands the gradient to lowering once above the goal.
    - The stillness term is additive and gated on ``is_obj_placed``
      (ManiSkill PickCube's ``static_reward * is_obj_placed``), so it shapes
      only the final settle instead of multiplying the transport gradient
      down to ~0 at realistic carry speeds, which a stillness *factor* does.

    See docs/superpowers/plans/2026-07-16-monotone-place-potential.md for the
    audit that motivated this shape. Accepts Python scalars, NumPy arrays, or
    torch tensors.

    Parameters
    ----------
    obj_to_target_xy : float or numpy.ndarray or torch.Tensor
        Horizontal distance from the object to the goal centre (metres).
    height_gap : float or numpy.ndarray or torch.Tensor
        Object height above its per-episode rest baseline, clamped >= 0 (metres).
    arm_speed : float or numpy.ndarray or torch.Tensor
        Norm of the arm joint velocities (rad/s).
    is_grasped : float or numpy.ndarray or torch.Tensor
        Whether the object is currently grasped (thresholded at 0.5).
    is_obj_placed : bool or numpy.ndarray or torch.Tensor
        Whether the object currently rests on the goal.
    scale : float
        Positive shaping steepness (``RewardConfig.tanh_shaping_scale``).
    velocity_scale : float
        Stillness shaping steepness (``RewardConfig.velocity_shaping_scale``).
    """
    gate = (is_grasped > 0.5) | is_obj_placed
    transport = reach_progress(_elementwise_max(obj_to_target_xy, height_gap), scale=scale)
    still = reach_progress(arm_speed, scale=velocity_scale)
    return gate * 0.5 * (transport + is_obj_placed * still)
