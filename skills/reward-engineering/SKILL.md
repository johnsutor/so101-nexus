---
name: reward-engineering
description: Use when designing or reviewing a reward function for a new so101-nexus environment (or auditing an existing one), especially any task with a multi-phase or dwelling-prone completion condition (grasp-then-release, reach-then-hold, multi-object sequencing). Covers dwelling-vs-potential-based shaping, keeping the potential monotone along the ideal trajectory, how to spot a reward-hacking trap before it costs a training run, and the primitives/citations to use.
---

# Reward engineering for so101-nexus environments

Procedural guide for writing a new task's reward, or auditing an existing one, so
dense shaping helps training instead of becoming the thing the policy learns to
game. Grounded in a real, previously-shipped bug: `PickAndPlaceEnv`'s
`task_progress` term let a policy collect ~90% of the reward budget forever by
hovering a grasped object above the goal without ever placing it (see
`docs/superpowers/plans/2026-07-12-potential-based-task-progress-shaping.md` for
the full writeup and the fix).

## The core failure mode: dwelling rewards

A **dwelling** reward term is a function of the *current state alone*,
recomputed fresh every step: `r_t = f(s_t)`. If `f` saturates near its maximum
before the task's actual `success` condition fires, and the episode does not
terminate until `success`, a policy that reaches a high-`f` state and then does
*nothing further* collects that near-maximum reward every remaining step of the
episode -- often far more total return than a policy that pushes on to finish
and terminate early. This is not a hypothetical: `Amodei, Olah, Steinhardt,
Christiano, Schulman & Mane, "Concrete Problems in AI Safety," arXiv:1606.06565
(2016)` name this *reward hacking* as one of five concrete AI-accident problem
classes, and `Krakovna, Uesato, Mikulik, Rahtz, Everitt, Kumar, Kenton, Leike &
Legg, "Specification gaming: the flip side of AI ingenuity" (DeepMind, 2020,
tinyurl.com/specification-gaming)` catalogue ~60 real instances, including a
Lego-stacking agent (`Popov, Heess, Lillicrap, Hafner, Barth-Maron, Vecerik,
Lampe, Tassa, Erez & Riedmiller, "Data-efficient Deep RL for Dexterous
Manipulation," arXiv:1704.03073, 2017`) rewarded for a block's height that
simply flipped the block instead of stacking it.

**Multi-phase tasks are the highest-risk shape.** A task whose `success`
requires state B *after* first passing through state A (e.g. grasp-then-lower,
reach-then-release, pick-then-place) is dangerous whenever the natural,
easiest-to-explore trajectory reaches a *dwelling-rewarding* state A and stops
there, because completing the task requires *reversing or complicating* the
just-learned behaviour (releasing a grasp, lowering back down, decelerating).
Single-phase tasks (reach a point, look at an object) are comparatively safe:
crossing the completion threshold is not a reversal of the most-explored
trajectory, so the same failure mode is far less likely to be where training
actually gets stuck -- but it is not categorically impossible, so still check.

## Checklist when writing a reward for a new task

1. **Read `so101_nexus/rewards.py` first.** Reuse `reach_progress`,
   `orientation_progress`, `lift_progress`, `simple_reward`,
   `potential_shaping`, `place_task_potential`, `place_reach_potential`,
   `place_grasp_potential` -- never inline a shaping formula. If none of these
   fit, propose a new one there (never in a backend env class); see
   `CLAUDE.md`/`AGENTS.md`'s "Reward functions live in `so101_nexus.rewards`" rule.
2. **Write down `success` first**, as a precise boolean predicate over state.
   Then ask: is there a *strict subset* of `success`'s conditions that a
   dwelling reward term would saturate on? If yes, that subset is the exploit
   surface -- this is exactly what happened when `task_progress` rewarded
   goal-xy proximity alone while `success` additionally required height and
   staticness.
3. **For any term that is part of, or a proxy for, the multi-phase completion
   surface identified in step 2**: make it a **potential-based delta**, not a
   raw state value.
   - Define a potential function `Phi(s)` in `[0, 1]` that is zero until every
     phase relevant to that term has genuinely started (e.g. gate on "grasped
     or already placed"). Build it as **boolean phase gates around an
     additive, staged progress sum** (ManiSkill PickCube's dense-reward shape:
     `reaching + is_grasped + place*is_grasped + static*is_obj_placed`), NOT
     as a product of continuous progress factors. Products are how you gate
     *raw dwelling* terms, where coasting on one maxed factor is the threat; a
     potential *delta* already pays ~0 for coasting, so the product buys
     nothing and instead lets any one small factor mute every other factor's
     gradient -- an arm-stillness factor at realistic carry speed multiplied
     pick-and-place's entire transport gradient down to ~1e-7/step (see the
     worked example below).
   - **Walk `Phi` through the ideal trajectory phase by phase before shipping
     it** (approach, grasp, lift, carry, lower, release, settle) and check it
     never decreases: any dip means `potential_shaping` pays *negative* reward
     for mandatory forward progress. Shaping is policy-invariant for ANY
     `Phi`, so you are free to pick one that ranks states in the right order
     -- delta-shaping fixes dwelling, it cannot fix a potential that ranks a
     later phase below an earlier one. The three traps found in the shipped
     pick-and-place potential
     (`docs/superpowers/plans/2026-07-16-monotone-place-potential.md`):
     1. *Undo-factors*: a height-back-near-rest factor pays negative on the
        lift the task requires. Measure transport with a distance under which
        the lift is free, e.g. Chebyshev `max(xy_dist, height_gap)` (a plain
        3D norm still dips slightly on lift-off).
     2. *Gradient-muting products*: stillness x transport zeroes the carry
        gradient at any realistic arm speed. Make stillness additive and gate
        it on `is_obj_placed` (ManiSkill's `static_reward * is_obj_placed`).
     3. *Missing successor holds*: a sub-potential a later phase must undo
        (grasp before release, reach before retreat) pays a negative delta at
        the finish unless held up by its successor condition: `grasped OR
        placed` (`rewards.place_grasp_potential`), `max(reach, placed)`
        (`rewards.place_reach_potential`). Keep the negative delta for
        genuine regressions -- if dropping the object paid 0 while grasping
        paid +, a grasp-drop-regrasp cycle would pump unbounded reward;
        symmetric deltas make every closed loop net exactly 0.
   - Feed `rewards.potential_shaping(Phi(s'), Phi(s))` as the term's value
     instead of `Phi(s')` directly. This requires tracking `Phi(s_prev)` as
     per-episode state on the env (a plain attribute for the scalar MuJoCo
     backend, a `(num_envs,)` tensor updated by index for the batched Warp
     backend), seeded post-settle in `_refresh_reset_reference_state` (mirror
     the existing `_initial_obj_z` baseline pattern) and updated once per real
     step inside `_compute_reward`/`_compute_reward_terminated` *after* reading
     the previous value.
   - This is the `gamma=1` telescoping special case of `Ng, Harada & Russell,
     "Policy Invariance Under Reward Transformations: Theory and Application to
     Reward Shaping," ICML 1999` (Theorem 1: `F(s,a,s') = gamma*Phi(s') -
     Phi(s)` is necessary and sufficient for the shaped MDP's optimal policy to
     equal the unshaped MDP's). Summed over an episode it telescopes to
     `Phi(final) - Phi(initial)`, bounded regardless of dwell time, so standing
     still nets ~0 further reward. Use `gamma=1` rather than a full discount
     because the env does not know the training algorithm's `gamma`; this is
     exact at `gamma=1` and a close approximation for `gamma` near 1 (typical
     PPO/FPO settings, 0.95-0.99). If a term's potential is genuinely
     phase/time-dependent (not just state-dependent), see `Devlin & Kudenko,
     "Dynamic Potential-Based Reward Shaping," AAMAS 2012` (Eq. 4), which
     extends the same guarantee to `Phi(s, t)`.
   - **Do not soften `success`/`terminated` itself into a continuous proxy to
     get partial credit.** Get partial credit from the *shaping* potential
     instead. A soft success proxy that never has to cross the real threshold
     just relocates the exploit to a new, not-yet-discovered shape (a policy
     that maximizes the soft proxy without ever really finishing) -- this is
     the mistake to avoid, not a hypothetical: see the "Concrete changes"
     section of the design note above for the reasoning trail.
4. **Terms that are NOT part of the completion surface** (e.g. a generic
   TCP-to-object reach term while the object is still far away) can stay flat
   dwelling terms at a modest budget weight; not everything needs potential
   shaping, only the specific surface a policy could exploit to avoid
   finishing. Reserve the biggest weight cut for `completion_bonus`, not the
   dwelling terms, if you are tuning a budget split.
5. **Verify `terminated == success`** in both backends' `step()` (already the
   base-class contract) so a real completion always ends the episode -- a
   potential-based term only closes the *dwelling* exploit; if the episode
   never ends on success, dwelling on the terminal state itself becomes the new
   exploit.
6. **Write the regression test before you trust the fix.** Simulate "hover"
   (call the reward computation twice with an unchanged potential) and assert
   the potential-shaped facet of `info["reward_components"]` is ~0 on the
   second call; then simulate genuine progress (potential increases) and assert
   real credit is paid. See `test_pick_and_place_hovering_earns_no_dwelling_task_objective_reward`
   in `tests/mujoco/test_envs.py` for the pattern.
7. Consider the sparse-reward alternative. `Andrychowicz, Wolski, Ray,
   Schneider, Fong, Welinder, McGrew, Tobin, Abbeel & Zaremba, "Hindsight
   Experience Replay," NeurIPS 2017 (arXiv:1707.01495)` is the other major
   line of attack on this class of problem: train off-policy on a sparse
   binary success signal and relabel failed trajectories with the goal they
   *actually* reached, removing the dense-shaping exploit surface entirely by
   not having one. Not applicable to this repo's on-policy training recipes
   today (HER needs an off-policy, replay-buffer algorithm and a
   goal-conditioned value function), but worth naming when proposing a new
   training recipe, not just a new env.

## Citations (primary sources, read in full before citing further)

- Amodei, Olah, Steinhardt, Christiano, Schulman & Mane, "Concrete Problems in
  AI Safety," arXiv:1606.06565 (2016). https://arxiv.org/abs/1606.06565
- Krakovna, Uesato, Mikulik, Rahtz, Everitt, Kumar, Kenton, Leike & Legg,
  "Specification gaming: the flip side of AI ingenuity," Google DeepMind blog
  (2020), with the maintained master list at tinyurl.com/specification-gaming.
  https://deepmind.google/discover/blog/specification-gaming-the-flip-side-of-ai-ingenuity/
- Ng, Harada & Russell, "Policy Invariance Under Reward Transformations: Theory
  and Application to Reward Shaping," ICML 1999, pp. 278-287.
  https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Devlin & Kudenko, "Dynamic Potential-Based Reward Shaping," AAMAS 2012,
  pp. 433-440. https://eprints.whiterose.ac.uk/id/eprint/75121/2/p433_devlin.pdf
- Popov, Heess, Lillicrap, Hafner, Barth-Maron, Vecerik, Lampe, Tassa, Erez &
  Riedmiller, "Data-efficient Deep Reinforcement Learning for Dexterous
  Manipulation," arXiv:1704.03073 (2017). https://arxiv.org/abs/1704.03073
- Andrychowicz, Wolski, Ray, Schneider, Fong, Welinder, McGrew, Tobin, Abbeel &
  Zaremba, "Hindsight Experience Replay," NeurIPS 2017, arXiv:1707.01495.
  https://arxiv.org/abs/1707.01495

## Worked example

`docs/superpowers/plans/2026-07-12-potential-based-task-progress-shaping.md` is
a complete worked application of this checklist to `PickAndPlaceEnv` and
`PickLiftEnv` (both backends): problem diagnosis, the decision with citations
mapped to concrete design choices, exact scope, and non-goals (why
Touch/Move/LookAt were left unchanged). Read it alongside
`so101_nexus/rewards.py`'s `potential_shaping` and `place_task_potential`
docstrings for the concrete implementation pattern to copy.

That plan's original "Non-goals" left `reaching`/`grasping` as flat dwelling
terms, reasoning they carried a smaller combined budget and were not (yet) an
exploited surface -- explicitly flagged to revisit "if evidence of the same
failure mode surfaces there." It did:
`docs/superpowers/plans/2026-07-16-pick-grasp-potential-shaping.md` is the
follow-up that converts `reaching`/`grasping` to potential-shaped deltas too,
once fixing `task_objective` alone made their combined flat budget the new
dominant dwelling ceiling. Read both plans together; the second is a second
application of this same checklist, not a different pattern.

`docs/superpowers/plans/2026-07-16-monotone-place-potential.md` is the third
application, and the origin of checklist step 3's monotonicity walk: with
every facet correctly delta-shaped, a recorded teleop episode's step-reward
graph exposed that the potential itself was not monotone along the ideal
trajectory -- the mandatory lift paid -0.09, carrying toward the goal paid
~1e-7 (stillness-muted product), and releasing on the goal paid -0.25, so a
perfect demonstration earned nothing but the terminal bonus. The fix swapped
the product potential for the staged additive `place_task_potential` and
added the `is_obj_placed` successor holds. Lesson: the 0.4.8/0.4.9 fixes
chose the right *mechanism* (deltas) around the wrong *potential*; audit both.

## Discovery

The skill source lives in-repo so it is versioned with the project. To make it
discoverable, copy or symlink this directory into
`${CODEX_HOME:-$HOME/.codex}/skills/reward-engineering`. Keeping the source
in-repo keeps it in sync with the codebase.
