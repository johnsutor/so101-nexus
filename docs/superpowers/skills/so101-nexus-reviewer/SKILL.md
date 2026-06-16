---
name: so101-nexus-reviewer
description: Use when reviewing changes to the so101-nexus sim backends (MuJoCo, ManiSkill, core) for cross-backend correctness, action-space and observation parity, seeded RNG, and reward-logic placement.
---

# so101-nexus reviewer

Procedural checklist for reviewing changes to the so101-nexus sim backends.

1. Read `AGENTS.md` or `CLAUDE.md` first to load project rules and the
   reviewer checklist.
2. For any change touching an env, check the relevant backend PAIR
   (MuJoCo and ManiSkill) for public API parity: action spaces, observation
   components, and task semantics. Note any simulator-forced divergence and
   confirm it is documented at the point of divergence.
3. Search for config fields touched by the change and verify a test proves
   their runtime effect (no dead config knobs).
4. Check that all RNG sources in reset, scene load, camera setup, and
   object/color sampling are seeded: `self.np_random` (MuJoCo), the ManiSkill
   episode RNG (`self._episode_rng`), or a seeded torch RNG. Flag any global
   `np.random` or Python `random` usage.
5. Check reward code against `so101_nexus_core.rewards` and
   `RewardConfig.compute` / `apply_penalties`. No inlined reward formulas in
   env classes; tensor-form backends must cite the core function in a comment.
6. Run the narrowest relevant test command first, then broaden:
   `make test-core`, `make test-mujoco`, or `make test-maniskill`, then
   `make test`.

## Discovery

The skill source lives in-repo so it is versioned with the project. To make it
discoverable, copy or symlink this directory into
`${CODEX_HOME:-$HOME/.codex}/skills/so101-nexus-reviewer`. Keeping the source
in-repo keeps it in sync with the codebase.
