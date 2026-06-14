# Simulation Drift Housekeeping Design

Date: 2026-06-14
Branch: `chore/sim-drift-housekeeping`

## Review Inputs

This housekeeping branch addresses validated findings from these local reviews:

- `LOGICAL-FLAWS.md`: runtime logic flaws, especially findings 1-11.
- `SIMULATION-DISCREPANCIES.md`: cross-backend behavior drift, especially findings 1, 3-7, 11-13, 16, 17, and 20.
- `CODE-INCONSISTENCIES.md`: style and public API consistency drift, especially findings 1, 5-8, 12, 16, and 19.
- `REVIEW.md`: ManiSkill menagerie follow-ups, especially seeded camera RNG, vectorized test skip hygiene, docs asset split, and no-em-dash prose.

Validated source anchors:

- Reach target workspace drift: `LOGICAL-FLAWS.md:19`, `SIMULATION-DISCREPANCIES.md:10`.
- Reward penalty drift: `LOGICAL-FLAWS.md:51`, `LOGICAL-FLAWS.md:83`.
- ManiSkill Pick sampling drift: `LOGICAL-FLAWS.md:112`, `LOGICAL-FLAWS.md:145`, `LOGICAL-FLAWS.md:170`, `SIMULATION-DISCREPANCIES.md:46`, `SIMULATION-DISCREPANCIES.md:63`, `SIMULATION-DISCREPANCIES.md:423`.
- Observation component drift: `LOGICAL-FLAWS.md:208`, `SIMULATION-DISCREPANCIES.md:229`.
- Episode limit drift: `LOGICAL-FLAWS.md:238`.
- Delta action-space drift: `LOGICAL-FLAWS.md:265`.
- Reset phase and physics drift: `SIMULATION-DISCREPANCIES.md:101`, `SIMULATION-DISCREPANCIES.md:334`, `SIMULATION-DISCREPANCIES.md:350`.
- Cleanup drift: `CODE-INCONSISTENCIES.md:13`, `CODE-INCONSISTENCIES.md:175`, `CODE-INCONSISTENCIES.md:194`, `CODE-INCONSISTENCIES.md:214`, `CODE-INCONSISTENCIES.md:415`, `CODE-INCONSISTENCIES.md:499`, `CODE-INCONSISTENCIES.md:605`.
- Extra review follow-ups: `REVIEW.md:5`, `REVIEW.md:24`, `REVIEW.md:41`, `REVIEW.md:63`.

Some review items were stale or already handled in this checkout:

- `CODE-INCONSISTENCIES.md` says 30+ source files miss `from __future__ import annotations`; a fresh count found 7 active source files missing it.
- `REVIEW.md` says SO101 ManiSkill wrist-camera randomization uses global `np.random`; current code uses the seeded episode RNG for SO101, but SO100 still uses global `np.random`.
- `REVIEW.md` says public docs still describe the old SO101 asset split; current source docs already describe `SO101_menagerie/so101.xml` as the model loaded by both MuJoCo and ManiSkill.
- `REVIEW.md` says vectorized fixture skips every construction exception; current shared helper narrows skips, but one local helper still catches broad exceptions and should be tightened.

## Goal

Make one beta housekeeping branch that fixes validated simulation semantics, cleans nearby public API and style drift, and adds guardrails that make future cross-backend drift harder to introduce.

## Non-Goals

- Do not add a new ManiSkill reach-only Pick env ID in this branch. That is a product/API decision, not a correctness fix.
- Do not try to make SO100 and SO101 physics constants identical. They are different robot models and should stay documented as such.
- Do not refactor all task logic into a new shared framework in this pass. Keep fixes scoped and test-backed.
- Do not normalize every historical style preference, such as all section comment styles or all dataclass/manual-init choices.

## Selected Approach

Use a staged housekeeping branch:

1. Add regression tests that reproduce the validated drift.
2. Fix reward penalties, episode limits, and public action semantics.
3. Fix target sampling, reset timing, Pick sampling, and observations.
4. Apply targeted public API and style cleanup.
5. Add prevention checks and reviewer guidance.

This keeps the branch coherent while preserving reviewable commits. It is broader than a minimal patch, but smaller and safer than a full shared task-spec refactor.

## Simulation Design

### Public Action Semantics

`ControlMode` should mean the same public action contract across backends. During beta, it is acceptable to break the MuJoCo delta action API to align with ManiSkill.

- Keep `pd_joint_pos` as physical joint-position targets.
- Change MuJoCo `pd_joint_delta_pos` and `pd_joint_target_delta_pos` action spaces to normalized `[-1, 1]` for all six joints.
- Scale normalized MuJoCo delta actions internally by the existing physical delta scale:
  `[0.05, 0.05, 0.05, 0.05, 0.05, 0.2]`.
- Compute `energy_norm` and `action_delta_norm` on the public normalized action so penalties are comparable across backends.
- Update tests that currently expect MuJoCo physical delta bounds.

### Rewards and Episode Limits

Both backends should honor the public `RewardConfig` and `EnvironmentConfig` fields.

- Track elapsed steps in each env instance and set `truncated=True` when `config.max_episode_steps` is reached.
- Reset elapsed step counters on full and partial resets.
- Store the previous public action per episode.
- Add `action_delta_norm` to `info`, with zero for the first step after reset.
- Add `energy_norm` to `info` everywhere rewards may use it.
- Route Reach, Move, and LookAt rewards through penalty-aware logic rather than bare `simple_reward`.
- Keep default rewards behaviorally unchanged when penalty coefficients are zero.

### Reach, Move, and LookAt

The public task semantics should match across backends unless a simulator limitation is explicitly documented.

- Make MuJoCo Reach sample targets from the same 3-D `target_workspace_half_extent` contract as ManiSkill.
- Update the existing MuJoCo floor-target test to assert 3-D workspace behavior instead of preserving the old floor-only implementation.
- Make MuJoCo Move compute its target after reset settling, so initial `tcp_to_target_dist` stays at `target_distance` and matches ManiSkill.
- Make the MuJoCo LookAt target non-colliding or otherwise kinematic-equivalent, so accidental contact cannot move the reference frame.

### ManiSkill Pick and PickAndPlace

ManiSkill manipulation tasks should honor the same config surface as MuJoCo.

- Remove standalone unseeded RNG state for Pick target/distractor identity.
- Use the seeded ManiSkill episode RNG for scene reconfiguration sampling.
- Sample distractors without replacement.
- Default variable-object Pick scenes to reconfigure each episode. A hidden-pool actor design could support identity changes without reconfiguration later, but it is larger than this housekeeping pass.
- Enforce `PickConfig.min_object_separation` for ManiSkill target and distractors using per-row rejection sampling with object bounding radii.
- Fix PickAndPlace vectorized separation sampling by tracking a boolean invalid-row mask and resampling only invalid rows.
- Route Pick and PickAndPlace observations through `_build_obs_extra_from_components()`.
- Add task-specific component handlers for `ObjectPose`, `ObjectOffset`, `TargetPosition`, and `TargetOffset`.
- Pass `config.robot.grasp_force_threshold` to `agent.is_grasping(...)`.
- Pass `config.robot.static_vel_threshold` to `agent.is_static(...)`.

### Physics and Cameras

- Set ManiSkill `SimConfig(sim_freq=200, control_freq=50)` to match MuJoCo's explicit 0.005 s simulation step and 0.02 s control interval.
- Match ManiSkill human render camera FOV to MuJoCo's 45 degree overhead camera.
- Replace SO100 wrist-camera global `np.random` sampling with ManiSkill's seeded episode RNG.
- Keep the existing SO101 camera parity conversion test as the guard for MJCF baseline conversion.
- Make PickAndPlace task descriptions agree with the sampled visual colors, or make color selection deterministic from the config. The implementation should choose the smaller change that produces stable text/scene agreement.

## Cleanup Design

Keep cleanup targeted to validated drift:

- Convert Google-style `Args:` blocks in touched public modules to concise NumPy-style `Parameters`.
- Add `__all__` to `so101_nexus_core`, `so101_nexus_mujoco`, and `so101_nexus_maniskill`.
- Remove redundant `as X` aliases in `so101_nexus_core.__init__` where practical.
- Add missing `from __future__ import annotations` to the 7 active source files that lack it.
- Introduce public `CameraObservation`, keep `_CameraObservation = CameraObservation` as a compatibility alias, and migrate internal imports.
- Replace production `assert` guards in runtime paths with explicit exceptions or validation branches.
- Normalize reward citation comments where tensor or backend-specific code mirrors core reward logic.
- Remove em dashes and other forbidden prose characters from source and test prose that is owned by this project. Vendored assets may be excluded from this check.

## Prevention Design

### Tests

Add narrow prevention tests rather than relying only on reviewer memory:

- Cross-backend action-space parity for all shared `ControlMode` values.
- Config-effect tests for Reach workspace, reward penalties, episode truncation, observation components, and seeded Pick sampling.
- A docs/style consistency test for forbidden `Args:` in public source docstrings.
- A public API test that every public package `__init__.py` defines `__all__`.
- A prose check that covers source and test comments/docstrings for em dashes and emoji, excluding vendored assets.
- A test or helper assertion that broad vectorized ManiSkill construction exceptions are not converted to skips.

### Agent Guidance

Update both `AGENTS.md` and `CLAUDE.md` so future reviewers check:

- Public config fields must have executable coverage showing they affect runtime behavior.
- Shared `ControlMode` names must expose the same public action-space contract across backends.
- All reset-time randomness must use `self.np_random`, ManiSkill episode RNG, or seeded torch RNG.
- Primitive task semantics must stay consistent across backends, with any simulator limitation documented at the divergence.
- Observation components must be honored by every task and backend.
- Dynamic task descriptions must reflect episode state when colors or object identities are sampled.
- New tests and docs must avoid em dashes and emoji.

### Reviewer Skill

Version-control a project-specific reviewer skill source under:

`docs/superpowers/skills/so101-nexus-reviewer/SKILL.md`

The skill should be concise and procedural:

- Read `AGENTS.md` or `CLAUDE.md`.
- Check the relevant backend pair for public API parity.
- Search for config fields touched by the change and verify tests prove their runtime effect.
- Check all RNG sources in reset, scene load, camera setup, and object sampling.
- Check reward code against `so101_nexus_core.rewards` and `RewardConfig.compute`.
- Run the narrowest relevant test command first, then broaden.

The skill source can later be copied or symlinked into `${CODEX_HOME:-$HOME/.codex}/skills/so101-nexus-reviewer` for automatic discovery. Keeping the source in the repo makes the review workflow versioned with the project.

## Validation Plan

Run narrow tests first:

- Core style and API tests that do not import backends.
- MuJoCo tests that cover action spaces, Reach, Move, LookAt, rewards, truncation, and observations.
- ManiSkill tests that cover action spaces, rewards, truncation, Pick sampling, observations, cameras, and vectorized separation.

Then broaden:

- `make lint`
- `make typecheck`
- `make test-core`
- `make test-mujoco`
- `make test-maniskill`
- `make test`

If visual or docs source changes affect docs rendering, run `make docs-check`. If visual camera framing changes are material, run the relevant visual tests.

## Commit Plan

1. `test: capture simulation drift regressions`
2. `fix: align reward penalties and episode limits`
3. `fix: normalize mujoco delta action spaces`
4. `fix: align reach move and pick reset semantics`
5. `chore: clean public api and docstring drift`
6. `test: add drift prevention checks`
7. `docs: refine agent review guidance`

