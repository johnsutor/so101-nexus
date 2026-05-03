# SOLID / DRY Audit

This document lists concrete spots in the codebase that currently work against SOLID and DRY principles, with an emphasis on areas that are making the repository bulkier than it needs to be.

It is intentionally focused on structure, not behavior changes. The fixes below describe how to simplify the design without prescribing exact code.

## Scope

Reviewed primarily:

- `packages/so101-nexus-core/src/so101_nexus_core`
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco`
- `packages/so101-nexus-maniskill/src/so101_nexus_maniskill`
- backend test suites under `packages/*/tests`

Skipped for this audit:

- asset trees
- docs site content
- generated or vendored data

## Highest-Value Issues

### 1. `teleop/app.py` is a god module

Files:

- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:1`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:98`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:237`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:578`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:713`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:824`

Why this violates SOLID:

- `Single Responsibility`: the file handles CLI fallback parsing, session state mutation, async worker startup, error handling, recording lifecycle, review lifecycle, dataset persistence, and Gradio view construction.
- `Open/Closed`: adding a new teleop step or changing initialization flow requires editing the same large file in multiple places.

Why this violates DRY:

- The same state keys and UI transition concepts are repeated across callbacks.
- The module manually threads the same session/init values through many callback signatures.

Suggested fix:

- Split the module into smaller units by responsibility:
  - `teleop/view.py` for Gradio layout
  - `teleop/controllers/init.py` for init flow
  - `teleop/controllers/recording.py` for recording/review flow
  - `teleop/state.py` for typed session/init state objects
- Replace raw `dict` session state with typed dataclasses or small state objects.
- Move callback composition into a thin coordinator so UI wiring does not own business logic.

Expected payoff:

- Smaller files, easier testing, less callback argument churn, and less risk when changing one teleop stage.

### 2. `config.py` has too many responsibilities and repeated subclass patterns

Files:

- `packages/so101-nexus-core/src/so101_nexus_core/config.py:1`
- `packages/so101-nexus-core/src/so101_nexus_core/config.py:376`
- `packages/so101-nexus-core/src/so101_nexus_core/config.py:474`
- `packages/so101-nexus-core/src/so101_nexus_core/config.py:536`
- `packages/so101-nexus-core/src/so101_nexus_core/config.py:606`
- `packages/so101-nexus-core/src/so101_nexus_core/config.py:633`
- `packages/so101-nexus-core/src/so101_nexus_core/config.py:673`

Why this violates SOLID:

- `Single Responsibility`: one file owns pose definitions, robot constants, camera presets, reward math, base environment settings, and all task-specific configs.
- `Open/Closed`: new task configs require editing a large central module instead of extending smaller focused modules.

Why this violates DRY:

- Multiple config subclasses repeat the same constructor shape: `super().__init__(**kwargs)`, field assignment, validation, and default observation setup.
- Object normalization logic appears in more than one config type, for example `PickConfig` and `LookAtConfig`.

Suggested fix:

- Split the module by concern:
  - `config/base.py`
  - `config/reward.py`
  - `config/robot.py`
  - `config/cameras.py`
  - `config/tasks/*.py`
- Extract shared helpers for:
  - object list normalization
  - default observation population
  - common scalar validation
- Prefer small immutable config models or dataclasses so subclasses stop re-implementing boilerplate initialization.

Expected payoff:

- Lower constructor duplication, smaller import surface, and safer task-specific changes.

### 3. Backend task environments are copy-heavy

Files:

- `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/reach_env.py:20`
- `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/move_env.py:20`
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/reach_env.py:47`
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/move_env.py:47`
- `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_env.py:122`
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_env.py:179`

Why this violates SOLID:

- `Open/Closed`: each new task tends to clone an existing environment file and modify a few details.
- `Single Responsibility`: task classes often combine scene construction, reset logic, observation assembly, success evaluation, and reward shaping in one class.

Why this violates DRY:

- The ManiSkill `ReachEnv` and `MoveEnv` files share most of their structure.
- The MuJoCo `ReachEnv` and `MoveEnv` files share the same pattern.
- The pick-family environments in both backends repeat the same task pipeline with backend-specific mechanics mixed into it.

Suggested fix:

- Separate task definition from backend implementation.
- Define shared task specs for:
  - target generation
  - success metric
  - reward composition
  - observation extras
- Let each backend implement only the adapter layer that turns a task spec into simulator objects.
- For tasks that differ only by target sampler or scene decoration, use strategy objects instead of separate full environment classes.

Expected payoff:

- Less copy-paste between tasks, easier addition of future backends, and thinner env modules.

### 4. Backend CLI entrypoints duplicate nearly everything

Files:

- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/cli.py:1`
- `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/cli.py:1`

Why this violates SOLID:

- `Open/Closed`: every new CLI option must be added in parallel to both backend packages.

Why this violates DRY:

- Parser creation is effectively duplicated line-for-line.
- Dispatch logic is identical except for backend name and one MuJoCo environment variable.

Suggested fix:

- Move shared parser creation and teleop dispatch into `so101_nexus_core.teleop.cli`.
- Keep backend packages as tiny wrappers that supply:
  - backend name
  - optional backend-specific environment setup

Expected payoff:

- One place to evolve teleop CLI behavior and fewer backend-specific test branches.

### 5. Teleop backend/robot selection relies on string conditionals

Files:

- `packages/so101-nexus-core/src/so101_nexus_core/teleop/leader.py:105`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/leader.py:119`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/leader.py:129`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/session.py:29`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/session.py:79`

Why this violates SOLID:

- `Open/Closed`: support for a new robot type or backend means editing `if/elif` branches and env-id prefix logic.
- `Dependency Inversion`: higher-level teleop flow depends on string conventions and import details rather than explicit registries/interfaces.

Why this also hurts maintainability:

- `_resolve_env_config()` in `session.py` uses constructor inspection and annotation-name lookups, which is brittle and hard to reason about.

Suggested fix:

- Introduce explicit registries:
  - backend id -> import hook
  - robot type -> leader factory
  - env class -> config factory
- Have envs expose config metadata directly instead of reconstructing it through reflection.

Expected payoff:

- Cleaner extension points and less hidden coupling to naming conventions.

## Medium-Value DRY Issues

### 6. Test coverage is shared in intent but duplicated in location

Files:

- `packages/so101-nexus-core/src/so101_nexus_core/testing/contract.py:15`
- `packages/so101-nexus-mujoco/tests/test_env_ids_filter.py:1`
- `packages/so101-nexus-maniskill/tests/test_env_ids_filter.py:1`
- `packages/so101-nexus-mujoco/tests/test_cli.py:1`
- `packages/so101-nexus-maniskill/tests/test_cli.py:1`
- `packages/so101-nexus-mujoco/tests/test_env_invariants.py:1`
- `packages/so101-nexus-maniskill/tests/test_env_invariants.py:1`

Why this violates DRY:

- There is already a shared contract test helper in core, but several backend tests still mirror each other structurally.
- CLI and env-filter tests repeat the same assertions with only backend names changed.

Suggested fix:

- Move backend-agnostic assertions into shared parametrized helpers in core test utilities.
- Keep backend package tests focused only on truly backend-specific behavior.
- Use a small table of backend descriptors instead of one test file per backend for the same contract.

Expected payoff:

- Smaller test surface and less update overhead when the CLI or env-id rules change.

### 7. Repeated raw dictionaries make behavior harder to evolve

Files:

- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:98`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:148`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:192`
- `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py:856`

Why this violates SOLID:

- `Single Responsibility`: ad hoc dictionaries blur the boundary between transport data, mutable runtime state, and validated configuration.

Why this violates DRY:

- The same keys are created, updated, and read across many functions.
- Every callback has to know the same implicit schema.

Suggested fix:

- Replace free-form dicts with typed request/state objects:
  - `InitConfig`
  - `InitState`
  - `TeleopSession`
- Centralize transitions on those objects instead of open-coded key mutation.

Expected payoff:

- Less schema drift, fewer argument lists, and less defensive code.

## Refactor Order

If the goal is to slim the codebase down with the best return first, I would do the work in this order:

1. Split `teleop/app.py` and introduce typed teleop state.
2. Extract shared backend CLI code into core.
3. Break up `config.py` and pull out repeated validation/defaulting helpers.
4. Introduce task-spec or strategy layers for backend environments.
5. Consolidate backend test duplication around shared parametrized helpers.
6. Replace string/reflective backend and config resolution with explicit registries.

## Summary

The biggest size drivers are not individual algorithms; they are structural repetition and oversized orchestration modules. The fastest path to a slimmer codebase is to centralize shared backend logic, stop cloning task skeletons, and replace dictionary/reflection-heavy orchestration with explicit small types and registries.
