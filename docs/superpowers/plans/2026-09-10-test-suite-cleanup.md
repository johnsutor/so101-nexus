# Test suite audit and cleanup

## Scope

Preserve behavioral assertions, input domains, physics regressions, and coverage
while reducing duplicate tests and repeated simulator construction. Production
APIs, coverage exclusions, coverage thresholds, and default test selection stay
unchanged. Preserve the existing edits to `tests/core/test_scene.py` and
`docs/next-env.d.ts`.

## Plan

1. Run the existing suite with coverage and pytest duration reporting. Read the
   slow tests, shared fixtures, and the production behavior they exercise.
2. Consolidate overlapping scalar reward and camera property tests into their
   owning modules. Keep unique properties, input ranges, and assertion tolerances.
3. Reduce redundant environment construction in invariant and observation tests
   using the existing environment factory and explicit seeded resets. Keep fresh
   environments between tests and independent instances where that is the contract.
4. Run focused tests, then the full coverage suite. Compare executed source lines
   as well as total coverage. Run `make format lint typecheck` and review the diff.

## Audit findings and changes

- The reward property module repeated scalar range, finiteness, zero-distance,
  and success-dominance checks. Move these into the existing formula tests and
  retain the unique monotonicity property. Keep the stricter dominance tolerance.
- The camera property module repeated framing and angled-distance checks. Retain
  these in the main camera module, along with the unique eye/target property.
  Generated input domains and retained 200-example budgets are unchanged or wider.
- MuJoCo invariant properties compiled 200 environments for 20 seeds across five
  tasks and two properties. Reuse a compiled environment within each property
  test, with explicit reset and seeded actions for each example. Two standalone
  checks retain coverage of the environment-ID helper APIs. This reduces those
  constructions to 12 without sharing mutable environments between tests.
- Observation tests repeated setup for shape, dtype, and privileged-state
  assertions. Combine these by task and camera configuration. Flat-array and
  dtype assertions now apply to every individually configured state component.
  Camera shape, dtype, and keys are checked at reset and after a step.
- Six config-validation tests exactly duplicated cases in
  `tests/core/test_config.py`. One LeRobot control-range test repeated another
  test in its own module. Keep one owner for each assertion.
- The baseline Warp dwell test called `_advance_physics()` without the device
  context supplied by the public `step()` method. Its CPU data triggered a CUDA
  kernel launch and a module-load failure after 147.40 seconds. Add the existing
  `wp.ScopedDevice(env._wp_device)` convention to the test; keep both dwell
  assertions and the simulated substep support-loss sequence intact.

## Validation results

- Original full suite: 1,773 passed, two failed, two skipped in 370.07 seconds.
  Coverage: 5,395 of 5,793 statements (93.1296%). The failures were the missing
  Warp device context and a later CUDA allocation failure.
- First focused property/invariant run: 41 passed.
- MuJoCo environment and observation-mode modules: 278 passed.
- Final focused core/config/invariant modules: 292 passed.
- Fixed Warp dwell test with CUDA hidden: one passed in 0.59 seconds, including
  0.01 seconds in the test call.
- `make format lint typecheck`: passed.
- Full verification with
  `CUDA_VISIBLE_DEVICES='' PYTEST_ADDOPTS='-q --durations=30 --cov-report=json:/tmp/so101-tests-after.json' make test`:
  1,733 passed, nine skipped in 171.68 seconds. The 84% coverage gate passed.
- Before/after coverage JSON comparison: exactly the same 5,395 executed source
  lines across all measured files, with no lost or newly covered lines.
- Matching before/after timing of the consolidated core and MuJoCo modules:
  428 cases in 25.73 seconds before, 393 cases in 18.54 seconds after (27.9%
  faster). Original test files were copied from `HEAD` to
  `/tmp/so101-tests-original`, including their original conftest. Both runs used
  the same production package, `pyproject.toml`, cached assets, and
  `uv run pytest -c pyproject.toml --hypothesis-seed=20260910 -q --durations=0`.
  These runs were sequential and started after full-suite verification finished.
- Net cleanup: 35 fewer collected cases, two redundant property modules removed,
  and 317 fewer test lines. Unique assertions remain in their owning modules.

The original full-suite time and final CPU-only time are not a controlled speed
comparison: the original run included a CUDA failure and kernel compilation.
The matched subset timings isolate the test consolidation from those effects.

CUDA-enabled retries became blocked in the NVIDIA driver's
`uvm_parent_gpu_replayable_faults_isr_lock` after the original failure. Full-suite
verification uses `CUDA_VISIBLE_DEVICES=''`, retaining all Warp CPU tests. Seven
existing CUDA-dependent cases skip under this environment setting; the other two
skips require a visual verification model. No test selection, skip condition,
coverage threshold, or dependency was changed in the repository. The two blocked
retry processes received termination signals, but the driver was still preventing
their exit. CUDA verification needs a healthy driver before retrying.
