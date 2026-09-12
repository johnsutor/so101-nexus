# Simulation throughput

Preserve physics, seeded resets, observation ownership, and placement semantics.
Use existing Torch/Warp graph and component conventions without new dependencies.

1. Add regression coverage for padded contacts, observation-scoped contact reuse,
   placement graph replay, and reset metadata. Batch ordinary MuJoCo substeps.
2. Keep contact counts on the GPU and reuse contact results within a transition,
   invalidating them before autoreset observations.
3. Capture placement-v2 physics and support/dwell updates after initialization.
   Keep final forward evaluation and every substep support check.
4. Reuse reset-time target metadata and defer PPO episode statistics to rollout
   boundaries. Preserve public diagnostic snapshots and seeded sampling order.
5. Benchmark warmed-up stepping and allocation budgets on CUDA. Reduce defaults
   only with sufficient contact/constraint evidence. Profile camera allocation;
   preserve returned image ownership and rendering quality.
6. Run focused regressions, CUDA replay checks, `make format lint typecheck`, and
   the full suite. Record measurements and limitations here before committing.

## Implementation and measurements

- Contact reductions mask padded rows on the device. Forces and grasp state are
  cached only across reward/observation evaluation and invalidated for autoreset.
- Placement-v2 and IK share a Torch-owned capture helper with Warp external-capture
  registration. Placement uses eager stepping if runtime support thresholds change,
  preserving configuration behavior without recapture-time physics advancement.
- Placement target metadata reuses reset-time host indices. Live device timesteps
  still require readback to honor the existing diagnostic contract.
- Camera outputs own Torch allocations; the caching allocator can reuse released
  storage without changing earlier observations. Rendering quality is unchanged.
- PPO diagnostics transfer once per rollout, retaining episode order and histories.
  Reset sampling and host task-description updates remain unchanged to preserve RNG
  order and same-step diagnostic snapshots.

Initial warmed-up measurements from `/tmp/so101_performance.py`: 100 steps after
10 warmup steps, seed 42, zero actions, 256 Warp worlds, one native MuJoCo world,
state observations, no reset settling or success termination. GPU timing boundaries
synchronize CUDA. Hardware: RTX 5090, MuJoCo 3.9.0, mujoco-warp 3.9.0.1,
warp-lang 1.14.0, Torch 2.9.1.

| Environment | Before, ms/step | After, ms/step |
| --- | ---: | ---: |
| MuJoCoPickLift-v1 | 0.083 | 0.082 |
| WarpPickLift-v1 | 2.95 | 2.81 |
| WarpPickAndPlace-v2 | 22.67 | 5.47 |

These measure environment stepping, not learned-policy success or PPO throughput.
Small differences need longer repeated measurements before claiming a speedup.

`/tmp/so101_camera_performance.py` measured 64 worlds at 128x128 wrist RGB,
100 renders after 10 warmups: 9.03 ms before and 8.87 ms after. This is a small
difference; output ownership and removal of explicit Warp output allocations are
the verified changes.

`/tmp/so101_contact_budgets.py` sampled 1024 random control steps with 256 worlds,
seed 42, resets every 128 steps, and contact/constraint maxima recorded within
every physics substep. One cube peaked at 4.20 pooled contacts per world and 136
constraints in an individual world; a cube/sphere/cylinder/pyramid pool peaked at
12.11 and 180 respectively. Halving the one-cube contact capacity to 104 while
retaining 416 constraints changed PickLift from 2.81 to 2.80 ms and placement-v2
from 5.47 to 5.22 ms in the short timing run. Keep general allocation defaults:
this does not establish sufficient safety across learned grasps and scanned meshes.

Final repeat of `/tmp/so101_performance.py` measured 2.85 ms for WarpPickLift and
5.59 ms for placement-v2. The placement improvement is about fourfold in this
workload; the smaller PickLift and camera differences remain preliminary.

## Validation

- Padded-contact and contact-reuse regressions failed before their fixes.
- CUDA graph replay and runtime support-threshold regressions failed before fixes.
- `make test`: 1819 passed, two optional visual-model tests skipped; 93.26% coverage
  against the 84% gate.
- Final focused Warp contact, placement, and IK checks: 45 passed, including the
  new device-only contact-query capture and live configuration regression.
- Independent camera-storage checks: two passed, on CPU and CUDA.
- PPO and BC/PPO smoke suites: 36 passed, including cross-rollout episode statistics.
- `make format lint typecheck`, documentation consistency, and `git diff --check`
  passed. No dependencies or public configuration fields were added.
