# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
See [Stability and versioning](https://so101-nexus.com/docs/getting-started/stability)
for the public-API and deprecation policy.

## [0.4.10] - 2026-07-16

### Added

- `so101_nexus.rewards.place_task_potential`, `place_reach_potential`, `place_grasp_potential`: shared, tensor-agnostic facet potentials for place tasks, now used by both pick-and-place backends instead of per-backend inline formulas.

### Changed

- Pick-and-place facet potentials (both backends) are now monotone non-decreasing along the ideal grasp-lift-carry-lower-release-settle trajectory, so every step of forward progress pays a non-negative shaping delta (previously a perfect demonstration paid -0.09 for the mandatory 5 cm lift, ~1e-7/step for carrying toward the goal, and -0.25 for releasing the object on the goal, with only the terminal bonus showing structure). The task potential (`info["task_potential"]`) is now a staged additive sum -- transport progress measured by Chebyshev distance `max(obj_to_goal_xy, height_gap)` plus an arm-stillness term gated on `is_obj_placed` -- instead of a product of xy-proximity x height-back-near-rest x stillness factors, and the reach/grasp potentials are held constant once the object is placed.

## [Unreleased]

### Added

- New stack-cube task: `StackCubeConfig`, `MuJoCoStackCube-v1`, and `WarpStackCube-v1`. Pick up cube A and stack it directly on top of cube B; success requires cube A to rest on cube B (within `stack_alignment_margin`), the arm to be static, cube A itself to be static (within `cube_static_lin_threshold`/`cube_static_ang_threshold`), and cube A to be released (a strict superset of ManiSkill's `StackCubeEnv`). `cube_a_colors`/`cube_b_colors` default to disjoint colors (red vs. blue) and warn on overlap, matching `PickAndPlaceConfig`'s cube/target overlap warning. Reuses the pick-and-place staged transport-then-settle potential (`so101_nexus.rewards.place_task_potential` and friends) with the goal generalized to a moving 3D point (`2 * cube_half_size` above cube B) instead of a fixed ground disc. Both backends resample both cube colors every episode: the MuJoCo backend via per-geom `geom_rgba`, the Warp backend by compiling one freejoint cube slot per configured color and selecting one slot per role per world at reset (the same slot-pool mechanism `WarpPickLift` uses for object identity), with unselected slots parked off-world. `WarpStackCubeVectorEnv.cube_a_color_names`/`cube_b_color_names` expose the selected color per world.
- `so101_nexus.rewards.cube_stack_offset_ok`: a shared, tensor-agnostic geometric predicate for "is cube A stacked on cube B" (mirrors ManiSkill `StackCubeEnv.evaluate`'s `xy_flag`/`z_flag` check), used by both `StackCubeEnv` and `WarpStackCubeVectorEnv`.
- `so101_nexus.rewards.cube_static_ok`: a shared, tensor-agnostic predicate for "is a cube's velocity near zero" (mirrors ManiSkill's `is_cubeA_static(lin_thresh=1e-2, ang_thresh=0.5)` check), used by both stack-cube backends.
- Side camera for human rendering and rollout video collection (MuJoCo backend; the Warp backend implements no `render()`). `RenderConfig` gains `camera` (`"overhead"` default, or `"side"` for an angled tabletop bystander view), `side_azimuth_deg`, and `side_elevation_deg`; the selected view drives `render_mode="rgb_array"` frames and the initial `render_mode="human"` viewer viewpoint. The side view is visualization-only and never enters the observation space or policy inputs.
- `RolloutRecorder(record_side_video=True)`: record the env's configured render view as an `observation.images.side` video channel alongside policy rollout datasets (requires `render_mode="rgb_array"`); `FieldSelection` gains `side_image` (default off, existing schemas unchanged) and `teleop.dataset` exposes `SIDE_KEY`.
- `SimCameraConfig(source="render")`: a sentinel source that reads `env.render()` instead of an observation key, so LeRobot recording flows can capture the visualization render view (e.g. the side camera) as a dataset camera.

### Changed

- Stack-cube success (both backends) now additionally requires cube A itself to be static -- linear speed below `StackCubeConfig.cube_static_lin_threshold` (default 0.01 m/s) and angular speed below `cube_static_ang_threshold` (default 0.5 rad/s, ManiSkill's `is_cubeA_static` thresholds) -- making the predicate a strict superset of ManiSkill's `StackCubeEnv.evaluate`. Previously a cube released inside the tolerance band while still descending or rocking counted as success (and terminated the episode) even if it later toppled. `info["is_cube_a_static"]` exposes the new gate.

### Fixed

- Color config fields (`cube_a_colors`, `cube_b_colors`, `cube_colors`, `target_colors`, `ground_colors`, `robot_colors`) now reject an empty list at construction with a clear `ValueError` instead of failing later with an opaque `IndexError` at env build or reset time.

## [0.4.9] - 2026-07-16

### Changed

- Teleop recorder UI: the episode progress counter is now one-indexed ("Episode 1 / 5" at the start of the first episode, matching the already-one-indexed "Recording episode 1/5..." status text) instead of showing "Episode 0 / 5" before any episode was recorded.
- Teleop recorder UI: the dataset Repo ID warning now also flags a repo ID that already has a dataset on local disk (`HF_LEROBOT_HOME/<repo_id>`), flags (on blur) a repo ID that already exists as a dataset on the HuggingFace Hub, and warns when no `username/` namespace is given that the recording will be local-only and cannot be pushed to the Hub.
- Pick-lift and pick-and-place `reaching`/`grasping` reward facets (both backends) are now potential-based shaping deltas, like `task_objective` since 0.4.8, instead of raw dwelling values: a policy that reaches and grasps but never finishes the task (lift/place) previously kept collecting up to 0.50/step (the combined `reaching + grasping` budget) indefinitely; it now collects that credit once, on genuine progress, same as `task_objective`. Non-terminal per-step reward for these two envs can now go as low as `-0.75` (default equal weights) instead of `0.0`, since these facets can swing negative on a genuine regression (e.g. losing a grasp) -- terminal (success) reward is unaffected, still clamped to `1.0`. See `docs/superpowers/plans/2026-07-16-pick-grasp-potential-shaping.md`.

### Fixed

- `examples/ppo_warp.py` / `examples/bc_ppo_warp.py`: `RunningMeanStd`, `ObsNormalizer`, and `RewardScaler` now sanitize non-finite (NaN/Inf) inputs, the raw `envs.step()` observation/reward are sanitized before use, and every `optimizer.step()` (including `bc_ppo_warp.py`'s BC-pretrain optimizer) is gated on a finite loss. One diverging parallel Warp world could previously emit a NaN/Inf observation or reward that permanently corrupted the shared running-stat accumulators, eventually crashing training with `ValueError: Normal(loc=NaN, ...)`.

## [0.4.8] - 2026-07-12

### Added

- `so101_nexus.rewards.potential_shaping`: a potential-based reward-shaping delta helper (`Phi(s') - Phi(s)`, the `gamma=1` case of Ng, Harada & Russell's policy-invariance theorem, ICML 1999). `RewardConfig.velocity_shaping_scale` (new field, default `15.0`) scales a dense arm-stillness shaping factor used by the pick-and-place potential below.
- `PickAndPlaceEnv`/`WarpPickAndPlaceVectorEnv` info now includes `task_potential`: the current value of the smooth completion-progress potential (goal-xy proximity x height-back-near-rest x arm-stillness x grasped-or-placed), useful for diagnosing reward shaping during training.
- `RewardConfig.velocity_shaping_scale` is only read by `PickAndPlaceEnv._task_potential`; customizing it on `PickConfig` (pick-lift), `TouchConfig`, `MoveConfig`, or `LookAtConfig` now warns (no dead knobs), matching the existing `reaching`/`grasping`/`task_objective`/`tanh_shaping_scale` inert-field warnings.

### Changed

- Pick-and-place and pick-lift `task_progress` (both backends) is now a potential-based shaping delta (`Phi(s') - Phi(s)`) instead of the raw potential value, closing a reward-hacking trap: a policy that reached a high-reward state and then stopped moving (e.g. carrying a grasped object to hover above the pick-and-place goal disc without lowering it) previously kept collecting up to 90% of the per-step reward budget indefinitely without ever completing the task, since `task_progress` was recomputed fresh from instantaneous state every step. Summed over an episode the new delta telescopes to `Phi(final) - Phi(initial)`, bounded regardless of dwell time, so hovering now earns ~0 further reward after the first step. Pick-and-place's potential additionally gates on height-back-near-rest and arm stillness (previously xy-only), a smooth relaxation of `success`'s `is_obj_placed & is_robot_static` AND condition. `reaching`/`grasping` and Touch/Move/LookAt's shaping mechanism are unchanged (see the default-weight entry below for a separate, shared-field magnitude change that does touch Touch/Move/LookAt).

- `RewardConfig`'s default weights are now equal across all four components: `reaching=0.25, grasping=0.25, task_objective=0.25, completion_bonus=0.25` (previously `task_objective=0.40, completion_bonus=0.10`). Matches the reference environment's actual reward structure: ManiSkill's `StackCubeEnv.compute_dense_reward` allocates an equal 2-of-8 (25%) budget to each of its four completion stages (reach, grasp, place, success), via sequential floor-jumps rather than this repo's flat weighted sum, but the per-stage split is the same. `task_objective`'s exploit-closing property (see the potential-based shaping entry above) does not depend on its weight's magnitude, so this rebalance is a pure precedent-alignment change, not a follow-up fix. `completion_bonus` is live for every task (`simple_reward` uses it too, not only `RewardConfig.compute`), so this also rescales Touch/Move/LookAt reward magnitudes (`shaped = (1 - completion_bonus) * progress` is now `0.75 * progress`, previously `0.90 * progress`); their reward *mechanism* (raw progress, no potential-based delta) is unchanged, only this shared weight's default value moved. Determinism goldens regenerated to reflect the new defaults.

## [0.4.7] - 2026-07-12

### Added

- PEP 561 `py.typed` marker so downstream type checkers consume the library's inline type hints.
- `so101_nexus.__version__`, resolved from installed package metadata.
- Golden-value determinism regression tests covering every MuJoCo environment, guarding reward and state-observation drift across dependency and code changes.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, a Stability and versioning documentation page, and GitHub issue/pull-request templates.
- Continuous integration coverage for Python 3.13 and macOS (MuJoCo backend).
- Teleoperation records privileged state and success/done signals by default (#105).
- `examples/bc_ppo_warp.py`: demo-seeded PPO for `WarpPickLift-v1` -- the same GPU-batched CleanRL PPO recipe as `ppo_warp.py`, plus behavior-cloning (BC) seeding from the 10-episode [`johnsutor/MuJoCoPickLift`](https://huggingface.co/datasets/johnsutor/MuJoCoPickLift) demonstrations: the actor is BC-pretrained on the demos before online PPO starts, and a persistent BC loss (`--bc-coef`) anchors the actor mean toward demo actions throughout training. Targets the one known weakness in `ppo_warp.py`'s current default recipe: a 5-seed sweep passed seeds 1-4 but seed 5 got stuck at a grasp-hold-at-table local optimum and never discovered the lift (`best_success=0.037`). Validated: same seed, same 30M-step recipe, demo-seeding alone rescues it to `best_success=0.993, final_success=0.983`. Demo actions are recomputed as the delta between consecutive recorded joint states (not the recorded absolute-position `action` column) since `ppo_warp.py`'s proven `pd_joint_delta_pos` control mode is left unchanged. `--use-demos false` recovers `ppo_warp.py` exactly.
- `docs/superpowers/specs/2026-07-11-rlpd-demo-augmented-sac-warp-design.md`: design doc for an RLPD-style demo-augmented off-policy alternative, deferred as a follow-up.
- `examples/ppo_warp.py` / `examples/bc_ppo_warp.py`: added `rollout_video_from_checkpoint()`, which renders one deterministic MuJoCo rollout of a saved Warp PPO policy to an mp4 (the Warp backend runs GPU-parallel worlds and does not render, so the rollout is shown in the matching MuJoCo backend as a transfer figure). Both Colab notebooks (`ppo_warp_colab.ipynb`, `bc_ppo_warp_colab.ipynb`) now finish with a "Watch a sample rollout" step that plays the mp4 inline via `IPython.display.Video`.
- Teleop and rollout-recorded LeRobot datasets now carry a per-facet reward breakdown alongside the existing scalar `reward` field: six always-on `reward_components.<name>` scalars (`reaching`, `grasping`, `task_objective`, `completion_bonus`, `action_delta_penalty`, `energy_penalty`) that sum exactly to `reward` on every frame. `RewardConfig.compute_components`/`compute_simple_components` produce the breakdown; unused buckets for single-objective envs (Touch/Move reach, LookAt orientation) are pinned at `0.0`.

### Removed

- An earlier `examples/tdmpc2_warp.py` (demo-augmented TD-MPC2, MPPI planning over a learned world model) was built, smoke-tested, and then dropped before landing on `bc_ppo_warp.py` above. TD-MPC2's MPC planning is kernel-launch-latency-bound (many small sequential forward passes per action), so it does not benefit from GPU-batched Warp collection the way PPO's rollout collection does: measured steady-state throughput was ~17-70 env-steps/sec versus PPO's ~100k+ on identical hardware, and it never reliably solved this task even with demo BC-anchoring. Kept as a documented decision, not shipped.

### Fixed

- Pick-and-place reward no longer collapses when the grasp is released to complete the task. Placement progress is now credited while grasped or once the object is set on the goal disc (both backends), so finishing the task is no longer scored below hovering the grasped object above the disc.
- Documentation consistency sweep (docs vs code): corrected `examples/README.md` PPO entropy defaults and PickLift results to match `ppo_warp.py` and `training/ppo`; removed two non-existent symbols (`SO_ARM100_DIR`, `get_so100_simulation_dir`) from the API overview; fixed the five `configs.mdx` default-observation lists (added `JointPositions()`, corrected dimensions); dropped a dead `agent.robot.get_qpos()` reference and documented the `observation.environment_state`, `success`, and `done` dataset fields plus the `Max Steps` and `Success Hold` teleop controls; documented `RobotCameraPreset` (in `configs.mdx`) and the reward/observation helper functions (in the API overview).
- `RewardConfig.apply_penalties` now floors a completed step's reward at `1 - completion_bonus` when `is_complete` is passed (all five envs, both backends now forward it): a nonzero `action_delta_penalty`/`energy_penalty` could previously push a successful terminal step below the best reward a non-terminal state can reach, silently reintroducing the same class of "completion is punished" bug the pick-and-place terminal-clamp fix eliminated. Defaults (`action_delta_penalty=energy_penalty=0.0`) are unaffected.
- Removed unreachable `SO101NexusMuJoCoBaseEnv._reach_only_reward` and `_orientation_toward_reward`: `PickEnv`, the only caller of the former, is never registered or directly instantiated (only its subclasses `PickLiftEnv`/`TouchEnv`, which each define their own `_compute_reward`, are); the latter had no callers anywhere.

### Changed

- Pinned upper version bounds on core runtime dependencies (`numpy`, `scipy`, `trimesh`, `huggingface_hub`, `mujoco`, `gymnasium`, `tyro`) so a transitive major release cannot silently break installs.
- Documented the MuJoCo Warp backend as experimental; its API may change between minor releases while the MuJoCo backend is stable.
- On success the reward is clamped to the full normalized budget (the weights sum to 1.0) in both `RewardConfig.compute` and `simple_reward`, so a successful terminal step is always the global maximum with `completion_bonus` as the guaranteed margin. This mirrors ManiSkill PickCube's `reward[success] = max`. Non-success rewards are unchanged (bounded by `1 - completion_bonus`).
- Tuned the Warp PickLift PPO default entropy schedule from `ent_coef=0.005, ent_coef_final=0.0` to `ent_coef=0.03, ent_coef_final=0.005`. A 5-seed sweep (30M steps each) showed the previous default solved only 2/3 seeds while the new schedule solves 4/5 (final success 0.97, 0.985, 0.965, 0.97). The strong warm-start plus nonzero floor keeps exploration alive so the policy can escape the reaching local optimum late in training.
- `RewardConfig.apply_penalties` gained an `is_complete` keyword (default `False`, backward compatible) so callers can opt into the completion-margin floor above.
- `TouchConfig`/`MoveConfig`/`LookAtConfig` now warn (`UserWarning`) when constructed with a `RewardConfig` that customizes `reaching`/`grasping`/`task_objective` (and, for `LookAtConfig`, `tanh_shaping_scale`): these envs reward via `so101_nexus.rewards.simple_reward`, not `RewardConfig.compute`, so those fields were silently inert.

## [0.4.5] - 2026-06-29

### Added

- LeRobot adapter helpers to decode a dataset row back into simulator qpos (#102).

## [0.4.4] - 2026-06-27

### Added

- Teleoperation Configure step surfaces leader-arm port status (#98).

### Fixed

- Spotlight key light removes floor shadow-map acne in camera renders (#99).

## [0.4.3] - 2026-06-24

### Added

- Warp backend camera rendering (#94).
- Warning when a render mode is requested on the Warp backend (#96).

### Fixed

- Doubled shadow and ground aliasing in camera renders (#95).
- Observation dtype and episode-length issues that added training friction (#93).

## [0.4.2] - 2026-06-22

### Fixed

- Teleoperation surfaces the current task and reward (#91).
- Episode buffer is located across LeRobot 0.5.x layouts (#90).

## [0.4.1] - 2026-06-22

### Added

- Warp backend heterogeneous objects (#84).
- Per-step reward recorded into datasets (#82).

### Changed

- Environments made more meaningful for teleoperation (#81).
- Example scripts converted to Hypothesis property tests (#87).

### Fixed

- README included in package metadata (#86).

## [0.4.0] - 2026-06-21

### Added

- Vendored MuJoCo Menagerie SO101 model as the default backend model (#64).
- MuJoCo Warp backend reaching full task parity with the MuJoCo backend, including a reach backend (#73, #75).

### Changed

- Consolidated to a single MuJoCo-based library, removing the SO100 submodule and the ManiSkill paths (#72, #77).
- README and documentation positioning refresh (#74).
- Hardened and cached GitHub Actions workflows (#66).

### Fixed

- Sharper render shadows and edges in menagerie scenes (#65).

## [0.3.12] - 2026-05-25

### Changed

- Refactored backend reward and task-description handling to share logic (DRY) (#62).
- MuJoCo test suite sped up by an order of magnitude (#60).

### Fixed

- Teleoperation distractor and YCB texture fixes (#59, #61).

## [0.3.11] - 2026-05-19

### Added

- YCB textures and improved spawn stability (#58).
- MolmoAct2 environment and teleoperation compatibility (#55, #56).
- LeRobot-compatible dataset recorder (#54).

### Fixed

- Documentation static search index alignment (#57).

## [0.3.10] - 2026-05-09

### Added

- Expanded teleoperation utilities (#51).

### Fixed

- Teleoperation finalize-before-push flow (#50).

## [0.3.9] - 2026-05-09

### Fixed

- Release smoke test Python version (#48).
- ManiSkill teleoperation issues (#47).

## [0.3.8] - 2026-05-08

### Changed

- Documentation and codebase consistency audit (#45).

### Fixed

- ManiSkill teleoperation issues (#44).

## [0.3.7] - 2026-05-08

### Fixed

- Type-checker issues resolved (#41).
- Teleoperation recorder fixes (#40).

## [0.3.6] - 2026-05-06

### Added

- LeRobot processors integration (#35).

### Fixed

- Correct MuJoCo backend selection (#37).
- Teleoperation friction fixes (#36).

## [0.3.5] - 2026-04-29

### Added

- `uvx` teleoperation entry point and documentation (#34).
- Teleoperation moved into the core library (#29).
- Gradio UI redesign (#30).

### Changed

- Testing overhaul (#31).

## [0.3.4] - 2026-03-24

### Fixed

- Maintenance release (#27).

## [0.3.3] - 2026-03-23

### Fixed

- Maintenance release (#26).

## [0.3.2] - 2026-03-22

### Added

- Overhead camera observation (#22).

### Removed

- Deprecated configuration parameters (#24).

## [0.3.1] - 2026-03-22

### Added

- State-observation environments and richer observation-space handling (#15, #17).
- Additional named poses (#20).
- Documentation search (#16).
- End-to-end tests (#19).

### Fixed

- Overhead camera spawn placement (#21).

## [0.3.0a1] - 2026-03-15

### Added

- Documentation site (#11).
- Teleoperation support (#7).
- PPO training examples (#5).
- Environments with multiple objects and color randomization (#4).

## [0.2.0] - 2026-03-06

### Added

- Pick-and-place environment.
- Configuration dataclasses with `__post_init__` validation, replacing loose keyword arguments.
- YCB object assets and additional MuJoCo control modes.
- Local Qwen-powered visual testing.

### Changed

- Degrees used consistently across public and configuration APIs.
- Deployment smoke test added.

## [0.1.0] - 2026-02-22

### Added

- Initial release: SO-101 MuJoCo simulation with cameras, GitHub Actions CI, and the core project structure.

[Unreleased]: https://github.com/johnsutor/so101-nexus/compare/0.4.9...HEAD
[0.4.9]: https://github.com/johnsutor/so101-nexus/compare/0.4.8...0.4.9
[0.4.8]: https://github.com/johnsutor/so101-nexus/compare/0.4.7...0.4.8
[0.4.7]: https://github.com/johnsutor/so101-nexus/compare/0.4.5...0.4.7
[0.4.5]: https://github.com/johnsutor/so101-nexus/compare/0.4.4...0.4.5
[0.4.4]: https://github.com/johnsutor/so101-nexus/compare/0.4.3...0.4.4
[0.4.3]: https://github.com/johnsutor/so101-nexus/compare/0.4.2...0.4.3
[0.4.2]: https://github.com/johnsutor/so101-nexus/compare/0.4.1...0.4.2
[0.4.1]: https://github.com/johnsutor/so101-nexus/compare/0.4.0...0.4.1
[0.4.0]: https://github.com/johnsutor/so101-nexus/compare/0.3.12...0.4.0
[0.3.12]: https://github.com/johnsutor/so101-nexus/compare/0.3.11...0.3.12
[0.3.11]: https://github.com/johnsutor/so101-nexus/compare/0.3.10...0.3.11
[0.3.10]: https://github.com/johnsutor/so101-nexus/compare/0.3.9...0.3.10
[0.3.9]: https://github.com/johnsutor/so101-nexus/compare/0.3.8...0.3.9
[0.3.8]: https://github.com/johnsutor/so101-nexus/compare/0.3.7...0.3.8
[0.3.7]: https://github.com/johnsutor/so101-nexus/compare/0.3.6...0.3.7
[0.3.6]: https://github.com/johnsutor/so101-nexus/compare/0.3.5...0.3.6
[0.3.5]: https://github.com/johnsutor/so101-nexus/compare/0.3.4...0.3.5
[0.3.4]: https://github.com/johnsutor/so101-nexus/compare/0.3.3...0.3.4
[0.3.3]: https://github.com/johnsutor/so101-nexus/compare/0.3.2...0.3.3
[0.3.2]: https://github.com/johnsutor/so101-nexus/compare/0.3.1...0.3.2
[0.3.1]: https://github.com/johnsutor/so101-nexus/compare/0.3.0a1...0.3.1
[0.3.0a1]: https://github.com/johnsutor/so101-nexus/compare/0.2.0...0.3.0a1
[0.2.0]: https://github.com/johnsutor/so101-nexus/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/johnsutor/so101-nexus/releases/tag/0.1.0
