# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
See [Stability and versioning](https://so101-nexus.com/docs/getting-started/stability)
for the public-API and deprecation policy.

## [Unreleased]

### Added

- `examples/bc_ppo_warp.py`: demo-seeded PPO for `WarpPickLift-v1` -- the same GPU-batched CleanRL PPO recipe as `ppo_warp.py`, plus behavior-cloning (BC) seeding from the 10-episode [`johnsutor/MuJoCoPickLift`](https://huggingface.co/datasets/johnsutor/MuJoCoPickLift) demonstrations: the actor is BC-pretrained on the demos before online PPO starts, and a persistent BC loss (`--bc-coef`) anchors the actor mean toward demo actions throughout training. Targets the one known weakness in `ppo_warp.py`'s current default recipe: a 5-seed sweep passed seeds 1-4 but seed 5 got stuck at a grasp-hold-at-table local optimum and never discovered the lift (`best_success=0.037`). Validated: same seed, same 30M-step recipe, demo-seeding alone rescues it to `best_success=0.993, final_success=0.983`. Demo actions are recomputed as the delta between consecutive recorded joint states (not the recorded absolute-position `action` column) since `ppo_warp.py`'s proven `pd_joint_delta_pos` control mode is left unchanged. `--use-demos false` recovers `ppo_warp.py` exactly.
- `docs/superpowers/specs/2026-07-11-rlpd-demo-augmented-sac-warp-design.md`: design doc for an RLPD-style demo-augmented off-policy alternative, deferred as a follow-up.
- `examples/ppo_warp.py` / `examples/bc_ppo_warp.py`: added `rollout_video_from_checkpoint()`, which renders one deterministic MuJoCo rollout of a saved Warp PPO policy to an mp4 (the Warp backend runs GPU-parallel worlds and does not render, so the rollout is shown in the matching MuJoCo backend as a transfer figure). Both Colab notebooks (`ppo_warp_colab.ipynb`, `bc_ppo_warp_colab.ipynb`) now finish with a "Watch a sample rollout" step that plays the mp4 inline via `IPython.display.Video`.

### Removed

- An earlier `examples/tdmpc2_warp.py` (demo-augmented TD-MPC2, MPPI planning over a learned world model) was built, smoke-tested, and then dropped before landing on `bc_ppo_warp.py` above. TD-MPC2's MPC planning is kernel-launch-latency-bound (many small sequential forward passes per action), so it does not benefit from GPU-batched Warp collection the way PPO's rollout collection does: measured steady-state throughput was ~17-70 env-steps/sec versus PPO's ~100k+ on identical hardware, and it never reliably solved this task even with demo BC-anchoring. Kept as a documented decision, not shipped.

### Fixed

- Pick-and-place reward no longer collapses when the grasp is released to complete the task. Placement progress is now credited while grasped or once the object is set on the goal disc (both backends), so finishing the task is no longer scored below hovering the grasped object above the disc.
- Documentation consistency sweep (docs vs code): corrected `examples/README.md` PPO entropy defaults and PickLift results to match `ppo_warp.py` and `training/ppo`; removed two non-existent symbols (`SO_ARM100_DIR`, `get_so100_simulation_dir`) from the API overview; fixed the five `configs.mdx` default-observation lists (added `JointPositions()`, corrected dimensions); dropped a dead `agent.robot.get_qpos()` reference and documented the `observation.environment_state`, `success`, and `done` dataset fields plus the `Max Steps` and `Success Hold` teleop controls; documented `RobotCameraPreset` (in `configs.mdx`) and the reward/observation helper functions (in the API overview).

### Changed

- On success the reward is clamped to the full normalized budget (the weights sum to 1.0) in both `RewardConfig.compute` and `simple_reward`, so a successful terminal step is always the global maximum with `completion_bonus` as the guaranteed margin. This mirrors ManiSkill PickCube's `reward[success] = max`. Non-success rewards are unchanged (bounded by `1 - completion_bonus`).
- Tuned the Warp PickLift PPO default entropy schedule from `ent_coef=0.005, ent_coef_final=0.0` to `ent_coef=0.03, ent_coef_final=0.005`. A 5-seed sweep (30M steps each) showed the previous default solved only 2/3 seeds while the new schedule solves 4/5 (final success 0.97, 0.985, 0.965, 0.97). The strong warm-start plus nonzero floor keeps exploration alive so the policy can escape the reaching local optimum late in training.

## [0.4.7] - 2026-07-02

### Added

- PEP 561 `py.typed` marker so downstream type checkers consume the library's inline type hints.
- `so101_nexus.__version__`, resolved from installed package metadata.
- Golden-value determinism regression tests covering every MuJoCo environment, guarding reward and state-observation drift across dependency and code changes.
- `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, a Stability and versioning documentation page, and GitHub issue/pull-request templates.
- Continuous integration coverage for Python 3.13 and macOS (MuJoCo backend).
- Teleoperation records privileged state and success/done signals by default (#105).

### Changed

- Pinned upper version bounds on core runtime dependencies (`numpy`, `scipy`, `trimesh`, `huggingface_hub`, `mujoco`, `gymnasium`, `tyro`) so a transitive major release cannot silently break installs.
- Documented the MuJoCo Warp backend as experimental; its API may change between minor releases while the MuJoCo backend is stable.

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

[Unreleased]: https://github.com/johnsutor/so101-nexus/compare/0.4.7...HEAD
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
