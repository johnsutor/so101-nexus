# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
See [Stability and versioning](https://so101-nexus.com/docs/api/stability)
for the public-API and deprecation policy.

## [Unreleased]

### Added

### Changed

### Fixed

### Removed

## [0.5.0] - 2026-07-31

### Added

- `so101_nexus.envhub.make_env`: a [LeRobot EnvHub](https://huggingface.co/docs/lerobot/en/envhub) entry point returning LeRobot's `{env_id: {0: vector_env}}` mapping for any registered environment on either backend. Observations use the gym-side convention `lerobot.envs.utils.preprocess_observation` consumes (`agent_pos`, `environment_state`, `pixels`), success is mirrored to `info["final_info"]["is_success"]`, and `task_description` is re-exposed on each sub-environment, so a LeRobot rollout reads instructions and success without any adaptation. Options come from an `EnvConfig` (`task`, `obs_type`, `observation_width`, `observation_height`, `episode_length`, `disable_env_checker`, plus a free-form `kwargs` dict) and from keyword arguments, which accept the same options under the name `env_id` instead of `task` and add `control_mode`, `render_mode`, `device` (Warp only) and `config` (a prebuilt `EnvironmentConfig` that takes precedence over `obs_type` and the camera resolution). Warp ids are bridged from batched torch tensors to NumPy, which costs a host copy per step; train against `gymnasium.make_vec` directly instead.
- `envhub/`: the Hub payload published to [johnsutor/so101-nexus-envs](https://huggingface.co/johnsutor/so101-nexus-envs), one entry-point file per registered env id plus a root `env.py`. Each file is a shim over `so101_nexus.envhub`, so the environment code stays in the released package. `python scripts/publish_envhub.py` uploads it.
- `so101_nexus.component_slice`: the flat-state slice occupied by an observation component, promoted out of `so101_nexus.testing` because the EnvHub adapter needs the same layout lookup at runtime. The public function takes the component list (`component_slice(config.observations, JointPositions)`); `so101_nexus.testing.component_slice` keeps its env-shaped signature, now delegates, and raises `ValueError` instead of `AssertionError` when the component is absent.
- `so101_nexus.env_ids.backend_for_env_id`: the backend that owns an env id.
- `GazeState` (1-dim) observation component on both backends and on by default for every task with a target object (pick, touch, pick-and-place, stack-cube, look-at): 1.0 when the target object lies inside the wrist camera's field of view. The FOV is read live from the model, so wrist-camera domain randomization moves the boundary with it, and this is exactly the predicate `LookAt` scores `success` on. It is a frustum test on the object's origin, not an occlusion test. Tasks with no target object (move) reject it.
- `ObjectVelocity` (6-dim) observation component on both backends, on by default for pick-and-place and stack-cube: the target object's free-joint velocity (linear in world coordinates, angular in the body frame). Both tasks make "the object has settled" a literal term of `success` (`object_static_lin_threshold` / `cube_static_lin_threshold` and their angular counterparts) while no component exposed object motion, so "settled on the goal" and "sliding across it" were the same observation.
- `so101_nexus.gaze` (`direction_to_object`, `gaze_cosine`, `gaze_angle_rad`, `object_in_view`): the tensor-friendly gaze primitives both backends' gaze components and the look-at predicate share, mirroring `so101_nexus.grasp`.
- `so101_nexus.observations_from_feature_names`: inverse of `privileged_state_feature_names`, rebuilding the observation component layout a recorded dataset declares. A dataset recorded before a component joined the defaults can be replayed against the layout it was recorded with, which is what behaviour cloning needs (`examples/bc_ppo_warp.py` now builds both its Warp training env and its MuJoCo evaluator from the demo dataset's own layout, and checkpoints record `env_state_names`).
- `default_config_cls` on the Warp env classes, matching the MuJoCo ones, so a caller can build a task's config from an env id.
- `n_distractors` and `distractors` on `StackCubeConfig` (both backends): place non-target clutter alongside cubes A and B, drawn without replacement each episode from the `distractors` pool (defaults to green, yellow, and purple cubes sharing `cube_half_size`/`cube_mass`, with colours disjoint from the default cube A/B colours). Slots are only compiled when `n_distractors > 0`, so the default two-cube scene is unchanged, and the count is exposed in the teleop UI and the `[stack_cube]` profile section. Passing a distractor cube whose colour is also a cube A/B colour warns, since the task description names cubes by colour alone.
- `JointEfforts` (6-dim) and `GripperContactForce` (3-dim) observation components on both backends: the actuator generalized force per controlled joint (`data.qfrc_actuator` / `qfrc_actuator`) and the world-frame resultant contact force on the two finger bodies. Under position control the commanded force grows with the unrealised tracking error, so these are what let a policy observe that it is pressing into something it cannot move; previously nothing in `so101_nexus.observations` exposed contact or effort, and a policy stuck in that absorbing state had no channel that distinguished it from holding a pose. Both are opt-in, so no default observation layout changes.
- `reset(seed=..., options={"target_index": k})` on the pick, pick-and-place, and touch tasks (both backends; the Warp backend also accepts one index per world and holds the pin across same-step autoresets, while the MuJoCo pin scopes to the one `reset` call): pin which pool slot is the episode's target. The active-slot draw is unchanged by the pin, so when slot `k` is already active the pin only relabels the target and object placements stay byte-identical, which is what counterfactual instruction pairs (same scene, different named target) require. Previously the layout and the target came out of one seeded draw and could not be varied independently. A `target_index` that the task never consumes raises on both backends rather than being silently ignored.
- `info["target_index"]` and `info["target_object"]` on the MuJoCo pick, pick-and-place, and touch tasks, and `info["target_index"]` on their Warp counterparts, so data-collection scripts can record the resolved target without reaching into private attributes. Warp omits the string field because its `info` values are per-world batched tensors; the per-world object identity is in `info["task_description"]`. Downstream suites asserting an exact `info` key set need updating.
- `RobotConfig.grasp_opposing_normal_threshold` (default `0.3`): how strongly the two finger sets must push against each other for `GraspState` to fire. `-1.0` restores the previous bilateral-contact-only predicate.
- `so101_nexus.opposing_normals_ok`: the tensor-friendly geometric primitive both backends' grasp predicates share.

### Changed

- **Breaking:** default observation dimensions grew: pick-lift and touch 30 -> 31, pick-and-place and stack-cube 36 -> 43, look-at 22 -> 23 (move is unchanged). Index the vector by name via `privileged_state_feature_names`, not by fixed offset. `gaze_state` cannot be reconstructed from an older recording (it needs the episode's realized wrist-camera pose), so `relabel_environment_state` raises for a dataset recorded before this release; rebuild the recorded layout with `observations_from_feature_names` or re-record.
- **Breaking:** the gaze ray now originates at the wrist camera instead of the gripper tip, in `GazeDirection` and in look-at's `orientation_error`, `success`, and reward on both backends. The camera sits 7.8 cm from the TCP, the same order as the distance to a graspable object, so the old mix of the camera's optical axis with a TCP-anchored direction did not compute the documented "target lies within the camera's field of view" test: on `MuJoCoLookAt-v1` seed 0 the two anchors disagree by 54 degrees at reset. Look-at success rates and reward values shift accordingly. Use `ObjectOffset` for the gripper-relative vector.
- The MuJoCo base env now resolves the wrist camera id in `_finish_model_setup` regardless of whether a `WristCamera` observation is configured. It was left `None` without one, and `LookAtEnv` indexed the camera arrays with it, which happened to work only because those scenes carry exactly one camera.
- **Breaking:** `GraspState` / `_is_grasping()` no longer report a grasp when both finger sets merely press the same side of the target. The predicate required only that a gripper geom and a jaw geom each contact the target above `grasp_force_threshold`, which an object too wide for the jaw to close on satisfies while resting on the table bearing no load; measured against YCB objects it fired on 0.70-0.75 of episodes for the two models whose convex hull exceeds the jaw opening, exactly the cases where lift height was negative. Both backends now also require the two sides' force-weighted mean contact normals to oppose (`RobotConfig.grasp_opposing_normal_threshold`). This is load-bearing beyond the observation: `PickLiftEnv` gates `success` on it and the reward's `grasping` term is potential-shaped on it, so the old predicate actively taught a policy to straddle. Cube pools are unaffected.
- `TargetOffset`'s docstring described it as gripper-relative, but the manipulation tasks (pick-and-place and stack-cube, both backends) override it to `goal - object`; only the object-free tasks return `goal - tcp`. Documented per task; no behaviour change. Anyone reasoning about what a privileged-state teacher actually consumes got the reference frame wrong from the docs alone.
- `obs_mode`'s docstring now states what `"visual"` moves where and points at `privileged_state_feature_names(...)` for the per-dimension names of `info["privileged_state"]`; the two APIs never referenced each other, which invited positional indexing into a vector whose layout is `observations`-dependent.
- `YCBObject` and `ensure_ycb_assets` now state that collision geometry is a single convex hull of the scan, and `PickConfig.objects` warns that this makes several YCB models ungraspable regardless of the policy. The failure was silent: the object builds, runs, and reports `success` normally while the fingers can only ever touch the hull enclosing the graspable feature.
- Documentation restructured for progressive disclosure: 36 pages across 8 sidebar sections collapsed to 18 across 5 (Get Started, Workflow, Environments, Concepts, API Reference). The six per-task `environments/mujoco-*` pages became one table-driven reference at `/docs/environments`, the `guides/` and `policies/` sections were folded into `concepts/` and `api/`, and teleoperation and training moved under `workflow/`. **Documentation URLs moved**: `/docs/teleoperation/overview` is now `/docs/workflow/teleoperation`, `/docs/training/ppo` is `/docs/workflow/training`, `/docs/getting-started/stability` is `/docs/api/stability`, `/docs/concepts/observations-and-camera-modes` is `/docs/concepts/observations`, `/docs/concepts/backend-support` is `/docs/concepts/backends`, `/docs/concepts/lerobot-compatibility` is `/docs/concepts/lerobot`, and the `guides/` pages are merged into `/docs/concepts/customization` and `/docs/api/objects`. The docs site is a static export, so these are not redirected.
- User-facing docs are now American English throughout, enforced by a new check in `tests/test_docs_consistency.py` alongside nav reachability, internal-link resolution, and API-heading existence.

### Removed

- **Breaking:** `so101_nexus.policy_adapters.MolmoActPolicy` and the `molmoact` optional extra. The adapter hardcoded `trust_remote_code=True` and pinned no revision, so `MolmoActPolicy.from_pretrained(repo_id)` executed whatever Python the named Hugging Face repo shipped, and no caller could turn that off. `ChunkedActionPolicy` and `RolloutRecorder` are unchanged: any object exposing `select_action(batch)` and `reset()` still drives the recorder, so wrap the checkpoint yourself and pin the revision you load. `scripts/smoke_molmoact_rollout.py` is removed with it.

### Fixed

- Documented pick-and-place success no longer claims the robot must be static. Since 0.4.13 the predicate has been `is_obj_placed and is_obj_static and is_grasped < 0.5` on both backends, so releasing the object is mandatory and `is_robot_static` is an `info` diagnostic only. The environment reference and the task-semantics page both still described the pre-0.4.13 behavior.
- Training docs no longer report `WarpPickAndPlace-v1` as "excluded until the environment is fixed" while the README reported it as solved. Both now state the same thing: `ppo_warp.py` has no baseline for it, `bc_ppo_warp.py` solves it (3 seeds, mean `best_success` 0.861), and `WarpStackCube-v1` has no baseline on either script.
- `MeshObject` is no longer documented as MuJoCo-only. Both backends build scenes through `so101_nexus.object_slots.build_object_scene_xml`.
- The API reference no longer documents `get_so100_simulation_dir()`, which does not exist, and now documents the exported `get_ycb_texture_file()`, which was missing. `all_registered_env_ids()` is documented as returning six ids per imported backend rather than five.
- `Accordion` and `Accordions` are registered in the docs MDX component map. Any page using them previously failed to render.

## [0.4.13] - 2026-07-26

### Added

- `terminate_on_success` on every environment config (default `True`, both backends): set `False` to keep an episode running to `max_episode_steps` while `info["success"]` still reports the shipped predicate every step. Previously `terminated = success` was hardcoded, so an alternative success predicate could not be evaluated offline against recorded rollouts (the shipped predicate ended the very episodes an alternative needed to keep observing).
- `object_static_lin_threshold` (default `0.01` m/s) and `object_static_ang_threshold` (default `0.5` rad/s) on `PickAndPlaceConfig`: the speeds below which the carried object counts as settled for the success check, matching `StackCubeConfig`'s corresponding fields and ManiSkill's `is_static` defaults.
- `so101_nexus.testing.assert_render_parity` / `measure_render_parity` (plus the `RenderParityReport` / `CameraParity` result types): compare MuJoCo and Warp camera observations at bit-identical simulator state and camera pose, so the backends' rendering divergence is measurable from the contract suite rather than from a failed training run. Tolerances are required arguments, not defaults, because the shipped backends are not pixel-interchangeable.
- `info["is_obj_static"]` on both pick-and-place backends.

### Changed

- **Breaking:** pick-and-place `success` is now `is_obj_placed and is_obj_static and not is_grasped`, replacing `is_obj_placed and is_robot_static` on both backends. The old predicate measured the arm rather than the object: the goal is a disc on the table, so the intended terminal behaviour is release-and-retreat, and gating on arm velocity scored the retreat itself as failure while still accepting a placement held motionless in a closed gripper. The new predicate is strictly stronger (it rejects flung placements the arm-velocity check never inspected, and adds the missing release check) and removes an unobservable variable from the objective, since object persistence is perceivable from a camera and arm velocity is not. `is_robot_static` remains in `info` as a diagnostic. Success rates reported against the old predicate are not comparable; no checkpoint can warm-start the change.
- **Breaking:** `so101_nexus.rewards.cube_static_ok` is renamed `object_static_ok`. It is now load-bearing for the arbitrary scene objects in the pick-and-place object pool (which includes YCB meshes), not just cubes.
- The Warp backend renders with shadow casting enabled and clears to MuJoCo's black background instead of mujoco_warp's blue-tinted default, closing the two render divergences that are correctable from this library. Camera observations are still not pixel-interchangeable across backends: mujoco_warp's rasteriser ignores per-light `diffuse` and applies every active light at unit intensity, so the Warp image remains brighter than MuJoCo's. This residual gap, and the contact-model divergence that costs a pick-and-place policy 6-14 success points transferring Warp to MuJoCo, are now documented in the Warp backend's module docstring instead of being contradicted by docstrings claiming parity.

### Fixed

### Removed

## [0.4.12] - 2026-07-25

### Added

- `so101_nexus.JointVelocities`: a 6-dimensional state observation component exposing the angular velocity (rad/s) of each controlled joint, mirroring `JointPositions` over the same joints on both backends. Reads `data.qvel` (MuJoCo) / `qvel` (Warp), the same per-DOF velocity `_is_robot_static()` and the `place_task_potential` settle term already consume. Tasks whose `success` includes `is_robot_static` (pick-and-place, stack-cube) previously asked a policy to regulate a quantity it could not observe: a single-frame position snapshot cannot distinguish "approaching the target" from "already settled".
- `so101_nexus.relabel_environment_state`: re-lay a recorded `observation.environment_state` matrix onto a newer observation component layout, matching columns by the dataset's declared per-dimension names. Missing `JointVelocities` columns are reconstructed offline as a backward finite difference of the recorded `JointPositions` columns (never across an episode boundary), the same relabeling a real SO-101 control loop performs on consecutive servo position readings. Rows must be grouped by episode and chronological within each (LeRobot's own row order); a mismatched `episode_index` length, a non-contiguous episode, a duplicated column name, or any other unreconstructible column raises rather than returning a plausible-looking wrong answer. Lets datasets recorded before this release train against the current observation layout.
- `env.control_dt` on both backends: simulated seconds advanced by one `step()` (physics timestep times substeps, 0.02 s). This is the correct denominator for finite-differencing recorded joint positions and is deliberately unrelated to a teleop recording's wall-clock fps, since the recorder sleeps to pace the operator but advances the simulation exactly one step per recorded frame.

### Changed

- **Breaking:** `JointVelocities()` is now part of every task config's default `observations` list, inserted immediately after `JointPositions()`. Default state dimensions grow accordingly: PickLift and Touch 24 to 30, PickAndPlace and StackCube 30 to 36, LookAt and Move 16 to 22. Policies and checkpoints trained against the old layout must be retrained; index the vector by name via `privileged_state_feature_names` rather than by fixed offset. Teleop recording picks the component up automatically, so `observation.environment_state` now carries `joint_velocities_0..5`.

### Fixed

### Removed

## [0.4.11] - 2026-07-24

### Added

- New stack-cube task: `StackCubeConfig`, `MuJoCoStackCube-v1`, and `WarpStackCube-v1`. Pick up cube A and stack it directly on top of cube B; success requires cube A to rest on cube B (within `stack_alignment_margin`), the arm to be static, cube A itself to be static (within `cube_static_lin_threshold`/`cube_static_ang_threshold`), and cube A to be released (a strict superset of ManiSkill's `StackCubeEnv`). `cube_a_colors`/`cube_b_colors` default to disjoint colors (red vs. blue) and warn on overlap, matching `PickAndPlaceConfig`'s cube/target overlap warning. Reuses the pick-and-place staged transport-then-settle potential (`so101_nexus.rewards.place_task_potential` and friends) with the goal generalized to a moving 3D point (`2 * cube_half_size` above cube B) instead of a fixed ground disc. Both backends resample both cube colors every episode: the MuJoCo backend via per-geom `geom_rgba`, the Warp backend by compiling one freejoint cube slot per configured color and selecting one slot per role per world at reset (the same slot-pool mechanism `WarpPickLift` uses for object identity), with unselected slots parked off-world. `WarpStackCubeVectorEnv.cube_a_color_names`/`cube_b_color_names` expose the selected color per world.
- `so101_nexus.rewards.cube_stack_offset_ok`: a shared, tensor-agnostic geometric predicate for "is cube A stacked on cube B" (mirrors ManiSkill `StackCubeEnv.evaluate`'s `xy_flag`/`z_flag` check), used by both `StackCubeEnv` and `WarpStackCubeVectorEnv`.
- `so101_nexus.rewards.cube_static_ok`: a shared, tensor-agnostic predicate for "is a cube's velocity near zero" (mirrors ManiSkill's `is_cubeA_static(lin_thresh=1e-2, ang_thresh=0.5)` check), used by both stack-cube backends.
- Side camera for human rendering and rollout video collection (MuJoCo backend; the Warp backend implements no `render()`). `RenderConfig` gains `camera` (`"overhead"` default, or `"side"` for an angled tabletop bystander view), `side_azimuth_deg`, and `side_elevation_deg`; the selected view drives `render_mode="rgb_array"` frames and the initial `render_mode="human"` viewer viewpoint. The side view is visualization-only and never enters the observation space or policy inputs.
- `RolloutRecorder(record_side_video=True)`: record the env's configured render view as an `observation.images.side` video channel alongside policy rollout datasets (requires `render_mode="rgb_array"`); `FieldSelection` gains `side_image` (default off, existing schemas unchanged) and `teleop.dataset` exposes `SIDE_KEY`.
- `SimCameraConfig(source="render")`: a sentinel source that reads `env.render()` instead of an observation key, so LeRobot recording flows can capture the visualization render view (e.g. the side camera) as a dataset camera.
- Teleop stack-cube customization: `TeleopConfigOverrides.cube_a_colors`/`cube_b_colors`, UI checkbox groups ("Stack Cube A Colors" / "Stack Cube B Colors") shown for `StackCubeConfig` environments, and a `[stack_cube]` config-profile section, mirroring the existing pick-and-place cube/target color controls.
- `so101-nexus[rocm]`: optional extra that routes the `train` extra's `torch` dependency (plus `pytorch-triton-rocm`/`triton-rocm`) to AMD's ROCm 7.2 PyTorch wheel index on Linux x86_64, via `tool.uv.sources`/`tool.uv.index`, for behavior-cloning and PPO training on the MuJoCo backend on AMD GPUs (`uv sync --extra train --extra rocm --no-default-groups`). No effect on the default CUDA/CPU install path for `train`/`warp`; the Warp backend remains NVIDIA-only (NVIDIA Warp has no ROCm support). `rocm` conflicts with `teleop`/`dev`/`test`, whose pinned `lerobot<0.6` requires `torch<2.11`, incompatible with the ROCm 7.2 build.
- End-effector action spaces on both backends: `control_mode="pd_ee_pose"` (absolute) and `control_mode="pd_ee_delta_pose"` (normalized delta). Both are 7-dimensional, `[x, y, z, wx, wy, wz, gripper]`, with orientation as a rotation vector to match LeRobot's `ee.wx`/`ee.wy`/`ee.wz` action features and the gripper last as in every other mode. `pd_ee_delta_pose` exposes the usual normalized `[-1, 1]` box scaled by `so101_nexus.kinematics.EE_DELTA_ACTION_SCALE` (2 cm per position axis, 0.1 rad per rotation axis, 0.2 rad for the gripper) and measures its delta against the current tool pose, mirroring `pd_joint_delta_pos`. Actions resolve to joint targets through three damped-least-squares iterations against MuJoCo's own tool Jacobian (`mj_jacSite`, or a batched `mujoco_warp.jac` on Warp); the two backends agree to 5.2e-07 rad. The SO-101 arm has five actuated joints, so its tool Jacobian is rank 5 and arbitrary orientations are unreachable: orientation error is de-weighted by `EE_ORIENTATION_WEIGHT` (0.01, LeRobot's default) and is best-effort, while position tracks to 0.16 mm median and 2.43 mm p90 over the reachable set for a 1 cm command. Out-of-reach targets resolve to the closest achievable pose rather than raising. See [Control Modes](https://so101-nexus.com/docs/concepts/control-modes).
- `so101_nexus.kinematics`: a tensor-agnostic tool-frame kinematics module (NumPy and torch, no torch import) providing `quat_multiply`, `quat_conjugate`, `quat_to_rotvec`, `rotvec_to_quat`, `orientation_error`, `pose_error`, `damped_least_squares`, and `ee_ik_delta_q`, plus the `EE_ACTION_DIM`, `EE_DELTA_ACTION_SCALE`, `EE_ORIENTATION_WEIGHT`, `EE_IK_DAMPING`, and `EE_IK_ITERATIONS` constants. Shared by both backends so end-effector semantics cannot drift between them.
- `so101_nexus.config` exports `JOINT_CONTROL_MODES` and `EE_CONTROL_MODES` (the two families partitioning `ControlMode`), `SO101_TCP_SITE_NAME`, and `SO101_TCP_FRAME_NAME`. `get_so101_urdf_path()` returns the vendored SO-101 URDF.
- `so101_new_calib.urdf` gains a `tcp_frame_link` matching the simulator's `gripperframe` site exactly, so end-effector poses mean the same physical point in simulation and in a URDF-based solver such as LeRobot's `RobotKinematics`. Upstream's `gripper_frame_link` is unchanged and still sits 19.9 mm away at the fixed fingertip; a regression test locks the two models together.
- `SimSOFollower` accepts LeRobot end-effector action features (`ee.x`, `ee.y`, `ee.z`, `ee.wx`, `ee.wy`, `ee.wz`, `ee.gripper_pos`) when the wrapped env uses `pd_ee_pose`, and reports them from `action_features`.
- `so101_nexus.ABSOLUTE_CONTROL_MODES` and `so101_nexus.DELTA_CONTROL_MODES`: the two families partitioning `ControlMode` by whether an action space carries physical units. Absolute modes bound physical targets; delta modes bound a normalized `[-1, 1]` increment, so nothing physical can be read off them.
- `RobotConfig.ee_orientation_weight` and `RobotConfig.ee_delta_action_scale`: the two end-effector solver parameters are now env config fields on both backends, defaulting to `so101_nexus.kinematics.EE_ORIENTATION_WEIGHT` (0.01) and `EE_DELTA_ACTION_SCALE` (2 cm / 0.1 rad / 0.2 rad), so tuning them no longer means rebinding module constants before construction. Raise `ee_orientation_weight` when a task needs the policy to command tool yaw (a randomized-yaw grasp, for example): at the default the rotation channels are attenuated roughly elevenfold, which pins them for much of an episode. It is a trade -- the weight scales both sides of the least-squares solve, so rotation authority costs position-tracking accuracy. Validated at construction: the weight must be in `(0, 1]` and the scale must be seven positive entries.

### Removed

- An earlier `examples/tdmpc2_warp.py` (demo-augmented TD-MPC2, MPPI planning over a learned world model) was built, smoke-tested, and then dropped before landing on `bc_ppo_warp.py` above. TD-MPC2's MPC planning is kernel-launch-latency-bound (many small sequential forward passes per action), so it does not benefit from GPU-batched Warp collection the way PPO's rollout collection does: measured steady-state throughput was ~17-70 env-steps/sec versus PPO's ~100k+ on identical hardware, and it never reliably solved this task even with demo BC-anchoring. Kept as a documented decision, not shipped.

### Changed

- Stack-cube success (both backends) now additionally requires cube A itself to be static -- linear speed below `StackCubeConfig.cube_static_lin_threshold` (default 0.01 m/s) and angular speed below `cube_static_ang_threshold` (default 0.5 rad/s, ManiSkill's `is_cubeA_static` thresholds) -- making the predicate a strict superset of ManiSkill's `StackCubeEnv.evaluate`. Previously a cube released inside the tolerance band while still descending or rocking counted as success (and terminated the episode) even if it later toppled. `info["is_cube_a_static"]` exposes the new gate.
- The Warp backend's end-effector inverse kinematics is captured into a CUDA graph at construction and replayed per control step, instead of relaunching the fixed-iteration solve kernel by kernel. Measured at 4096 worlds on an RTX 5090: the solve drops from 3.19 ms to 1.27 ms and a full end-effector step from 8.41 ms to 6.54 ms, cutting the cost of an end-effector step against a joint-space step from 1.81x to 1.41x. The captured graph is bit-identical to the direct solve. The Warp CPU device and any capture failure fall back to the direct loop with a `RuntimeWarning`, so this is an optimization and never a requirement.
- Teleop recording now pins the environment to `control_mode="pd_joint_pos"`, overriding any control mode a registry entry or config profile supplies, so a recorded dataset always stores absolute joint positions and can never contain `ee.*` action columns. Joint positions are the only label the leader arm produces; delta and end-effector labels are recomputed offline, and `pd_ee_delta_pose` labels in particular are a finite difference of the already-recorded `EndEffectorPose` observation, which targets the same `gripperframe` TCP the end-effector control modes do. See `docs/concepts/control-modes.mdx`.

### Fixed

- Color config fields (`cube_a_colors`, `cube_b_colors`, `cube_colors`, `target_colors`, `ground_colors`, `robot_colors`) now reject an empty list at construction with a clear `ValueError` instead of failing later with an opaque `IndexError` at env build or reset time.
- `SimSOFollower` and the adapter's control-bound helpers no longer silently mis-scale in the delta control modes. The follower always sends an absolute joint target in radians, but `action_for_env` clipped it to whatever the env's action space advertised: in `pd_joint_delta_pos` and `pd_joint_target_delta_pos` that is the normalized `[-1, 1]` box, so a commanded 1.1 rad target was clipped to 1.0 and then applied by the env as a full-scale 0.05 rad increment, discarding the commanded pose entirely. The same bounds fed `read_gripper_limits_rad`, which reported `(-1.0, 1.0)` as the jaw travel in radians and corrupted every tick-to-radian conversion built on it, recorded observations included. `read_gripper_limits_rad`, `clip_qpos_to_env_ctrlrange`, and `action_for_env` now raise `ValueError` for delta-mode envs, and `SimSOFollower` rejects every delta mode at construction (previously only `pd_ee_delta_pose` was rejected). Record demonstrations in `pd_joint_pos` and recompute deltas offline, as the behavior-cloning workflow already does.

## [0.4.10] - 2026-07-16

### Added

- `so101_nexus.rewards.place_task_potential`, `place_reach_potential`, `place_grasp_potential`: shared, tensor-agnostic facet potentials for place tasks, now used by both pick-and-place backends instead of per-backend inline formulas.

### Changed

- Pick-and-place facet potentials (both backends) are now monotone non-decreasing along the ideal grasp-lift-carry-lower-release-settle trajectory, so every step of forward progress pays a non-negative shaping delta (previously a perfect demonstration paid -0.09 for the mandatory 5 cm lift, ~1e-7/step for carrying toward the goal, and -0.25 for releasing the object on the goal, with only the terminal bonus showing structure). The task potential (`info["task_potential"]`) is now a staged additive sum -- transport progress measured by Chebyshev distance `max(obj_to_goal_xy, height_gap)` plus an arm-stillness term gated on `is_obj_placed` -- instead of a product of xy-proximity x height-back-near-rest x stillness factors, and the reach/grasp potentials are held constant once the object is placed.

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

[Unreleased]: https://github.com/johnsutor/so101-nexus/compare/0.5.0...HEAD
[0.5.0]: https://github.com/johnsutor/so101-nexus/compare/0.4.13...0.5.0
[0.4.13]: https://github.com/johnsutor/so101-nexus/compare/0.4.12...0.4.13
[0.4.12]: https://github.com/johnsutor/so101-nexus/compare/0.4.11...0.4.12
[0.4.11]: https://github.com/johnsutor/so101-nexus/compare/0.4.10...0.4.11
[0.4.10]: https://github.com/johnsutor/so101-nexus/compare/0.4.9...0.4.10
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
