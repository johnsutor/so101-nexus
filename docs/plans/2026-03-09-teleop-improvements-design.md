# Teleop Improvements Design

Date: 2026-03-09

## Summary

Three targeted improvements to `examples/teleop.py` and the core environment configuration to make teleoperation more comfortable and data quality higher.

---

## 1. Steps-Based Episode Duration

### Problem
The current UI configures episode length in seconds (max 300s). The user thinks in steps (currently 256, wants 1024). The seconds-based slider is an indirect and confusing control.

### Design
- Change `max_episode_steps` default in `EnvironmentConfig` (core config) from 256 → 1024.
- Update all MuJoCo `gym.register` calls in `so101_nexus_mujoco/__init__.py` to pass `max_episode_steps=1024`.
- Replace the `episode_time_input` seconds-slider in `teleop.py` with a `max_steps_input` number field (default 1024, min 1).
- The `recording_thread` signature gains `max_steps: int` (replacing `max_duration: float`). The loop terminates when `len(episode_actions) >= max_steps` instead of checking wall time.
- `RecordingState` keeps `episode_duration` for display purposes (wall time elapsed).

### Files changed
- `packages/so101-nexus-core/src/so101_nexus_core/config.py` — default 256 → 1024
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/__init__.py` — all registrations 256 → 1024
- `examples/teleop.py` — UI + recording thread

---

## 2. Larger Live Camera Feed

### Problem
The `gr.Image` component for the live wrist-camera feed uses Gradio's default height, which is too small to comfortably monitor during recording.

### Design
- Set `height=640` on the `gr.Image("Live Camera Feed")` component in `teleop.py`.
- Gradio scales the image to fit the specified height regardless of underlying camera resolution.

### Files changed
- `examples/teleop.py` — one-line change on the `gr.Image` component

---

## 3. Robot Initializes to Leader Arm Pose

### Problem
On episode start, the sim robot jumps to `REST_QPOS` (a fixed default), which is often far from the physical leader arm's current pose. This creates a discontinuity that is jarring and wastes the first seconds of each episode.

### Design

#### Environment layer (`base_env.py`)
- Extend `reset(options=...)` to check for `options["init_qpos"]` (a `np.ndarray` of joint positions in radians, length = number of joints).
- If provided, `_reset_robot_joints` uses that array as the target pose (no noise added — the user intentionally placed the arm there).
- If not provided, behavior is unchanged (REST_QPOS + noise).

#### Teleop layer (`teleop.py`)
- In `recording_thread`, before calling `env.reset()`:
  1. Call `leader.get_action()` to read current physical joint positions.
  2. Convert via `convert_leader_action(...)` to radians.
  3. Pass as `env.reset(options={"init_qpos": init_qpos})`.

### Files changed
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py` — `reset()` and `_reset_robot_joints()`
- `examples/teleop.py` — `recording_thread`

---

## Acceptance Criteria

1. The Gradio UI shows a "Max Steps" number input instead of "Max Episode Duration (s)".
2. Default `max_episode_steps` is 1024 everywhere.
3. A recording with 1024 steps at 30 FPS runs for ~34 seconds without early truncation from the gym wrapper.
4. The live camera feed displays at 640px height.
5. On episode start, the sim robot's joint positions match the leader arm's physical pose (not REST_QPOS).
