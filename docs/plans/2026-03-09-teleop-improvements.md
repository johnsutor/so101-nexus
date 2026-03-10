# Teleop Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve the teleop recorder by switching episode limits to steps, enlarging the live camera view, and initializing the sim robot to the leader arm's current pose.

**Architecture:** Three independent changes: (1) a default value bump in core config + gym registrations + UI/thread logic, (2) a one-line Gradio component tweak, (3) a new `init_qpos` option flowing from teleop → `env.reset()` → `_reset_robot_joints`.

**Tech Stack:** Python, Gymnasium, MuJoCo, Gradio, uv (package manager)

---

## Test runner reference

```bash
# core config tests
uv run --package so101-nexus-core pytest packages/so101-nexus-core/tests/test_config.py -v

# top-level tests (includes tests/test_teleop_wrist_offset.py)
uv run --package so101-nexus-mujoco --group teleop pytest tests/ -v
```

---

### Task 1: Update `max_episode_steps` default to 1024 in core config

**Files:**
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/config.py:178`
- Modify: `packages/so101-nexus-core/tests/test_config.py`

**Step 1: Write the failing test**

Add this test to the `TestConfigConsistency` class in `packages/so101-nexus-core/tests/test_config.py`:

```python
def test_default_max_episode_steps_is_1024(self):
    cfg = EnvironmentConfig()
    assert cfg.max_episode_steps == 1024
```

**Step 2: Run test to verify it fails**

```bash
uv run --package so101-nexus-core pytest packages/so101-nexus-core/tests/test_config.py::TestConfigConsistency::test_default_max_episode_steps_is_1024 -v
```

Expected: FAIL with `AssertionError: assert 256 == 1024`

**Step 3: Update the default**

In `packages/so101-nexus-core/src/so101_nexus_core/config.py`, change line 178:

```python
# Before
max_episode_steps: int = 256

# After
max_episode_steps: int = 1024
```

**Step 4: Run test to verify it passes**

```bash
uv run --package so101-nexus-core pytest packages/so101-nexus-core/tests/test_config.py -v
```

Expected: all tests PASS (the existing test at line 22 uses `max_episode_steps=512` explicitly, so it is unaffected)

**Step 5: Commit**

```bash
git add packages/so101-nexus-core/src/so101_nexus_core/config.py \
        packages/so101-nexus-core/tests/test_config.py
git commit -m "feat: change default max_episode_steps from 256 to 1024"
```

---

### Task 2: Update MuJoCo gym registrations to 1024

**Files:**
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/__init__.py`

**Step 1: No new test needed** — the gymnasium spec is tested implicitly by the env. The core config default already drives correctness; this just keeps the registration in sync.

**Step 2: Replace all `max_episode_steps=256` with `max_episode_steps=1024`**

In `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/__init__.py`, do a find-and-replace of every occurrence of `max_episode_steps=256` → `max_episode_steps=1024`. There are 16 occurrences (lines 8, 14, 20, 26, 32, 38, 44, 50, 56, 63, 69, 75, 81, 87, 93 for the loop body).

**Step 3: Verify no 256 remains**

```bash
grep -n "max_episode_steps=256" packages/so101-nexus-mujoco/src/so101_nexus_mujoco/__init__.py
```

Expected: no output

**Step 4: Commit**

```bash
git add packages/so101-nexus-mujoco/src/so101_nexus_mujoco/__init__.py
git commit -m "feat: update MuJoCo gym registrations to max_episode_steps=1024"
```

---

### Task 3: Switch recording thread to steps-based termination

**Files:**
- Modify: `examples/teleop.py`

This task changes `recording_thread` to count steps instead of wall-clock seconds, and updates the UI to expose a "Max Steps" number input.

**Step 1: Update `recording_thread` signature and loop**

Find `recording_thread` in `examples/teleop.py` (line 160). Change its signature and loop:

```python
# Old signature
def recording_thread(
    state: RecordingState,
    env_id: str,
    leader,
    joint_names: tuple[str, ...],
    fps: int,
    max_duration: float,   # <-- remove
    countdown: int,
    wrist_roll_offset_deg: float,
) -> None:

# New signature
def recording_thread(
    state: RecordingState,
    env_id: str,
    leader,
    joint_names: tuple[str, ...],
    fps: int,
    max_steps: int,        # <-- add
    countdown: int,
    wrist_roll_offset_deg: float,
) -> None:
```

Inside the loop, replace the duration check:

```python
# Old (remove these two lines)
if step_start - start_time >= max_duration:
    break

# New (add after the step_start assignment)
if len(state.episode_actions) >= max_steps:
    break
```

Also remove `start_time = time.monotonic()` — no longer needed for termination. Keep it only for `episode_duration` tracking:

```python
# Keep this for the duration display:
start_time = time.monotonic()
# ... (rest of loop unchanged)
state.episode_duration = time.monotonic() - start_time
```

**Step 2: Update `_init_worker` to accept `max_steps` instead of `episode_time`**

Find `_init_worker` (line 323). In the function signature, rename `episode_time` → `max_steps`. Update the `session.update(...)` call:

```python
# Old
session.update(
    ...
    episode_time=episode_time,
    ...
)

# New
session.update(
    ...
    max_steps=max_steps,
    ...
)
```

**Step 3: Update `start_init` function**

In `start_init` (line 404), rename `episode_time` → `max_steps` in the signature and the validation line:

```python
# Old
num_episodes, episode_time, countdown = int(num_episodes), int(episode_time), int(countdown)

# New
num_episodes, max_steps, countdown = int(num_episodes), int(max_steps), int(countdown)
```

Update the `threading.Thread` call arguments to pass `max_steps` in place of `episode_time`.

**Step 4: Update `start_recording` to pass `max_steps`**

In `start_recording` (line 509), update the `recording_thread` call:

```python
# Old
session["episode_time"],

# New
session["max_steps"],
```

**Step 5: Update the Gradio UI**

Replace `episode_time_input` slider with a number input. Find the `episode_time_input` block (around line 720):

```python
# Old
episode_time_input = gr.Slider(
    minimum=5,
    maximum=300,
    value=60,
    step=5,
    label="Max Episode Duration (s)",
)

# New
max_steps_input = gr.Number(
    value=1024,
    minimum=1,
    precision=0,
    label="Max Steps",
)
```

Update all references to `episode_time_input` → `max_steps_input` in `init_btn.click(inputs=[...])` and the `start_init` call outputs.

Also update the session header string in `poll_init` (around line 497):

```python
# Old
f"**FPS:** {s['fps']} | **Max episode:** {s['episode_time']}s"

# New
f"**FPS:** {s['fps']} | **Max steps:** {s['max_steps']}"
```

**Step 6: Run existing tests**

```bash
uv run --package so101-nexus-mujoco --group teleop pytest tests/ -v
```

Expected: all existing tests PASS (they test `convert_leader_action`, not the thread)

**Step 7: Commit**

```bash
git add examples/teleop.py
git commit -m "feat: switch teleop episode limit from seconds to steps (default 1024)"
```

---

### Task 4: Enlarge live camera feed in Gradio UI

**Files:**
- Modify: `examples/teleop.py`

**Step 1: Set height on the Image component**

Find line ~752 in `examples/teleop.py`:

```python
# Old
live_feed = gr.Image(label="Live Camera Feed")

# New
live_feed = gr.Image(label="Live Camera Feed", height=640)
```

**Step 2: Run existing tests**

```bash
uv run --package so101-nexus-mujoco --group teleop pytest tests/ -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add examples/teleop.py
git commit -m "feat: enlarge live camera feed to 640px height in Gradio UI"
```

---

### Task 5: Initialize sim robot to leader arm's current pose

**Files:**
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py`
- Modify: `examples/teleop.py`

#### Part A: Extend `base_env` to accept `init_qpos` via reset options

**Step 1: Write the failing test**

Create a new file `packages/so101-nexus-mujoco/tests/test_base_env_init_qpos.py`:

```python
"""Test that base_env.reset() respects options['init_qpos']."""
from __future__ import annotations

import numpy as np
import pytest


def test_reset_uses_init_qpos_from_options():
    """Env reset should set joints to the provided init_qpos, not REST_QPOS."""
    import so101_nexus_mujoco  # noqa: F401 — registers envs
    import gymnasium as gym

    env = gym.make("MuJoCoPickCubeGoal-v1", camera_mode="wrist", render_mode="rgb_array")
    try:
        custom_qpos = np.array([0.1, -0.5, 0.8, 0.2, 0.0, 0.05], dtype=np.float64)
        obs, _ = env.reset(options={"init_qpos": custom_qpos})
        actual_qpos = env.unwrapped._get_current_qpos()
        np.testing.assert_allclose(actual_qpos, custom_qpos, atol=1e-6)
    finally:
        env.close()


def test_reset_without_init_qpos_uses_rest_pose():
    """Without init_qpos, reset should use the default REST_QPOS (within noise)."""
    import so101_nexus_mujoco  # noqa: F401
    import gymnasium as gym
    from so101_nexus_core.config import EnvironmentConfig

    rest = np.array(EnvironmentConfig().robot.rest_qpos_rad, dtype=np.float64)
    env = gym.make("MuJoCoPickCubeGoal-v1", camera_mode="wrist", render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=0)
        actual_qpos = env.unwrapped._get_current_qpos()
        # Should be close to rest pose (within default noise 0.02 rad)
        np.testing.assert_allclose(actual_qpos, rest, atol=0.025)
    finally:
        env.close()
```

**Step 2: Run tests to verify they fail**

```bash
uv run --package so101-nexus-mujoco pytest packages/so101-nexus-mujoco/tests/test_base_env_init_qpos.py -v
```

Expected: FAIL — `test_reset_uses_init_qpos_from_options` fails because `init_qpos` option is ignored.

**Step 3: Update `_reset_robot_joints` in `base_env.py`**

In `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py`, change `_reset_robot_joints`:

```python
def _reset_robot_joints(self, init_qpos: np.ndarray | None = None) -> None:
    target = init_qpos if init_qpos is not None else _REST_QPOS
    for i, jid in enumerate(self._joint_ids):
        qpos_addr = self.model.jnt_qposadr[jid]
        noise = (
            0.0
            if init_qpos is not None
            else self.np_random.uniform(
                -self.robot_init_qpos_noise, self.robot_init_qpos_noise
            )
        )
        self.data.qpos[qpos_addr] = target[i] + noise
    self.data.ctrl[self._actuator_ids] = target
```

**Step 4: Update `reset()` in `base_env.py`**

In the `reset()` method, extract `init_qpos` from options and pass it through:

```python
def reset(
    self, *, seed: int | None = None, options: dict[str, Any] | None = None
) -> tuple[np.ndarray | dict[str, np.ndarray], dict]:
    super().reset(seed=seed, options=options)
    mujoco.mj_resetData(self.model, self.data)

    init_qpos: np.ndarray | None = None
    if options is not None:
        raw = options.get("init_qpos")
        if raw is not None:
            init_qpos = np.asarray(raw, dtype=np.float64)

    self._reset_robot_joints(init_qpos=init_qpos)
    self._task_reset()
    self._randomize_wrist_camera()

    # _prev_target should match actual starting pose for delta control
    self._prev_target = (init_qpos if init_qpos is not None else _REST_QPOS).copy()
    mujoco.mj_forward(self.model, self.data)

    obs = self._get_obs()
    info = self._get_info()
    return obs, info
```

**Step 5: Run tests**

```bash
uv run --package so101-nexus-mujoco pytest packages/so101-nexus-mujoco/tests/test_base_env_init_qpos.py -v
```

Expected: both tests PASS

**Step 6: Commit the env change**

```bash
git add packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py \
        packages/so101-nexus-mujoco/tests/test_base_env_init_qpos.py
git commit -m "feat: support init_qpos option in MuJoCo base env reset"
```

#### Part B: Read leader arm pose in `recording_thread`

**Step 7: Update `recording_thread` in `examples/teleop.py`**

At the start of `recording_thread`, before `env.reset()`, add:

```python
env = gym.make(env_id, camera_mode="wrist", render_mode="rgb_array")
try:
    # Read leader arm's current pose to initialize sim robot
    leader_action = leader.get_action()
    init_qpos = convert_leader_action(
        leader_action,
        joint_names,
        wrist_roll_offset_deg=wrist_roll_offset_deg,
    )
    obs, _ = env.reset(options={"init_qpos": init_qpos})
    # ... rest unchanged
```

Remove the `leader.get_action()` call that was previously the first line inside the while loop for the first iteration — it remains in the loop as-is since each loop iteration still reads the leader.

**Step 8: Run all tests**

```bash
uv run --package so101-nexus-mujoco --group teleop pytest tests/ packages/so101-nexus-mujoco/tests/ -v
```

Expected: all PASS

**Step 9: Commit**

```bash
git add examples/teleop.py
git commit -m "feat: initialize sim robot to leader arm pose at episode start"
```

---

## Final verification

Run the full test suite:

```bash
uv run --package so101-nexus-core pytest packages/so101-nexus-core/tests/ -v
uv run --package so101-nexus-mujoco pytest packages/so101-nexus-mujoco/tests/ -v
uv run --package so101-nexus-mujoco --group teleop pytest tests/ -v
```

All should pass. Smoke-test the UI manually if hardware is available:

```bash
uv run --package so101-nexus-mujoco --group teleop python examples/teleop.py --leader-port /dev/ttyACM0
```

Verify:
1. "Max Steps" input shows in Advanced Settings (default 1024)
2. Live camera feed is visibly larger (640px)
3. On episode start, sim robot matches the physical leader arm's current pose
