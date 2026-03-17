# Teleop UI Improvements — Design Spec

**Date:** 2026-03-16
**File:** `examples/teleop.py`
**Scope:** Bug fixes + live telemetry feature, pure Gradio, no new files

## Context

The Gradio-based teleop recorder has several UX issues and is missing live
recording feedback. This spec covers four focused improvements to the single
`examples/teleop.py` file.

**Note:** The current codebase has a partially-applied edit where
`poll_recording` already returns 9-element tuples but `rec_timer.tick` only
lists 7 outputs. The last two values (intended for `stop_btn` and a countdown
display) are silently discarded by Gradio. This spec accounts for that state
and completes the wiring.

## 1. Fix Keyboard Shortcuts

**Problem:** Keyboard shortcuts (Enter, Esc, A, D) do not work. The current
implementation injects a `<script>` tag via `gr.HTML()`, but browsers do not
execute `<script>` tags inserted via `innerHTML`.

**Solution:**
- Replace `_keyboard_shortcuts_script()` with `_keyboard_shortcuts_js()` that
  returns a JS function body (no `<script>` wrapper).
- Pass it to `gr.Blocks(js=_keyboard_shortcuts_js())` which executes on page
  load.
- Remove the `gr.HTML(...)` call.
- Shortcut mapping is unchanged: Enter → start recording, Esc → stop recording,
  A → approve, D → discard.

## 2. Countdown UX

**Problem:** Clicking "Start Recording" causes jarring layout shift (ready
screen disappears, 640px image appears) and the countdown renders ugly CV2
white text on a black numpy image.

**Solution:**
- During countdown, **hide** `live_feed` (gr.Image) and `stop_btn` (gr.Button).
- Show countdown in `recording_status` (gr.Markdown) using large inline-styled
  HTML text (e.g., `<div style="font-size:120px;...">3</div>`).
- When recording starts, **show** `live_feed` and `stop_btn`, switch
  `recording_status` back to normal recording status text.
- Remove the CV2-based countdown image rendering entirely.

**Output changes for `poll_recording`:** The existing `rec_timer.tick` outputs
list must be extended to include `stop_btn` and `countdown_display` to match
the 9 return values already present in `poll_recording`. `countdown_display`
is **not** a new component — it is the existing `recording_status` Markdown
being used for the countdown. The 9th output controls a dedicated
`countdown_area` (gr.Group) that wraps the countdown content, providing
visibility toggling between the countdown view and the recording view within
the same `recording_screen`.

Revised output list for `rec_timer.tick`:
```python
outputs=[
    recording_status, live_feed,
    recording_screen, review_screen,
    review_video, state_plot, episode_metadata,
    stop_btn, countdown_area,
]
```

## 3. Saving Indicator

**Problem:** After clicking Approve or Discard, `dataset.save_episode()` runs
synchronously with no visual feedback, making the UI appear frozen.

**Solution:**
- Add `progress=gr.Progress()` as a default parameter to `approve_episode()`
  and `discard_episode()`.
- Call `progress(0, desc="Saving episode...")` before the save operation and
  `progress(1.0, desc="Done")` after.
- Gradio renders its built-in progress bar at the top of the interface during
  the operation.
- **Important:** `progress` is auto-injected by Gradio — it must NOT appear
  in the `inputs=` list of the `.click()` handler. Gradio detects the
  `gr.Progress` default and injects it automatically when calling the function.

## 4. Live Telemetry During Recording

### 4a. Data Capture

**Problem:** `recording_thread` discards the reward and info dict returned by
`env.step()` (line 212). No live feedback is available during recording.

**Solution — RecordingState additions:**

```python
episode_rewards: list[float] = field(default_factory=list)
episode_infos: list[dict] = field(default_factory=list)
cumulative_reward: float = 0.0
```

**Solution — recording_thread changes:**
- Capture `reward` and `info` from `env.step()`.
- Append reward to `episode_rewards`, accumulate into `cumulative_reward`.
- Append a shallow copy of info dict to `episode_infos`.

### 4b. UI Components

Add to the recording screen (`recording_screen` group), **below** the camera
feed and stop button, inside a `gr.Row`. Placing them inside `recording_screen`
ensures they hide/show automatically with the recording screen group:

| Component | Type | Purpose |
|-----------|------|---------|
| `joint_live_plot` | `gr.Plot` | Plotly line chart of commanded joint positions over time |
| `reward_live_plot` | `gr.Plot` | Plotly line chart of cumulative reward over time |
| `metrics_panel` | `gr.Markdown` | Formatted table of latest info dict values |

**Note on joint positions:** `episode_states` stores the leader arm's
commanded joint positions (same values written to the dataset as
`observation.state`), not the simulator's internal state. The live plot
label should reflect this (e.g., "Commanded Joint Positions").

### 4c. Update Strategy

- **Camera feed timer** (`rec_timer`): 100ms interval, updates `live_feed`
  and `recording_status`. Unchanged from current behavior.
- **Telemetry timer** (`telemetry_timer`): new, ~1s interval, updates
  `joint_live_plot`, `reward_live_plot`, and `metrics_panel`.

This split avoids the cost of re-serializing Plotly charts at 10Hz. The
telemetry timer reads snapshot data from `RecordingState` and builds plots
from the accumulated lists.

Both timers run continuously (matching existing `rec_timer` and `init_timer`
patterns). When not recording, `poll_telemetry` returns `gr.update()` no-ops.

### 4d. Telemetry Poller (`poll_telemetry`)

New function with outputs `[joint_live_plot, reward_live_plot, metrics_panel]`.

Behavior per tick:
1. If not recording or no data yet, return `gr.update()` for all three.
2. Build joint trajectory plot from `episode_states` — same style as existing
   `make_state_plot` but using data accumulated so far. If more than 200
   points, downsample with uniform stride (`arr[::step]`) to keep Plotly
   responsive.
3. Build reward curve from `episode_rewards` — simple line chart, x = time
   (frame index / fps), y = cumulative reward.
4. Format latest info dict entry as a Markdown table showing key metrics
   (e.g., `tcp_to_target_dist`, `success`, `is_grasped`). Keys vary by
   environment, so render whatever keys are present. Format floats to 3
   decimal places, booleans as check/cross marks.

### 4e. Clearing Telemetry

`RecordingState.clear_episode()` must be extended to also clear
`episode_rewards`, `episode_infos`, and reset `cumulative_reward` to 0.0.

The telemetry components are inside `recording_screen`, so they hide
automatically when the screen transitions to review or ready. No explicit
visibility management is needed for them.

## Thread Safety

`RecordingState` is accessed from both the recording thread and Gradio
callbacks. The existing code uses this pattern for images, actions, and states.
New fields follow the same append-only pattern:

- `list.append()` is atomic under CPython's GIL.
- The telemetry poller snapshots lists via `list(state.episode_rewards)` etc.
- `cumulative_reward` (float) is read/written from different threads — this is
  safe under CPython GIL but relies on GIL semantics. This matches the
  existing pattern used for `live_frame`, `episode_duration`, etc.

This would not be safe under free-threaded Python (PEP 703). A comment in the
code should note this GIL dependency, matching the existing implicit contract.

## Non-Goals

- Custom environment config editing in the UI (deferred).
- Changes to the dataset format or recording logic.
- New files, custom CSS, or non-Gradio UI elements.
- Changes to the review screen layout.

## Testing

- Manual testing only (this is a UI-heavy example script, not library code).
- Verify: keyboard shortcuts work in Chrome/Firefox.
- Verify: countdown shows styled text, transitions smoothly to camera feed.
- Verify: progress bar appears during approve/discard.
- Verify: live plots and metrics update during recording without excessive lag.

## Risks

- **Plotly serialization cost at 1Hz:** At 30 FPS with 1024 max steps, the
  maximum is 1024 data points. Downsampling to ~200 points via uniform stride
  keeps rendering fast. Plotly is already a dependency (imported at line 31).
- **Info dict key variability:** Different environments expose different info
  keys. The metrics panel renders whatever keys are present — no hardcoded
  key list.
- **Timer overhead:** Both timers tick continuously even when idle, returning
  no-ops. This matches the existing `init_timer` and `rec_timer` patterns and
  has negligible cost.
