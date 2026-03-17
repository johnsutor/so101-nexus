# Teleop UI Improvements — Design Spec

**Date:** 2026-03-16
**File:** `examples/teleop.py`
**Scope:** Bug fixes + live telemetry feature, pure Gradio, no new files

## Context

The Gradio-based teleop recorder has several UX issues and is missing live
recording feedback. This spec covers four focused improvements to the single
`examples/teleop.py` file.

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

**Output changes for `poll_recording`:** Two new outputs are needed
(`stop_btn` and a new `countdown_display` Markdown component) to control
visibility during the countdown-to-recording transition. All return paths in
`poll_recording` must be updated for the new output count.

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
- Append info dict to `episode_infos`.
- Store the latest info dict for quick access by the telemetry poller.

### 4b. UI Components

Add to the recording screen, below the camera feed, inside a `gr.Row`:

| Component | Type | Purpose |
|-----------|------|---------|
| `joint_live_plot` | `gr.Plot` | Plotly line chart of joint positions over time |
| `reward_live_plot` | `gr.Plot` | Plotly line chart of cumulative reward over time |
| `metrics_panel` | `gr.Markdown` | Formatted table of latest info dict values |

### 4c. Update Strategy

- **Camera feed timer** (`rec_timer`): 100ms interval, updates `live_feed`
  and `recording_status`. Unchanged from current behavior.
- **Telemetry timer** (`telemetry_timer`): new, ~1s interval, updates
  `joint_live_plot`, `reward_live_plot`, and `metrics_panel`.

This split avoids the cost of re-serializing Plotly charts at 10Hz. The
telemetry timer reads snapshot data from `RecordingState` and builds plots
from the accumulated lists.

### 4d. Telemetry Poller (`poll_telemetry`)

New function with outputs `[joint_live_plot, reward_live_plot, metrics_panel]`.

Behavior per tick:
1. If not recording or no data yet, return `gr.update()` for all three.
2. Build joint trajectory plot from `episode_states` (same style as existing
   `make_state_plot` but using data accumulated so far).
3. Build reward curve from `episode_rewards` (simple line chart, x = time,
   y = cumulative reward).
4. Format latest info dict entry as a Markdown table showing key metrics
   (e.g., `tcp_to_target_dist`, `success`, `is_grasped`). Keys vary by
   environment, so render whatever keys are present.

### 4e. Clearing Telemetry

When an episode ends (recording_finished) or is approved/discarded, the
telemetry data is already cleared by `RecordingState.clear_episode()`. The
`clear_episode` method needs to be extended to also clear `episode_rewards`,
`episode_infos`, and reset `cumulative_reward`.

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

- **Plotly serialization cost at 1Hz:** If episodes are long with many frames,
  building the full plot each tick could get slow. Mitigation: downsample to
  ~200 points max before plotting.
- **Info dict key variability:** Different environments expose different info
  keys. The metrics panel must handle arbitrary key sets gracefully.
- **Thread safety:** `RecordingState` is accessed from both the recording
  thread and Gradio callbacks. Current code already does this for images and
  actions. New fields (rewards, infos) follow the same append-only pattern,
  which is safe for Python lists with GIL protection.
