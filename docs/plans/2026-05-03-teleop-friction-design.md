# Teleop Friction Reduction Design

**Date:** 2026-05-03

**Branch:** `fix/teleop-friction`

## Goal

Reduce first-run teleop friction so the MuJoCo recorder is recoverable in-place when the leader-arm serial port is unavailable or lacks permissions, and clean up record-stage UI state so countdown and recording transitions feel intentional instead of sloppy.

## Current Problems

1. Serial connection failures are only surfaced after the user advances into initialization.
2. The current init failure path is effectively terminal for that attempt. Users have to fix `/dev/ttyACM*` permissions outside the app and restart or back out manually.
3. Permission and device errors are shown as raw backend failures with weak recovery guidance.
4. The record step shows duplicate “Get ready” text because both the main status region and the countdown area render the same phase.
5. Startup logs can include noisy GUI-library warnings that look fatal even when teleop is otherwise healthy.

## Constraints

- The solution must stay non-privileged. Teleop should not run `sudo`, `chmod`, or mutate host permissions.
- `uvx` must remain a first-class path.
- The core teleop package must stay importable without optional teleop dependencies at import time.
- Existing user changes in the repo must remain untouched.

## Approach Options

### Option 1: In-app preflight and retry loop

Add a serial-port preflight and structured init diagnostics inside the teleop app, then keep the Initialize step reusable after failures.

Pros:
- Fixes the actual user flow.
- No restart required after permission fixes.
- Keeps recovery guidance close to the failure.

Cons:
- Requires modest UI state refactoring.

### Option 2: CLI-only preflight before launching Gradio

Detect permission issues before the browser opens and fail fast in the terminal.

Pros:
- Simpler implementation.

Cons:
- Still forces the user out of the app.
- Does not help with reconnect/retry or record-step polish.

### Option 3: Full teleop state-machine refactor

Replace the ad hoc callback state with explicit teleop phases.

Pros:
- Cleanest long-term design.

Cons:
- Too large for the issues at hand.

## Chosen Design

Use Option 1 with a narrow state cleanup.

### Serial preflight and diagnostics

- Add helper functions in `so101_nexus_core.teleop.leader` to:
  - inspect whether the requested serial path exists
  - detect whether the current user likely lacks read/write access
  - format recovery instructions for Linux `/dev/ttyACM*` devices
- Convert connection failures into structured diagnostics instead of raw strings.
- Preserve the original exception as the root cause for logs and tests.

### Initialize-step recovery

- Keep the app on the Initialize step after a failure.
- Show a clear status message plus exact recovery commands.
- Replace the current “Back to Configure” behavior with a retry flow that:
  - keeps previously entered config values intact
  - clears the failed init state
  - allows the same browser session to try again after the user fixes the device

### Record-step cleanup

- Make countdown render in one place only.
- Ensure the main record status region owns idle/live/completion messages.
- Ensure the dedicated countdown area is only visible during the countdown phase.
- Keep camera feeds and stop button hidden until actual recording begins.

### Non-fatal GUI warning handling

- Investigate whether teleop is accidentally triggering a human-viewer path in MuJoCo.
- If the warning is external and non-fatal, prevent the app from presenting it as a teleop failure.
- If the warning is emitted during expected `rgb_array` operation, document and, where practical, suppress the misleading path.

## Components

### `packages/so101-nexus-core/src/so101_nexus_core/teleop/leader.py`

- Add serial-port inspection and diagnostic formatting helpers.

### `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py`

- Use preflight diagnostics before `leader.connect()`.
- Improve init error handling and retry behavior.
- Fix countdown/status visibility updates.

### Tests

- Extend pure helper tests for serial diagnostics.
- Add callback tests covering:
  - permission-denied init flow
  - retry-in-place state reset
  - countdown uses a single visible text surface

### Docs

- Update the teleop guide to explain the improved retry flow and exact permission recovery path.

## Error Handling

- Missing serial path: show “device not found” guidance and suggest `lerobot-find-port`.
- Permission denied: show exact non-privileged next steps and example commands.
- Other connect failures: preserve the original backend message and attach generic recovery guidance.
- Dataset creation failures: keep disconnect-on-failure behavior.

## Testing Strategy

- TDD for new helper logic and callback behavior.
- Run focused pytest targets for teleop app, leader helpers, and MuJoCo CLI if launch semantics change.
- Verify no duplicate countdown content through callback-return assertions.

## Success Criteria

- A permission issue no longer forces a browser restart.
- The UI tells the user exactly what to do when `/dev/ttyACM*` access fails.
- Re-initialization works in the same session after the user fixes permissions.
- The Record step shows one countdown display, not two competing “Get ready” boxes.
