# SO-101 Calibration UI Handoff

Build a guided Gradio UI for SO-101 robot profiling, reference alignment,
excitation collection, and system identification in:

```text
/home/johnsutor/Desktop/so101-nexus
```

Read `CLAUDE.md` and repository instructions before working. Use subagents for
independent, bounded work when useful.

## Start from the latest code

- Fetch origin and inspect branch history before editing.
- The latest implementation commit when this handoff was prepared was `be4c9e9`
  on `fix/unreleased-bug-hunt`. This handoff can have a newer documentation-only
  commit above it.
- That branch contains four important unpublished fix commits after `origin/main`,
  which was `7118871`:
  - `e70366b`
  - `84cbbaa`
  - `7c80b53`
  - `be4c9e9`
- Preserve those fixes. Do not start from `origin/main` alone if they remain
  unmerged.
- Incorporate any newer relevant commits, then create `feat/calibration-ui` from
  the resulting latest state.
- Record the actual base commit in your final report.
- Preserve the existing unrelated modification to `docs/next-env.d.ts`.

## What the user wants

The existing CLI requires copying a very specific physical pose from a static
reference image. The user found this impractical and requested a UI that makes
the expected positioning clear.

Use Gradio, consistent with the existing teleoperation UI. This is a local
hardware application, not a hosted website. Reuse the repository's Gradio
patterns, components, dependencies, and visual conventions. Do not introduce a
separate frontend stack.

The central interaction must show:

- A translucent "ghost" robot representing the reference pose.
- A clearly distinguishable robot representing the current measured pose.
- Both in the same coordinate frame and viewport.
- A persistent legend distinguishing reference from measured state.
- Live joint feedback with clear units and useful alignment guidance.

Visual references:

- [Playground SO-101 system identification](https://github.com/google-deepmind/mujoco_playground/tree/main/mujoco_playground/tasks/sysid/so101)
- [Rollout comparison animation](https://www.gstatic.com/reaf-sim/demos/so101_sysid/0/so101_sysid.webp)

Inspect these references. The animation compares measured and simulated rollouts.
Borrow the overlay concept for live setup, rather than copying its entire
workflow.

## Important UX requirement

Do not just wrap the existing numerical CLI inputs in a form. Remove the need to
transcribe encoder readings or mentally reconstruct six angles from one image.

Consider guided joint-by-joint alignment, a few understandable reference
postures, adjustable viewing angles, and clear instructions for the active
joint. Explain which controls adjust the simulated reference and which actions
affect hardware.

Resolve the alignment bootstrap explicitly:

- Encoder calibration does not establish the model's physical joint zero.
- A measured-pose rendering that uses unverified offsets is only provisional.
- Do not falsely declare alignment successful by assuming the mapping being
  calibrated.
- Do not silently map an arbitrary current pose to the reference.
- Reference capture must correspond to a physically understood joint position.
- Keep body-joint alignment separate from gripper endpoint mapping.

Inspect the existing code deeply, then propose a short implementation plan
before editing. Make routine implementation decisions autonomously.

## Existing implementation to reuse

- `src/so101_nexus/teleop/app.py`: Existing Gradio app, lazy dependency imports,
  walkthroughs, progress, and cleanup.
- `src/so101_nexus/teleop/cli.py`
- `src/so101_nexus/cli.py`
- `src/so101_nexus/profiles.py`: `RobotProfile`, immutable calibration snapshots,
  alignment transforms, profile persistence, model fingerprinting, and dynamics.
- `src/so101_nexus/excitation.py`: Trajectory generation, simulation preflight,
  hardware collection, calibration checks, timestamps, tracking checks, and
  cleanup.
- `src/so101_nexus/sysid.py`: Candidate fitting, held-out evaluation, and
  diagnostics.
- `src/so101_nexus/physics.py`: Shared timing configuration.
- `examples/calibrate_robot.py`: Existing `create`, `check`, `collect`, and `fit`
  commands.
- `docs/content/docs/workflow/teleoperation.mdx`: Robot identification workflow.
- `docs/content/docs/api/configs.mdx`: Canonical profile and configuration
  reference.

## Suggested end-to-end flow

1. Discover the serial device and read motor information.
2. Identify the saved calibration that actually matches the motor registers.
3. Guide physical reference alignment with the live measured and ghost overlay.
4. Capture the reference and save a robot profile.
5. Choose a practical excitation center and preview the planned motion.
6. Run the existing simulation preflight and explain any failure concretely.
7. Start collection through an explicit user action, with clear preparation,
   progress, stop behavior, and hardware status.
8. Fit candidate dynamics and present baseline versus candidate errors,
   held-out results, and poorly identified parameters.
9. Let the user download or locate the resulting profile and recordings.

Do not automatically accept fitted dynamics just because optimization completed.

## Hardware behavior

- Loading the UI, discovering devices, and displaying live measurements must
  not enable torque, write calibration, or send movement commands.
- Do not call `SOFollower.connect()` merely to read encoders. It configures
  motors and enables torque. Use the read-only motor bus path and close it
  without changing torque.
- Explicitly define serial-port ownership across live monitoring and collection.
  Avoid competing readers or overlapping workers.
- Handle disconnects, exceptions, cancellation, stale UI state, and browser
  reconnects without leaving misleading controls or uncontrolled background
  work.
- Define stop, hold, disconnect, and motor-release behavior separately.
  Disabling torque can let a raised arm fall.
- Address the physical handoff from a supported torque-free pose to active
  holding and then excitation. The user needs time and clear feedback to remove
  their hands before motion.
- Preview and preflight the exact trajectory and configuration that collection
  uses.
- Reject stale preflight results after relevant settings or alignment change.
- Do not execute physical motion during development or browser QA without
  checking the user's current physical setup and readiness.

## Known hardware state at handoff

Read-only validation succeeded:

- Device: `/dev/ttyACM0`.
- Stable path:
  `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C118974-if00`.
- The user confirmed this is the follower arm.
- All six motors responded.
- Matching calibration:
  `/home/johnsutor/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower.json`.
- Another file, `my_so101_follower.json`, exists but does not match the hardware.
  Do not delete it.
- PID coefficients read as P=16, I=0, D=32 on all six motors.
- Torque was off on every motor, and the bus was closed.
- No motor writes or physical motion were performed.
- For 100 encoder reads at 50 Hz, median latency was 1.56 ms and maximum latency
  was 2.15 ms.
- The last measured shoulder-lift position was near its calibrated lower limit,
  so the resting pose was unsuitable as an excitation center.
- Re-read all hardware state before relying on it. These are historical
  observations.

Live validation artifacts:

```text
runs/hardware-validation/20261112T033913Z/
```

This directory contains `report.json`, `readback.json`, `read-timing.json`, and
`reference-pose.png`. The static reference image is precisely the approach the
user found hard to use.

## Existing fixes to preserve

- `RobotProfile` serialization works in training checkpoints.
- Incomplete recording retries cannot contaminate later episodes.
- BC labels use recorded model joint positions when available, preserving
  custom profile gripper ranges.
- Nonfinite reset poses are rejected before simulation state changes.
- Profile command periods drive implicit physics timing, including 30 Hz.
- Excitation stops after a missed command period.
- Failed collection removes only an empty newly created output directory.
- Limit errors identify the offending joint and requested and allowed degree
  ranges.

## Implementation and validation requirements

- Keep the top-level package import lightweight.
- Reuse the existing profiling, excitation, and sysid core rather than
  duplicating it.
- Public, configuration, and user-facing joint angles use degrees. Gripper
  hardware readings use percent. Model gripper angles use degrees. Label these
  explicitly.
- Introduce a new dependency only when necessary and explain it first.
- Use red-green regression tests for fixes. Test a happy path and an edge case
  for new behavior.
- Test hardware ownership, no-write read-only behavior, reference capture,
  stale-state invalidation, cancellation, and cleanup with fake hardware.
- Verify the actual Gradio UI in a local browser. Test initial and empty,
  connected, alignment, preflight-failure, running, stopped, and disconnected
  states.
- Run `make format lint typecheck` and appropriate broader tests.
- Before this UI work, 3,032 tests passed, 2 visual-model tests were skipped,
  and coverage was 94%. Global typechecking already reports 22 backend
  inheritance diagnostics, confirmed on untouched main. Do not weaken types to
  hide them.
- Update the changelog and canonical documentation.
- Commit meaningful steps with conventional commit messages.

Deliver a working local UI with a simple launch command. Launch it for the user.
Report what changed, why, validation, and remaining risks. Clearly distinguish
fake-hardware and browser verification from actual physical-motion validation.

## Remove this handoff after implementation

Once implementation, validation, and durable documentation are complete, delete
`CALIBRATION_UI_HANDOFF.md` and include its removal in the final implementation
commit. Transfer any necessary lasting information to the appropriate source,
tests, or canonical documentation before deleting it. This file is temporary
agent context, not user-facing project documentation.
