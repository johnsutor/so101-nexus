# Teleop Friction Reduction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make teleop initialization recoverable in-app after serial port issues and clean up record-step countdown/status behavior so first-run teleop feels polished.

**Architecture:** Add serial-port diagnostic helpers in teleop core, then thread structured failures through the Gradio init callbacks so users can fix and retry without restarting. Keep the UI changes narrow by preserving the existing callback structure while tightening the record-step visibility contract.

**Tech Stack:** Python, Gradio, pytest, LeRobot integration, MuJoCo backend registration

---

### Task 1: Add failing tests for serial diagnostics

**Files:**
- Modify: `packages/so101-nexus-core/tests/test_teleop_leader.py`
- Modify: `packages/so101-nexus-core/tests/test_teleop_app_helpers.py`

**Step 1: Write the failing test**

Add tests that expect:

```python
diag = diagnose_leader_port("/dev/ttyACM0")
assert diag.kind == "permission_denied"
assert "chmod" in diag.recovery_hint
```

and:

```python
with pytest.raises(RuntimeError, match="permission"):
    _connect_leader("so101", "/dev/ttyACM0", "leader_a")
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/so101-nexus-core/tests/test_teleop_leader.py packages/so101-nexus-core/tests/test_teleop_app_helpers.py -q`
Expected: FAIL because the diagnostic helpers and richer messaging do not exist yet.

**Step 3: Write minimal implementation**

Add a small diagnostic model/helper path in `teleop.leader` and update `_connect_leader` to include structured recovery guidance.

**Step 4: Run test to verify it passes**

Run: `pytest packages/so101-nexus-core/tests/test_teleop_leader.py packages/so101-nexus-core/tests/test_teleop_app_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/so101-nexus-core/tests/test_teleop_leader.py \
        packages/so101-nexus-core/tests/test_teleop_app_helpers.py \
        packages/so101-nexus-core/src/so101_nexus_core/teleop/leader.py \
        packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py
git commit -m "fix: add teleop serial diagnostics"
```

### Task 2: Add failing tests for init retry and countdown UI state

**Files:**
- Modify: `packages/so101-nexus-core/tests/test_teleop_app_helpers.py`

**Step 1: Write the failing test**

Add callback tests that expect:

```python
updates = _cb_poll_init(session, init_state)
assert "permission denied" in updates[0]["value"].lower()
assert updates[1]["visible"] is True
```

and:

```python
updates = _cb_poll_recording(session)
assert updates[0]["visible"] is False or "Get ready" not in str(updates[0])
assert updates[2]["visible"] is True
```

and:

```python
updates = _cb_retry_init(init_state)
assert init_state["processed"] is False
assert init_state["error"] is None
```

**Step 2: Run test to verify it fails**

Run: `pytest packages/so101-nexus-core/tests/test_teleop_app_helpers.py -q`
Expected: FAIL because the callback outputs do not yet meet the new state contract.

**Step 3: Write minimal implementation**

Refine init-state reset behavior and record-step callback outputs to expose one countdown surface and a clean retry path.

**Step 4: Run test to verify it passes**

Run: `pytest packages/so101-nexus-core/tests/test_teleop_app_helpers.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/so101-nexus-core/tests/test_teleop_app_helpers.py \
        packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py
git commit -m "fix: streamline teleop init and countdown states"
```

### Task 3: Investigate and fix misleading startup warning behavior

**Files:**
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py`
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/teleop/recorder.py`
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py` (only if root cause is in env launch path)

**Step 1: Write the failing test**

If the warning is caused by an app-controlled code path, add the smallest regression test available for that path. If it is emitted by an external native dependency and not testable in pytest, document the evidence and skip direct unit coverage.

**Step 2: Run test to verify it fails**

Run: targeted pytest command for the touched module, or record the inability to reproduce under unit test if the issue is native-library-only.
Expected: FAIL or explicit reproduction evidence.

**Step 3: Write minimal implementation**

Avoid triggering the misleading GUI path during normal teleop, or keep the warning from being surfaced as an app failure if it is non-fatal.

**Step 4: Run test to verify it passes**

Run: targeted pytest command for the touched module.
Expected: PASS, or documented evidence that the warning path is external and unchanged but now better isolated from teleop errors.

**Step 5: Commit**

```bash
git add packages/so101-nexus-core/src/so101_nexus_core/teleop/app.py \
        packages/so101-nexus-core/src/so101_nexus_core/teleop/recorder.py \
        packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py
git commit -m "fix: reduce teleop startup warning friction"
```

### Task 4: Update docs for the smoother flow

**Files:**
- Modify: `docs/content/docs/guides/teleop-dataset-recording.mdx`

**Step 1: Write the failing test**

Documentation-only change. No automated failing test required.

**Step 2: Run test to verify it fails**

Skip. Not applicable.

**Step 3: Write minimal implementation**

Document the new retry-in-place flow and explicit permission recovery commands.

**Step 4: Run test to verify it passes**

Run: `pytest packages/so101-nexus-core/tests/test_teleop_ui.py -q`
Expected: PASS as a light regression guard around the teleop UI structure.

**Step 5: Commit**

```bash
git add docs/content/docs/guides/teleop-dataset-recording.mdx
git commit -m "docs: clarify teleop recovery flow"
```

### Task 5: Final verification

**Files:**
- Modify: none

**Step 1: Write the failing test**

No new tests. This task is verification-only.

**Step 2: Run test to verify it fails**

Skip.

**Step 3: Write minimal implementation**

None.

**Step 4: Run test to verify it passes**

Run: `pytest packages/so101-nexus-core/tests/test_teleop_leader.py packages/so101-nexus-core/tests/test_teleop_app_helpers.py packages/so101-nexus-core/tests/test_teleop_ui.py packages/so101-nexus-core/tests/test_teleop_recorder.py packages/so101-nexus-core/tests/test_teleop_session.py packages/so101-nexus-core/tests/test_teleop_dataset.py packages/so101-nexus-mujoco/tests/test_cli.py -q`
Expected: PASS

**Step 5: Commit**

```bash
git add -A
git commit -m "fix: reduce teleop setup friction"
```
