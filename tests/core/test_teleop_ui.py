"""Smoke tests for the teleop recorder Gradio UI structure."""

from __future__ import annotations

import pytest


@pytest.fixture
def _require_gradio():
    """Skip if gradio is not (functionally) installed.

    ``importorskip`` alone is not enough: a broken install can leave an
    empty namespace package that passes ``import gradio`` but lacks the
    public API. Verify ``Blocks`` is present before proceeding.
    """
    gr = pytest.importorskip("gradio")
    if not hasattr(gr, "Blocks"):
        pytest.skip("gradio is present but missing public API (Blocks)")


@pytest.mark.usefixtures("_require_gradio")
def test_app_builds_with_walkthrough() -> None:
    """Verify the app builds and contains a Walkthrough with 5 steps."""
    import gradio as gr

    from so101_nexus.env_ids import env_ids_for_backend
    from so101_nexus.teleop.app import (
        _build_complete_step,
        _build_init_step,
        _build_record_step,
        _build_review_step,
        _build_setup_screen,
    )

    all_env_ids = env_ids_for_backend("mujoco")

    with gr.Blocks(), gr.Walkthrough(selected=0) as wt:
        with gr.Step("Configure", id=0):
            _build_setup_screen(gr, all_env_ids, "test_leader", -90.0, "/dev/null")
        with gr.Step("Initialize", id=1):
            _build_init_step(gr)
        with gr.Step("Record", id=2):
            _build_record_step(gr)
        with gr.Step("Review", id=3):
            _build_review_step(gr)
        with gr.Step("Complete", id=4):
            _build_complete_step(gr)

    step_children = [c for c in wt.children if isinstance(c, gr.Step)]
    assert len(step_children) == 5


@pytest.mark.usefixtures("_require_gradio")
def test_record_step_has_combined_camera_preview() -> None:
    """Verify the Record step contains the combined three-camera preview image."""
    import gradio as gr

    from so101_nexus.teleop.app import _build_record_step

    with gr.Blocks():
        (
            _record_status,
            _start_btn,
            _countdown_area,
            preview_feed,
            _stop_btn,
            task_status,
            _rec_timer,
        ) = _build_record_step(gr)

    assert isinstance(preview_feed, gr.Image)
    assert preview_feed.label == "Live Preview (wrist | overhead | side)"
    assert isinstance(task_status, gr.Markdown)


@pytest.mark.usefixtures("_require_gradio")
def test_complete_step_has_editable_hub_destination() -> None:
    import gradio as gr

    from so101_nexus.teleop.app import _build_complete_step

    with gr.Blocks():
        _, _, hub_repo_id_input, push_btn, _ = _build_complete_step(gr)

    assert isinstance(hub_repo_id_input, gr.Textbox)
    assert hub_repo_id_input.label == "Hub Repo ID"
    assert hub_repo_id_input.interactive is not False
    assert "original directory" in hub_repo_id_input.info
    assert push_btn.value == "Push to Hub"


@pytest.mark.usefixtures("_require_gradio")
def test_app_upload_uses_complete_step_destination(monkeypatch) -> None:
    import gradio as gr

    from so101_nexus.teleop.app import _cb_push_to_hub, main
    from so101_nexus.teleop.cli import TeleopArgs

    apps = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda app, **kwargs: apps.append(app))
    main(TeleopArgs(leader_port="/dev/null"), backend="mujoco")

    app = apps[0]
    destination = next(
        component
        for component in app.blocks.values()
        if isinstance(component, gr.Textbox) and component.label == "Hub Repo ID"
    )
    upload = next(fn for fn in app.fns.values() if getattr(fn.fn, "func", None) is _cb_push_to_hub)
    assert upload.inputs == [destination]
    populate = next(fn for fn in app.fns.values() if fn.outputs == [destination])
    assert populate.trigger_after is not None


@pytest.mark.usefixtures("_require_gradio")
def test_intervention_controls_request_handoffs_without_queueing(monkeypatch):
    import gradio as gr

    from so101_nexus.teleop.app import _cb_request_intervention, main
    from so101_nexus.teleop.cli import TeleopArgs

    apps = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda app, **kwargs: apps.append(app))
    main(TeleopArgs(leader_port="/dev/null", policy_factory="example:build"), backend="mujoco")
    app = apps[0]
    buttons = [
        component
        for component in app.blocks.values()
        if isinstance(component, gr.Button)
        and component.elem_id
        in {"teleop-takeover", "teleop-correct", "teleop-hold", "teleop-resume"}
    ]
    assert len(buttons) == 4
    handlers = [
        fn for fn in app.fns.values() if getattr(fn.fn, "func", None) is _cb_request_intervention
    ]
    assert len(handlers) == 4
    assert all(fn.queue is False for fn in handlers)


@pytest.mark.parametrize("key", ["t", "c", "h", "p"])
def test_intervention_shortcuts_handle_direct_buttons_and_editable_fields(key):
    import shutil
    import subprocess

    from so101_nexus.teleop.app import _INTERVENTION_SHORTCUTS

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is needed to execute viewer keyboard shortcuts")
    target_id = {
        "t": "teleop-takeover",
        "c": "teleop-correct",
        "h": "teleop-hold",
        "p": "teleop-resume",
    }[key]
    script = (
        """
const assert = require('node:assert/strict');
let handler;
let clicks = 0;
let prevented = 0;
const button = {disabled: false, getClientRects: () => [1], click: () => clicks++};
global.window = {};
global.document = {
    addEventListener: (_, callback) => handler = callback,
    querySelector: selector => selector.split(',').some(s => s.trim() === 'button#' + targetId)
        ? button : null,
};
"""
        + f"const targetId = {target_id!r};"
        + f"const shortcutKey = {key!r};"
        + f"({_INTERVENTION_SHORTCUTS})();"
        + """
const event = {key: shortcutKey, target: {closest: () => null}, preventDefault: () => prevented++};
handler(event);
assert.equal(clicks, 1);
assert.equal(prevented, 1);
handler({...event, target: {closest: () => ({})}});
handler({...event, repeat: true});
handler({...event, ctrlKey: true});
button.disabled = true;
handler(event);
assert.equal(clicks, 1);
"""
    )
    subprocess.run([node, "-e", script], check=True, capture_output=True, text=True)


@pytest.mark.usefixtures("_require_gradio")
@pytest.mark.parametrize(
    ("status", "enabled", "message"),
    [
        ("ready", (False, False, False, False), "Ready"),
        ("policy", (True, False, True, False), "Running: policy"),
        ("synchronizing", (False, False, True, True), "Paused: aligning leader"),
        ("human paused", (False, True, True, True), "Paused: ready for correction"),
        ("human", (False, False, True, True), "Running: human correction"),
        ("hold", (True, False, False, True), "Paused"),
        ("waiting for policy", (True, False, True, False), "Policy active: waiting for prediction"),
    ],
)
def test_intervention_status_explains_simulation_and_available_actions(status, enabled, message):
    from types import SimpleNamespace

    from so101_nexus.teleop.app import _cb_poll_intervention

    session = {
        "intervention": SimpleNamespace(status=status),
        "state": SimpleNamespace(is_recording=status != "ready"),
    }
    text, *updates = _cb_poll_intervention(session)
    assert message in text
    assert tuple(update["interactive"] for update in updates) == enabled
    assert updates[0]["variant"] == ("primary" if enabled[0] else "secondary")
    assert updates[1]["variant"] == ("primary" if enabled[1] else "secondary")


@pytest.mark.usefixtures("_require_gradio")
def test_takeover_callback_requests_paused_alignment():
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.teleop.app import _cb_request_intervention

    intervention = Mock()
    session = {"intervention": intervention, "state": SimpleNamespace(is_recording=True)}
    message = _cb_request_intervention(session, "human", pause_after_sync=True)
    intervention.request.assert_called_once_with("human", pause_after_sync=True)
    assert "requested" in message.lower()


@pytest.mark.usefixtures("_require_gradio")
@pytest.mark.parametrize("remaining", [0, 300])
def test_intervention_status_shows_simulation_budget(remaining):
    from types import SimpleNamespace

    from so101_nexus.teleop.app import _cb_poll_intervention

    state = SimpleNamespace(is_recording=True, episode_actions=[None] * (600 - remaining))
    session = {
        "intervention": SimpleNamespace(status="human paused"),
        "state": state,
        "fps": 30,
        "max_steps": 600,
    }
    message, *_ = _cb_poll_intervention(session)
    assert f"{remaining / 30:.1f}s of simulation remaining" in message
    assert f"({remaining} steps)" in message
    assert "automatically after task success" in message


@pytest.mark.usefixtures("_require_gradio")
@pytest.mark.parametrize("aligned", [False, True])
def test_correction_callback_uses_guarded_start(aligned):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.teleop.app import _cb_request_intervention

    intervention = Mock()
    intervention.start_correction.return_value = aligned
    message = _cb_request_intervention(
        {"intervention": intervention, "state": SimpleNamespace(is_recording=True)}, "human"
    )
    intervention.start_correction.assert_called_once_with()
    intervention.request.assert_not_called()
    assert ("Human correction requested" if aligned else "Correction not started") in message


@pytest.mark.usefixtures("_require_gradio")
def test_policy_record_step_explains_autonomous_start_and_review():
    import gradio as gr

    from so101_nexus.teleop.app import _build_record_step

    with gr.Blocks():
        status, start, _, _, stop, *_ = _build_record_step(gr, policy_collection=True)
    assert "policy drives the simulated arm after the countdown" in status.value
    assert start.value == "Start policy episode"
    assert stop.value == "Stop and review episode"
