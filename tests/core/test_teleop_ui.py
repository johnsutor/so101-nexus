"""Smoke tests for the teleop recorder Gradio UI structure."""

from __future__ import annotations

import pytest


def _descendants(component):
    for child in getattr(component, "children", []):
        yield child
        yield from _descendants(child)


@pytest.mark.usefixtures("_require_gradio")
def test_setup_discloses_optional_recording_settings():
    import gradio as gr

    from so101_nexus.teleop.app import _build_setup_screen
    from so101_nexus.teleop.dataset import ENV_STATE_KEY, OVERHEAD_KEY, SIDE_KEY, WRIST_KEY

    with gr.Blocks() as app:
        controls = _build_setup_screen(gr, ["MuJoCoTouch-v1"], "leader", -90, "/dev/null")

    options = next(
        c
        for c in app.blocks.values()
        if isinstance(c, gr.Accordion) and c.label == "Recording settings"
    )
    optional = list(_descendants(options))
    assert options.open is False
    assert all(controls[i] in optional for i in (4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 27))
    assert all(controls[i] not in optional for i in (1, 2, 3, 5))
    assert controls[3].minimum == 1
    assert controls[7].value == "joint_pos"
    assert controls[24].value == 5
    assert controls[27].value == 0.5
    assert controls[7].choices == [
        ("Joint positions", "joint_pos"),
        ("Changes in joint position", "joint_pos_delta"),
    ]
    assert controls[15].choices == [
        ("Wrist camera", WRIST_KEY),
        ("Overhead camera", OVERHEAD_KEY),
        ("Side camera", SIDE_KEY),
        ("Environment state", ENV_STATE_KEY),
    ]
    assert controls[15].value == [WRIST_KEY, OVERHEAD_KEY, SIDE_KEY, ENV_STATE_KEY]


@pytest.mark.usefixtures("_require_gradio")
@pytest.mark.parametrize("enabled", [False, True])
def test_customization_disclosure_tracks_enablement(monkeypatch, enabled):
    import gradio as gr

    from so101_nexus.teleop import app as teleop_app

    monkeypatch.setattr(
        teleop_app,
        "_customization_ui_state_for_env",
        lambda *args, **kwargs: teleop_app.CustomizationUIState(
            customize_visible=True, customize_value=enabled, common_visible=True
        ),
    )
    with gr.Blocks() as app:
        controls = teleop_app._build_setup_screen(gr, ["custom-v1"], "leader", -90, "/dev/null")

    handler = next(fn for fn in app.fns.values() if fn.inputs == [controls[16]])
    panel = handler.outputs[0]
    assert panel.visible is enabled
    assert controls[19] in list(_descendants(panel))
    assert handler.fn(not enabled)["visible"] is not enabled
    assert handler.fn(enabled)["visible"] is enabled
    assert controls[29].visible is True
    assert controls[30].visible is False


@pytest.mark.usefixtures("_require_gradio")
def test_review_prioritizes_replay_and_explicit_save():
    import gradio as gr

    from so101_nexus.teleop.app import _build_review_step

    with gr.Blocks() as app:
        video, plot, _, _, save, discard = _build_review_step(gr)

    diagnostics = next(c for c in app.blocks.values() if isinstance(c, gr.Accordion))
    assert diagnostics.open is False
    assert plot in list(_descendants(diagnostics))
    assert video not in list(_descendants(diagnostics))
    assert video.interactive is False
    assert save.value == "Save episode"
    assert discard.value == "Discard and retry"


@pytest.mark.usefixtures("_require_gradio")
@pytest.mark.parametrize("policy_collection", [False, True])
def test_completion_prioritizes_local_finish(policy_collection):
    import gradio as gr

    from so101_nexus.teleop.app import _build_complete_step

    with gr.Blocks() as app:
        _, _, destination, upload, finish = _build_complete_step(
            gr, policy_collection=policy_collection
        )

    optional = next(c for c in app.blocks.values() if isinstance(c, gr.Accordion))
    assert optional.label == "Upload to Hugging Face (optional)"
    assert optional.open is False
    assert destination in list(_descendants(optional))
    assert upload in list(_descendants(optional))
    assert finish not in list(_descendants(optional))
    assert finish.variant == "primary"
    assert finish.value == "Finish session"
    guidance = [c.value for c in app.blocks.values() if isinstance(c, gr.Markdown)]
    assert any("Leader motors" in text for text in guidance) is policy_collection


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
def test_initialization_and_retry_keep_native_progress_visible(monkeypatch):
    import gradio as gr

    from so101_nexus.teleop.app import main
    from so101_nexus.teleop.cli import TeleopArgs

    apps = []
    monkeypatch.setattr(gr.Blocks, "launch", lambda app, **kwargs: apps.append(app))
    main(TeleopArgs(leader_port="/dev/null"), backend="mujoco")
    app = apps[0]
    handlers = [fn for fn in app.fns.values() if fn.name == "show_init_progress"]
    assert len(handlers) == 2
    assert handlers[0].outputs == handlers[1].outputs
    for handler in handlers:
        assert handler.trigger_after is not None
        assert handler.trigger_only_on_success
        assert handler.show_progress == "full"
        assert isinstance(handler.fn.__defaults__[0], gr.Progress)
    details = next(
        component
        for component in app.blocks.values()
        if isinstance(component, gr.Accordion) and component.label == "Initialization details"
    )
    assert not details.open


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
    assert all(len(fn.inputs) == 1 and isinstance(fn.inputs[0], gr.State) for fn in handlers)


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
const checkbox = {closest: () => ({matches: selector => selector.includes('checkbox')})};
handler({...event, target: checkbox});
assert.equal(clicks, 2);
handler({...event, target: {closest: () => ({matches: () => false})}});
handler({...event, repeat: true});
handler({...event, ctrlKey: true});
button.disabled = true;
handler(event);
assert.equal(clicks, 2);
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
        ("alignment paused", (True, True, True, True), "Paused: alignment needs attention"),
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
        "intervention": SimpleNamespace(
            status=status, synchronization_error=None, support_token=(1, 1)
        ),
        "state": SimpleNamespace(is_recording=status != "ready"),
    }
    _cb_poll_intervention(session)
    text, *updates = _cb_poll_intervention(session)
    assert message in text
    assert tuple(update["interactive"] for update in updates[:4]) == enabled
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

    state = SimpleNamespace(is_recording=True, transitions=[None] * (600 - remaining))
    session = {
        "intervention": SimpleNamespace(status="human paused", support_token=(1, 1)),
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

    intervention = Mock(support_token=(1, 1))
    intervention.start_correction.return_value = aligned
    message = _cb_request_intervention(
        {"intervention": intervention, "state": SimpleNamespace(is_recording=True)},
        "human",
        support_token=intervention.support_token,
    )
    intervention.start_correction.assert_called_once_with(support_token=intervention.support_token)
    intervention.request.assert_not_called()
    assert ("Human correction requested" if aligned else "Correction not started") in message


@pytest.mark.usefixtures("_require_gradio")
def test_policy_record_step_explains_autonomous_start_and_review():
    import gradio as gr

    from so101_nexus.teleop.app import _build_record_step

    with gr.Blocks():
        status, start, _, _, stop, *_ = _build_record_step(gr, policy_collection=True)
    assert "policy then drives the simulated arm" in status.value
    assert "default resting pose over four seconds" in status.value
    assert start.value == "Start policy episode"
    assert stop.value == "Stop and review episode"


@pytest.mark.usefixtures("_require_gradio")
@pytest.mark.parametrize("token", [None, (1, 0)])
def test_start_correction_rejects_missing_or_stale_token(token):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.teleop.app import _cb_request_intervention

    intervention = Mock(support_token=(1, 1))
    message = _cb_request_intervention(
        {"intervention": intervention, "state": SimpleNamespace(is_recording=True)},
        "human",
        support_token=token,
    )
    intervention.start_correction.assert_not_called()
    assert "changed" in message.lower()


@pytest.mark.usefixtures("_require_gradio")
@pytest.mark.parametrize("supported", [False, True])
def test_idle_release_requires_support_and_an_idle_worker(supported):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.teleop.app import _cb_release_leader

    intervention = Mock()
    worker = Mock()
    worker.wait.return_value = False
    session = {
        "intervention": intervention,
        "state": SimpleNamespace(is_recording=False),
        "recording_worker": worker,
    }
    _cb_release_leader(session, supported, intervention.support_token)
    intervention.release_leader.assert_not_called()
    worker.wait.return_value = True
    _cb_release_leader(session, supported, intervention.support_token)
    assert intervention.release_leader.call_count == int(supported)


@pytest.mark.usefixtures("_require_gradio")
def test_finalize_refuses_to_disconnect_an_unreleased_physical_leader():
    from unittest.mock import Mock

    import gradio as gr

    from so101_nexus.teleop.app import _cb_finalize_and_close

    intervention = Mock(leader_released=False)
    leader = Mock()
    dataset = Mock()
    with pytest.raises(gr.Error, match="release its motors"):
        _cb_finalize_and_close({"intervention": intervention, "leader": leader, "dataset": dataset})
    intervention.close.assert_not_called()
    leader.disconnect.assert_not_called()
    dataset.finalize.assert_not_called()


@pytest.mark.usefixtures("_require_gradio")
def test_unconfirmed_hold_is_never_reported_as_holding():
    from types import SimpleNamespace

    from so101_nexus.teleop.app import _cb_poll_leader_support

    intervention = SimpleNamespace(
        leader_released=False,
        status="hold",
        synchronizer=SimpleNamespace(holding=False),
        support_token=(1, 1),
    )
    state = SimpleNamespace(is_recording=False, error="Bus read failed")
    message, update, *_ = _cb_poll_leader_support({"intervention": intervention, "state": state})
    assert "hold unconfirmed" in message
    assert "Support the arm and gripper now" in message
    assert update["interactive"] is False


@pytest.mark.usefixtures("_require_gradio")
def test_uninitialized_support_panel_resets_confirmation():
    from so101_nexus.teleop.app import _cb_poll_leader_support

    _, release, support, token, *_ = _cb_poll_leader_support({}, True)
    assert release["interactive"] is False
    assert support["interactive"] is False
    assert support["value"] is False
    assert token is None


@pytest.mark.usefixtures("_require_gradio")
def test_automatic_episode_end_expires_correction_token():
    from types import SimpleNamespace

    from so101_nexus.teleop.app import _cb_poll_intervention

    intervention = SimpleNamespace(status="human paused", support_token=(1, 1))
    state = SimpleNamespace(is_recording=True)
    session = {"intervention": intervention, "state": state}
    _, _, correction, _, _, token, _ = _cb_poll_intervention(session)
    assert correction["interactive"] is True
    assert token == (1, 1)
    intervention.support_token = (1, 2)
    intervention.status = "hold"
    state.is_recording = False
    _, _, correction, _, _, token, _ = _cb_poll_intervention(session)
    assert correction["interactive"] is False
    assert token is None


@pytest.mark.usefixtures("_require_gradio")
def test_reinitialization_requires_explicit_motor_release(monkeypatch):
    from unittest.mock import Mock

    import gradio as gr

    from so101_nexus.teleop import app

    close = Mock()
    monkeypatch.setattr(app, "_close_recording_worker", close)
    with pytest.raises(gr.Error, match="release its motors"):
        app._start_init_attempt({"intervention": Mock(leader_released=False)}, {}, "/dev/null", {})
    close.assert_not_called()


@pytest.mark.usefixtures("_require_gradio")
def test_alignment_pause_shows_joint_diagnostics_and_recovery_actions():
    from types import SimpleNamespace

    from so101_nexus.teleop.app import _cb_poll_intervention

    session = {
        "intervention": SimpleNamespace(
            status="alignment paused",
            synchronization_error="wrist_flex: measured 6.0, target 10.0 deg",
            support_token=(1, 1),
        ),
        "state": SimpleNamespace(is_recording=True),
    }
    message, takeover, correct, _, policy, *_ = _cb_poll_intervention(session)
    assert "wrist_flex" in message
    assert "6.0" in message
    assert "10.0 deg" in message
    assert "retries" in message.lower()
    assert takeover["interactive"] is True
    assert correct["interactive"] is True
    assert policy["interactive"] is True


@pytest.mark.usefixtures("_require_gradio")
@pytest.mark.parametrize("countdown", [0, 3])
def test_pending_episode_keeps_stop_available(countdown):
    from unittest.mock import Mock

    from so101_nexus.teleop.app import _cb_poll_recording
    from so101_nexus.teleop.recorder import RecordingState

    worker = Mock()
    worker.wait.return_value = False
    updates = _cb_poll_recording(
        {"state": RecordingState(countdown_value=countdown), "fps": 30, "recording_worker": worker}
    )
    assert updates[4]["visible"] is True


@pytest.mark.usefixtures("_require_gradio")
def test_policy_correction_contains_optional_motor_controls():
    import gradio as gr

    from so101_nexus.teleop.app import _build_policy_correction

    with gr.Blocks():
        panel, _, support = _build_policy_correction(gr, enabled=True)

    children = list(_descendants(panel))
    assert panel.variant == "default"
    assert not any(isinstance(child, gr.Group) for child in children)
    motor_controls = next(
        c for c in children if isinstance(c, gr.Accordion) and c.label == "Leader motors"
    )
    assert motor_controls.open is False
    assert support[2] in list(_descendants(motor_controls))
    assert panel.visible is False
    assert (
        sum(isinstance(c, gr.Markdown) and c.value == "## Policy correction" for c in children) == 1
    )


@pytest.mark.usefixtures("_require_gradio")
@pytest.mark.parametrize(
    ("status", "countdown", "recording"),
    [("returning to rest", 0, False), ("ready", 0, False), ("policy", 3, True)],
)
def test_preparing_episode_disables_policy_handoff(status, countdown, recording):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.teleop.app import _cb_poll_intervention

    worker = Mock()
    worker.wait.return_value = False
    session = {
        "intervention": SimpleNamespace(
            status=status, synchronization_error=None, support_token=(1, 1)
        ),
        "state": SimpleNamespace(is_recording=recording, countdown_value=countdown, error=None),
        "recording_worker": worker,
    }
    message, *updates = _cb_poll_intervention(session)
    expected = "returning" if status == "returning to rest" else "preparing"
    assert expected in message.lower()
    assert all(update["interactive"] is False for update in updates[:4])
    assert updates[-1]["visible"] is False


@pytest.mark.usefixtures("_require_gradio")
def test_rest_return_failure_shows_recovery_and_motor_diagnostics():
    from types import SimpleNamespace

    from so101_nexus.teleop.app import _cb_poll_intervention

    session = {
        "intervention": SimpleNamespace(
            status="hold", synchronization_error=None, support_token=(1, 1)
        ),
        "state": SimpleNamespace(
            is_recording=False, error="Leader return timed out: elbow: measured 6, target 10 deg"
        ),
    }
    message, *updates = _cb_poll_intervention(session)
    assert updates[-1]["visible"] is False
    assert "elbow" in message
    assert "6" in message
    assert "10 deg" in message
    assert "discard" in message.lower()
    assert "Leader motors" in message


@pytest.mark.usefixtures("_require_gradio")
def test_policy_handoff_controls_visible_only_during_active_recording():
    from types import SimpleNamespace

    from so101_nexus.teleop.app import _cb_poll_intervention

    session = {
        "intervention": SimpleNamespace(status="policy", support_token=(1, 1)),
        "state": SimpleNamespace(is_recording=True),
    }
    *_, group = _cb_poll_intervention(session)
    assert group["visible"] is True
    session["state"].is_recording = False
    *_, group = _cb_poll_intervention(session)
    assert group["visible"] is False


@pytest.mark.usefixtures("_require_gradio")
def test_correction_disclaimer_replaces_takeover_checkbox():
    import gradio as gr

    from so101_nexus.teleop.app import _build_intervention_controls

    with gr.Blocks():
        _, _, correct, _, _, token, handoff = _build_intervention_controls(gr, enabled=True)
    assert isinstance(token, gr.State)
    assert not any(isinstance(child, gr.Checkbox) for child in handoff.children)
    disclaimer = handoff.children[handoff.children.index(correct) + 1]
    assert isinstance(disclaimer, gr.Markdown)
    assert disclaimer.value == (
        "Support the arm and gripper before taking control. This releases the motors."
    )
