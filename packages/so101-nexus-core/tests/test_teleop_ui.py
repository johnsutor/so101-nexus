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

    from so101_nexus_core.env_ids import env_ids_for_backend
    from so101_nexus_core.teleop.app import (
        _build_complete_step,
        _build_init_step,
        _build_record_step,
        _build_review_step,
        _build_setup_screen,
    )

    all_env_ids = env_ids_for_backend("mujoco")

    with gr.Blocks(), gr.Walkthrough(selected=0) as wt:
        with gr.Step("Configure", id=0):
            _build_setup_screen(gr, all_env_ids, "test_leader", -90.0)
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
def test_record_step_has_dual_camera_feeds() -> None:
    """Verify the Record step contains two Image components for dual camera."""
    import gradio as gr

    from so101_nexus_core.teleop.app import _build_record_step

    with gr.Blocks():
        (
            _record_status,
            _start_btn,
            _countdown_area,
            wrist_feed,
            overhead_feed,
            _stop_btn,
            _rec_timer,
        ) = _build_record_step(gr)

    assert isinstance(wrist_feed, gr.Image)
    assert isinstance(overhead_feed, gr.Image)
    assert wrist_feed.label == "Wrist Camera"
    assert overhead_feed.label == "Overhead Camera"
