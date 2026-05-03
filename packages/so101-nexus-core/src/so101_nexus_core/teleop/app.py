"""Vanilla-Gradio teleop recorder UI: thin coordinator.

Imports the view builders, controller callbacks, and typed state, then
wires them into a Blocks layout. All Gradio-heavy and lerobot-heavy
imports are deferred into individual callbacks (in controllers.py) or
into ``_import_gradio`` so importing this module on a base install does
not fail.
"""

from __future__ import annotations

import argparse
import functools

from so101_nexus_core.env_ids import Backend, env_ids_for_backend
from so101_nexus_core.teleop import controllers, view
from so101_nexus_core.teleop.leader import DEFAULT_WRIST_ROLL_OFFSET_DEG
from so101_nexus_core.teleop.state import InitState, TeleopSession

_TELEOP_INSTALL_HINT = (
    "Teleop dependencies are not installed. Install them with one of:\n"
    "  uv sync --package so101-nexus-mujoco --extra teleop\n"
    "  uv sync --package so101-nexus-maniskill --extra teleop --prerelease=allow\n"
    "  pip install 'so101-nexus-core[teleop]'"
)


def _import_gradio():
    """Import ``gradio`` and verify it is the real package, not an orphaned namespace."""
    try:
        import gradio as gr
    except ImportError as exc:
        raise SystemExit(_TELEOP_INSTALL_HINT) from exc
    if not hasattr(gr, "Blocks"):
        raise SystemExit(_TELEOP_INSTALL_HINT)
    return gr


def _wire_events(
    *,
    walkthrough,
    init_btn,
    init_inputs,
    init_log,
    retry_btn,
    init_timer,
    record_status,
    start_btn,
    countdown_area,
    wrist_feed,
    overhead_feed,
    stop_btn,
    rec_timer,
    review_video,
    state_plot,
    episode_metadata,
    approve_btn,
    discard_btn,
    done_info,
    done_status,
    push_btn,
    finalize_btn,
    session_header,
    progress_status,
    session: TeleopSession,
    init_state: InitState,
    leader_port: str,
    leader_id_default: str,
):
    """Bind controllers to view components."""
    start_init = functools.partial(
        controllers.cb_start_init, session, init_state, leader_port, leader_id_default
    )
    poll_init = functools.partial(controllers.cb_poll_init, session, init_state)
    start_recording = functools.partial(controllers.cb_start_recording, session)
    poll_recording = functools.partial(controllers.cb_poll_recording, session)
    approve_episode = functools.partial(controllers.cb_approve_episode, session)
    discard_episode = functools.partial(controllers.cb_discard_episode, session)
    push_to_hub = functools.partial(controllers.cb_push_to_hub, session)
    finalize_and_close = functools.partial(controllers.cb_finalize_and_close, session)

    def stop_recording() -> None:
        session.state.should_stop = True

    retry_init = functools.partial(
        controllers.cb_retry_init_with_session, session, init_state, leader_port
    )

    init_btn.click(fn=start_init, inputs=init_inputs, outputs=[walkthrough, init_log, retry_btn])
    init_timer.tick(
        fn=poll_init,
        outputs=[init_log, retry_btn, walkthrough, session_header, progress_status],
    )
    retry_btn.click(fn=retry_init, outputs=[walkthrough, init_log, retry_btn])
    start_btn.click(
        fn=start_recording,
        outputs=[record_status, start_btn, wrist_feed, overhead_feed],
    )
    stop_btn.click(fn=stop_recording)
    rec_timer.tick(
        fn=poll_recording,
        outputs=[
            record_status,
            start_btn,
            countdown_area,
            wrist_feed,
            overhead_feed,
            stop_btn,
            walkthrough,
            review_video,
            state_plot,
            episode_metadata,
        ],
    )
    approve_btn.click(
        fn=approve_episode,
        outputs=[walkthrough, record_status, progress_status, done_info],
    )
    discard_btn.click(
        fn=discard_episode,
        outputs=[walkthrough, record_status, progress_status],
    )
    push_btn.click(fn=push_to_hub, outputs=[done_status])
    finalize_btn.click(fn=finalize_and_close, outputs=[done_status])


def main(
    args: argparse.Namespace | None = None,
    backend: Backend | None = None,
) -> None:
    """Launch the Gradio teleop recorder app.

    Parameters
    ----------
    args
        Parsed CLI args (``--leader-port``, ``--leader-id``,
        ``--wrist-roll-offset-deg``). If ``None``, parses ``sys.argv``.
    backend
        Restrict the env dropdown to a single backend (``"mujoco"`` or
        ``"maniskill"``). When ``None``, all registered envs are shown.
    """
    gr = _import_gradio()

    if args is None:
        parser = argparse.ArgumentParser(description="Gradio-based teleop recorder")
        parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
        parser.add_argument("--leader-id", type=str, default="so101_leader")
        parser.add_argument(
            "--wrist-roll-offset-deg",
            type=float,
            default=DEFAULT_WRIST_ROLL_OFFSET_DEG,
        )
        args = parser.parse_args()

    leader_port: str = getattr(args, "leader_port", "/dev/ttyACM0")
    leader_id_default: str = getattr(args, "leader_id", "so101_leader")
    wrist_roll_offset: float = getattr(args, "wrist_roll_offset_deg", DEFAULT_WRIST_ROLL_OFFSET_DEG)

    session = TeleopSession()
    init_state = InitState()

    all_env_ids = env_ids_for_backend(backend)

    with gr.Blocks(title="SO Nexus Teleop Recorder", fill_width=True) as app:
        gr.Markdown("# SO Nexus Teleop Recorder")
        session_header = gr.Markdown(visible=False)
        progress_status = gr.Markdown(visible=False)

        with gr.Walkthrough(selected=0) as walkthrough:
            with gr.Step("Configure", id=0):
                (
                    init_btn,
                    env_id_input,
                    robot_type_input,
                    num_episodes_input,
                    leader_id_input,
                    repo_id_input,
                    fps_input,
                    action_space_input,
                    wrist_roll_offset_deg_input,
                    wrist_camera_width_input,
                    wrist_camera_height_input,
                    overhead_camera_width_input,
                    overhead_camera_height_input,
                    max_steps_input,
                    countdown_input,
                    field_selection_input,
                ) = view.build_setup_screen(gr, all_env_ids, leader_id_default, wrist_roll_offset)

            with gr.Step("Initialize", id=1):
                init_log, retry_btn, init_timer = view.build_init_step(gr)

            with gr.Step("Record", id=2):
                (
                    record_status,
                    start_btn,
                    countdown_area,
                    wrist_feed,
                    overhead_feed,
                    stop_btn,
                    rec_timer,
                ) = view.build_record_step(gr)

            with gr.Step("Review", id=3):
                (
                    review_video,
                    state_plot,
                    episode_metadata,
                    approve_btn,
                    discard_btn,
                ) = view.build_review_step(gr)

            with gr.Step("Complete", id=4):
                (
                    done_info,
                    done_status,
                    push_btn,
                    finalize_btn,
                ) = view.build_complete_step(gr)

        init_inputs = [
            env_id_input,
            robot_type_input,
            leader_id_input,
            fps_input,
            wrist_camera_width_input,
            wrist_camera_height_input,
            overhead_camera_width_input,
            overhead_camera_height_input,
            repo_id_input,
            num_episodes_input,
            action_space_input,
            max_steps_input,
            countdown_input,
            wrist_roll_offset_deg_input,
            field_selection_input,
        ]

        _wire_events(
            walkthrough=walkthrough,
            init_btn=init_btn,
            init_inputs=init_inputs,
            init_log=init_log,
            retry_btn=retry_btn,
            init_timer=init_timer,
            record_status=record_status,
            start_btn=start_btn,
            countdown_area=countdown_area,
            wrist_feed=wrist_feed,
            overhead_feed=overhead_feed,
            stop_btn=stop_btn,
            rec_timer=rec_timer,
            review_video=review_video,
            state_plot=state_plot,
            episode_metadata=episode_metadata,
            approve_btn=approve_btn,
            discard_btn=discard_btn,
            done_info=done_info,
            done_status=done_status,
            push_btn=push_btn,
            finalize_btn=finalize_btn,
            session_header=session_header,
            progress_status=progress_status,
            session=session,
            init_state=init_state,
            leader_port=leader_port,
            leader_id_default=leader_id_default,
        )

    app.launch(inbrowser=True)
