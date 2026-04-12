"""Vanilla-Gradio teleop recorder UI.

All heavy dependencies (gradio, lerobot, cv2) are imported lazily inside
:func:`main` or individual callbacks so that importing this module on a
base install does not fail.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading
from typing import TYPE_CHECKING, cast

import numpy as np

if TYPE_CHECKING:
    from so101_nexus_core.teleop.recorder import _WritableTextStream

from so101_nexus_core.env_ids import Backend, env_ids_for_backend
from so101_nexus_core.teleop.dataset import (
    OVERHEAD_KEY,
    WRIST_KEY,
    FieldSelection,
    build_features,
    build_frame,
)
from so101_nexus_core.teleop.leader import (
    DEFAULT_WRIST_ROLL_OFFSET_DEG,
    ROBOT_JOINT_NAMES,
    check_robot_env_mismatch,
    get_leader,
    import_backend_for_env_id,
)
from so101_nexus_core.teleop.recorder import (
    RecordingState,
    TeeStream,
    compute_delta_actions,
    recording_thread,
)
from so101_nexus_core.teleop.session import (
    _default_repo_id,
    make_review_video,
    make_state_plot,
)

_OPTIONAL_FIELD_CHOICES = [WRIST_KEY, OVERHEAD_KEY, "task"]


def _progress_text(completed: int, total: int) -> str:
    """Format a progress string for the episode counter."""
    return f"**Episode {completed} / {total}**"


def _build_field_selection(field_selection_value: list[str]) -> FieldSelection:
    """Construct a :class:`FieldSelection` from the checkbox-group value list."""
    return FieldSelection(
        wrist_image=WRIST_KEY in field_selection_value,
        overhead_image=OVERHEAD_KEY in field_selection_value,
        task="task" in field_selection_value,
    )


# ---------------------------------------------------------------------------
# Init-worker helpers
# ---------------------------------------------------------------------------


def _connect_leader(robot_type: str, leader_port: str, leader_id: str):
    """Connect and return the leader arm, or raise RuntimeError."""
    print(f"Connecting leader arm on {leader_port} (id={leader_id})...")
    leader = get_leader(robot_type, leader_port, leader_id)
    try:
        leader.connect()
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect on {leader_port}: {exc}\n"
            "Is the arm plugged in? Run 'lerobot-find-port' to list ports."
        ) from exc
    return leader


def _create_dataset(repo_id: str, fps: int, robot_type: str, features: dict, leader):
    """Create and return a LeRobotDataset, disconnecting *leader* on failure."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
        )
    except Exception as exc:
        leader.disconnect()
        raise RuntimeError(f"Failed to create dataset: {exc}") from exc


def _run_init_worker(
    session: dict,
    init_state: dict,
    leader_port: str,
    env_id: str,
    robot_type: str,
    leader_id: str,
    fps: int,
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
    repo_id: str,
    num_episodes: int,
    action_space: str,
    max_steps: int,
    countdown: int,
    wrist_roll_offset_deg: float,
    field_selection: FieldSelection,
) -> None:
    """Body of the background init worker."""
    joint_names = ROBOT_JOINT_NAMES[robot_type]
    try:
        import_backend_for_env_id(env_id)
        leader = _connect_leader(robot_type, leader_port, leader_id)
        print("Creating LeRobot dataset...")
        features = build_features(field_selection, joint_names, wrist_wh, overhead_wh)
        dataset = _create_dataset(repo_id, fps, robot_type, features, leader)
        session.update(
            leader=leader,
            dataset=dataset,
            state=RecordingState(num_episodes=num_episodes),
            joint_names=joint_names,
            fps=fps,
            action_space=action_space,
            max_steps=max_steps,
            countdown=countdown,
            wrist_wh=wrist_wh,
            overhead_wh=overhead_wh,
            env_id=env_id,
            robot_type=robot_type,
            wrist_roll_offset_deg=wrist_roll_offset_deg,
            field_selection=field_selection,
        )
        print("Initialization complete.")
        init_state["done"] = True
    except Exception as exc:
        init_state["error"] = str(exc)
        init_state["done"] = True


# ---------------------------------------------------------------------------
# Gradio callbacks (module-level; receive session/init_state as parameters)
# ---------------------------------------------------------------------------


def _cb_start_init(
    session: dict,
    init_state: dict,
    leader_port: str,
    leader_id_default: str,
    env_id: str,
    robot_type: str,
    leader_id: str,
    fps: float,
    wrist_camera_width: float,
    wrist_camera_height: float,
    overhead_camera_width: float,
    overhead_camera_height: float,
    repo_id: str,
    num_episodes: float,
    action_space: str,
    max_steps: float,
    countdown: float,
    wrist_roll_offset_deg: float,
    field_selection_value: list[str],
):
    """Validate inputs and launch the init worker thread."""
    import gradio as gr

    fps_i = int(fps)
    wrist_wh = (int(wrist_camera_width), int(wrist_camera_height))
    overhead_wh = (int(overhead_camera_width), int(overhead_camera_height))
    num_ep_i, max_steps_i, countdown_i = int(num_episodes), int(max_steps), int(countdown)
    if max_steps_i < 1:
        raise gr.Error("Max Steps must be at least 1.")
    repo_id_value = (repo_id or "").strip() or _default_repo_id(env_id)
    leader_id_value = (leader_id or "").strip() or leader_id_default
    field_selection = _build_field_selection(field_selection_value)

    init_state.update(
        warning=check_robot_env_mismatch(env_id, robot_type),
        running=True,
        done=False,
        processed=False,
        error=None,
    )

    tee_out = TeeStream(cast("_WritableTextStream", sys.stdout))
    tee_err = TeeStream(cast("_WritableTextStream", sys.stderr))
    init_state["tee_stdout"], init_state["tee_stderr"] = tee_out, tee_err
    sys.stdout, sys.stderr = tee_out, tee_err  # type: ignore[assignment]

    threading.Thread(
        target=_run_init_worker,
        args=(
            session,
            init_state,
            leader_port,
            env_id,
            robot_type,
            leader_id_value,
            fps_i,
            wrist_wh,
            overhead_wh,
            repo_id_value,
            num_ep_i,
            action_space,
            max_steps_i,
            countdown_i,
            float(wrist_roll_offset_deg),
            field_selection,
        ),
        daemon=True,
    ).start()

    return gr.Walkthrough(selected=1)


def _cb_poll_init(session: dict, init_state: dict):
    """Poll the init thread and stream log output to the UI."""
    import gradio as gr

    if init_state.get("processed"):
        return (gr.update(), gr.update(), gr.update(), gr.update())

    tee_out = init_state.get("tee_stdout")
    tee_err = init_state.get("tee_stderr")
    log = tee_out.get_output() if tee_out else ""
    if tee_err:
        err_text = tee_err.get_output()
        if err_text:
            log += "\n" + err_text

    if not init_state["done"]:
        return (gr.update(value=log or "Starting..."), gr.update(), gr.update(), gr.update())

    if tee_out:
        sys.stdout = tee_out._original  # type: ignore[assignment]
    if tee_err:
        sys.stderr = tee_err._original  # type: ignore[assignment]
    init_state["running"] = False
    init_state["processed"] = True

    if init_state["error"]:
        return (
            gr.update(value=log + f"\n\nERROR: {init_state['error']}"),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    if init_state["warning"]:
        gr.Warning(init_state["warning"])

    s = session
    header = (
        f"**Env:** `{s['env_id']}` | **Robot:** `{s['robot_type']}` | "
        f"**FPS:** {s['fps']} | **Max steps:** {s['max_steps']}"
    )
    n = s["state"].num_episodes
    return (
        gr.update(value=log),
        gr.Walkthrough(selected=2),
        gr.update(value=header, visible=True),
        gr.update(visible=True, value=_progress_text(0, n)),
    )


def _cb_start_recording(session: dict):
    """Launch the recording thread and update UI to recording state."""
    import gradio as gr

    s = session["state"]
    s.should_stop = False
    s.recording_finished = False
    threading.Thread(
        target=recording_thread,
        args=(
            s,
            session["env_id"],
            session["leader"],
            session["joint_names"],
            session["fps"],
            session["max_steps"],
            session["countdown"],
            session["wrist_roll_offset_deg"],
            session["wrist_wh"],
            session["overhead_wh"],
        ),
        daemon=True,
    ).start()
    return (
        gr.update(value="Starting..."),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def _cb_poll_recording(session: dict):
    """Poll the recording thread for live frames and detect completion."""
    import cv2
    import gradio as gr

    s = session.get("state")
    if s is None:
        return tuple(gr.update() for _ in range(10))
    fps = session["fps"]
    if s.countdown_value > 0:
        return (
            gr.update(value="Get ready..."),
            gr.update(visible=False),
            gr.update(visible=True, value=f"**{s.countdown_value}**\n\nGet ready..."),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    if s.is_recording:
        wrist_w, wrist_h = session["wrist_wh"]
        overhead_w, overhead_h = session["overhead_wh"]

        wrist_frame = s.live_frame
        if wrist_frame is None:
            wrist_frame = np.zeros((wrist_h, wrist_w, 3), dtype=np.uint8)
        elif wrist_frame.shape[0] != wrist_h or wrist_frame.shape[1] != wrist_w:
            wrist_frame = cv2.resize(
                wrist_frame, (wrist_w, wrist_h), interpolation=cv2.INTER_LINEAR
            )

        overhead_frame = s.live_overhead_frame
        if overhead_frame is None:
            overhead_frame = np.zeros((overhead_h, overhead_w, 3), dtype=np.uint8)
        elif overhead_frame.shape[0] != overhead_h or overhead_frame.shape[1] != overhead_w:
            overhead_frame = cv2.resize(
                overhead_frame, (overhead_w, overhead_h), interpolation=cv2.INTER_LINEAR
            )

        n = len(s.episode_actions)
        ep = s.episodes_completed + 1
        status = f"Recording episode {ep}/{s.num_episodes}: {n} frames ({n / fps:.1f}s)"
        return (
            gr.update(value=status),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=wrist_frame, visible=True),
            gr.update(value=overhead_frame, visible=True),
            gr.update(visible=True),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    if s.recording_finished:
        return _recording_finished_updates(session, s, fps)
    return tuple(gr.update() for _ in range(10))


def _recording_finished_updates(session: dict, s: RecordingState, fps: int):
    """Return the review-screen update tuple after recording ends."""
    import gradio as gr

    jn = session["joint_names"]
    video_path = make_review_video(s.episode_wrist_images, fps)
    fig = make_state_plot(s.episode_states, jn, fps)
    n = len(s.episode_actions)
    meta = (
        f"**Episode {s.episodes_completed + 1} / {s.num_episodes}**\n\n"
        f"- Frames: {n}\n"
        f"- Duration: {s.episode_duration:.1f}s\n"
        f"- Task: {s.task_description}"
    )
    s.recording_finished = False
    return (
        gr.update(value="Recording complete."),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.Walkthrough(selected=3),
        gr.update(value=video_path),
        gr.update(value=fig),
        gr.update(value=meta),
    )


def _cb_approve_episode(session: dict):
    """Save the recorded episode to the dataset and advance."""
    import gradio as gr

    s = session["state"]
    dataset = session["dataset"]
    actions = list(s.episode_actions)
    if session["action_space"] == "joint_pos_delta":
        actions = compute_delta_actions(actions)

    sel = session["field_selection"]
    for i in range(len(actions)):
        wrist_img = s.episode_wrist_images[i] if i < len(s.episode_wrist_images) else None
        overhead_img = s.episode_overhead_images[i] if i < len(s.episode_overhead_images) else None
        try:
            frame = build_frame(
                sel,
                state=s.episode_states[i],
                action=actions[i],
                task=s.task_description,
                wrist_image=wrist_img,
                overhead_image=overhead_img,
            )
        except ValueError as exc:
            raise gr.Error(str(exc)) from exc
        dataset.add_frame(frame)

    dataset.save_episode()
    s.episodes_completed += 1
    s.clear_episode()

    progress = _progress_text(s.episodes_completed, s.num_episodes)
    if s.episodes_completed >= s.num_episodes:
        ds = session["dataset"]
        repo_id = getattr(ds, "repo_id", "unknown")
        info = (
            f"**Dataset:** `{repo_id}`\n\n"
            f"**Episodes:** {s.episodes_completed} | "
            f"**Env:** `{session['env_id']}` | "
            f"**FPS:** {session['fps']}"
        )
        return (
            gr.Walkthrough(selected=4),
            gr.update(),
            gr.update(value=progress),
            gr.update(value=info),
        )
    return (
        gr.Walkthrough(selected=2),
        gr.update(value="Episode saved! Ready to record the next one. Click the button below."),
        gr.update(value=progress),
        gr.update(),
    )


def _cb_discard_episode(session: dict):
    """Discard the current episode buffer without saving."""
    import gradio as gr

    s = session["state"]
    session["dataset"].clear_episode_buffer()
    s.clear_episode()
    return (
        gr.Walkthrough(selected=2),
        gr.update(value="Episode discarded. Ready to re-record. Click the button below."),
        gr.update(value=_progress_text(s.episodes_completed, s.num_episodes)),
    )


def _cb_push_to_hub(session: dict):
    """Push the completed dataset to HuggingFace Hub."""
    import gradio as gr

    try:
        session["dataset"].push_to_hub()
    except Exception as exc:
        msg = str(exc).strip() or type(exc).__name__
        raise gr.Error(
            f"Failed to push to Hub: {msg}\n\n"
            "Make sure you are logged in (`huggingface-cli login`) and have "
            "write access to the target repository."
        ) from exc
    return "Dataset pushed to HuggingFace Hub!"


def _cb_finalize_and_close(session: dict):
    """Finalize the dataset and disconnect all hardware."""
    import gradio as gr

    try:
        session["dataset"].finalize()
    except Exception as exc:
        raise gr.Error(f"Failed to finalize dataset: {exc}") from exc
    with contextlib.suppress(Exception):
        session["leader"].disconnect()
    return "Session finalized. You can close this tab."


# ---------------------------------------------------------------------------
# UI builder helpers
# ---------------------------------------------------------------------------


def _build_setup_screen(
    gr,
    all_env_ids: list[str],
    default_leader_id: str,
    wrist_roll_offset: float,
):
    """Build the Configure step contents and return all input components."""
    gr.Markdown("### Environment & Robot")
    env_id_input = gr.Dropdown(
        choices=all_env_ids,
        value=all_env_ids[0] if all_env_ids else None,
        label="Environment",
    )
    with gr.Row():
        robot_type_input = gr.Radio(choices=["so100", "so101"], value="so101", label="Robot Type")
        action_space_input = gr.Radio(
            choices=["joint_pos", "joint_pos_delta"],
            value="joint_pos_delta",
            label="Action Space",
        )

    gr.Markdown("### Recording")
    with gr.Row():
        num_episodes_input = gr.Number(value=5, precision=0, label="Number of Episodes")
        fps_input = gr.Slider(minimum=1, maximum=60, value=30, step=1, label="FPS")
        countdown_input = gr.Slider(minimum=0, maximum=10, value=3, step=1, label="Countdown (s)")

    gr.Markdown("### Dataset & Storage")
    repo_id_input = gr.Textbox(
        label="HuggingFace Repo ID (optional)",
        placeholder="username/dataset-name",
        info="Leave blank for local-only recording",
    )
    field_selection_input = gr.CheckboxGroup(
        choices=_OPTIONAL_FIELD_CHOICES,
        value=list(_OPTIONAL_FIELD_CHOICES),
        label="Dataset fields",
        info="observation.state and action are always saved",
    )

    with gr.Accordion("Advanced Settings", open=False):
        leader_id_input = gr.Textbox(label="Leader Arm ID", value=default_leader_id)
        wrist_roll_offset_deg_input = gr.Slider(
            minimum=-180,
            maximum=180,
            value=wrist_roll_offset,
            step=1,
            label="Wrist Roll Offset (deg)",
        )
        with gr.Row():
            wrist_camera_width_input = gr.Slider(
                minimum=64, maximum=1024, value=480, step=32, label="Wrist Camera Width"
            )
            wrist_camera_height_input = gr.Slider(
                minimum=64, maximum=1024, value=480, step=32, label="Wrist Camera Height"
            )
        with gr.Row():
            overhead_camera_width_input = gr.Slider(
                minimum=64, maximum=1024, value=640, step=32, label="Overhead Camera Width"
            )
            overhead_camera_height_input = gr.Slider(
                minimum=64, maximum=1024, value=480, step=32, label="Overhead Camera Height"
            )
        max_steps_input = gr.Number(value=1024, minimum=1, precision=0, label="Max Steps")

    init_btn = gr.Button("Initialize Session", variant="primary")

    return (
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
    )


def _build_init_step(gr):
    """Build the Initialize step contents."""
    gr.Markdown("### Initializing Session...")
    init_log = gr.Textbox(label="Log Output", lines=12, max_lines=20, interactive=False)
    init_timer = gr.Timer(value=0.25)
    return init_log, init_timer


def _build_record_step(gr):
    """Build the Record step contents."""
    record_status = gr.Markdown("Ready to record. Click the button below to begin.")
    start_btn = gr.Button("Start Recording", variant="primary")
    countdown_area = gr.Markdown("Get ready...", visible=False)
    with gr.Row():
        wrist_feed = gr.Image(label="Wrist Camera", height=480, visible=False)
        overhead_feed = gr.Image(label="Overhead Camera", height=480, visible=False)
    stop_btn = gr.Button("Stop Recording", variant="stop", visible=False)
    rec_timer = gr.Timer(value=0.1)
    return record_status, start_btn, countdown_area, wrist_feed, overhead_feed, stop_btn, rec_timer


def _build_review_step(gr):
    """Build the Review step contents."""
    gr.Markdown("### Review Episode")
    with gr.Row():
        with gr.Column(scale=2):
            review_video = gr.Video(label="Episode Video")
        with gr.Column(scale=1):
            state_plot = gr.Plot(label="Joint States")
            episode_metadata = gr.Markdown()
    with gr.Row():
        approve_btn = gr.Button("Approve", variant="primary")
        discard_btn = gr.Button("Discard", variant="stop")
    return review_video, state_plot, episode_metadata, approve_btn, discard_btn


def _build_complete_step(gr):
    """Build the Complete step contents."""
    gr.Markdown("### All episodes recorded!")
    done_info = gr.Markdown("")
    done_status = gr.Markdown("")
    with gr.Row():
        push_btn = gr.Button("Push to Hub", variant="primary")
        finalize_btn = gr.Button("Finalize & Close")
    return done_info, done_status, push_btn, finalize_btn


def _wire_events(
    gr,
    *,
    walkthrough,
    init_btn,
    init_inputs,
    init_log,
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
    session,
    init_state,
    leader_port,
    leader_id_default,
):
    """Wire all Gradio event handlers."""
    import functools

    start_init = functools.partial(
        _cb_start_init, session, init_state, leader_port, leader_id_default
    )
    poll_init = functools.partial(_cb_poll_init, session, init_state)
    start_recording = functools.partial(_cb_start_recording, session)
    poll_recording = functools.partial(_cb_poll_recording, session)
    approve_episode = functools.partial(_cb_approve_episode, session)
    discard_episode = functools.partial(_cb_discard_episode, session)
    push_to_hub = functools.partial(_cb_push_to_hub, session)
    finalize_and_close = functools.partial(_cb_finalize_and_close, session)

    def stop_recording() -> None:
        session["state"].should_stop = True

    init_btn.click(fn=start_init, inputs=init_inputs, outputs=[walkthrough])
    init_timer.tick(
        fn=poll_init,
        outputs=[init_log, walkthrough, session_header, progress_status],
    )
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

    session: dict = {}
    init_state: dict = {
        "running": False,
        "done": False,
        "processed": False,
        "error": None,
        "tee_stdout": None,
        "tee_stderr": None,
        "warning": None,
    }

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
                ) = _build_setup_screen(gr, all_env_ids, leader_id_default, wrist_roll_offset)

            with gr.Step("Initialize", id=1):
                init_log, init_timer = _build_init_step(gr)

            with gr.Step("Record", id=2):
                (
                    record_status,
                    start_btn,
                    countdown_area,
                    wrist_feed,
                    overhead_feed,
                    stop_btn,
                    rec_timer,
                ) = _build_record_step(gr)

            with gr.Step("Review", id=3):
                (
                    review_video,
                    state_plot,
                    episode_metadata,
                    approve_btn,
                    discard_btn,
                ) = _build_review_step(gr)

            with gr.Step("Complete", id=4):
                (
                    done_info,
                    done_status,
                    push_btn,
                    finalize_btn,
                ) = _build_complete_step(gr)

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
            gr,
            walkthrough=walkthrough,
            init_btn=init_btn,
            init_inputs=init_inputs,
            init_log=init_log,
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
