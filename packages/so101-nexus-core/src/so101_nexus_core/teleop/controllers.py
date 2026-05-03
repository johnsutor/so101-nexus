"""Teleop callbacks and async worker, decoupled from view construction.

Every callback consumes the typed ``TeleopSession`` and ``InitState`` from
``so101_nexus_core.teleop.state`` rather than dicts.
"""

from __future__ import annotations

import contextlib
import threading

import numpy as np

from so101_nexus_core.teleop.dataset import (
    OVERHEAD_KEY,
    WRIST_KEY,
    FieldSelection,
    build_features,
    build_frame,
)
from so101_nexus_core.teleop.leader import (
    ROBOT_JOINT_NAMES,
    check_robot_env_mismatch,
    format_leader_connection_error,
    get_leader,
    import_backend_for_env_id,
)
from so101_nexus_core.teleop.recorder import (
    RecordingState,
    compute_delta_actions,
    recording_thread,
)
from so101_nexus_core.teleop.session import (
    _default_repo_id,
    make_review_video,
    make_state_plot,
)
from so101_nexus_core.teleop.state import InitConfig, InitState, TeleopSession


def progress_text(completed: int, total: int) -> str:
    """Format a progress string for the episode counter."""
    return f"**Episode {completed} / {total}**"


def build_field_selection(field_selection_value: list[str]) -> FieldSelection:
    """Construct a :class:`FieldSelection` from the checkbox-group value list."""
    return FieldSelection(
        wrist_image=WRIST_KEY in field_selection_value,
        overhead_image=OVERHEAD_KEY in field_selection_value,
        task="task" in field_selection_value,
    )


def connect_leader(robot_type: str, leader_port: str, leader_id: str):
    """Connect and return the leader arm, or raise RuntimeError."""
    leader = get_leader(robot_type, leader_port, leader_id)
    try:
        leader.connect()
    except Exception as exc:
        raise RuntimeError(format_leader_connection_error(leader_port, exc)) from exc
    return leader


def create_dataset(repo_id: str, fps: int, robot_type: str, features: dict, leader):
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


def normalized_init_config(
    *,
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
) -> InitConfig:
    """Validate UI inputs and return a frozen InitConfig."""
    fps_i = int(fps)
    wrist_wh = (int(wrist_camera_width), int(wrist_camera_height))
    overhead_wh = (int(overhead_camera_width), int(overhead_camera_height))
    repo_id_value = (repo_id or "").strip() or _default_repo_id(env_id)
    leader_id_value = (leader_id or "").strip() or leader_id_default
    return InitConfig(
        env_id=env_id,
        robot_type=robot_type,
        leader_id=leader_id_value,
        fps=fps_i,
        wrist_wh=wrist_wh,
        overhead_wh=overhead_wh,
        repo_id=repo_id_value,
        num_episodes=int(num_episodes),
        action_space=action_space,
        max_steps=int(max_steps),
        countdown=int(countdown),
        wrist_roll_offset_deg=float(wrist_roll_offset_deg),
        field_selection=build_field_selection(field_selection_value),
    )


def run_init_worker(
    session: TeleopSession,
    init_state: InitState,
    leader_port: str,
    config: InitConfig,
) -> None:
    """Body of the background init worker. Mutates session/init_state in place."""
    joint_names = ROBOT_JOINT_NAMES[config.robot_type]
    try:
        init_state.append_log(f"Connecting leader arm on {leader_port} (id={config.leader_id})...")
        import_backend_for_env_id(config.env_id)
        leader = connect_leader(config.robot_type, leader_port, config.leader_id)
        init_state.append_log("Creating LeRobot dataset...")
        features = build_features(
            config.field_selection, joint_names, config.wrist_wh, config.overhead_wh
        )
        dataset = create_dataset(config.repo_id, config.fps, config.robot_type, features, leader)
        session.leader = leader
        session.dataset = dataset
        session.state = RecordingState(num_episodes=config.num_episodes)
        session.joint_names = joint_names
        session.fps = config.fps
        session.action_space = config.action_space
        session.max_steps = config.max_steps
        session.countdown = config.countdown
        session.wrist_wh = config.wrist_wh
        session.overhead_wh = config.overhead_wh
        session.env_id = config.env_id
        session.robot_type = config.robot_type
        session.wrist_roll_offset_deg = config.wrist_roll_offset_deg
        session.field_selection = config.field_selection
        init_state.append_log("Initialization complete.")
        init_state.done = True
    except Exception as exc:
        init_state.error = str(exc)
        init_state.done = True


def start_init_attempt(
    session: TeleopSession,
    init_state: InitState,
    leader_port: str,
    config: InitConfig,
):
    """Reset init state and launch a fresh init attempt. Returns gr update tuple."""
    import gradio as gr

    init_state.reset_for_new_attempt(
        warning=check_robot_env_mismatch(config.env_id, config.robot_type),
        last_config=config,
    )

    threading.Thread(
        target=run_init_worker,
        args=(session, init_state, leader_port, config),
        daemon=True,
    ).start()

    return (
        gr.Walkthrough(selected=1),
        gr.update(value="Starting..."),
        gr.update(visible=False),
    )


def cb_start_init(
    session: TeleopSession,
    init_state: InitState,
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

    config = normalized_init_config(
        leader_id_default=leader_id_default,
        env_id=env_id,
        robot_type=robot_type,
        leader_id=leader_id,
        fps=fps,
        wrist_camera_width=wrist_camera_width,
        wrist_camera_height=wrist_camera_height,
        overhead_camera_width=overhead_camera_width,
        overhead_camera_height=overhead_camera_height,
        repo_id=repo_id,
        num_episodes=num_episodes,
        action_space=action_space,
        max_steps=max_steps,
        countdown=countdown,
        wrist_roll_offset_deg=wrist_roll_offset_deg,
        field_selection_value=field_selection_value,
    )
    if config.max_steps < 1:
        raise gr.Error("Max Steps must be at least 1.")
    return start_init_attempt(session, init_state, leader_port, config)


def cb_poll_init(session: TeleopSession, init_state: InitState):
    """Poll the init thread and stream log output to the UI."""
    import gradio as gr

    if init_state.processed:
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    log = init_state.log_text

    if not init_state.done:
        return (
            gr.update(value=log or "Starting..."),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    init_state.running = False
    init_state.processed = True

    if init_state.error:
        return (
            gr.update(value=log + f"\n\nERROR: {init_state.error}"),
            gr.update(visible=True),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    if init_state.warning:
        gr.Warning(init_state.warning)

    header = (
        f"**Env:** `{session.env_id}` | **Robot:** `{session.robot_type}` | "
        f"**FPS:** {session.fps} | **Max steps:** {session.max_steps}"
    )
    n = session.state.num_episodes  # state is non-None on success path
    return (
        gr.update(value=log),
        gr.update(),
        gr.Walkthrough(selected=2),
        gr.update(value=header, visible=True),
        gr.update(visible=True, value=progress_text(0, n)),
    )


def cb_start_recording(session: TeleopSession):
    """Launch the recording thread and update UI to recording state."""
    import gradio as gr

    s = session.state
    s.should_stop = False
    s.recording_finished = False
    threading.Thread(
        target=recording_thread,
        args=(
            s,
            session.env_id,
            session.leader,
            session.joint_names,
            session.fps,
            session.max_steps,
            session.countdown,
            session.wrist_roll_offset_deg,
            session.wrist_wh,
            session.overhead_wh,
        ),
        kwargs={"action_pipeline": session.action_pipeline},
        daemon=True,
    ).start()
    return (
        gr.update(value="Starting..."),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    )


def cb_poll_recording(session: TeleopSession):
    """Poll the recording thread for live frames and detect completion."""
    import gradio as gr

    s = session.state
    if s is None:
        return tuple(gr.update() for _ in range(10))
    fps = session.fps
    if s.countdown_value > 0:
        return (
            gr.update(value=""),
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
        import cv2

        wrist_w, wrist_h = session.wrist_wh
        overhead_w, overhead_h = session.overhead_wh

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
        return recording_finished_updates(session, s, fps)
    return tuple(gr.update() for _ in range(10))


def recording_finished_updates(session: TeleopSession, s: RecordingState, fps: int):
    """Return the review-screen update tuple after recording ends."""
    import gradio as gr

    jn = session.joint_names
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


def cb_approve_episode(session: TeleopSession):
    """Save the recorded episode to the dataset and advance."""
    import gradio as gr

    s = session.state
    dataset = session.dataset
    actions = list(s.episode_actions)
    if session.action_space == "joint_pos_delta":
        actions = compute_delta_actions(actions)

    sel = session.field_selection
    assert sel is not None, "field_selection not populated — init must succeed first"
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

    progress = progress_text(s.episodes_completed, s.num_episodes)
    if s.episodes_completed >= s.num_episodes:
        repo_id = getattr(session.dataset, "repo_id", "unknown")
        info = (
            f"**Dataset:** `{repo_id}`\n\n"
            f"**Episodes:** {s.episodes_completed} | "
            f"**Env:** `{session.env_id}` | "
            f"**FPS:** {session.fps}"
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


def cb_discard_episode(session: TeleopSession):
    """Discard the current episode buffer without saving."""
    import gradio as gr

    s = session.state
    session.dataset.clear_episode_buffer()
    s.clear_episode()
    return (
        gr.Walkthrough(selected=2),
        gr.update(value="Episode discarded. Ready to re-record. Click the button below."),
        gr.update(value=progress_text(s.episodes_completed, s.num_episodes)),
    )


def cb_retry_init(init_state: InitState):
    """Reset init state when no last_config is set; otherwise use the with-session variant."""
    import gradio as gr

    if init_state.last_config is None:
        init_state.running = False
        init_state.done = False
        init_state.processed = False
        init_state.error = None
        return (
            gr.Walkthrough(selected=0),
            gr.update(value=""),
            gr.update(visible=False),
        )
    raise RuntimeError("cb_retry_init requires session and leader_port when last_config is set")


def cb_retry_init_with_session(session: TeleopSession, init_state: InitState, leader_port: str):
    """Retry init with the last submitted config, falling back to cold reset."""
    last_config = init_state.last_config
    if last_config is None:
        return cb_retry_init(init_state)
    return start_init_attempt(session, init_state, leader_port, last_config)


def cb_push_to_hub(session: TeleopSession):
    """Push the completed dataset to HuggingFace Hub."""
    import gradio as gr

    try:
        session.dataset.push_to_hub()
    except Exception as exc:
        msg = str(exc).strip() or type(exc).__name__
        raise gr.Error(
            f"Failed to push to Hub: {msg}\n\n"
            "Make sure you are logged in (`huggingface-cli login`) and have "
            "write access to the target repository."
        ) from exc
    return "Dataset pushed to HuggingFace Hub!"


def cb_finalize_and_close(session: TeleopSession):
    """Finalize the dataset and disconnect all hardware."""
    import gradio as gr

    try:
        session.dataset.finalize()
    except Exception as exc:
        raise gr.Error(f"Failed to finalize dataset: {exc}") from exc
    with contextlib.suppress(Exception):
        session.leader.disconnect()
    return "Session finalized. You can close this tab."
