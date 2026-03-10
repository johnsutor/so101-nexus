"""Gradio-based teleop recorder for SO101-Nexus environments.

Records LeRobot v3 datasets from a physical SO leader arm controlling a
simulated follower in any registered gymnasium environment. All recording
parameters are configured through the Gradio web UI.

Usage::

    uv run --package so101-nexus-mujoco --group teleop python examples/teleop.py
    uv run --package so101-nexus-mujoco --group teleop \\
        python examples/teleop.py --leader-port /dev/ttyACM1
"""

from __future__ import annotations

import argparse
import datetime
import io
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol, cast

import cv2
import gymnasium as gym
import numpy as np
import plotly.graph_objects as go

from so101_nexus_core.config import SO101_JOINT_NAMES
from so101_nexus_core.env_ids import all_registered_env_ids

ROBOT_JOINT_NAMES: dict[str, tuple[str, ...]] = {
    "so100": SO101_JOINT_NAMES,
    "so101": SO101_JOINT_NAMES,
}

DEFAULT_WRIST_ROLL_OFFSET_DEG = -90.0


def get_leader(robot_type: str, port: str, leader_id: str):
    """Create and return the appropriate ``SOLeader`` for *robot_type*."""
    if robot_type == "so100":
        from lerobot.teleoperators.so_leader.config_so_leader import SO100LeaderConfig
        from lerobot.teleoperators.so_leader.so_leader import SO100Leader

        return SO100Leader(SO100LeaderConfig(port=port, use_degrees=True, id=leader_id))

    from lerobot.teleoperators.so_leader.config_so_leader import SO101LeaderConfig
    from lerobot.teleoperators.so_leader.so_leader import SO101Leader

    return SO101Leader(SO101LeaderConfig(port=port, use_degrees=True, id=leader_id))


def check_robot_env_mismatch(env_id: str, robot_type: str) -> str | None:
    """Return a warning string if *env_id* encodes a robot that contradicts *robot_type*."""
    upper = env_id.upper()
    if robot_type == "so100" and "SO101" in upper:
        return f"Robot type is so100 but env '{env_id}' appears to target SO101."
    if robot_type == "so101" and "SO100" in upper:
        return f"Robot type is so101 but env '{env_id}' appears to target SO100."
    return None


def import_backend_for_env_id(env_id: str) -> None:
    """Import the simulator backend that matches the *env_id* prefix."""
    if env_id.startswith("ManiSkill"):
        import so101_nexus_maniskill  # noqa: F401
    elif env_id.startswith("MuJoCo"):
        import so101_nexus_mujoco  # noqa: F401
    else:
        raise ValueError("env_id must start with 'ManiSkill' or 'MuJoCo'")


def convert_leader_action(
    action: dict,
    joint_names: tuple[str, ...],
    wrist_roll_offset_deg: float = DEFAULT_WRIST_ROLL_OFFSET_DEG,
) -> np.ndarray:
    """Convert leader arm joint readings (degrees) to radians."""
    converted: list[float] = []
    wrist_roll_offset_rad = np.deg2rad(wrist_roll_offset_deg)
    for name in joint_names:
        value = np.deg2rad(action[f"{name}.pos"])
        if name == "wrist_roll":
            value += wrist_roll_offset_rad
        converted.append(value)
    return np.array(converted, dtype=np.float64)


def compute_delta_actions(actions: list[np.ndarray]) -> list[np.ndarray]:
    """Convert absolute joint positions to frame-to-frame deltas."""
    deltas = [np.zeros_like(actions[0])]
    for i in range(1, len(actions)):
        deltas.append(actions[i] - actions[i - 1])
    return deltas


class TeeStream(io.TextIOBase):
    """Writable stream that duplicates writes to both *original* and an internal buffer."""

    def __init__(self, original: _WritableTextStream) -> None:
        self._original = original
        self._buffer = io.StringIO()
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self._original.write(s)
            self._buffer.write(s)
        return len(s)

    def flush(self) -> None:
        self._original.flush()

    def get_output(self) -> str:
        with self._lock:
            return self._buffer.getvalue()


class _WritableTextStream(Protocol):
    """Protocol for text-like streams used by TeeStream."""

    def write(self, s: str) -> object: ...

    def flush(self) -> object: ...


@dataclass
class RecordingState:
    """Shared mutable state between the recording thread and the Gradio UI."""

    is_recording: bool = False
    should_stop: bool = False
    countdown_value: int = 0
    recording_finished: bool = False

    episode_actions: list[np.ndarray] = field(default_factory=list)
    episode_states: list[np.ndarray] = field(default_factory=list)
    episode_images: list[np.ndarray] = field(default_factory=list)
    task_description: str = ""
    episode_duration: float = 0.0
    live_frame: np.ndarray | None = None

    episodes_completed: int = 0
    num_episodes: int = 0

    def clear_episode(self) -> None:
        """Reset all per-episode buffers."""
        self.episode_actions.clear()
        self.episode_states.clear()
        self.episode_images.clear()
        self.task_description = ""
        self.episode_duration = 0.0
        self.live_frame = None
        self.recording_finished = False


def recording_thread(
    state: RecordingState,
    env_id: str,
    leader,
    joint_names: tuple[str, ...],
    fps: int,
    max_steps: int,
    countdown: int,
    wrist_roll_offset_deg: float,
) -> None:
    """Background thread that runs the countdown then records frames at *fps*."""
    for i in range(countdown, 0, -1):
        state.countdown_value = i
        time.sleep(1.0)
    state.countdown_value = 0

    env = gym.make(env_id, camera_mode="wrist", render_mode="rgb_array")
    try:
        obs, _ = env.reset()
        state.task_description = getattr(env.unwrapped, "task_description", "")
        state.is_recording = True
        state.clear_episode()

        frame_duration = 1.0 / fps
        start_time = time.monotonic()

        while not state.should_stop:
            step_start = time.monotonic()
            if len(state.episode_actions) >= max_steps:
                break

            leader_action = leader.get_action()
            action = convert_leader_action(
                leader_action,
                joint_names,
                wrist_roll_offset_deg=wrist_roll_offset_deg,
            )
            obs, _, terminated, truncated, _ = env.step(action)

            if isinstance(obs, dict):
                obs_state = obs.get("state", obs.get("observation.state", action))
                wrist_image = obs.get("wrist_camera", obs.get("observation.images.wrist_cam"))
            else:
                obs_state = obs
                wrist_image = None

            state.episode_actions.append(action.astype(np.float32))
            state.episode_states.append(
                obs_state
                if isinstance(obs_state, np.ndarray)
                else np.array(obs_state, dtype=np.float32)
            )
            if wrist_image is not None:
                state.episode_images.append(wrist_image)
                state.live_frame = wrist_image.copy()

            if terminated or truncated:
                break

            sleep_time = frame_duration - (time.monotonic() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        state.episode_duration = time.monotonic() - start_time
    finally:
        env.close()
        state.is_recording = False
        state.should_stop = False
        state.recording_finished = True


def make_review_video(images: list[np.ndarray], fps: int) -> str | None:
    """Write *images* to a temporary MP4 file and return its path."""
    if not images:
        return None
    from so101_nexus_core.visualization import save_video

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
        path = temp_video.name
    save_video(images, path, fps=fps)
    return path


def make_state_plot(states: list[np.ndarray], joint_names: tuple[str, ...], fps: int) -> go.Figure:
    """Return a Plotly figure showing joint state trajectories over time."""
    arr = np.array(states)
    t = np.arange(arr.shape[0]) / fps
    fig = go.Figure()
    for j, name in enumerate(joint_names):
        fig.add_trace(go.Scatter(x=t, y=arr[:, j], mode="lines", name=name))
    fig.update_layout(
        title="Joint States Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Position (rad)",
        height=400,
    )
    return fig


def _default_repo_id(env_id: str) -> str:
    """Generate a local-only dataset repo ID from *env_id* and the current timestamp."""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = env_id.replace("/", "-").replace(" ", "_")
    return f"local/teleop-{safe}-{ts}"


def _noop(*_n: int):
    """Return *n* ``gr.update()`` sentinels (import ``gr`` lazily to avoid top-level dep)."""
    import gradio as gr

    return tuple(gr.update() for _ in range(_n[0] if _n else 1))


def _enter_to_start_script() -> str:
    """Return JS that maps Enter to Start Recording on the ready screen."""
    return """
<script>
(() => {
  if (window.__so_nexus_enter_listener_installed) return;
  window.__so_nexus_enter_listener_installed = true;
  document.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" || event.repeat) return;
    const active = document.activeElement;
    const tag = active?.tagName?.toLowerCase();
    if (tag === "input" || tag === "textarea" || active?.isContentEditable) return;

    const ready = document.getElementById("ready-screen");
    if (!ready || ready.offsetParent === null) return;

    const startButton = document.querySelector("#start-recording-btn button");
    if (!startButton) return;
    event.preventDefault();
    startButton.click();
  });
})();
</script>
"""


def main() -> None:
    """Launch the Gradio teleop recorder app."""
    import gradio as gr

    parser = argparse.ArgumentParser(description="Gradio-based teleop recorder")
    parser.add_argument("--leader-port", type=str, default="/dev/ttyACM0")
    parser.add_argument("--leader-id", type=str, default="so101_leader")
    parser.add_argument(
        "--wrist-roll-offset-deg",
        type=float,
        default=DEFAULT_WRIST_ROLL_OFFSET_DEG,
    )
    args = parser.parse_args()

    session: dict = {}
    init_state: dict = {
        "running": False,
        "done": False,
        "error": None,
        "tee_stdout": None,
        "tee_stderr": None,
        "warning": None,
    }

    def _init_worker(
        env_id,
        robot_type,
        leader_id,
        fps,
        camera_width,
        camera_height,
        repo_id,
        num_episodes,
        action_space,
        max_steps,
        countdown,
        wrist_roll_offset_deg,
    ):
        """Background worker that creates the env, connects the leader, and creates the dataset."""
        try:
            import_backend_for_env_id(env_id)
            joint_names = ROBOT_JOINT_NAMES[robot_type]

            print(f"Connecting leader arm on {args.leader_port} (id={leader_id})...")
            leader = get_leader(robot_type, args.leader_port, leader_id)
            try:
                leader.connect()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to connect on {args.leader_port}: {exc}\n"
                    "Is the arm plugged in? Run 'lerobot-find-port' to list ports."
                ) from exc

            print("Creating LeRobot dataset...")
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            try:
                dataset = LeRobotDataset.create(
                    repo_id=repo_id,
                    fps=fps,
                    robot_type=robot_type,
                    features={
                        "observation.state": {
                            "dtype": "float32",
                            "shape": (len(joint_names),),
                            "names": {"axes": list(joint_names)},
                        },
                        "observation.images.wrist_cam": {
                            "dtype": "video",
                            "shape": (camera_height, camera_width, 3),
                            "names": {"axes": ["height", "width", "channels"]},
                        },
                        "action": {
                            "dtype": "float32",
                            "shape": (len(joint_names),),
                            "names": {"axes": list(joint_names)},
                        },
                    },
                )
            except Exception as exc:
                leader.disconnect()
                raise RuntimeError(f"Failed to create dataset: {exc}") from exc

            session.update(
                leader=leader,
                dataset=dataset,
                state=RecordingState(num_episodes=num_episodes),
                joint_names=joint_names,
                fps=fps,
                action_space=action_space,
                max_steps=max_steps,
                countdown=countdown,
                camera_width=camera_width,
                camera_height=camera_height,
                env_id=env_id,
                robot_type=robot_type,
                wrist_roll_offset_deg=wrist_roll_offset_deg,
            )
            print("Initialization complete.")
            init_state["done"] = True

        except Exception as exc:
            init_state["error"] = str(exc)
            init_state["done"] = True

    def start_init(
        env_id,
        robot_type,
        leader_id,
        fps,
        camera_width,
        camera_height,
        repo_id,
        num_episodes,
        action_space,
        max_steps,
        countdown,
        wrist_roll_offset_deg,
    ):
        """Validate inputs and launch the init worker thread."""
        fps, camera_width, camera_height = int(fps), int(camera_width), int(camera_height)
        num_episodes, max_steps, countdown = int(num_episodes), int(max_steps), int(countdown)
        if max_steps < 1:
            raise ValueError("Max Steps must be at least 1.")
        wrist_roll_offset_deg = float(wrist_roll_offset_deg)
        repo_id_value = (repo_id or "").strip()
        if not repo_id_value:
            repo_id_value = _default_repo_id(env_id)
        leader_id_value = (leader_id or "").strip() or args.leader_id

        init_state.update(
            warning=check_robot_env_mismatch(env_id, robot_type),
            running=True,
            done=False,
            error=None,
        )

        tee_out = TeeStream(cast(_WritableTextStream, sys.stdout))
        tee_err = TeeStream(cast(_WritableTextStream, sys.stderr))
        init_state["tee_stdout"], init_state["tee_stderr"] = tee_out, tee_err
        sys.stdout, sys.stderr = tee_out, tee_err

        threading.Thread(
            target=_init_worker,
            args=(
                env_id,
                robot_type,
                leader_id_value,
                fps,
                camera_width,
                camera_height,
                repo_id_value,
                num_episodes,
                action_space,
                max_steps,
                countdown,
                wrist_roll_offset_deg,
            ),
            daemon=True,
        ).start()

        return (
            gr.update(visible=False),
            gr.update(visible=True),
        )

    def poll_init():
        """Poll the init thread and stream log output to the UI."""
        tee_out = init_state.get("tee_stdout")
        tee_err = init_state.get("tee_stderr")
        log = tee_out.get_output() if tee_out else ""
        if tee_err:
            err = tee_err.get_output()
            if err:
                log += "\n" + err

        if not init_state["done"]:
            return (gr.update(value=log or "Starting..."), *_noop(5))

        if tee_out:
            sys.stdout = tee_out._original
        if tee_err:
            sys.stderr = tee_err._original
        init_state["running"] = False

        if init_state["error"]:
            return (
                gr.update(value=log + f"\n\nERROR: {init_state['error']}"),
                gr.update(visible=False),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(visible=True),
            )

        if init_state["warning"]:
            gr.Warning(init_state["warning"])

        s = session
        header = (
            f"**Env:** `{s['env_id']}` | **Robot:** `{s['robot_type']}` | "
            f"**FPS:** {s['fps']} | **Max steps:** {s['max_steps']}"
        )
        return (
            gr.update(value=log),
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(value=header, visible=True),
            gr.update(value=f"Ready to record. 0/{s['state'].num_episodes} episodes complete."),
            gr.update(),
        )

    def start_recording():
        """Launch the recording thread and switch to the recording screen."""
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
            ),
            daemon=True,
        ).start()
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
        )

    def poll_recording():
        """Poll the recording thread for live frames and detect completion."""
        s = session.get("state")
        if s is None:
            return _noop(7)

        w, h, fps = session["camera_width"], session["camera_height"], session["fps"]

        if s.countdown_value > 0:
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.putText(
                img,
                str(s.countdown_value),
                (w // 2 - 40, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                4,
                (255, 255, 255),
                6,
            )
            return (
                gr.update(value=f"Starting in {s.countdown_value}..."),
                gr.update(value=img),
                *_noop(5),
            )

        if s.is_recording:
            frame = s.live_frame
            if frame is None:
                frame = np.zeros((h, w, 3), dtype=np.uint8)
            n = len(s.episode_actions)
            return (
                gr.update(value=f"Recording — {n} frames ({n / fps:.1f}s)"),
                gr.update(value=frame),
                *_noop(5),
            )

        if s.recording_finished:
            jn = session["joint_names"]
            video_path = make_review_video(s.episode_images, fps)
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
                gr.update(value=None),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value=video_path),
                gr.update(value=fig),
                gr.update(value=meta),
            )

        return _noop(7)

    def stop_recording():
        """Signal the recording thread to stop."""
        session["state"].should_stop = True

    def approve_episode():
        """Save the recorded episode to the dataset and advance."""
        s = session["state"]
        dataset = session["dataset"]
        actions = list(s.episode_actions)
        if session["action_space"] == "joint_pos_delta":
            actions = compute_delta_actions(actions)

        for i in range(len(actions)):
            frame = {
                "observation.state": s.episode_states[i].astype(np.float32),
                "action": actions[i].astype(np.float32),
            }
            if s.episode_images:
                frame["observation.images.wrist_cam"] = s.episode_images[i]
            dataset.add_frame(frame)

        dataset.save_episode(task=s.task_description)
        s.episodes_completed += 1
        s.clear_episode()

        if s.episodes_completed >= s.num_episodes:
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(),
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(value=f"Episode saved. {s.episodes_completed}/{s.num_episodes} complete."),
        )

    def discard_episode():
        """Discard the current episode buffer without saving."""
        s = session["state"]
        session["dataset"].clear_episode_buffer()
        s.clear_episode()
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(
                value=f"Episode discarded. {s.episodes_completed}/{s.num_episodes} complete."
            ),
        )

    def push_to_hub():
        """Push the completed dataset to HuggingFace Hub."""
        session["dataset"].push_to_hub()
        return "Dataset pushed to HuggingFace Hub!"

    def finalize_and_close():
        """Finalize the dataset and disconnect all hardware."""
        session["dataset"].finalize()
        session["leader"].disconnect()
        return "Dataset finalized. You can close this window."

    all_env_ids = all_registered_env_ids()

    with gr.Blocks(title="SO Nexus Teleop Recorder") as app:
        gr.Markdown("# SO Nexus Teleop Recorder")
        gr.HTML(_enter_to_start_script())
        session_header = gr.Markdown(visible=False)

        with gr.Group(visible=True) as setup_screen:
            gr.Markdown("## Session Configuration")
            env_id_input = gr.Dropdown(
                choices=all_env_ids,
                value=all_env_ids[0] if all_env_ids else None,
                label="Environment",
            )
            with gr.Row():
                robot_type_input = gr.Radio(
                    choices=["so100", "so101"],
                    value="so101",
                    label="Robot Type",
                )
                num_episodes_input = gr.Number(value=5, precision=0, label="Number of Episodes")

            with gr.Accordion("Advanced Settings", open=False):
                with gr.Row():
                    leader_id_input = gr.Textbox(label="Leader Arm ID", value=args.leader_id)
                    repo_id_input = gr.Textbox(
                        label="HuggingFace Repo ID (optional)",
                        placeholder="username/dataset-name",
                        info="Leave blank for local-only recording",
                    )
                with gr.Row():
                    fps_input = gr.Slider(minimum=1, maximum=60, value=30, step=1, label="FPS")
                    action_space_input = gr.Radio(
                        choices=["joint_pos", "joint_pos_delta"],
                        value="joint_pos_delta",
                        label="Action Space",
                    )
                with gr.Row():
                    wrist_roll_offset_deg_input = gr.Slider(
                        minimum=-180,
                        maximum=180,
                        value=args.wrist_roll_offset_deg,
                        step=1,
                        label="Wrist Roll Offset (deg)",
                    )
                with gr.Row():
                    camera_width_input = gr.Slider(
                        minimum=64,
                        maximum=1024,
                        value=480,
                        step=32,
                        label="Camera Width",
                    )
                    camera_height_input = gr.Slider(
                        minimum=64,
                        maximum=1024,
                        value=480,
                        step=32,
                        label="Camera Height",
                    )
                with gr.Row():
                    max_steps_input = gr.Number(
                        value=1024,
                        minimum=1,
                        precision=0,
                        label="Max Steps",
                    )
                    countdown_input = gr.Slider(
                        minimum=0,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Countdown (s)",
                    )

            init_btn = gr.Button("Initialize Session", variant="primary")

        with gr.Group(visible=False) as init_screen:
            gr.Markdown("## Initializing Session...")
            init_log = gr.Textbox(label="Log Output", lines=12, max_lines=20, interactive=False)
            init_timer = gr.Timer(value=0.25)

        with gr.Group(visible=False, elem_id="ready-screen") as ready_screen:
            ready_status = gr.Markdown("Ready to record.")
            start_btn = gr.Button(
                "Start Recording (or press Enter)",
                variant="primary",
                elem_id="start-recording-btn",
            )

        with gr.Group(visible=False) as recording_screen:
            recording_status = gr.Markdown("Starting...")
            live_feed = gr.Image(label="Live Camera Feed")
            stop_btn = gr.Button("Stop Recording", variant="stop")
            rec_timer = gr.Timer(value=0.1)

        with gr.Group(visible=False) as review_screen:
            gr.Markdown("## Review Episode")
            with gr.Row():
                with gr.Column():
                    review_video = gr.Video(label="Episode Video")
                with gr.Column():
                    state_plot = gr.Plot(label="Joint States")
            episode_metadata = gr.Markdown()
            with gr.Row():
                approve_btn = gr.Button("Approve", variant="primary")
                discard_btn = gr.Button("Discard", variant="stop")

        with gr.Group(visible=False) as done_screen:
            gr.Markdown("## All episodes recorded!")
            done_status = gr.Markdown("")
            with gr.Row():
                push_btn = gr.Button("Push to Hub", variant="primary")
                finalize_btn = gr.Button("Finalize & Close")

        init_btn.click(
            fn=start_init,
            inputs=[
                env_id_input,
                robot_type_input,
                leader_id_input,
                fps_input,
                camera_width_input,
                camera_height_input,
                repo_id_input,
                num_episodes_input,
                action_space_input,
                max_steps_input,
                countdown_input,
                wrist_roll_offset_deg_input,
            ],
            outputs=[setup_screen, init_screen],
        )
        init_timer.tick(
            fn=poll_init,
            outputs=[
                init_log,
                init_screen,
                ready_screen,
                session_header,
                ready_status,
                setup_screen,
            ],
        )
        start_btn.click(
            fn=start_recording,
            outputs=[ready_screen, recording_screen, review_screen, done_screen],
        )
        stop_btn.click(fn=stop_recording)
        rec_timer.tick(
            fn=poll_recording,
            outputs=[
                recording_status,
                live_feed,
                recording_screen,
                review_screen,
                review_video,
                state_plot,
                episode_metadata,
            ],
        )
        approve_btn.click(
            fn=approve_episode,
            outputs=[review_screen, ready_screen, done_screen, ready_status],
        )
        discard_btn.click(
            fn=discard_episode,
            outputs=[review_screen, ready_screen, done_screen, ready_status],
        )
        push_btn.click(fn=push_to_hub, outputs=[done_status])
        finalize_btn.click(fn=finalize_and_close, outputs=[done_status])

    app.launch(inbrowser=True)


if __name__ == "__main__":
    main()
