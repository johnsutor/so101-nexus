"""Pure Gradio layout helpers for the teleop recorder UI.

These functions only build view components and return them. No callbacks,
no state mutation, no business logic.
"""

from __future__ import annotations

from so101_nexus_core.teleop.dataset import OVERHEAD_KEY, WRIST_KEY

_OPTIONAL_FIELD_CHOICES = [WRIST_KEY, OVERHEAD_KEY, "task"]


def build_setup_screen(
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


def build_init_step(gr):
    """Build the Initialize step contents."""
    gr.Markdown("### Initializing Session...")
    init_log = gr.Textbox(label="Log Output", lines=12, max_lines=20, interactive=False)
    retry_btn = gr.Button("Retry Initialization", variant="secondary", visible=False)
    init_timer = gr.Timer(value=0.25)
    return init_log, retry_btn, init_timer


def build_record_step(gr):
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


def build_review_step(gr):
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


def build_complete_step(gr):
    """Build the Complete step contents."""
    gr.Markdown("### All episodes recorded!")
    done_info = gr.Markdown("")
    done_status = gr.Markdown("")
    with gr.Row():
        push_btn = gr.Button("Push to Hub", variant="primary")
        finalize_btn = gr.Button("Finalize & Close")
    return done_info, done_status, push_btn, finalize_btn
