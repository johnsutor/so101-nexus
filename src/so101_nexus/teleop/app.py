"""Vanilla-Gradio teleop recorder UI.

All heavy dependencies (gradio, lerobot, cv2) are imported lazily inside
:func:`main` or individual callbacks so that importing this module on a
base install does not fail.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from so101_nexus.constants import COLOR_MAP, CUBE_COLOR_MAP, ColorName
from so101_nexus.env_ids import Backend, env_ids_for_backend
from so101_nexus.objects import CubeObject, SceneObject, YCBObject
from so101_nexus.teleop.config_customization import (
    TeleopConfigOverrides,
    color_tuple_from_names,
    default_color_choices,
    default_cube_color_choices,
    default_object_choices,
    load_config_factory,
    overrides_to_mapping,
)
from so101_nexus.teleop.dataset import (
    OVERHEAD_KEY,
    WRIST_KEY,
    FieldSelection,
    build_features,
    build_frame,
)
from so101_nexus.teleop.leader import (
    DEFAULT_WRIST_ROLL_OFFSET_DEG,
    ROBOT_JOINT_NAMES,
    check_robot_env_mismatch,
    format_leader_connection_error,
    get_leader,
    import_backend_for_env_id,
)
from so101_nexus.teleop.recorder import (
    RecordingState,
    compute_delta_actions,
    recording_thread,
)
from so101_nexus.teleop.session import (
    _default_repo_id,
    _resolve_env_config,
    _resolve_env_ctor,
    make_review_video,
    make_state_plot,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

_OPTIONAL_FIELD_CHOICES = [WRIST_KEY, OVERHEAD_KEY]
_FOLLOWER_ROBOT_ID = "teleop_sim"
_TASK_PENDING_TEXT = "_Task will appear after reset._"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CustomizationUIState:
    """UI defaults and capability flags for environment customization controls."""

    customize_visible: bool = False
    customize_value: bool = False
    common_visible: bool = False
    pick_visible: bool = False
    pick_and_place_visible: bool = False
    object_specs: list[str] = field(default_factory=lambda: ["cube:red"])
    n_distractors: int = 0
    ground_colors: list[str] = field(default_factory=lambda: ["gray"])
    robot_colors: list[str] = field(default_factory=lambda: ["yellow"])
    spawn_min_radius: float = 0.10
    spawn_max_radius: float = 0.30
    spawn_angle_half_range_deg: float = 90.0
    reset_settle_frames: int = 5
    success_hold_seconds: float = 0.5
    cube_colors: list[str] = field(default_factory=lambda: ["red"])
    target_colors: list[str] = field(default_factory=lambda: ["blue"])


def _progress_text(completed: int, total: int) -> str:
    """Format a progress string for the episode counter."""
    return f"**Episode {completed} / {total}**"


def _format_hub_links(repo_id: str) -> str:
    """Return Markdown links for a pushed HuggingFace dataset."""
    from urllib.parse import quote

    dataset_url = f"https://huggingface.co/datasets/{quote(repo_id, safe='/')}"
    viewer_path = quote(f"/{repo_id}/episode_0", safe="")
    viewer_url = f"https://huggingface.co/spaces/lerobot/visualize_dataset?path={viewer_path}"
    return (
        "Pushed to HuggingFace Hub.\n\n"
        f"- [View the dataset page]({dataset_url})\n"
        f"- [Open in the dataset viewer]({viewer_url})"
    )


def _default_follower_calibration_dir() -> Path:
    """Return the default LeRobot calibration directory for the sim follower."""
    from lerobot.utils.constants import HF_LEROBOT_CALIBRATION, ROBOTS

    root = Path(os.environ.get("HF_LEROBOT_CALIBRATION", HF_LEROBOT_CALIBRATION))
    return root.expanduser() / ROBOTS / "sim_so_follower"


def _build_field_selection(field_selection_value: list[str]) -> FieldSelection:
    """Construct a :class:`FieldSelection` from the checkbox-group value list."""
    return FieldSelection(
        wrist_image=WRIST_KEY in field_selection_value,
        overhead_image=OVERHEAD_KEY in field_selection_value,
        task="task" in field_selection_value,
    )


def _default_env_id(all_env_ids: list[str], robot_type: str) -> str | None:
    """Return the first env ID matching *robot_type*, falling back to the first env."""
    robot_token = robot_type.upper()
    return next((env_id for env_id in all_env_ids if robot_token in env_id), None) or (
        all_env_ids[0] if all_env_ids else None
    )


def _import_env_modules(module_names: list[str]) -> None:
    """Import user-specified modules that register custom Gymnasium envs."""
    for module_name in module_names:
        importlib.import_module(module_name)


def _merge_extra_env_ids(base: list[str], extra: list[str]) -> list[str]:
    """Return base env ids plus unique extra ids, preserving order."""
    out = list(base)
    for env_id in extra:
        if env_id not in out:
            out.append(env_id)
    return out


def _color_config_to_names(value: object, default: list[str]) -> list[str]:
    """Return a UI-friendly list of color names from a config color value."""
    if value is None:
        return list(default)
    if isinstance(value, str):
        return [value]
    return [str(item) for item in cast("Iterable[object]", value)]


def _object_spec_from_scene_object(obj: SceneObject) -> str | None:
    """Return a compact UI object spec for built-in representable objects."""
    if isinstance(obj, CubeObject):
        return f"cube:{obj.color}"
    if isinstance(obj, YCBObject):
        return f"ycb:{obj.model_id}"
    return None


def _object_specs_from_config(value: object) -> list[str]:
    """Return object specs for config objects that can be represented in the UI."""
    if value is None:
        return ["cube:red"]
    objects = value if isinstance(value, list | tuple) else [value]
    return [
        spec
        for obj in objects
        if isinstance(obj, SceneObject)
        for spec in [_object_spec_from_scene_object(obj)]
        if spec is not None
    ]


def _customization_ui_state_from_config(config: object | None) -> CustomizationUIState:
    """Return capability flags and base defaults for the env customization UI."""
    if config is None:
        return CustomizationUIState()

    attrs = vars(config)
    common_keys = {
        "ground_colors",
        "robot_colors",
        "spawn_min_radius",
        "spawn_max_radius",
        "spawn_angle_half_range_deg",
        "reset_settle_frames",
    }
    common_visible = bool(common_keys & set(attrs))
    pick_visible = "objects" in attrs and "n_distractors" in attrs
    pick_and_place_visible = "cube_colors" in attrs or "target_colors" in attrs
    customize_visible = common_visible or pick_visible or pick_and_place_visible

    return CustomizationUIState(
        customize_visible=customize_visible,
        customize_value=customize_visible,
        common_visible=common_visible,
        pick_visible=pick_visible,
        pick_and_place_visible=pick_and_place_visible,
        object_specs=_object_specs_from_config(attrs.get("objects")),
        n_distractors=int(attrs.get("n_distractors", 0)),
        ground_colors=_color_config_to_names(attrs.get("ground_colors"), ["gray"]),
        robot_colors=_color_config_to_names(attrs.get("robot_colors"), ["yellow"]),
        spawn_min_radius=float(attrs.get("spawn_min_radius", 0.10)),
        spawn_max_radius=float(attrs.get("spawn_max_radius", 0.30)),
        spawn_angle_half_range_deg=float(attrs.get("spawn_angle_half_range_deg", 90.0)),
        reset_settle_frames=int(attrs.get("reset_settle_frames", 5)),
        success_hold_seconds=float(attrs.get("success_hold_seconds", 0.5)),
        cube_colors=_color_config_to_names(attrs.get("cube_colors"), ["red"]),
        target_colors=_color_config_to_names(attrs.get("target_colors"), ["blue"]),
    )


def _customization_ui_state_for_env(env_id: str | None) -> CustomizationUIState:
    """Resolve an env's base config and return UI customization defaults."""
    if not env_id:
        return CustomizationUIState()
    try:
        env_ctor, kwargs = _resolve_env_ctor(env_id)
        base_config = _resolve_env_config(env_ctor) if isinstance(env_ctor, type) else None
        if base_config is None:
            base_config = kwargs.get("config")
    except Exception as exc:
        logger.warning(
            "Failed to resolve customization defaults for env %s: %s",
            env_id,
            exc,
        )
        return CustomizationUIState()
    return _customization_ui_state_from_config(base_config)


def _optional_color_tuple(
    value: list[str],
    *,
    field_name: str,
    cube_only: bool = False,
) -> tuple[ColorName, ...] | None:
    """Return UI-selected colors as ColorName literals for typed overrides."""
    return color_tuple_from_names(
        value,
        field_name=field_name,
        valid_colors=CUBE_COLOR_MAP if cube_only else COLOR_MAP,
    )


def _append_init_log(init_state: dict, message: str) -> None:
    """Append a line of init log text captured only for this teleop session."""
    lines = init_state.setdefault("log_lines", [])
    lines.append(message)
    init_state["log_text"] = "\n".join(lines)


def _current_init_log(init_state: dict) -> str:
    """Return the accumulated init log text."""
    return init_state.get("log_text", "")


def _connect_leader(robot_type: str, leader_port: str, leader_id: str):
    """Connect and return the leader arm, or raise RuntimeError."""
    leader = get_leader(robot_type, leader_port, leader_id)
    try:
        leader.connect()
    except Exception as exc:
        raise RuntimeError(format_leader_connection_error(leader_port, exc)) from exc
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
    env_overrides: TeleopConfigOverrides | None,
    success_hold_seconds: float = 0.5,
) -> None:
    """Body of the background init worker."""
    joint_names = ROBOT_JOINT_NAMES[robot_type]
    try:
        _append_init_log(init_state, f"Connecting leader arm on {leader_port} (id={leader_id})...")
        import_backend_for_env_id(env_id)
        leader = _connect_leader(robot_type, leader_port, leader_id)
        _append_init_log(init_state, "Creating LeRobot dataset...")
        action_features = {f"{name}.pos": float for name in joint_names}
        follower_features = dict(action_features)
        follower_features["wrist"] = (wrist_wh[1], wrist_wh[0], 3)
        follower_features["overhead"] = (overhead_wh[1], overhead_wh[0], 3)
        features = build_features(field_selection, follower_features, action_features)
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
            env_overrides=env_overrides,
            success_hold_seconds=success_hold_seconds,
        )
        _append_init_log(init_state, "Initialization complete.")
        init_state["done"] = True
    except Exception as exc:
        init_state["error"] = str(exc)
        init_state["done"] = True


def _normalized_init_config(
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
    customize_env_config: bool,
    object_pool_value: list[str],
    n_distractors: float,
    ground_color_value: list[str],
    robot_color_value: list[str],
    spawn_min_radius: float,
    spawn_max_radius: float,
    spawn_angle_half_range_deg: float,
    reset_settle_frames: float,
    cube_color_value: list[str],
    target_color_value: list[str],
    *,
    success_hold_seconds: float = 0.5,
) -> dict:
    """Validate UI inputs and return a normalized init config dict."""
    fps_i = int(fps)
    wrist_wh = (int(wrist_camera_width), int(wrist_camera_height))
    overhead_wh = (int(overhead_camera_width), int(overhead_camera_height))
    num_ep_i, max_steps_i, countdown_i = int(num_episodes), int(max_steps), int(countdown)
    repo_id_value = (repo_id or "").strip() or _default_repo_id(env_id)
    leader_id_value = (leader_id or "").strip() or leader_id_default
    field_selection = _build_field_selection(field_selection_value)
    env_overrides = None
    if customize_env_config:
        reset_settle_frames_i = round(float(reset_settle_frames))
        env_overrides = TeleopConfigOverrides(
            object_specs=tuple(object_pool_value) or None,
            n_distractors=int(n_distractors),
            ground_colors=_optional_color_tuple(
                ground_color_value,
                field_name="ground_colors",
            ),
            robot_colors=_optional_color_tuple(
                robot_color_value,
                field_name="robot_colors",
            ),
            spawn_min_radius=float(spawn_min_radius),
            spawn_max_radius=float(spawn_max_radius),
            spawn_angle_half_range_deg=float(spawn_angle_half_range_deg),
            reset_settle_frames=reset_settle_frames_i,
            cube_colors=_optional_color_tuple(
                cube_color_value,
                field_name="cube_colors",
                cube_only=True,
            ),
            target_colors=_optional_color_tuple(
                target_color_value,
                field_name="target_colors",
                cube_only=True,
            ),
        )

    return {
        "env_id": env_id,
        "robot_type": robot_type,
        "leader_id": leader_id_value,
        "fps": fps_i,
        "wrist_wh": wrist_wh,
        "overhead_wh": overhead_wh,
        "repo_id": repo_id_value,
        "num_episodes": num_ep_i,
        "action_space": action_space,
        "max_steps": max_steps_i,
        "countdown": countdown_i,
        "wrist_roll_offset_deg": float(wrist_roll_offset_deg),
        "field_selection": field_selection,
        "env_overrides": env_overrides,
        "success_hold_seconds": float(success_hold_seconds),
    }


def _start_init_attempt(session: dict, init_state: dict, leader_port: str, config: dict):
    """Reset init state and launch a fresh initialization attempt."""
    import gradio as gr

    init_state.update(
        warning=check_robot_env_mismatch(config["env_id"], config["robot_type"]),
        running=True,
        done=False,
        processed=False,
        error=None,
        log_lines=[],
        log_text="",
        last_config=config,
    )

    threading.Thread(
        target=_run_init_worker,
        args=(
            session,
            init_state,
            leader_port,
            config["env_id"],
            config["robot_type"],
            config["leader_id"],
            config["fps"],
            config["wrist_wh"],
            config["overhead_wh"],
            config["repo_id"],
            config["num_episodes"],
            config["action_space"],
            config["max_steps"],
            config["countdown"],
            config["wrist_roll_offset_deg"],
            config["field_selection"],
            config["env_overrides"],
            config["success_hold_seconds"],
        ),
        daemon=True,
    ).start()

    return (
        gr.Walkthrough(selected=1),
        gr.update(value="Starting..."),
        gr.update(visible=False),
    )


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
    customize_env_config: bool,
    object_pool_value: list[str],
    n_distractors: float,
    ground_color_value: list[str],
    robot_color_value: list[str],
    spawn_min_radius: float,
    spawn_max_radius: float,
    spawn_angle_half_range_deg: float,
    reset_settle_frames: float,
    cube_color_value: list[str],
    target_color_value: list[str],
    success_hold_seconds: float = 0.5,
):
    """Validate inputs and launch the init worker thread."""
    import gradio as gr

    config = _normalized_init_config(
        leader_id_default,
        env_id,
        robot_type,
        leader_id,
        fps,
        wrist_camera_width,
        wrist_camera_height,
        overhead_camera_width,
        overhead_camera_height,
        repo_id,
        num_episodes,
        action_space,
        max_steps,
        countdown,
        wrist_roll_offset_deg,
        field_selection_value,
        customize_env_config,
        object_pool_value,
        n_distractors,
        ground_color_value,
        robot_color_value,
        spawn_min_radius,
        spawn_max_radius,
        spawn_angle_half_range_deg,
        reset_settle_frames,
        cube_color_value,
        target_color_value,
        success_hold_seconds=success_hold_seconds,
    )
    if config["max_steps"] < 1:
        raise gr.Error("Max Steps must be at least 1.")

    return _start_init_attempt(session, init_state, leader_port, config)


def _cb_update_customization_for_env(env_id: str):
    """Return Gradio updates for customization controls when the env changes."""
    import gradio as gr

    state = _customization_ui_state_for_env(env_id)
    return (
        gr.update(
            value=state.customize_value,
            visible=state.customize_visible,
            interactive=state.customize_visible,
        ),
        gr.update(value=state.object_specs),
        gr.update(value=state.n_distractors),
        gr.update(value=state.ground_colors),
        gr.update(value=state.robot_colors),
        gr.update(value=state.spawn_min_radius),
        gr.update(value=state.spawn_max_radius),
        gr.update(value=state.spawn_angle_half_range_deg),
        gr.update(value=state.reset_settle_frames),
        gr.update(value=state.cube_colors),
        gr.update(value=state.target_colors),
        gr.update(value=state.success_hold_seconds),
        gr.update(visible=state.common_visible),
        gr.update(visible=state.pick_visible),
        gr.update(visible=state.pick_and_place_visible),
    )


def _cb_validate_repo_id(repo_id: str):
    """Return a warning update for the repo ID input."""
    import gradio as gr

    from so101_nexus.teleop.session import RepoIdStatus, validate_hub_repo_id

    status = validate_hub_repo_id(repo_id)
    if status in (RepoIdStatus.OK, RepoIdStatus.LOCAL_ONLY):
        return gr.update(visible=False, value="")
    if status is RepoIdStatus.MISSING_NAMESPACE:
        msg = (
            "Use `username/dataset` format if you plan to push to the Hub. "
            "Leave blank for local-only recording."
        )
    else:
        msg = (
            "Invalid repo ID: must be alphanumeric plus `-`, `_`, or `.`, "
            "with no spaces and a maximum length of 96 characters."
        )
    return gr.update(visible=True, value=f"_{msg}_")


def _cb_poll_init(session: dict, init_state: dict):
    """Poll the init thread and stream log output to the UI."""
    import gradio as gr

    if init_state.get("processed"):
        return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    log = _current_init_log(init_state)

    if not init_state["done"]:
        return (
            gr.update(value=log or "Starting..."),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    init_state["running"] = False
    init_state["processed"] = True

    if init_state["error"]:
        return (
            gr.update(value=log + f"\n\nERROR: {init_state['error']}"),
            gr.update(visible=True),
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
        gr.update(),
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
        kwargs={
            "state": s,
            "env_id": session["env_id"],
            "leader": session["leader"],
            "joint_names": session["joint_names"],
            "fps": session["fps"],
            "max_steps": session["max_steps"],
            "countdown": session["countdown"],
            "wrist_roll_offset_deg": session["wrist_roll_offset_deg"],
            "wrist_wh": session["wrist_wh"],
            "overhead_wh": session["overhead_wh"],
            "follower_calibration_dir": _default_follower_calibration_dir(),
            "follower_robot_id": _FOLLOWER_ROBOT_ID,
            "customization_overrides": session.get("env_overrides"),
            "env_config_profile": session.get("env_config_profile"),
            "env_config_factory": session.get("env_config_factory"),
            "success_hold_seconds": session.get("success_hold_seconds", 0.5),
        },
        daemon=True,
    ).start()
    return (
        gr.update(value="Starting..."),
        gr.update(visible=False),
        gr.update(value=None, visible=False),
        gr.update(value=_TASK_PENDING_TEXT),
    )


def _task_status_text(task_description: str) -> str:
    """Return the record-step task Markdown for the current episode."""
    return f"**Task:** {task_description}" if task_description else _TASK_PENDING_TEXT


def _cb_poll_recording(session: dict):
    """Poll the recording thread for live frames and detect completion."""
    import gradio as gr

    s = session.get("state")
    if s is None:
        return tuple(gr.update() for _ in range(13))
    fps = session["fps"]
    if s.countdown_value > 0:
        return (
            gr.update(value=""),
            gr.update(visible=False),
            gr.update(visible=True, value=f"**{s.countdown_value}**\n\nGet ready..."),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=_TASK_PENDING_TEXT),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    if s.is_recording:
        preview = s.live_preview
        n = len(s.episode_actions)
        ep = s.episodes_completed + 1
        status = f"Recording episode {ep}/{s.num_episodes}: {n} frames ({n / fps:.1f}s)"
        if s.terminated_at_frame is not None:
            status = f"Success. {status} (finishing...)"
        if preview is None:
            return (
                gr.update(value=f"{status}\n\nWaiting for camera frame..."),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(value=_task_status_text(s.task_description)),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
            )
        return (
            gr.update(value=status),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(value=preview, visible=True),
            gr.update(visible=True),
            gr.update(value=_task_status_text(s.task_description)),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )
    if s.error:
        gr.Warning(f"Recording failed: {s.error}")
        s.error = None
    if s.recording_finished:
        return _recording_finished_updates(session, s, fps)
    return tuple(gr.update() for _ in range(13))


def _recording_finished_updates(session: dict, s: RecordingState, fps: int):
    """Return the review-screen update tuple after recording ends."""
    import gradio as gr

    jn = session["joint_names"]
    video_path = make_review_video(s.episode_wrist_images, fps)
    fig = make_state_plot(s.episode_states, jn, fps) if s.episode_states else None
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
        gr.update(value=None, visible=False),
        gr.update(visible=False),
        gr.update(value=_task_status_text(s.task_description)),
        gr.Walkthrough(selected=3),
        gr.update(value=video_path),
        gr.update(value=fig),
        gr.update(value=meta),
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


def _cb_prepare_episode_approval():
    """Show immediate UI feedback before saving/converting an approved episode."""
    import gradio as gr

    return (
        gr.update(value="Saving episode...", visible=True),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def _cb_approve_episode(session: dict):
    """Save the recorded episode to the dataset and advance."""
    import gradio as gr

    s = session["state"]
    dataset = session["dataset"]

    try:
        actions = list(s.episode_actions)
        if session["action_space"] == "joint_pos_delta":
            actions = compute_delta_actions(actions)

        sel = session["field_selection"]
        for i in range(len(actions)):
            wrist_img = s.episode_wrist_images[i] if i < len(s.episode_wrist_images) else None
            overhead_img = (
                s.episode_overhead_images[i] if i < len(s.episode_overhead_images) else None
            )
            frame = build_frame(
                sel,
                state=s.episode_states[i],
                action=actions[i],
                task=s.task_description,
                wrist_image=wrist_img,
                overhead_image=overhead_img,
            )
            dataset.add_frame(frame)

        dataset.save_episode()
    except Exception as exc:
        with contextlib.suppress(Exception):
            dataset.clear_episode_buffer()
        return (
            gr.Walkthrough(selected=3),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(value=f"Failed to save episode: {exc}", visible=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

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
            gr.update(visible=False),
            gr.update(value=None, visible=False),
            gr.update(value="", visible=False),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
    return (
        gr.Walkthrough(selected=2),
        gr.update(value="Episode saved! Ready to record the next one. Click the button below."),
        gr.update(value=progress),
        gr.update(),
        gr.update(visible=True),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
        gr.update(interactive=True),
        gr.update(interactive=True),
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
        gr.update(visible=True),
        gr.update(value=None, visible=False),
        gr.update(value="", visible=False),
    )


def _cb_retry_init(init_state: dict):
    """Reset init state and start a fresh initialization attempt."""
    import gradio as gr

    last_config = init_state.get("last_config")
    if last_config is None:
        init_state.update(running=False, done=False, processed=False, error=None)
        return (
            gr.Walkthrough(selected=0),
            gr.update(value=""),
            gr.update(visible=False),
        )
    raise RuntimeError("_cb_retry_init requires session and leader_port")


def _cb_retry_init_with_session(session: dict, init_state: dict, leader_port: str):
    """Retry initialization with the last submitted config."""
    last_config = init_state.get("last_config")
    if last_config is None:
        return _cb_retry_init(init_state)
    return _start_init_attempt(session, init_state, leader_port, last_config)


def _write_env_config_meta(session: dict) -> None:
    """Persist the env id and customization overrides as a reloadable profile.

    Written to the dataset ``meta/`` dir so a recording can be reproduced via
    ``--env-config-profile``. JSON, not TOML, because the stdlib cannot write
    TOML and both are accepted profile formats.
    """
    import json

    dataset = session["dataset"]
    env_id = session.get("env_id")
    overrides = session.get("env_overrides")
    if env_id is None:
        return
    section = overrides_to_mapping(overrides) if overrides is not None else {}
    profile = {"envs": {env_id: section}}
    meta_dir = Path(dataset.root) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "so101_nexus_env.json").write_text(json.dumps(profile, indent=2))


def _cb_push_to_hub(session: dict):
    """Push the completed dataset to HuggingFace Hub.

    Finalize first so LeRobot v3.0 per-episode metadata is flushed to disk
    before the Hub upload scans the dataset directory.
    """
    import gradio as gr

    from so101_nexus.teleop.session import RepoIdStatus, validate_hub_repo_id

    repo_id = str(getattr(session["dataset"], "repo_id", "")).strip()
    if repo_id.startswith("local/"):
        raise gr.Error(
            "Cannot push a local-only dataset repo ID. "
            "Start a new recording with a `username/dataset` Repo ID to push to the Hub."
        )
    status = validate_hub_repo_id(repo_id)
    if status is not RepoIdStatus.OK:
        raise gr.Error(
            f"Cannot push: repo ID must be `username/dataset`. Current value: {repo_id!r}."
        )

    try:
        session["dataset"].finalize()
        _write_env_config_meta(session)
        session["dataset"].push_to_hub()
    except Exception as exc:
        msg = str(exc).strip() or type(exc).__name__
        raise gr.Error(
            f"Failed to push to Hub: {msg}\n\n"
            "Make sure you are logged in (`huggingface-cli login`) and have "
            "write access to the target repository."
        ) from exc
    return _format_hub_links(repo_id)


def _cb_prepare_push_to_hub():
    """Show immediate UI feedback before the blocking Hub push starts."""
    import gradio as gr

    return gr.update(value="Pushing dataset to HuggingFace Hub...", visible=True)


def _cb_finalize_and_close(session: dict):
    """Finalize the dataset and disconnect all hardware."""
    import gradio as gr

    try:
        session["dataset"].finalize()
        _write_env_config_meta(session)
    except Exception as exc:
        raise gr.Error(f"Failed to finalize dataset: {exc}") from exc
    with contextlib.suppress(Exception):
        session["leader"].disconnect()
    return "Session finalized. You can close this tab."


def _cb_prepare_finalize_and_close():
    """Show immediate UI feedback before dataset finalization starts."""
    import gradio as gr

    return gr.update(value="Finalizing dataset...", visible=True)


def _build_setup_screen(
    gr,
    all_env_ids: list[str],
    default_leader_id: str,
    wrist_roll_offset: float,
):
    """Build the Configure step contents and return all input components."""
    gr.Markdown("### Environment & Robot")
    default_robot_type = "so101"
    default_env = _default_env_id(all_env_ids, default_robot_type)
    customization_state = _customization_ui_state_for_env(default_env)
    env_id_input = gr.Dropdown(
        choices=all_env_ids,
        value=default_env,
        label="Environment",
    )
    with gr.Row():
        robot_type_input = gr.Radio(
            choices=["so100", "so101"], value=default_robot_type, label="Robot Type"
        )
        action_space_input = gr.Radio(
            choices=["joint_pos", "joint_pos_delta"],
            value="joint_pos",
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
    repo_id_warning = gr.Markdown("", visible=False)
    field_selection_input = gr.CheckboxGroup(
        choices=_OPTIONAL_FIELD_CHOICES,
        value=list(_OPTIONAL_FIELD_CHOICES),
        label="Dataset fields",
        info="observation.state, action, and task are always saved",
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
                minimum=64, maximum=1024, value=640, step=32, label="Wrist Camera Width"
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
        success_hold_seconds_input = gr.Slider(
            minimum=0.0,
            maximum=2.0,
            value=customization_state.success_hold_seconds,
            step=0.1,
            label="Success Hold (s)",
            info=(
                "After the env reports success, keep recording this many seconds "
                "before auto-stopping. Set 0 to stop immediately."
            ),
        )
        customize_env_input = gr.Checkbox(
            value=customization_state.customize_value,
            label="Apply Environment Customization",
            visible=customization_state.customize_visible,
            interactive=customization_state.customize_visible,
            info=(
                "When off, the environment's default config is used and the "
                "fields below are ignored."
            ),
        )
        with gr.Group(visible=customization_state.pick_visible) as pick_customization_group:
            object_pool_input = gr.CheckboxGroup(
                choices=default_object_choices(),
                value=customization_state.object_specs,
                label="Pick Object Pool",
            )
            n_distractors_input = gr.Number(
                value=customization_state.n_distractors,
                minimum=0,
                precision=0,
                label="Pick Distractors",
            )
        with gr.Group(visible=customization_state.common_visible) as common_customization_group:
            with gr.Row():
                ground_colors_input = gr.CheckboxGroup(
                    choices=default_color_choices(),
                    value=customization_state.ground_colors,
                    label="Ground Colors",
                )
                robot_colors_input = gr.CheckboxGroup(
                    choices=default_color_choices(),
                    value=customization_state.robot_colors,
                    label="Robot Colors",
                )
            with gr.Row():
                spawn_min_radius_input = gr.Slider(
                    minimum=0.0,
                    maximum=0.5,
                    value=customization_state.spawn_min_radius,
                    step=0.01,
                    label="Spawn Min Radius",
                )
                spawn_max_radius_input = gr.Slider(
                    minimum=0.01,
                    maximum=0.8,
                    value=customization_state.spawn_max_radius,
                    step=0.01,
                    label="Spawn Max Radius",
                )
                spawn_angle_half_range_input = gr.Slider(
                    minimum=0,
                    maximum=180,
                    value=customization_state.spawn_angle_half_range_deg,
                    step=1,
                    label="Spawn Angle Half Range (deg)",
                )
                reset_settle_frames_input = gr.Slider(
                    minimum=0,
                    maximum=60,
                    value=customization_state.reset_settle_frames,
                    step=1,
                    label="Reset Settle Frames",
                )
        with (
            gr.Group(
                visible=customization_state.pick_and_place_visible
            ) as pick_and_place_customization_group,
            gr.Row(),
        ):
            cube_colors_input = gr.CheckboxGroup(
                choices=default_cube_color_choices(),
                value=customization_state.cube_colors,
                label="Pick-and-Place Cube Colors",
            )
            target_colors_input = gr.CheckboxGroup(
                choices=default_cube_color_choices(),
                value=customization_state.target_colors,
                label="Pick-and-Place Target Colors",
            )

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
        customize_env_input,
        object_pool_input,
        n_distractors_input,
        ground_colors_input,
        robot_colors_input,
        spawn_min_radius_input,
        spawn_max_radius_input,
        spawn_angle_half_range_input,
        reset_settle_frames_input,
        cube_colors_input,
        target_colors_input,
        success_hold_seconds_input,
        repo_id_warning,
        common_customization_group,
        pick_customization_group,
        pick_and_place_customization_group,
    )


def _build_init_step(gr):
    """Build the Initialize step contents."""
    gr.Markdown("### Initializing Session...")
    init_log = gr.Textbox(label="Log Output", lines=12, max_lines=20, interactive=False)
    retry_btn = gr.Button("Retry Initialization", variant="secondary", visible=False)
    init_timer = gr.Timer(value=0.25)
    return init_log, retry_btn, init_timer


def _build_record_step(gr):
    """Build the Record step contents."""
    record_status = gr.Markdown("Ready to record. Click the button below to begin.")
    start_btn = gr.Button("Start Recording", variant="primary")
    countdown_area = gr.Markdown("", visible=False)
    preview_feed = gr.Image(
        label="Live Preview (wrist | overhead)",
        height=320,
        visible=False,
        show_label=True,
        interactive=False,
    )
    stop_btn = gr.Button("Stop Recording", variant="stop", visible=False)
    task_status = gr.Markdown(_TASK_PENDING_TEXT)
    rec_timer = gr.Timer(value=0.1)
    return record_status, start_btn, countdown_area, preview_feed, stop_btn, task_status, rec_timer


def _build_review_step(gr):
    """Build the Review step contents."""
    gr.Markdown("### Review Episode")
    with gr.Row():
        with gr.Column(scale=2):
            review_video = gr.Video(label="Episode Video")
        with gr.Column(scale=1):
            state_plot = gr.Plot(label="Joint States")
            episode_metadata = gr.Markdown()
    review_status = gr.Markdown("", visible=False)
    with gr.Row():
        approve_btn = gr.Button("Approve", variant="primary")
        discard_btn = gr.Button("Discard", variant="stop")
    return review_video, state_plot, episode_metadata, review_status, approve_btn, discard_btn


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
    env_id_input,
    repo_id_input,
    repo_id_warning,
    customization_outputs,
    init_log,
    retry_btn,
    init_timer,
    record_status,
    start_btn,
    countdown_area,
    preview_feed,
    stop_btn,
    task_status,
    rec_timer,
    review_video,
    state_plot,
    episode_metadata,
    review_status,
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

    retry_init = functools.partial(_cb_retry_init_with_session, session, init_state, leader_port)

    env_id_input.change(
        fn=_cb_update_customization_for_env,
        inputs=[env_id_input],
        outputs=customization_outputs,
    )
    repo_id_input.change(
        fn=_cb_validate_repo_id,
        inputs=[repo_id_input],
        outputs=[repo_id_warning],
    )
    init_btn.click(fn=start_init, inputs=init_inputs, outputs=[walkthrough, init_log, retry_btn])
    init_timer.tick(
        fn=poll_init,
        outputs=[init_log, retry_btn, walkthrough, session_header, progress_status],
    )
    retry_btn.click(
        fn=retry_init,
        outputs=[walkthrough, init_log, retry_btn],
    )
    start_btn.click(
        fn=start_recording,
        outputs=[record_status, start_btn, preview_feed, task_status],
    )
    stop_btn.click(fn=stop_recording)
    rec_timer.tick(
        fn=poll_recording,
        outputs=[
            record_status,
            start_btn,
            countdown_area,
            preview_feed,
            stop_btn,
            task_status,
            walkthrough,
            review_video,
            state_plot,
            episode_metadata,
            review_status,
            approve_btn,
            discard_btn,
        ],
    )
    approve_btn.click(
        fn=_cb_prepare_episode_approval,
        outputs=[review_status, approve_btn, discard_btn],
    ).then(
        fn=approve_episode,
        outputs=[
            walkthrough,
            record_status,
            progress_status,
            done_info,
            start_btn,
            preview_feed,
            review_status,
            approve_btn,
            discard_btn,
        ],
    )
    discard_btn.click(
        fn=discard_episode,
        outputs=[
            walkthrough,
            record_status,
            progress_status,
            start_btn,
            preview_feed,
            review_status,
        ],
    )
    push_btn.click(fn=_cb_prepare_push_to_hub, outputs=[done_status]).then(
        fn=push_to_hub,
        outputs=[done_status],
    )
    finalize_btn.click(fn=_cb_prepare_finalize_and_close, outputs=[done_status]).then(
        fn=finalize_and_close,
        outputs=[done_status],
    )


_TELEOP_INSTALL_HINT = (
    "Teleop dependencies are not installed. Install them with one of:\n"
    "  uv sync --extra teleop\n"
    "  pip install 'so101-nexus[teleop]'"
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
        Restrict the env dropdown to a single backend (``"mujoco"``). When
        ``None``, all registered envs are shown.
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
        parser.add_argument("--env-config-profile", type=str, default=None)
        parser.add_argument("--env-config-factory", type=str, default=None)
        parser.add_argument("--env-module", action="append", default=[], dest="env_modules")
        parser.add_argument("--extra-env-id", action="append", default=[], dest="extra_env_ids")
        args = parser.parse_args()

    leader_port: str = getattr(args, "leader_port", "/dev/ttyACM0")
    leader_id_default: str = getattr(args, "leader_id", "so101_leader")
    wrist_roll_offset: float = getattr(args, "wrist_roll_offset_deg", DEFAULT_WRIST_ROLL_OFFSET_DEG)
    env_config_profile: str | None = getattr(args, "env_config_profile", None)
    env_config_factory = load_config_factory(getattr(args, "env_config_factory", None))
    env_modules: list[str] = list(getattr(args, "env_modules", []) or [])
    extra_env_ids: list[str] = list(getattr(args, "extra_env_ids", []) or [])
    _import_env_modules(env_modules)

    session: dict = {
        "env_config_profile": env_config_profile,
        "env_config_factory": env_config_factory,
    }
    init_state: dict = {
        "running": False,
        "done": False,
        "processed": False,
        "error": None,
        "warning": None,
        "log_lines": [],
        "log_text": "",
        "last_config": None,
    }

    all_env_ids = _merge_extra_env_ids(env_ids_for_backend(backend), extra_env_ids)

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
                    customize_env_input,
                    object_pool_input,
                    n_distractors_input,
                    ground_colors_input,
                    robot_colors_input,
                    spawn_min_radius_input,
                    spawn_max_radius_input,
                    spawn_angle_half_range_input,
                    reset_settle_frames_input,
                    cube_colors_input,
                    target_colors_input,
                    success_hold_seconds_input,
                    repo_id_warning,
                    common_customization_group,
                    pick_customization_group,
                    pick_and_place_customization_group,
                ) = _build_setup_screen(gr, all_env_ids, leader_id_default, wrist_roll_offset)

            with gr.Step("Initialize", id=1):
                init_log, retry_btn, init_timer = _build_init_step(gr)

            with gr.Step("Record", id=2):
                (
                    record_status,
                    start_btn,
                    countdown_area,
                    preview_feed,
                    stop_btn,
                    task_status,
                    rec_timer,
                ) = _build_record_step(gr)

            with gr.Step("Review", id=3):
                (
                    review_video,
                    state_plot,
                    episode_metadata,
                    review_status,
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
            customize_env_input,
            object_pool_input,
            n_distractors_input,
            ground_colors_input,
            robot_colors_input,
            spawn_min_radius_input,
            spawn_max_radius_input,
            spawn_angle_half_range_input,
            reset_settle_frames_input,
            cube_colors_input,
            target_colors_input,
            success_hold_seconds_input,
        ]

        _wire_events(
            gr,
            walkthrough=walkthrough,
            init_btn=init_btn,
            init_inputs=init_inputs,
            env_id_input=env_id_input,
            repo_id_input=repo_id_input,
            repo_id_warning=repo_id_warning,
            customization_outputs=[
                customize_env_input,
                object_pool_input,
                n_distractors_input,
                ground_colors_input,
                robot_colors_input,
                spawn_min_radius_input,
                spawn_max_radius_input,
                spawn_angle_half_range_input,
                reset_settle_frames_input,
                cube_colors_input,
                target_colors_input,
                success_hold_seconds_input,
                common_customization_group,
                pick_customization_group,
                pick_and_place_customization_group,
            ],
            init_log=init_log,
            retry_btn=retry_btn,
            init_timer=init_timer,
            record_status=record_status,
            start_btn=start_btn,
            countdown_area=countdown_area,
            preview_feed=preview_feed,
            stop_btn=stop_btn,
            task_status=task_status,
            rec_timer=rec_timer,
            review_video=review_video,
            state_plot=state_plot,
            episode_metadata=episode_metadata,
            review_status=review_status,
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
