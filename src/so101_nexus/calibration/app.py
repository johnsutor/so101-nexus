"""A profile-first Gradio walkthrough for physical SO-101 alignment."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from so101_nexus.calibration.guide import CalibrationGuide
from so101_nexus.calibration.session import CalibrationSession
from so101_nexus.calibration.view import expand_bounds, overlay_bounds, render_overlay
from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.excitation import ExcitationConfig, generate_excitation

if TYPE_CHECKING:
    from so101_nexus.profiles import RobotProfile

_DEFAULT_CENTER = (0, -15, 30, 45, 0, 50)
_JOINT_LABELS = ("Base direction", "Upper arm tilt", "Elbow bend", "Wrist bend", "Jaw twist")
_GUIDANCE = (
    "Turn the shoulder around the vertical base axis. Match its facing direction to the "
    "cyan reference. The other joints do not need to match yet.",
    "Support the upper arm. Match its inclination to the cyan upper arm, using the "
    "highlighted hinge. Ignore the elbow and jaws for now.",
    "Support the forearm. Match the bend between the upper arm and forearm to the cyan "
    "reference. Leave the wrist for the next steps.",
    "This hinge bends the hand up and down. In the close-up, match the fixed jaw's tilt "
    "relative to the forearm. A twist around the forearm belongs to the next joint.",
    "This joint twists the jaws around the forearm. Match the fixed jaw's orientation to "
    "the cyan reference. Use Front, Side, or Top if the jaw faces are hard to distinguish.",
)
_MEASURE_VIEW = """(...values) => {
    const container = document.querySelector('#calibration-robot-view');
    if (container && !container.nexusCamera) {
        const state = {plot: null, camera: null, revision: null, restoring: false};
        container.nexusCamera = state;
        const remember = (plot) => {
            if (plot !== state.plot || state.restoring) return;
            const scene = plot?._fullLayout?.scene?._scene;
            if (scene) state.camera = structuredClone(scene.getCamera());
        };
        const scan = () => {
            const plot = container.querySelector('.js-plotly-plot');
            if (!plot?._fullLayout?.scene?._scene) return;
            if (plot === state.plot) { remember(plot); return; }
            const revision = plot.layout.uirevision;
            if (revision !== state.revision) state.camera = null;
            state.plot = plot;
            state.revision = revision;
            plot.on('plotly_relayout', () => remember(plot));
            plot.on('plotly_relayouting', () => remember(plot));
            if (state.camera && window.Plotly) {
                state.restoring = true;
                Promise.resolve(window.Plotly.relayout(plot, {'scene.camera': state.camera}))
                    .finally(() => { state.restoring = false; remember(plot); });
            } else { remember(plot); }
        };
        const observer = new MutationObserver(() => requestAnimationFrame(scan));
        observer.observe(container, {subtree: true, childList: true});
        const timer = setInterval(() => {
            if (container.isConnected) scan();
            else { clearInterval(timer); observer.disconnect(); }
        }, 50);
        scan();
    }
    const plot = container?.querySelector('.js-plotly-plot');
    const width = plot?.clientWidth || container?.clientWidth || 975;
    return [...values.slice(0, -1), width / 650];
}"""


@dataclass
class _Viewport:
    bounds: Any = None
    key: Any = None
    revision: int = 0
    tuning: bool = False
    tuning_step: int = 0
    previous_artifacts: list[str] = field(default_factory=list)

    def fit(self) -> None:
        self.bounds = None
        self.revision += 1


def make_motion_config(
    profile: RobotProfile,
    center_deg: list[float],
    amplitudes_deg: list[float],
    duration_s: float,
    table_z_m: float,
    seed: int,
) -> ExcitationConfig:
    """Normalize episode duration to the profile's whole command periods."""
    duration_s = round(duration_s / profile.control_period_s) * profile.control_period_s
    return ExcitationConfig(
        center_qpos_deg=tuple(center_deg),
        amplitudes_deg=tuple(amplitudes_deg),
        duration_s=duration_s,
        ramp_s=min(2.0, duration_s / 5),
        table_z_m=table_z_m,
        seed=int(seed),
    )


def _call(gr, session, function, *args, **kwargs):
    try:
        return function(*args, **kwargs)
    except Exception as error:
        session.status = str(error)
        gr.Warning(str(error))
        return None


def _plan(session, *values):
    if session.profile is None:
        raise ValueError("Connect a follower or load its matched profile before previewing motion.")
    try:
        if len(values) == 16:
            session.set_period(values[15])
        center, amplitudes = list(values[:6]), list(values[6:12])
        session.set_plan(make_motion_config(session.profile, center, amplitudes, *values[12:15]))
    except Exception:
        session.verified = False
        if session.worker is not None and session.worker.is_alive():
            session.stop()
        raise


def _current_center(gr, session):
    def read_center():
        with session.lock:
            center = session.use_current_center()
            session.verified = False
            session.status = "Live center selected. Preview and run simulation preflight again."
            return list(center)

    result = _call(gr, session, read_center)
    return result if result is not None else [gr.skip()] * 6


def _show_plan(gr, session, *values):
    import plotly.graph_objects as go

    try:
        _plan(session, *values)
        assert session.config is not None
        assert session.profile is not None
        trajectories = generate_excitation(session.config, session.profile)
    except Exception as error:
        session.status = str(error)
        gr.Warning(str(error))
        return gr.skip()
    figure = go.Figure()
    for trajectory in trajectories:
        for index, name in enumerate(SO101_JOINT_NAMES):
            figure.add_scatter(
                x=trajectory.timestamps_s,
                y=trajectory.commands_deg[:, index],
                name=f"{trajectory.name}: {name}",
                visible=True if trajectory.name == "multisine" else "legendonly",
            )
    figure.update_layout(
        xaxis_title="Time (seconds)", yaxis_title="Model joint angle (deg)", height=350
    )
    return figure


def _fresh_recording_path(requested):
    path = Path(requested).expanduser().resolve()
    if not path.exists():
        return path
    match = re.fullmatch(r"(.+)_([0-9]{3})", path.name)
    name, index = (match[1], int(match[2]) + 1) if match else (path.name, 2)
    candidate = path.with_name(f"{name}_{index:03d}")
    while candidate.exists():
        index += 1
        candidate = path.with_name(f"{name}_{index:03d}")
    return candidate


def _check_motion_pose(session, values, *, manual_center):
    settings = list(values)
    if not manual_center:
        settings[:6] = session.use_current_center()
    _plan(session, *settings)
    session.preflight()
    return list(session.config.center_qpos_deg)


def _ensure_saved_profile(session):
    session._aligned()
    if session.profile_path is None:
        session.save_profile(_suggest_path(session))
    return session.profile_path


def _prepare_collection(session, path, supported, values):
    with session.lock:
        session._idle()
        _plan(session, *values)
        if not session.verified:
            raise ValueError("Check the current pose and motion before enabling hold.")
        _ensure_saved_profile(session)
        destination = _fresh_recording_path(path)
        session.arm(destination, supported=supported)
        return str(destination)


def _dynamics_readiness(session):
    active_messages = {
        "arming": "Preparing holding. Keep supporting the arm until Holding appears.",
        "holding": (
            "Holding is ready. Remove your hands, confirm they are clear, then start excitation."
        ),
        "running": "Recording motion. Stop commands ends this attempt and preserves torque.",
    }
    if session.phase in active_messages:
        return active_messages[session.phase]
    if session.monitor is None:
        return "Live readings are closed. Reconnect to prepare another attempt."
    if any(session.monitor.torque.values()):
        return "Motors are holding. Support the arm and release motors before a new attempt."
    if not session.verified:
        return "Read the current pose and check motion before enabling hold."
    return (
        "Motion check passed. Keep the arm at the checked pose, confirm support, and enable hold."
    )


def _start_motion(gr, session, hands_clear, *values):
    with session.lock:
        try:
            if session.profile is None or len(values) != 16:
                raise ValueError("A complete, aligned motion plan is required.")
            visible_profile = replace(session.profile, control_period_s=values[15])
            visible_config = make_motion_config(
                visible_profile, list(values[:6]), list(values[6:12]), *values[12:15]
            )
            if (
                visible_profile.control_period_s != session.profile.control_period_s
                or visible_config != session.config
            ):
                raise ValueError("Visible motion settings differ from the armed plan.")
        except Exception as error:
            session.stop()
            session.status = (
                f"Motion start cancelled: {error} "
                "Reconnect read-only, preview and run preflight again before enabling hold."
            )
            gr.Warning(session.status)
            return
        _call(gr, session, session.start_motion, hands_clear=hands_clear)


def _fit_candidate(gr, session, *values):
    recordings, output, max_iters = values
    result = _call(
        gr, session, session.fit, recordings, str(_fresh_recording_path(output)), max_iters
    )
    return result if result is not None else ([], [], session.status)


def _connect_controls(gr, ui):
    gr.Markdown(
        "### Connect your follower\n\nConnect the USB cable, then open the live view. "
        "Connecting does not enable torque or move the robot."
    )
    ui["port"] = gr.Dropdown(
        choices=["/dev/ttyACM0"],
        value="/dev/ttyACM0",
        allow_custom_value=True,
        label="Follower serial port",
    )
    ui["connect"] = gr.Button("Connect robot", variant="primary")
    with gr.Accordion("Different device or existing profile", open=False):
        from so101_nexus.calibration.hardware import default_calibration_dir

        ui["discover"] = gr.Button("Find serial devices")
        ui["calibration_dir"] = gr.Textbox(
            str(default_calibration_dir()), label="LeRobot follower calibration directory"
        )
        ui["saved_profile"] = gr.Textbox(label="Existing confirmed profile JSON")
        ui["load"] = gr.Button("Load matched profile")
        gr.Markdown(
            "Connect first. Loading a profile trusts its previous physical alignment and "
            "requires matching motor calibration."
        )


def _alignment_controls(gr, ui):
    ui["joint_title"] = gr.Markdown("### Joint 1 of 5: Base direction")
    ui["instruction"] = gr.Markdown(_GUIDANCE[0])
    gr.Markdown(
        "Support the arm with torque off. Move only the highlighted joint by hand until it "
        "matches the cyan reference. The amber pose is provisional until you confirm that "
        "physical reference."
    )
    with gr.Group(visible=False, elem_id="wrist-correction") as ui["rotation_group"]:
        gr.Markdown(
            "**Wrist looks a quarter-turn off?** Preview a correction to the amber display."
            " These buttons do not move the robot or confirm alignment."
        )
        with gr.Row():
            ui["rotate_minus"] = gr.Button("Preview -90 deg")
            ui["rotate_plus"] = gr.Button("Preview +90 deg")
        gr.Markdown(
            "After the preview, physically match this joint to the cyan reference before "
            "capturing it."
        )
    ui["capture_notice"] = gr.Markdown("", visible=False)
    ui["capture"] = gr.Button("Matches, capture and continue", variant="primary")
    ui["previous"] = gr.Button("Previous joint", interactive=False)
    with gr.Accordion("Use a different known reference angle", open=False):
        ui["reference_angle"] = gr.Number(
            0,
            label="Reference angle (model deg)",
            info="Use a known physical angle. This changes the cyan reference only.",
        )


def _gripper_controls(gr, ui):
    gr.Markdown(
        "### Check the jaw endpoints\n\nOpen and close the jaws by hand while watching the "
        "live view. Compare the closed and open physical endpoints with the cyan reference."
        " This is separate from wrist alignment."
    )
    with gr.Row():
        ui["closed"] = gr.Button("Show closed reference")
        ui["open"] = gr.Button("Show open reference")
    ui["grip_confirm"] = gr.Checkbox(
        label="I checked both physical jaw endpoints against the reference"
    )
    ui["map_gripper"] = gr.Button("Jaw endpoints match, continue", variant="primary")
    ui["back_body"] = gr.Button("Back to wrist alignment")
    with gr.Accordion("Adjust model jaw geometry", open=False):
        ui["grip_low"] = gr.Number(-10, label="Model angle at hardware 0 percent (deg)")
        ui["grip_high"] = gr.Number(100, label="Model angle at hardware 100 percent (deg)")
        gr.Markdown(
            "The defaults use nominal SO-101 geometry. Confirm the physical endpoints or "
            "enter independently measured angles."
        )


def _profile_controls(gr, ui):
    ui["profile_message"] = gr.Markdown(
        "### Save your aligned profile\n\nYour five body joints and jaw mapping are "
        "confirmed. Save them together so simulation can use this robot's joint "
        "coordinates."
    )
    ui["save"] = gr.Button("Save profile", variant="primary")
    ui["edit_profile"] = gr.Button("Recheck alignment")
    with gr.Accordion("Save location and control timing", open=False):
        ui["profile_output"] = gr.Textbox(
            "runs/calibration/robot_profile.json", label="New profile JSON path"
        )
        ui["period"] = gr.Number(0.02, label="Command period (seconds)")
    ui["tune"] = gr.Button("Dynamics tuning", visible=False)
    gr.Markdown(
        "An aligned profile establishes coordinates. Dynamics tuning improves how "
        "the simulator predicts motor response; neither step proves policy transfer "
        "performance."
    )


def _motion_controls(gr, ui):
    gr.Markdown(
        "### Preview a small motion\n\nPlace and support the arm at a raised pose with room "
        "to move. Read that pose as the center, then preview and check all three episodes "
        "in simulation."
    )
    with gr.Accordion("Motion settings and workspace", open=False):
        ui["manual_center"] = gr.Checkbox(
            value=False, label="Use a manually entered center instead of the live pose"
        )
        ui["current"] = gr.Button("Read live center")
        ui["centers"] = [
            gr.Number(value, label=f"Center {name.replace('_', ' ')} (model deg)")
            for name, value in zip(SO101_JOINT_NAMES, _DEFAULT_CENTER, strict=True)
        ]
        ui["amplitudes"] = [
            gr.Number(value, label=f"Amplitude {name.replace('_', ' ')} (model deg)")
            for name, value in zip(SO101_JOINT_NAMES, (8, 8, 8, 8, 10, 15), strict=True)
        ]
        ui["duration"] = gr.Number(20, label="Duration per episode (seconds)")
        ui["table"] = gr.Number(0, label="Table height relative to base (meters)")
        ui["seed"] = gr.Number(0, precision=0, label="Trajectory seed")
    ui["preview"] = gr.Button("Read current pose and check motion", variant="primary")
    ui["trajectory_plot"] = gr.Plot(label="Planned joint motion (model deg)")
    with gr.Accordion("Inspect an episode", open=False):
        ui["episode"] = gr.Radio(
            ["multisine", "joint_chirps", "heldout"], value="multisine", label="Preview episode"
        )
        ui["fraction"] = gr.Slider(0, 1, value=0, step=0.01, label="Position within episode")
    gr.Markdown(
        "The check includes the robot and a flat table. It cannot see other workspace obstacles."
    )


def _collection_controls(gr, ui):
    gr.Markdown(
        "### Hold first, then start motion\n\nSupport the arm at the checked center. Enable "
        "hold, wait for Holding, then remove your hands before starting excitation."
    )
    with gr.Column() as ui["hold_setup"]:
        ui["support"] = gr.Checkbox(
            label="Arm supported at the center, torque off, workspace clear"
        )
        ui["hold"] = gr.Button("Enable hold", interactive=False)
    with gr.Column(visible=False) as ui["start_setup"]:
        ui["hands_clear"] = gr.Checkbox(label="Holding is stable, my hands and workspace are clear")
        ui["start"] = gr.Button("Start excitation", variant="primary", interactive=False)
    ui["retry"] = gr.Button("Reconnect and prepare another attempt", visible=False)
    with gr.Column(visible=False) as ui["release_setup"]:
        ui["tuning_release_supported"] = gr.Checkbox(
            label="I am supporting the arm before turning torque off"
        )
        ui["tuning_release"] = gr.Button("Release motors and prepare another attempt")
    ui["review"] = gr.Button("Review completed recordings")
    ui["back_preview_collect"] = gr.Button("Back to motion preview")
    with gr.Accordion("Recording location", open=False):
        ui["output_dir"] = gr.Textbox(
            "runs/calibration/recordings", label="New recordings directory"
        )
    gr.Markdown(
        "Stop ends further commands and preserves torque at the last target. Hold times out"
        " after 60 seconds without a start action."
    )


def _fit_controls(gr, ui):
    gr.Markdown(
        "### Review a candidate\n\nFit the recorded motor response and compare baseline and "
        "candidate errors, including the independent held-out episode. A completed fit does"
        " not accept the candidate or change hardware."
    )
    ui["fit"] = gr.Button("Fit candidate", variant="primary")
    ui["back_preview_review"] = gr.Button("Check another motion")
    ui["fit_summary"] = gr.Markdown("")
    ui["errors"] = gr.Dataframe(
        headers=["Episode", "Split", "Joint", "Baseline RMSE deg", "Candidate RMSE deg"],
        interactive=False,
    )
    ui["diagnostics"] = gr.Dataframe(
        headers=["Parameter", "Candidate", "At bound", "Poorly identified", "95% CI half-width"],
        interactive=False,
    )
    with gr.Accordion("Fit settings and recording paths", open=False):
        ui["fit_recordings"] = gr.Textbox(
            "runs/calibration/recordings", label="Recordings directory"
        )
        ui["fit_output"] = gr.Textbox("runs/calibration/fit", label="New fit output directory")
        ui["max_iters"] = gr.Number(100, precision=0, label="Maximum optimizer iterations")


def _build_guide(gr, ui):
    with gr.Walkthrough(selected=0) as ui["walkthrough"]:
        for index, (label, builder) in enumerate(
            (
                ("Connect", _connect_controls),
                ("Match joints", _alignment_controls),
                ("Check jaws", _gripper_controls),
                ("Save profile", _profile_controls),
            )
        ):
            with gr.Step(label, id=index, interactive=True):
                builder(gr, ui)
    with gr.Column(visible=False, min_width=0, elem_id="dynamics-workflow") as ui["dynamics"]:
        ui["tuning_status"] = gr.Markdown("")
        gr.Markdown(
            "### Dynamics tuning\n\nYour aligned profile is already saved. These "
            "extra steps identify the robot's motion response."
        )
        with gr.Walkthrough(selected=0) as ui["tuning_steps"]:
            for index, (label, builder) in enumerate(
                (
                    ("Preview motion", _motion_controls),
                    ("Collect motion", _collection_controls),
                    ("Review dynamics", _fit_controls),
                )
            ):
                with gr.Step(label, id=index, interactive=True):
                    builder(gr, ui)
        ui["return_profile"] = gr.Button("Return to your profile")


def _build_scene(gr, ui):
    ui["overlay"] = gr.Plot(
        render_overlay((0, 0, 0, 0, 0, 45)),
        label="Your robot and its physical reference",
        elem_id="calibration-robot-view",
    )
    gr.Markdown(
        "**Cyan: physical reference to match. Amber: live encoder pose.** Unconfirmed joint"
        " offsets are provisional. Drag to orbit; use Fit view to reset."
    )
    with gr.Row():
        ui["camera"] = gr.Radio(
            [("Front", "front"), ("Side", "side"), ("Top", "top")],
            value="front",
            label="View",
            scale=3,
        )
        ui["fit_view"] = gr.Button("Fit view", scale=1)
    ui["focus"] = gr.Checkbox(label="Close-up of the active joint", value=False)
    with gr.Accordion("Joint readings and connection details", open=False):
        ui["feedback"] = gr.Dataframe(
            headers=[
                "Joint",
                "Hardware",
                "Unit",
                "Model deg",
                "Reference deg",
                "Difference deg",
                "Alignment",
            ],
            interactive=False,
            label="Live joint feedback",
        )
        ui["details"] = gr.Markdown("")


def _build_layout(gr, ui):
    gr.Markdown(
        "# Calibrate your robot\n\nConnect, match the physical arm to the model, and save its"
        " profile."
    )
    ui["status"] = gr.Markdown(
        "Disconnected. Connect your follower to begin.", elem_id="hardware-status"
    )
    with gr.Row():
        with gr.Column(scale=8, min_width=320, elem_classes="calibration-scene"):
            _build_scene(gr, ui)
        with gr.Column(scale=3, min_width=320):
            _build_guide(gr, ui)
            ui["progress"] = gr.Markdown("")
            ui["reconnect"] = gr.Button("Reconnect saved robot", visible=False)
            ui["stop"] = gr.Button("Stop commands", variant="stop", visible=False)
            with gr.Accordion("Disconnect or release motors", open=False):
                ui["disconnect"] = gr.Button("Disconnect")
                ui["release_supported"] = gr.Checkbox(
                    label="I am supporting the arm and ready for torque to turn off"
                )
                ui["release"] = gr.Button("Release motors")
                gr.Markdown(
                    "Disconnect preserves torque. Release turns torque off and can let an "
                    "unsupported arm fall."
                )
            ui["files"] = gr.File(
                label="Your profile and recordings",
                file_count="multiple",
                interactive=False,
                visible=False,
            )
    ui["aspect"] = gr.Number(1.5, visible=False, label="Robot viewport aspect")
    ui["shown_reference"] = gr.JSON([0, 0], visible=False, label="Displayed reference frame")
    ui["instruction_joint"] = gr.Number(0, precision=0, visible=False, label="Instruction joint")


def _guide_updates(gr, guide):
    return (
        gr.Walkthrough(selected=guide.stage),
        f"### Joint {guide.joint_index + 1} of 5: {_JOINT_LABELS[guide.joint_index]}",
        _GUIDANCE[guide.joint_index],
        guide.reference_deg[guide.joint_index],
        gr.update(visible=guide.stage == 1 and guide.joint_index >= 3),
        guide.focused,
        guide.view,
        gr.update(interactive=guide.joint_index > 0),
        guide.joint_index,
    )


def _suggest_path(session):
    name = session.profile.robot_id if session.profile else "robot"
    path = Path("runs/calibration") / f"{name}_profile.json"
    if path.exists():
        path = path.with_stem(f"{path.stem}_{datetime.now(UTC):%Y%m%dT%H%M%SZ}")
    return str(path)


def _profile_message(session):
    if session.profile_path is not None:
        return (
            f"### Profile saved\n\n`{session.profile_path}`\n\n"
            "Your confirmed joint mapping is ready to use in simulation. "
            "You can stop here or improve motion dynamics below."
        )
    if len(session.confirmed) != 5 or not session.gripper_confirmed:
        return "### Save your aligned profile\n\nFinish the physical joint and jaw checks first."
    return (
        "### Save your aligned profile\n\n"
        "Your body joints and jaw mapping are confirmed. "
        "Save them together so simulation can use this robot's joint coordinates."
    )


class _ReferenceViewChanged(ValueError):
    """Reject captures against a stale form or rendered reference."""


def _capture_visible_reference(guide, angle, displayed, instruction_joint):
    with guide.session.lock:
        expected = [guide.joint_index, guide.reference_deg[guide.joint_index]]
        if displayed != expected or [instruction_joint, angle] != expected:
            raise _ReferenceViewChanged(
                "Wait for the reference view to update, "
                "then check the physical joint before capturing."
            )
        guide.capture_next(angle)


def _tick(session, guide, viewport, episode, fraction, aspect):
    import gradio as gr

    session.poll()
    with session.lock:
        if session.phase in ("disconnected", "fault"):
            guide.stage = 0
            viewport.tuning = False
        target = np.array(guide.reference_deg, dtype=float)
        if viewport.tuning and session.config is not None and session.profile is not None:
            trajectory = next(
                item
                for item in generate_excitation(session.config, session.profile)
                if item.name == episode
            )
            target = trajectory.commands_deg[round(fraction * (len(trajectory.commands_deg) - 1))]
        measured = (
            session.profile.to_model_deg(session.measured)
            if session.profile is not None and session.measured is not None
            else None
        )
        hardware = None if session.measured is None else session.measured.copy()
        focus = (
            SO101_JOINT_NAMES[guide.joint_index] if guide.focused and not viewport.tuning else None
        )
        if guide.stage == 2 and guide.focused and not viewport.tuning:
            focus = "gripper"
        key = (
            tuple(target),
            guide.view,
            focus,
            measured is not None,
            None
            if session.profile is None
            else (session.profile.joint_offsets_deg, session.profile.gripper_limits_deg),
        )
        if viewport.bounds is None or key != viewport.key:
            viewport.bounds = overlay_bounds(target, measured, focus_joint=focus)
            viewport.key = key
        else:
            viewport.bounds = expand_bounds(viewport.bounds, target, measured, focus_joint=focus)
        view = guide.view
        camera_revision = f"guided:{view}:{focus}:{viewport.revision}"
        active_joint = (
            None
            if viewport.tuning
            else ("gripper" if guide.stage == 2 else SO101_JOINT_NAMES[guide.joint_index])
        )
        bounds = viewport.bounds
        confirmed = session.confirmed.copy()
        gripper_confirmed = session.gripper_confirmed
        phase = session.phase
        active = session.worker is not None and session.worker.is_alive()
        motor_state = (
            "motors off"
            if session.monitor is not None and not any(session.monitor.torque.values())
            else "torque may be enabled"
        )
        status = (
            f"Connected, {motor_state}. {len(confirmed)}/5 body references confirmed."
            if phase == "monitoring" and session.monitor is not None
            else phase.replace("_", " ").capitalize()
        )
        if phase == "disconnected" and session.torque_possible:
            status += ". Torque may still be enabled. Support the arm before release."
        if phase in ("arming", "holding", "running", "stopped") or session.status.startswith(
            "Preflight failed"
        ):
            status += f". {session.status}"
        rows = [
            [
                name,
                None if hardware is None else round(float(hardware[index]), 2),
                "deg" if index < 5 else "percent",
                None if measured is None else round(float(measured[index]), 2),
                round(float(target[index]), 2),
                None
                if measured is None or not (index in confirmed if index < 5 else gripper_confirmed)
                else round(float(measured[index] - target[index]), 2),
                "Physically confirmed"
                if (index in confirmed if index < 5 else gripper_confirmed)
                else "Provisional",
            ]
            for index, name in enumerate(SO101_JOINT_NAMES)
        ]
        outputs = (
            status,
            session.progress,
            rows,
            gr.update(
                interactive=session.verified
                and phase == "monitoring"
                and session.monitor is not None
                and not any(session.monitor.torque.values())
            ),
            gr.update(interactive=phase == "holding" and not session.should_stop()),
            gr.update(
                value=list(dict.fromkeys([*session.artifacts, *viewport.previous_artifacts]))
                or None,
                visible=bool(session.artifacts or viewport.previous_artifacts),
            ),
            session.status,
            gr.update(visible=active),
            gr.Walkthrough(selected=guide.stage, visible=not viewport.tuning),
            gr.update(visible=viewport.tuning),
            gr.update(
                visible=session.profile_path is not None
                and len(confirmed) == 5
                and gripper_confirmed
            ),
            _profile_message(session),
            gr.update(
                visible=session.profile_path is None
                and guide.stage == 3
                and len(confirmed) == 5
                and gripper_confirmed
            ),
            gr.update(visible=guide.stage > 0 and session.monitor is None and not active),
            [guide.joint_index, float(target[guide.joint_index])],
            gr.update(interactive=guide.stage == 1 and session.monitor is not None),
            f"{session.status}\n\n**Next:** {_dynamics_readiness(session)}",
            gr.update(
                visible=phase == "monitoring"
                and session.monitor is not None
                and not any(session.monitor.torque.values())
            ),
            gr.update(visible=phase == "holding"),
            gr.update(visible=not active and session.monitor is None),
            gr.update(visible=session.monitor is not None and any(session.monitor.torque.values())),
        )
    figure = render_overlay(
        target,
        measured,
        provisional=len(confirmed) != 5 or not gripper_confirmed,
        active_joint=active_joint,
        view=view,
        focus_joint=focus,
        bounds_m=bounds,
        viewport_aspect=aspect,
    )
    if measured is None and phase in ("arming", "stopped"):
        text = (
            "Preparing holding. Keep supporting the arm."
            if phase == "arming"
            else "No live pose after this attempt. Reconnect to read the arm again."
        )
        figure.update_annotations(text=text)
    figure.update_layout(uirevision=camera_revision)
    return (*outputs[:3], figure, *outputs[3:])


def _wire_scene(gr, ui, guide, viewport, perform):
    def reference(angle):
        perform(guide.set_reference, angle)
        viewport.fit()
        return gr.update(interactive=False)

    def rotation(delta):
        perform(guide.preview_rotation, delta)
        viewport.fit()

    def jaw(angle):
        if not isinstance(angle, (int, float)) or isinstance(angle, bool) or not np.isfinite(angle):
            raise gr.Error("Enter a finite jaw angle in model degrees.")
        guide.reference_deg[5] = angle
        viewport.fit()

    ui["reference_angle"].input(
        reference, ui["reference_angle"], ui["capture"], trigger_mode="always_last"
    )
    ui["rotate_minus"].click(lambda: rotation(-90))
    ui["rotate_plus"].click(lambda: rotation(90))
    ui["closed"].click(jaw, ui["grip_low"])
    ui["open"].click(jaw, ui["grip_high"])
    for endpoint in ("grip_low", "grip_high"):
        ui[endpoint].input(lambda: False, outputs=ui["grip_confirm"], queue=False)

    def camera(value):
        guide.view = value
        viewport.fit()

    def focus(value):
        guide.focused = value
        viewport.fit()

    ui["camera"].input(camera, ui["camera"], trigger_mode="always_last")
    ui["focus"].input(focus, ui["focus"], trigger_mode="always_last")
    ui["fit_view"].click(viewport.fit)


def _wire_restore(gr, app, ui, session, guide, viewport, perform, sync, guide_outputs):
    def restore():
        profile, config = session.profile, session.config
        motion = (
            (
                *config.center_qpos_deg,
                *config.amplitudes_deg,
                config.duration_s,
                config.table_z_m,
                config.seed,
            )
            if config is not None
            else (gr.skip(),) * 15
        )
        return (
            profile.control_period_s if profile else 0.02,
            *(profile.gripper_limits_deg if profile else (-10, 100)),
            str(session.profile_path) if session.profile_path else _suggest_path(session),
            str(session.profile_path) if session.profile_path else "",
            gr.Walkthrough(selected=viewport.tuning_step),
            *motion,
        )

    restore_outputs = [
        ui[name]
        for name in (
            "period",
            "grip_low",
            "grip_high",
            "profile_output",
            "saved_profile",
            "tuning_steps",
        )
    ] + [*ui["centers"], *ui["amplitudes"], ui["duration"], ui["table"], ui["seed"]]

    def reconnect(port, directory):
        saved = session.profile_path
        perform(guide.connect, port, directory)
        if saved is not None:
            perform(guide.load_profile, saved)

    ui["reconnect"].click(reconnect, [ui["port"], ui["calibration_dir"]]).then(
        sync, outputs=guide_outputs
    ).then(
        restore,
        outputs=restore_outputs,
    )
    app.load(sync, outputs=guide_outputs).then(
        restore,
        outputs=restore_outputs,
    )


def _wire_profile(gr, app, ui, session, guide, viewport):
    from so101_nexus.calibration.hardware import discover_ports

    guide_outputs = [
        ui[name]
        for name in (
            "walkthrough",
            "joint_title",
            "instruction",
            "reference_angle",
            "rotation_group",
            "focus",
            "camera",
            "previous",
            "instruction_joint",
        )
    ]

    def perform(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except Exception as error:
            raise gr.Error(str(error)) from error

    def sync():
        viewport.fit()
        return _guide_updates(gr, guide)

    def connect(port, directory):
        perform(guide.connect, port, directory)
        return _suggest_path(session), False, False, False

    ui["connect"].click(
        connect,
        [ui["port"], ui["calibration_dir"]],
        [ui["profile_output"], ui["grip_confirm"], ui["support"], ui["hands_clear"]],
    ).then(sync, outputs=guide_outputs)
    ui["discover"].click(lambda: gr.update(choices=discover_ports()), outputs=ui["port"])

    def load(path, port, directory):
        with session.lock:
            if session.monitor is None:
                perform(guide.connect, port, directory)
            perform(guide.load_profile, path)
            return session.profile.control_period_s, *session.profile.gripper_limits_deg

    ui["load"].click(
        load,
        [ui["saved_profile"], ui["port"], ui["calibration_dir"]],
        [ui["period"], ui["grip_low"], ui["grip_high"]],
    ).then(sync, outputs=guide_outputs).then(
        lambda: str(session.profile_path), outputs=ui["profile_output"]
    )

    def capture(angle, displayed, instruction_joint):
        try:
            _capture_visible_reference(guide, angle, displayed, instruction_joint)
        except _ReferenceViewChanged:
            return gr.update(interactive=False), gr.update(
                value="The reference changed. This panel has been refreshed. "
                "Check the highlighted physical joint against the cyan reference, "
                "then capture again.",
                visible=True,
            )
        except Exception as error:
            raise gr.Error(str(error)) from error
        return gr.update(interactive=False), gr.update(visible=False)

    ui["capture"].click(
        capture,
        [ui["reference_angle"], ui["shown_reference"], ui["instruction_joint"]],
        [ui["capture"], ui["capture_notice"]],
    ).then(sync, outputs=guide_outputs)
    for name in ("previous", "back_body", "edit_profile"):
        ui[name].click(lambda: perform(guide.previous_joint)).then(
            sync, outputs=guide_outputs
        ).then(lambda: False, outputs=ui["grip_confirm"])

    ui["map_gripper"].click(
        lambda lo, hi, confirmed: perform(guide.confirm_gripper, lo, hi, confirmed=confirmed),
        [ui["grip_low"], ui["grip_high"], ui["grip_confirm"]],
    ).then(sync, outputs=guide_outputs)

    def save(path, period):
        saved = perform(guide.save_profile, path, period)
        return (
            f"### Profile saved\n\n`{saved}`\n\n"
            "Your confirmed joint mapping is ready to use in simulation. "
            "You can stop here or improve motion dynamics below.",
            gr.update(visible=True),
        )

    ui["save"].click(
        save, [ui["profile_output"], ui["period"]], [ui["profile_message"], ui["tune"]]
    )

    _wire_scene(gr, ui, guide, viewport, perform)
    _wire_restore(gr, app, ui, session, guide, viewport, perform, sync, guide_outputs)
    return perform


def _wire_attempts(gr, ui, session, guide, viewport, perform, settings):
    def check(manual_center, *values):
        try:
            center = _check_motion_pose(session, values, manual_center=manual_center)
            settings_values = [*center, *values[6:]]
            plot = _show_plan(gr, session, *settings_values)
        except Exception as error:
            session.verified = False
            session.status = (
                f"Motion check failed: {error}. Adjust the supported pose and check again."
            )
            raise gr.Error(session.status) from error
        viewport.tuning_step = 1
        return plot, gr.Walkthrough(selected=1), *center, False, False

    ui["preview"].click(
        check,
        [ui["manual_center"], *settings],
        [
            ui["trajectory_plot"],
            ui["tuning_steps"],
            *ui["centers"],
            ui["support"],
            ui["hands_clear"],
        ],
    )

    def hold(path, supported, *values):
        try:
            return _prepare_collection(session, path, supported, values)
        except Exception as error:
            session.status = f"Could not enable hold: {error}"
            raise gr.Error(session.status) from error

    ui["hold"].click(hold, [ui["output_dir"], ui["support"], *settings], ui["output_dir"]).then(
        lambda: (False, False), outputs=[ui["support"], ui["hands_clear"]]
    )
    ui["start"].click(
        partial(_start_motion, gr, session), [ui["hands_clear"], *settings], queue=False
    ).then(lambda: False, outputs=ui["hands_clear"])

    def retry(port, directory):
        perform(session._idle)
        saved = perform(_ensure_saved_profile, session)
        viewport.previous_artifacts.extend(
            path for path in session.artifacts if Path(path).exists()
        )
        perform(guide.connect, port, directory)
        perform(guide.load_profile, saved)
        viewport.tuning_step = 1 if any(session.monitor.torque.values()) else 0
        viewport.fit()
        session.status = (
            "Read-only connection restored. Check the arm's current supported pose again."
        )
        return gr.Walkthrough(selected=viewport.tuning_step), False, False

    retry_inputs = [ui["port"], ui["calibration_dir"]]
    retry_outputs = [ui["tuning_steps"], ui["support"], ui["hands_clear"]]
    ui["retry"].click(retry, retry_inputs, retry_outputs)

    def release_and_retry(supported, port, directory):
        perform(session.release, supported=supported)
        result = retry(port, directory)
        return *result, False

    ui["tuning_release"].click(
        release_and_retry,
        [ui["tuning_release_supported"], *retry_inputs],
        [*retry_outputs, ui["tuning_release_supported"]],
    )


def _wire_tuning(gr, ui, session, guide, viewport, perform):
    settings = [
        *ui["centers"],
        *ui["amplitudes"],
        ui["duration"],
        ui["table"],
        ui["seed"],
        ui["period"],
    ]

    def begin():
        perform(session._idle)
        perform(session._aligned)
        if session.profile_path is None or guide.stage != 3:
            raise gr.Error("Save your aligned profile first.")
        session.status = (
            "Position and support the arm in a raised pose. Check its current pose to plan motion."
        )
        viewport.tuning = True
        viewport.tuning_step = 0
        viewport.fit()
        return gr.update(visible=True), gr.update(visible=False), gr.Walkthrough(selected=0)

    ui["tune"].click(begin, outputs=[ui["dynamics"], ui["walkthrough"], ui["tuning_steps"]])

    def finish():
        perform(session._idle)
        viewport.tuning = False
        viewport.fit()
        return gr.update(visible=False), gr.Walkthrough(visible=True, selected=3)

    ui["return_profile"].click(finish, outputs=[ui["dynamics"], ui["walkthrough"]])
    ui["current"].click(partial(_current_center, gr, session), outputs=ui["centers"]).then(
        lambda: (False, False), outputs=[ui["support"], ui["hands_clear"]]
    )

    def edited(*values):
        _call(gr, session, _plan, session, *values)
        return False, False

    for setting in settings:
        setting.input(edited, settings, [ui["support"], ui["hands_clear"]], queue=False)

    _wire_attempts(gr, ui, session, guide, viewport, perform, settings)

    def review():
        perform(session._idle)
        if session.recordings_dir is None or not list(session.recordings_dir.glob("*.npz")):
            raise gr.Error("No completed recordings yet. Collect all three episodes first.")
        viewport.tuning_step = 2
        return gr.Walkthrough(selected=2), str(session.recordings_dir)

    def back_preview():
        perform(session._idle)
        session.verified = False
        viewport.tuning_step = 0
        return gr.Walkthrough(selected=0)

    for button in ("back_preview_collect", "back_preview_review"):
        ui[button].click(back_preview, outputs=ui["tuning_steps"])

    ui["review"].click(review, outputs=[ui["tuning_steps"], ui["fit_recordings"]])
    ui["fit"].click(
        partial(_fit_candidate, gr, session),
        [ui["fit_recordings"], ui["fit_output"], ui["max_iters"]],
        [ui["errors"], ui["diagnostics"], ui["fit_summary"]],
    )


def build_app(session: CalibrationSession | None = None) -> Any:
    """Build a guided local profile workflow without opening a hardware connection."""
    import gradio as gr

    session = session or CalibrationSession()
    guide, viewport, ui = CalibrationGuide(session), _Viewport(), {}
    with gr.Blocks(title="SO Nexus Calibration", fill_width=True) as app:
        _build_layout(gr, ui)
        perform = _wire_profile(gr, app, ui, session, guide, viewport)
        _wire_tuning(gr, ui, session, guide, viewport, perform)
        tick = partial(_tick, session, guide, viewport)
        inputs = [ui["episode"], ui["fraction"], ui["aspect"]]
        outputs = [
            ui[name]
            for name in (
                "status",
                "progress",
                "feedback",
                "overlay",
                "hold",
                "start",
                "files",
                "details",
                "stop",
                "walkthrough",
                "dynamics",
                "tune",
                "profile_message",
                "save",
                "reconnect",
                "shown_reference",
                "capture",
                "tuning_status",
                "hold_setup",
                "start_setup",
                "retry",
                "release_setup",
            )
        ]
        timer = gr.Timer(0.5)
        timer.tick(tick, inputs, outputs, js=_MEASURE_VIEW, queue=False, show_progress="hidden")
        app.load(tick, inputs, outputs, js=_MEASURE_VIEW)
        ui["stop"].click(session.stop, queue=False)
        ui["disconnect"].click(lambda: perform(session.disconnect), queue=False)
        ui["release"].click(
            lambda supported: perform(session.release, supported=supported),
            ui["release_supported"],
            queue=False,
        ).then(lambda: False, outputs=ui["release_supported"])
        ui["walkthrough"].select(
            lambda: gr.Walkthrough(selected=guide.stage), outputs=ui["walkthrough"], queue=False
        )
        ui["tuning_steps"].select(
            lambda: gr.Walkthrough(selected=viewport.tuning_step),
            outputs=ui["tuning_steps"],
            queue=False,
        )
        app.unload(session.stop)
    return app
