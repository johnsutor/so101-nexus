"""Gradio-first guided calibration, collection, and candidate review."""

from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np

from so101_nexus.calibration.session import CalibrationSession
from so101_nexus.calibration.view import render_overlay
from so101_nexus.config import SO101_JOINT_NAMES
from so101_nexus.excitation import ExcitationConfig, generate_excitation

if TYPE_CHECKING:
    from so101_nexus.profiles import RobotProfile
_JOINT_GUIDANCE = (
    "Shoulder pan: turn the shoulder around the vertical base axis. Match its direction "
    "relative to the clamp, viewed from above.",
    "Shoulder lift: support the upper arm and match its inclination. View from the side, "
    "with the highlighted hinge axis pointing toward you.",
    "Elbow flex: support the forearm and match the bend between upper arm and forearm. Keep"
    " the upstream joints fixed while comparing.",
    "Wrist flex: match the fixed jaw's tilt relative to the forearm. Compare the side "
    "silhouettes around the highlighted wrist hinge.",
    "Wrist roll: match the fixed jaw's rotation around the forearm. Look along the forearm "
    "toward the jaws to compare orientation.",
)
_DEFAULT_CENTER = (0, -15, 30, 45, 0, 50)


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


def _build_connect(gr):
    from so101_nexus.calibration.hardware import default_calibration_dir

    gr.Markdown(
        "### Read-only connection\n\nSelect the follower, then match its saved calibration "
        "against motor registers. Discovery and live monitoring leave torque unchanged."
    )
    port = gr.Dropdown(choices=[], allow_custom_value=True, label="Follower serial port")
    discover = gr.Button("Discover serial devices")
    calibration_dir = gr.Textbox(
        str(default_calibration_dir()), label="LeRobot follower calibration directory"
    )
    connect = gr.Button("Connect read-only", variant="primary")
    saved_profile = gr.Textbox(
        label="Existing confirmed profile JSON", placeholder="/path/to/robot_profile.json"
    )
    load = gr.Button("Load matched profile")
    gr.Markdown(
        "Loading a saved profile trusts its previous physical alignment and gripper "
        "mapping. Its encoder snapshot must match the connected arm."
    )
    return (port, discover, calibration_dir, connect, saved_profile, load)


def _build_align(gr):
    gr.Markdown(
        "### Match one joint at a time\n\nKeep the arm supported with torque off. Select a "
        "joint, choose its reference angle, then physically match that joint to the ghost. "
        "Rotate the view to compare silhouettes. Use an angle gauge if placement is "
        "uncertain. Capturing reads the encoder for you."
    )
    active = gr.Radio(
        list(SO101_JOINT_NAMES[:5]), value=SO101_JOINT_NAMES[0], label="Active body joint"
    )
    guidance = gr.Markdown(_JOINT_GUIDANCE[0])
    posture = gr.Dropdown(
        ["Model zero", "Raised example"],
        value="Model zero",
        label="Reference posture (visual aid only)",
    )
    refs = [
        gr.Slider(
            -180, 180, value=0, step=0.5, label=f"Reference {name.replace('_', ' ')} (model deg)"
        )
        for name in SO101_JOINT_NAMES[:5]
    ]
    ref_gripper = gr.Slider(-10, 100, value=45, label="Reference jaw angle (model deg)")
    physically_matched = gr.Checkbox(
        label="I physically matched the active joint to this reference"
    )
    capture = gr.Button("Capture active joint", variant="primary")
    gr.Markdown(
        "### Gripper endpoints\n\nBody alignment does not calibrate the jaws. Enter "
        "independently measured model angles at the calibrated hardware endpoints. Hardware"
        " readings remain 0 to 100 percent."
    )
    grip_low = gr.Number(-10, label="Model angle at hardware 0 percent (deg)")
    grip_high = gr.Number(100, label="Model angle at hardware 100 percent (deg)")
    grip_confirm = gr.Checkbox(label="I verified both gripper endpoint angles separately")
    map_gripper = gr.Button("Confirm gripper mapping")
    period = gr.Number(0.02, label="Command period (seconds)")
    profile_output = gr.Textbox(
        "runs/calibration/robot_profile.json", label="New profile JSON path"
    )
    save = gr.Button("Save confirmed profile")
    return (
        active,
        guidance,
        posture,
        refs,
        ref_gripper,
        physically_matched,
        capture,
        grip_low,
        grip_high,
        grip_confirm,
        map_gripper,
        period,
        profile_output,
        save,
    )


def _build_motion(gr):
    gr.Markdown(
        "### Preview the exact excitation\n\nCenter and amplitude controls change simulation "
        "targets only. Use the live pose after alignment, or adjust a raised example with "
        "room for motion. Preflight includes the robot and a flat table; it cannot see "
        "other workspace obstacles."
    )
    use_current = gr.Button("Use live pose as center")
    centers = [
        gr.Number(value, label=f"Center {name.replace('_', ' ')} (model deg)")
        for name, value in zip(SO101_JOINT_NAMES, _DEFAULT_CENTER, strict=True)
    ]
    with gr.Accordion("Excitation settings", open=False):
        amplitudes = [
            gr.Number(value, label=f"Amplitude {name.replace('_', ' ')} (model deg)")
            for name, value in zip(SO101_JOINT_NAMES, (8, 8, 8, 8, 10, 15), strict=True)
        ]
        duration = gr.Number(20, label="Duration per episode (seconds)")
        table = gr.Number(0, label="Table height relative to base (meters)")
        seed = gr.Number(0, precision=0, label="Trajectory seed")
    preview_episode = gr.Radio(
        ["multisine", "joint_chirps", "heldout"], value="multisine", label="Preview episode"
    )
    preview_fraction = gr.Slider(
        0, 1, value=0, step=0.01, label="Scrub planned motion (fraction of episode)"
    )
    trajectory_plot = gr.Plot(label="Exact generated commands (model deg)")
    preview = gr.Button("Preview planned motion")
    preflight = gr.Button("Run simulation preflight", variant="primary")
    output_dir = gr.Textbox("runs/calibration/recordings", label="New recordings directory")
    support = gr.Checkbox(label="Arm supported at the center, torque off, workspace clear")
    hold = gr.Button("Enable hold", interactive=False)
    hands_clear = gr.Checkbox(label="Holding is stable, my hands and workspace are clear")
    start = gr.Button("Start excitation", variant="primary", interactive=False)
    gr.Markdown(
        "Enabling hold writes the measured servo goal and configures motors before holding."
        " Keep supporting until the status says Holding, then remove your hands. Start "
        "excitation is a separate action. Hold times out after 60 seconds."
    )
    return (
        use_current,
        centers,
        amplitudes,
        duration,
        table,
        seed,
        preview_episode,
        preview_fraction,
        trajectory_plot,
        preview,
        preflight,
        output_dir,
        support,
        hold,
        hands_clear,
        start,
    )


def _build_fit(gr):
    gr.Markdown(
        "### Review a candidate\n\nFit the two training episodes and evaluate the independent"
        " held-out episode. A completed optimization does not accept the candidate or "
        "change hardware."
    )
    fit_recordings = gr.Textbox("runs/calibration/recordings", label="Recordings directory")
    fit_output = gr.Textbox("runs/calibration/fit", label="New fit output directory")
    max_iters = gr.Number(100, precision=0, label="Maximum optimizer iterations")
    fit = gr.Button("Fit candidate", variant="primary")
    fit_summary = gr.Markdown("")
    errors = gr.Dataframe(
        headers=["Episode", "Split", "Joint", "Baseline RMSE deg", "Candidate RMSE deg"],
        interactive=False,
    )
    diagnostics = gr.Dataframe(
        headers=["Parameter", "Candidate", "At bound", "Poorly identified", "95% CI half-width"],
        interactive=False,
    )
    return (fit_recordings, fit_output, max_iters, fit, fit_summary, errors, diagnostics)


def _build_interface(gr, session):
    gr.Markdown(
        "# SO Nexus Calibration\n\nMatch a physical reference, preview motion, then identify "
        "your follower. Opening this app does not connect or write to motors."
    )
    status = gr.Markdown(session.status, elem_id="hardware-status")
    progress = gr.Markdown("")
    with gr.Row():
        with gr.Column(scale=5):
            overlay = gr.Plot(
                render_overlay((0, 0, 0, 0, 0, 45), active_joint=SO101_JOINT_NAMES[0]),
                label="Reference and live arm in the same coordinate frame",
            )
            gr.Markdown(
                "**Cyan translucent: reference. Amber solid: measured.** Drag to orbit, "
                "scroll to zoom, double-click to reset the view. The highlighted axis "
                "belongs to the selected reference joint.\n\n**Provisional until physically "
                "confirmed:** encoder calibration gives travel, not model zero. Visual "
                "overlap alone cannot prove alignment."
            )
            feedback = gr.Dataframe(
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
            view_mode = gr.Radio(
                ["Alignment reference", "Motion preview"],
                value="Alignment reference",
                label="Viewport",
            )
        with gr.Column(scale=4):
            with gr.Tabs():
                with gr.Tab("Connect"):
                    port, discover, calibration_dir, connect, saved_profile, load = _build_connect(
                        gr
                    )
                with gr.Tab("Align"):
                    (
                        active,
                        guidance,
                        posture,
                        refs,
                        ref_gripper,
                        physically_matched,
                        capture,
                        grip_low,
                        grip_high,
                        grip_confirm,
                        map_gripper,
                        period,
                        profile_output,
                        save,
                    ) = _build_align(gr)
                with gr.Tab("Motion"):
                    (
                        use_current,
                        centers,
                        amplitudes,
                        duration,
                        table,
                        seed,
                        preview_episode,
                        preview_fraction,
                        trajectory_plot,
                        preview,
                        preflight,
                        output_dir,
                        support,
                        hold,
                        hands_clear,
                        start,
                    ) = _build_motion(gr)
                with gr.Tab("Fit and review"):
                    fit_recordings, fit_output, max_iters, fit, fit_summary, errors, diagnostics = (
                        _build_fit(gr)
                    )
            with gr.Row():
                stop = gr.Button("Stop commands", variant="stop")
                disconnect = gr.Button("Disconnect")
            gr.Markdown(
                "Stop ends future commands and holds the last servo target. Disconnect also"
                " closes the port. Both preserve torque. If the browser disappears for 5 "
                "seconds, commands stop and torque remains. A serial failure can prevent "
                "commands from reaching the arm."
            )
            release_supported = gr.Checkbox(
                label="I am supporting the arm and ready for torque to turn off"
            )
            release = gr.Button("Release motors")
            gr.Markdown("Release turns torque off. A raised, unsupported arm can fall.")
            files = gr.File(
                label="Profiles, recordings, and fit diagnostics",
                file_count="multiple",
                interactive=False,
            )
    return (
        status,
        progress,
        overlay,
        feedback,
        view_mode,
        release_supported,
        release,
        files,
        stop,
        disconnect,
        port,
        discover,
        calibration_dir,
        connect,
        saved_profile,
        load,
        active,
        guidance,
        posture,
        refs,
        ref_gripper,
        physically_matched,
        capture,
        grip_low,
        grip_high,
        grip_confirm,
        map_gripper,
        period,
        profile_output,
        save,
        use_current,
        centers,
        amplitudes,
        duration,
        table,
        seed,
        preview_episode,
        preview_fraction,
        trajectory_plot,
        preview,
        preflight,
        output_dir,
        support,
        hold,
        hands_clear,
        start,
        fit_recordings,
        fit_output,
        max_iters,
        fit,
        fit_summary,
        errors,
        diagnostics,
    )


def _refresh(session, selected, mode, episode, fraction, *reference):
    import gradio as gr

    session.poll()
    with session.lock:
        target = np.asarray(reference, dtype=float)
        if (
            mode == "Motion preview"
            and session.profile is not None
            and (session.config is not None)
        ):
            trajectories = generate_excitation(session.config, session.profile)
            trajectory = next(item for item in trajectories if item.name == episode)
            target = trajectory.commands_deg[round(fraction * (len(trajectory.commands_deg) - 1))]
        measured = (
            session.profile.to_model_deg(session.measured)
            if session.profile is not None and session.measured is not None
            else None
        )
        rows = []
        for index, name in enumerate(SO101_JOINT_NAMES):
            confirmed = index in session.confirmed if index < 5 else session.gripper_confirmed
            rows.append(
                [
                    name,
                    None if session.measured is None else round(float(session.measured[index]), 2),
                    "deg" if index < 5 else "percent",
                    None if measured is None else round(float(measured[index]), 2),
                    round(float(target[index]), 2),
                    None
                    if measured is None or not confirmed
                    else round(float(measured[index] - target[index]), 2),
                    "Operator confirmed" if confirmed else "Provisional",
                ]
            )
        return (
            session.status,
            session.progress,
            rows,
            render_overlay(
                target,
                measured,
                provisional=len(session.confirmed) != 5 or not session.gripper_confirmed,
                active_joint=selected,
            ),
            gr.update(interactive=session.verified and session.phase == "monitoring"),
            gr.update(interactive=session.phase == "holding" and (not session.should_stop())),
            session.artifacts or None,
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


def _load_profile(gr, session, path):
    def load_values():
        with session.lock:
            session.load_profile(path)
            assert session.profile is not None
            return session.profile.control_period_s, *session.profile.gripper_limits_deg

    result = _call(gr, session, load_values)
    return result if result is not None else (gr.skip(),) * 3


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


def _verify(gr, session, *values):
    try:
        _plan(session, *values)
        session.preflight()
    except Exception as error:
        session.verified = False
        session.status = (
            f"Preflight failed: {error}. "
            "Adjust the center, amplitudes, or table height, then retry."
        )
        gr.Warning(session.status)


def _enable_hold(gr, session, path, supported, *values):

    def arm():
        _plan(session, *values)
        session.arm(path, supported=supported)

    _call(gr, session, arm)


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
    result = _call(gr, session, session.fit, *values)
    return result if result is not None else ([], [], session.status)


def _wire_events(gr, app, session, components):
    from so101_nexus.calibration.hardware import discover_ports

    (
        status,
        progress,
        overlay,
        feedback,
        view_mode,
        release_supported,
        release,
        files,
        stop,
        disconnect,
        port,
        discover,
        calibration_dir,
        connect,
        saved_profile,
        load,
        active,
        guidance,
        posture,
        refs,
        ref_gripper,
        physically_matched,
        capture,
        grip_low,
        grip_high,
        grip_confirm,
        map_gripper,
        period,
        profile_output,
        save,
        use_current,
        centers,
        amplitudes,
        duration,
        table,
        seed,
        preview_episode,
        preview_fraction,
        trajectory_plot,
        preview,
        preflight,
        output_dir,
        support,
        hold,
        hands_clear,
        start,
        fit_recordings,
        fit_output,
        max_iters,
        fit,
        fit_summary,
        errors,
        diagnostics,
    ) = components
    call = partial(_call, gr, session)
    plan = partial(_plan, session)
    current_center = partial(_current_center, gr, session)
    show_plan = partial(_show_plan, gr, session)
    verify = partial(_verify, gr, session)
    enable_hold = partial(_enable_hold, gr, session)
    fit_candidate = partial(_fit_candidate, gr, session)
    settings = [*centers, *amplitudes, duration, table, seed, period]
    refresh = partial(_refresh, session)
    refresh_inputs = [active, view_mode, preview_episode, preview_fraction, *refs, ref_gripper]
    refresh_outputs = [status, progress, feedback, overlay, hold, start, files]
    timer = gr.Timer(0.5)
    timer.tick(refresh, refresh_inputs, refresh_outputs, queue=False, show_progress="hidden")
    app.load(refresh, refresh_inputs, refresh_outputs)
    discover.click(lambda: gr.update(choices=discover_ports()), outputs=port)
    connect.click(lambda p, c: call(session.connect, p, c), [port, calibration_dir]).then(
        lambda: (False,) * 5,
        outputs=[physically_matched, grip_confirm, support, hands_clear, release_supported],
    ).then(refresh, refresh_inputs, refresh_outputs)
    load.click(
        partial(_load_profile, gr, session), saved_profile, [period, grip_low, grip_high]
    ).then(refresh, refresh_inputs, refresh_outputs)
    active.change(
        lambda name: (_JOINT_GUIDANCE[SO101_JOINT_NAMES.index(name)], False),
        active,
        [guidance, physically_matched],
    )
    posture.change(
        lambda name: [*(list(_DEFAULT_CENTER[:5]) if name == "Raised example" else [0] * 5), False],
        posture,
        [*refs, physically_matched],
    )
    for reference in refs:
        reference.input(lambda: False, outputs=physically_matched, queue=False)
    capture.click(
        lambda name, confirmed, *values: call(
            session.capture_joint,
            SO101_JOINT_NAMES.index(name),
            values[SO101_JOINT_NAMES.index(name)],
            confirmed=confirmed,
        ),
        [active, physically_matched, *refs],
    ).then(lambda: False, outputs=physically_matched).then(refresh, refresh_inputs, refresh_outputs)
    map_gripper.click(
        lambda lo, hi, confirmed: call(session.set_gripper, lo, hi, confirmed=confirmed),
        [grip_low, grip_high, grip_confirm],
    ).then(refresh, refresh_inputs, refresh_outputs)
    period.input(lambda value: call(session.set_period, value), period, queue=False)

    def save_current(path, period_s):
        def save_profile():
            session.set_period(period_s)
            session.save_profile(path)

        call(save_profile)

    save.click(save_current, [profile_output, period]).then(
        refresh, refresh_inputs, refresh_outputs
    )
    use_current.click(current_center, outputs=centers).then(
        lambda: (False, False), outputs=[support, hands_clear]
    )

    def edited_plan(*values):
        call(plan, *values)
        return False, False

    for setting in settings:
        setting.input(edited_plan, settings, [support, hands_clear], queue=False)
    preview.click(show_plan, settings, trajectory_plot).then(
        lambda: "Motion preview", outputs=view_mode
    )
    preflight.click(verify, settings).then(refresh, refresh_inputs, refresh_outputs)
    hold.click(enable_hold, [output_dir, support, *settings]).then(
        lambda: (False, False), outputs=[support, hands_clear]
    ).then(refresh, refresh_inputs, refresh_outputs)
    start.click(partial(_start_motion, gr, session), [hands_clear, *settings], queue=False).then(
        lambda: False, outputs=hands_clear
    )
    stop.click(session.stop, queue=False)
    disconnect.click(lambda: call(session.disconnect), queue=False).then(
        refresh, refresh_inputs, refresh_outputs
    )
    release.click(
        lambda supported: call(session.release, supported=supported), release_supported, queue=False
    ).then(lambda: False, outputs=release_supported).then(refresh, refresh_inputs, refresh_outputs)
    fit.click(
        fit_candidate, [fit_recordings, fit_output, max_iters], [errors, diagnostics, fit_summary]
    )
    app.unload(session.disconnect)


def build_app(session: CalibrationSession | None = None) -> Any:
    """Build a local, shared hardware session without connecting to any serial port."""
    import gradio as gr

    session = session or CalibrationSession()
    with gr.Blocks(title="SO Nexus Calibration", fill_width=True) as app:
        components = _build_interface(gr, session)
        _wire_events(gr, app, session, components)
    return app
