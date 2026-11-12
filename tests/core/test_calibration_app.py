"""The guided app plans exact motion and stays read-only until operator actions."""

import numpy as np
import pytest

from so101_nexus import SO101_JOINT_NAMES, RobotProfile, generate_excitation


@pytest.fixture
def profile():
    return RobotProfile(
        "ui-arm",
        {
            name: {
                "id": index,
                "drive_mode": 0,
                "homing_offset": 0,
                "range_min": 0,
                "range_max": 4095,
            }
            for index, name in enumerate(SO101_JOINT_NAMES, 1)
        },
    )


@pytest.mark.parametrize("period", [0.02, 1 / 30, 0.03])
def test_motion_settings_round_duration_to_exact_command_periods(profile, period):
    from dataclasses import replace

    from so101_nexus.calibration.app import make_motion_config

    profile = replace(profile, control_period_s=period)
    center = (0, -15, 30, 45, 0, 50)
    amplitudes = (8, 8, 8, 8, 10, 15)
    config = make_motion_config(profile, list(center), list(amplitudes), 1, 0.1, 4)
    episodes = generate_excitation(config, profile)
    assert config.duration_s == round(1 / period) * period
    assert config.ramp_s == min(2, config.duration_s / 5)
    assert config.center_qpos_deg == center
    assert config.amplitudes_deg == amplitudes
    assert config.table_z_m == 0.1
    assert config.seed == 4
    for episode in episodes:
        np.testing.assert_allclose(np.diff(episode.timestamps_s), period, atol=1e-12)
        assert episode.timestamps_s[-1] == config.duration_s
        np.testing.assert_array_equal(episode.commands_deg[[0, -1]], [center, center])


@pytest.mark.parametrize(
    "center,amplitudes,duration",
    [
        ((0, -15, 30, 45, 0, float("nan")), (8,) * 6, 1),
        ((0, -15, 30, 45, 0, 50), (0,) * 6, 1),
        ((0, -15, 30, 45, 0, 50), (8,) * 6, 0.001),
    ],
)
def test_motion_settings_reject_invalid_or_too_short_motion(profile, center, amplitudes, duration):
    from so101_nexus.calibration.app import make_motion_config

    with pytest.raises(ValueError, match=r"finite|positive|periods"):
        generate_excitation(
            make_motion_config(profile, list(center), list(amplitudes), duration, 0, 0), profile
        )


def test_app_initialization_and_refresh_do_not_open_or_write_hardware(monkeypatch):
    from unittest.mock import Mock

    gr = pytest.importorskip("gradio")
    if not hasattr(gr, "Blocks"):
        pytest.skip("Gradio public Blocks API is unavailable")
    from so101_nexus.calibration.app import build_app
    from so101_nexus.calibration.session import CalibrationSession

    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    monitor_factory = Mock(side_effect=AssertionError("unexpected hardware connection"))
    session = CalibrationSession(monitor_factory=monitor_factory)
    actions = ["connect", "arm", "release", "start_motion"]
    for name in actions:
        monkeypatch.setattr(session, name, Mock(side_effect=AssertionError(name)))
    app = build_app(session)
    assert isinstance(app, gr.Blocks)
    config = app.get_config_file()
    components = config["components"]
    assert any(component["type"] == "plot" for component in components)
    assert any(component["type"] == "dataframe" for component in components)
    buttons = [
        str(component["props"].get("value", ""))
        for component in components
        if component["type"] == "button"
    ]
    for action in ("hold", "excitation", "stop", "disconnect", "release", "fit"):
        assert any(action in value.lower() for value in buttons), action
    for dependency in config["dependencies"]:
        events = {event for _, event in dependency["targets"]}
        if events & {"load", "tick"}:
            app.fns[dependency["id"]].fn(
                *(app.blocks[identifier].value for identifier in dependency["inputs"])
            )
    monitor_factory.assert_not_called()
    for name in actions:
        getattr(session, name).assert_not_called()
    assert session.phase == "disconnected"
    assert session.worker is None


def test_calibration_launch_import_stays_lightweight():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import so101_nexus.calibration.__main__; "
            "assert not ({'gradio', 'lerobot', 'mujoco'} & sys.modules.keys())",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_launcher_binds_loopback_and_passes_browser_preference(monkeypatch):
    from unittest.mock import Mock

    import tyro

    from so101_nexus.calibration import app
    from so101_nexus.calibration.__main__ import LaunchOptions, main

    server = Mock()
    builder = Mock(return_value=server)
    monkeypatch.setattr(app, "build_app", builder)
    monkeypatch.setattr(tyro, "cli", lambda _: LaunchOptions(port=7862, open_browser=False))
    main()
    builder.assert_called_once_with()
    server.queue.assert_called_once_with()
    server.queue.return_value.launch.assert_called_once_with(
        server_name="127.0.0.1", server_port=7862, inbrowser=False
    )


def test_invalid_preview_cannot_render_previous_valid_plan(profile):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.calibration.app import _show_plan, make_motion_config
    from so101_nexus.calibration.session import CalibrationSession

    session = CalibrationSession()
    session.profile = profile
    center: list[float] = [0, -15, 30, 45, 0, 50]
    amplitudes: list[float] = [8] * 6
    previous = make_motion_config(profile, center, amplitudes, 1, 0, 0)
    session.set_plan(previous)
    skipped = object()
    gr = SimpleNamespace(Warning=Mock(), skip=Mock(return_value=skipped))
    result = _show_plan(gr, session, *center, *([0] * 6), 1, 0, 0)
    assert result is skipped
    assert session.config is previous
    gr.Warning.assert_called_once()
    assert "positive" in session.status


def test_valid_preview_contains_exact_generated_commands(profile):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.calibration.app import _show_plan
    from so101_nexus.calibration.session import CalibrationSession

    session = CalibrationSession()
    session.profile = profile
    gr = SimpleNamespace(Warning=Mock(), skip=Mock())
    figure = _show_plan(gr, session, *[0, -15, 30, 45, 0, 50], *([8] * 6), 1, 0, 4)
    assert session.config is not None
    trajectories = generate_excitation(session.config, profile)
    assert len(figure.data) == 18
    for index, trajectory in enumerate(trajectories):
        for joint in range(6):
            trace = figure.data[index * 6 + joint]
            np.testing.assert_array_equal(trace.x, trajectory.timestamps_s)
            np.testing.assert_array_equal(trace.y, trajectory.commands_deg[:, joint])
    gr.Warning.assert_not_called()
    gr.skip.assert_not_called()


def test_invalid_visible_settings_revoke_previous_preflight(profile):
    from so101_nexus.calibration.app import _plan
    from so101_nexus.calibration.session import CalibrationSession

    session = CalibrationSession()
    session.profile = profile
    session.verified = True
    with pytest.raises(ValueError, match="positive"):
        _plan(session, *([0] * 6), *([0] * 6), 1, 0, 0)
    assert not session.verified


def test_plan_uses_visible_command_period_and_rejects_invalid_period(profile):
    from so101_nexus.calibration.app import _plan
    from so101_nexus.calibration.session import CalibrationSession

    session = CalibrationSession()
    session.profile = profile
    values = [0, -15, 30, 45, 0, 50, *([8] * 6), 1, 0, 0]
    _plan(session, *values, 1 / 30)
    assert session.profile.control_period_s == 1 / 30
    assert session.config is not None
    assert session.config.duration_s == 1.0
    session.verified = True
    with pytest.raises(ValueError, match="control_period_s"):
        _plan(session, *values, 0)
    assert not session.verified


def test_new_connection_and_motion_edits_clear_stale_operator_confirmations():
    pytest.importorskip("gradio")
    from so101_nexus.calibration.app import build_app

    app = build_app()
    config = app.get_config_file()
    components = {item["id"]: item for item in config["components"]}
    connect_id = next(
        identifier
        for identifier, item in components.items()
        if item["type"] == "button" and item["props"].get("value") == "Connect read-only"
    )
    connect_event = next(
        item
        for item in config["dependencies"]
        if any(
            identifier == connect_id and event == "click" for identifier, event in item["targets"]
        )
    )
    reset_events = [
        item
        for item in config["dependencies"]
        if item.get("trigger_after") == connect_event["id"]
        and any(components[identifier]["type"] == "checkbox" for identifier in item["outputs"])
    ]
    assert reset_events, "Reconnecting must discard previous physical and motion confirmations"
    reset = reset_events[0]
    assert len(reset["outputs"]) == 5
    assert app.fns[reset["id"]].fn() == (False,) * 5
    edit_events = [
        item
        for item in config["dependencies"]
        if len(item["inputs"]) == 16 and any(event == "input" for _, event in item["targets"])
    ]
    assert len(edit_events) == 16
    for item in edit_events:
        assert len(item["outputs"]) == 2, "Edited plans must clear support and hands-clear claims"

    start_id = next(
        identifier
        for identifier, item in components.items()
        if item["type"] == "button" and item["props"].get("value") == "Start excitation"
    )
    start_event = next(
        item
        for item in config["dependencies"]
        if any(identifier == start_id and event == "click" for identifier, event in item["targets"])
    )
    assert len(start_event["inputs"]) == 17
    labels = {components[identifier]["props"].get("label") for identifier in start_event["inputs"]}
    assert "Command period (seconds)" in labels
    assert "Trajectory seed" in labels
    assert "Holding is stable, my hands and workspace are clear" in labels


@pytest.fixture
def holding_session(profile, monkeypatch):
    import threading

    from so101_nexus.calibration.app import make_motion_config
    from so101_nexus.calibration.session import CalibrationSession

    session = CalibrationSession()
    session.profile = profile
    values = [0, -15, 30, 45, 0, 50, *([8] * 6), 1, 0, 0, profile.control_period_s]
    session.config = make_motion_config(profile, values[:6], values[6:12], *values[12:15])
    session.phase = "holding"
    session.worker = threading.Thread()
    monkeypatch.setattr(session.worker, "is_alive", lambda: True)
    return session, values


def test_editing_motion_settings_cancels_the_frozen_holding_plan(holding_session):
    from so101_nexus.calibration.app import _plan

    session, values = holding_session
    previous = session.config
    values[0] = 10
    with pytest.raises(RuntimeError, match="active worker"):
        _plan(session, *values)
    assert session.cancel.is_set()
    assert not session.motion.is_set()
    assert session.config is previous


@pytest.mark.parametrize("setting,value", [(0, 10), (12, 2), (15, 1 / 30), (15, 0), (6, 0)])
def test_start_checks_submitted_visible_motion_before_any_pending_change_event(
    holding_session, setting, value
):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.calibration.app import _start_motion

    session, values = holding_session
    previous_profile, previous_config = session.profile, session.config
    values[setting] = value
    gr = SimpleNamespace(Warning=Mock())
    _start_motion(gr, session, True, *values)
    assert session.cancel.is_set()
    assert not session.motion.is_set()
    assert session.profile is previous_profile
    assert session.config is previous_config
    assert "preflight" in session.status.lower()
    gr.Warning.assert_called_once()


def test_start_accepts_matching_visible_plan_with_explicit_hands_clear(holding_session):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.calibration.app import _start_motion

    session, values = holding_session
    gr = SimpleNamespace(Warning=Mock())
    _start_motion(gr, session, True, *values)
    assert session.motion.is_set()
    assert not session.cancel.is_set()
    gr.Warning.assert_not_called()


def test_command_period_input_preserves_loaded_profile_clock_exactly():
    pytest.importorskip("gradio")
    from so101_nexus.calibration.app import build_app

    app = build_app()
    period = next(
        component
        for component in app.blocks.values()
        if getattr(component, "label", None) == "Command period (seconds)"
    )
    assert period.preprocess(1 / 30) == 1 / 30


def test_loading_profile_returns_its_clock_and_gripper_in_one_event(profile, tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    pytest.importorskip("gradio")
    from so101_nexus.calibration.app import build_app
    from so101_nexus.calibration.session import CalibrationSession

    saved = replace(profile, control_period_s=1 / 30, gripper_limits_deg=(-5, 80))
    path = tmp_path / "profile.json"
    saved.save(path)
    session = CalibrationSession()
    session.profile = profile
    session.monitor = SimpleNamespace(profile=profile, read=lambda: np.array([0, 0, 0, 0, 0, 50]))
    app = build_app(session)
    config = app.get_config_file()
    load_id = next(
        component["id"]
        for component in config["components"]
        if component["type"] == "button"
        and component["props"].get("value") == "Load matched profile"
    )
    event = next(
        item
        for item in config["dependencies"]
        if any(
            identifier == load_id and trigger == "click" for identifier, trigger in item["targets"]
        )
    )
    assert len(event["outputs"]) == 3
    assert app.fns[event["id"]].fn(str(path)) == (1 / 30, -5, 80)
    assert session.profile.control_period_s == 1 / 30
    assert session.profile.gripper_limits_deg == (-5, 80)
    settings_ids = set(
        next(item for item in config["dependencies"] if len(item["inputs"]) == 16)["inputs"]
    )
    assert not any(
        identifier in settings_ids and trigger == "change"
        for item in config["dependencies"]
        for identifier, trigger in item["targets"]
    ), "Loading control values must not replay stale motion settings into the profile"


def test_using_live_center_revokes_preflight_without_programmatic_change_event(profile):
    from types import SimpleNamespace
    from unittest.mock import Mock

    from so101_nexus.calibration.app import _current_center
    from so101_nexus.calibration.session import CalibrationSession

    session = CalibrationSession()
    session.profile = profile
    session.monitor = SimpleNamespace(read=lambda: np.array([0, 0, 0, 0, 0, 50]))
    session.confirmed = set(range(5))
    session.gripper_confirmed = True
    session.verified = True
    result = _current_center(SimpleNamespace(Warning=Mock(), skip=Mock()), session)
    np.testing.assert_array_equal(result, [0, 0, 0, 0, 0, 45])
    assert not session.verified
