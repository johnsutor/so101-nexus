"""Registered environment configs remain authoritative during teleoperation."""

import pytest

from so101_nexus import PickAndPlaceConfig, PickAndPlaceV2Config, PickConfig, StackCubeConfig
from so101_nexus.teleop import app, session
from so101_nexus.teleop.config_customization import TeleopConfigOverrides


@pytest.mark.parametrize("override", [None, TeleopConfigOverrides(reset_settle_frames=2)])
def test_recording_preserves_registered_config(monkeypatch, override) -> None:
    class RegisteredEnv:
        default_config_cls = PickConfig

    config = PickConfig(ground_colors="black", reset_settle_frames=1)
    monkeypatch.setattr(session, "_resolve_env_ctor", lambda _: (RegisteredEnv, {"config": config}))

    kwargs = session._recording_env_kwargs(
        "CustomPick-v1", (320, 240), (320, 240), overrides=override
    )

    assert kwargs["config"].ground_colors == "black"
    assert kwargs["config"].reset_settle_frames == (1 if override is None else 2)
    assert config.reset_settle_frames == 1


def test_customization_defaults_use_registered_config(monkeypatch) -> None:
    class RegisteredEnv:
        default_config_cls = PickConfig

    monkeypatch.setattr(
        app,
        "_resolve_env_ctor",
        lambda _: (RegisteredEnv, {"config": PickConfig(ground_colors="black")}),
    )

    assert app._customization_ui_state_for_env("CustomPick-v1").ground_colors == ["black"]


def test_customization_defaults_use_profile(monkeypatch, tmp_path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text('{"common":{"ground_colors":["black"],"reset_settle_frames":12}}')

    class RegisteredEnv:
        default_config_cls = PickConfig

    monkeypatch.setattr(app, "_resolve_env_ctor", lambda _: (RegisteredEnv, {}))

    defaults = app._customization_ui_state_for_env("CustomPick-v1", profile_path=str(profile))

    assert defaults.ground_colors == ["black"]
    assert defaults.reset_settle_frames == 12

    kwargs = session._build_recording_config(
        PickConfig(),
        (320, 240),
        (320, 240),
        profile_path=str(profile),
        env_id="CustomPick-v1",
        overrides=TeleopConfigOverrides(ground_colors=("white",)),
    )
    assert kwargs.ground_colors == ["white"]
    assert kwargs.reset_settle_frames == 12


@pytest.mark.parametrize("config_cls", [PickAndPlaceConfig, PickAndPlaceV2Config, StackCubeConfig])
@pytest.mark.parametrize("align_fps", [False, True])
@pytest.mark.parametrize("overrides", [None, TeleopConfigOverrides(reset_settle_frames=2)])
def test_recording_preserves_millimeter_cube_size(
    monkeypatch, config_cls, align_fps, overrides
) -> None:
    class RegisteredEnv:
        default_config_cls = config_cls

    config = config_cls(cube_side_length_mm=30)
    monkeypatch.setattr(session, "_resolve_env_ctor", lambda _: (RegisteredEnv, {"config": config}))

    follower = session.build_sim_follower_config(
        env_id="CustomCube-v1",
        robot_id="test",
        wrist_wh=(64, 48),
        overhead_wh=(96, 72),
        fps=25,
        align_fps=align_fps,
        overrides=overrides,
    )

    updated = follower.env_kwargs["config"]
    assert type(updated) is config_cls
    assert updated is not config
    assert updated.cube_side_length_mm == config.cube_side_length_mm == 30
    assert updated.cube_half_size == config.cube_half_size == pytest.approx(0.015)
    assert updated.reset_settle_frames == (5 if overrides is None else 2)
    assert config.reset_settle_frames == 5
    cameras = {type(obs).__name__: obs for obs in updated.observations}
    assert (cameras["WristCamera"].width, cameras["WristCamera"].height) == (64, 48)
    assert (cameras["OverheadCamera"].width, cameras["OverheadCamera"].height) == (96, 72)
    if align_fps:
        assert updated.physics.control_period_s == pytest.approx(1 / 25)
