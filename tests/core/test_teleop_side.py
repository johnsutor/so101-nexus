"""Side-camera selection, buffering, and dataset approval regressions."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from so101_nexus.teleop.app import (
    _OPTIONAL_FIELD_CHOICES,
    _build_field_selection,
    _cb_approve_episode,
)
from so101_nexus.teleop.dataset import SIDE_KEY
from so101_nexus.teleop.recorder import RecordingState, _publish_camera_frames


@pytest.mark.parametrize("selected", [True, False])
def test_side_camera_saved_only_when_selected(monkeypatch, tmp_path, selected):
    monkeypatch.setitem(
        sys.modules,
        "gradio",
        SimpleNamespace(update=lambda **kwargs: kwargs, Walkthrough=SimpleNamespace),
    )
    assert SIDE_KEY in _OPTIONAL_FIELD_CHOICES
    assert _build_field_selection(_OPTIONAL_FIELD_CHOICES).side_image
    state = RecordingState(num_episodes=2)
    side = np.full((24, 32, 3), 17, dtype=np.uint8)
    _publish_camera_frames(state, {"side": side})
    state.episode_states.append(np.zeros(6, dtype=np.float32))
    state.episode_actions.append(np.zeros(6, dtype=np.float32))
    frames = []
    dataset = SimpleNamespace(
        root=tmp_path,
        repo_id="local/test",
        num_episodes=0,
        add_frame=frames.append,
        save_episode=lambda: None,
    )
    _cb_approve_episode(
        {
            "state": state,
            "dataset": dataset,
            "action_space": "joint_pos",
            "fps": 30,
            "env_id": "MuJoCoTouch-v1",
            "field_selection": _build_field_selection([SIDE_KEY] if selected else []),
        }
    )
    assert state.episodes_completed == 1
    assert (SIDE_KEY in frames[0]) == selected
    if selected:
        np.testing.assert_array_equal(frames[0][SIDE_KEY], side)
    state.clear_episode()
    assert state.episode_side_images == []


def test_recording_config_preserves_side_camera_randomization():
    from so101_nexus import RenderConfig, TouchConfig
    from so101_nexus.teleop.session import _build_recording_config

    render = RenderConfig(
        side_azimuth_range_deg=(150, 150),
        side_elevation_range_deg=(-35, -35),
        side_distance_range=(0.8, 1.2),
    )
    original = TouchConfig(render=render)
    config = _build_recording_config(original, (32, 24), (64, 48))
    assert config.render.camera == "side"
    assert (config.render.width, config.render.height) == (64, 48)
    assert config.render.side_azimuth_range_deg == (150, 150)
    assert config.render.side_elevation_range_deg == (-35, -35)
    assert config.render.side_distance_range == (0.8, 1.2)
    assert original.render.camera == "overhead"


def test_recording_preserves_custom_config_without_render_settings():
    from dataclasses import dataclass

    from so101_nexus.observations import OverheadCamera, WristCamera
    from so101_nexus.teleop.session import _build_recording_config

    @dataclass
    class CustomConfig:
        observations: list

    config = _build_recording_config(CustomConfig(observations=[]), (32, 24), (64, 48))
    assert isinstance(config, CustomConfig)
    assert [type(component) for component in config.observations] == [WristCamera, OverheadCamera]
