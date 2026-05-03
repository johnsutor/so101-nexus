"""Invariants for the teleop typed state dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from so101_nexus_core.teleop.dataset import FieldSelection
from so101_nexus_core.teleop.state import InitConfig, InitState, TeleopSession


def _example_init_config() -> InitConfig:
    return InitConfig(
        env_id="MuJoCoReach-v1",
        robot_type="so101",
        leader_id="so101_leader",
        fps=30,
        wrist_wh=(480, 480),
        overhead_wh=(640, 480),
        repo_id="local/teleop-test",
        num_episodes=3,
        action_space="joint_pos_delta",
        max_steps=512,
        countdown=3,
        wrist_roll_offset_deg=-90.0,
        field_selection=FieldSelection(),
    )


def test_init_config_is_frozen():
    cfg = _example_init_config()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.fps = 60  # type: ignore[misc]


def test_init_state_log_text_joins_log_lines():
    state = InitState()
    state.append_log("first")
    state.append_log("second")
    assert state.log_text == "first\nsecond"


def test_init_state_reset_for_new_attempt_clears_log_keeps_config():
    state = InitState()
    state.append_log("old")
    cfg = _example_init_config()
    state.reset_for_new_attempt(warning=None, last_config=cfg)
    assert state.log_lines == []
    assert state.last_config is cfg
    assert state.running is True
    assert state.done is False


def test_teleop_session_default_is_empty():
    s = TeleopSession()
    assert s.leader is None
    assert s.dataset is None
    assert s.state is None
    assert s.joint_names == ()
    assert s.fps == 0
