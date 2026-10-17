"""Regression tests for conflicting teleop episode operations."""

import threading
from unittest.mock import Mock

import numpy as np
import pytest

from so101_nexus.teleop import app
from so101_nexus.teleop.dataset import FieldSelection
from so101_nexus.teleop.recorder import RecordedTransition, RecordingState
from tests.core.test_teleop_app_helpers import fake_gradio  # noqa: F401

pytestmark = pytest.mark.usefixtures("fake_gradio")


@pytest.fixture
def review_session(tmp_path):
    state = RecordingState(num_episodes=2)
    state.transitions.append(RecordedTransition(action=np.zeros(6), state=np.zeros(6)))
    dataset = Mock(root=tmp_path, num_episodes=0)

    def save():
        dataset.num_episodes += 1

    dataset.save_episode.side_effect = save
    return {
        "state": state,
        "dataset": dataset,
        "action_space": "joint_pos",
        "field_selection": FieldSelection(
            wrist_image=False, overhead_image=False, environment_state=False
        ),
        "env_id": "MuJoCoTouch-v1",
        "fps": 30,
    }


def test_duplicate_approval_cannot_save_an_empty_episode(review_session):
    app._cb_approve_episode(review_session)
    result = app._cb_approve_episode(review_session)
    assert review_session["state"].episodes_completed == 1
    review_session["dataset"].save_episode.assert_called_once()
    review_session["dataset"].clear_episode_buffer.assert_not_called()
    assert result == ({},) * 9


def test_stale_start_cannot_extend_a_completed_session(review_session):
    state = review_session["state"]
    state.clear_episode()
    state.episodes_completed = state.num_episodes
    with pytest.raises(RuntimeError, match="complete"):
        app._cb_start_recording(review_session)
    assert "recording_worker" not in review_session


@pytest.mark.parametrize("operation", [app._cb_approve_episode, app._cb_discard_episode])
def test_review_callbacks_cannot_modify_a_recording_episode(review_session, operation):
    review_session["recording_worker"] = Mock()
    review_session["recording_worker"].wait.return_value = False
    result = operation(review_session)
    assert len(review_session["state"].transitions) == 1
    review_session["dataset"].add_frame.assert_not_called()
    review_session["dataset"].clear_episode_buffer.assert_not_called()
    assert all(update == {} for update in result)


@pytest.mark.parametrize("operation", [app._cb_discard_episode, app._cb_start_recording])
def test_episode_mutations_cannot_interrupt_saving(review_session, operation):
    entered, release = threading.Event(), threading.Event()

    def add_frame(_frame):
        entered.set()
        assert release.wait(3)

    review_session["dataset"].add_frame.side_effect = add_frame
    errors = []

    def approve():
        try:
            app._cb_approve_episode(review_session)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=approve)
    worker.start()
    try:
        assert entered.wait(3)
        with pytest.raises(RuntimeError, match="already being"):
            operation(review_session)
        assert len(review_session["state"].transitions) == 1
        review_session["dataset"].clear_episode_buffer.assert_not_called()
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert not errors
    review_session["dataset"].save_episode.assert_called_once()


@pytest.mark.parametrize("camera", ["wrist", "overhead", "side"])
def test_missing_middle_camera_does_not_shift_later_frames(review_session, camera):
    from so101_nexus.config import SO101_JOINT_NAMES
    from so101_nexus.teleop.recorder import _append_transition

    state = review_session["state"]
    state.clear_episode()
    review_session["field_selection"] = FieldSelection(
        wrist_image=camera == "wrist",
        overhead_image=camera == "overhead",
        side_image=camera == "side",
        environment_state=False,
    )
    for index in range(3):
        observation = dict.fromkeys((f"{name}.pos" for name in SO101_JOINT_NAMES), float(index))
        if index != 1:
            observation[camera] = np.full((2, 2, 3), index, dtype=np.uint8)
        _append_transition(state, observation, observation, None, SO101_JOINT_NAMES)

    result = app._cb_approve_episode(review_session)

    assert f"no {camera} frame" in result[6]["value"]
    assert len(state.transitions) == 3
    assert state.episodes_completed == 0
    assert getattr(state.transitions[1], f"{camera}_image") is None
    np.testing.assert_array_equal(
        getattr(state.transitions[2], f"{camera}_image"), np.full((2, 2, 3), 2, dtype=np.uint8)
    )
    assert review_session["dataset"].add_frame.call_count == 1
    review_session["dataset"].save_episode.assert_not_called()
    review_session["dataset"].clear_episode_buffer.assert_called_once()
