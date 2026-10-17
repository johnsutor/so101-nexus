"""Regression tests for conflicting teleop episode operations."""

import threading
from unittest.mock import Mock

import numpy as np
import pytest

from so101_nexus.teleop import app
from so101_nexus.teleop.dataset import FieldSelection
from so101_nexus.teleop.recorder import RecordingState
from tests.core.test_teleop_app_helpers import fake_gradio  # noqa: F401

pytestmark = pytest.mark.usefixtures("fake_gradio")


@pytest.fixture
def review_session(tmp_path):
    state = RecordingState(num_episodes=2)
    state.episode_actions.append(np.zeros(6))
    state.episode_states.append(np.zeros(6))
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
    assert len(review_session["state"].episode_actions) == 1
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
        assert len(review_session["state"].episode_actions) == 1
        review_session["dataset"].clear_episode_buffer.assert_not_called()
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert not errors
    review_session["dataset"].save_episode.assert_called_once()
