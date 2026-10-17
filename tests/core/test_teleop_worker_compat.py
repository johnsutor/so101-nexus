"""Native simulator checks for persistent teleop recording workers."""

import threading

import numpy as np
import pytest

from so101_nexus.config import SO101_JOINT_NAMES


def _use_reproducible_rendering(monkeypatch):
    from so101_nexus import RenderingQuality
    from so101_nexus.teleop import session

    build = session.build_sim_follower_config

    def build_config(**kwargs):
        config = build(**kwargs)
        config.env_kwargs["config"].render.quality = RenderingQuality(profile="reproducible")
        return config

    monkeypatch.setattr(session, "build_sim_follower_config", build_config)


@pytest.mark.slow
@pytest.mark.parametrize("env_id", ["MuJoCoTouch-v1", "MuJoCoPickAndPlace-v2"])
def test_worker_reuses_native_simulator_and_preserves_seeded_resets(monkeypatch, tmp_path, env_id):
    import gymnasium as gym

    from so101_nexus.teleop.recorder import RecordingState, RecordingWorker, recording_thread

    _use_reproducible_rendering(monkeypatch)
    environments = []
    owner_threads = []
    make = gym.make

    def track_make(*args, **kwargs):
        environment = make(*args, **kwargs)
        environments.append(environment)
        owner_threads.append(threading.get_ident())
        return environment

    monkeypatch.setattr(gym, "make", track_make)
    state = RecordingState(num_episodes=2)
    kwargs = {
        "state": state,
        "env_id": env_id,
        "leader": _Leader(),
        "joint_names": SO101_JOINT_NAMES,
        "fps": 30,
        "max_steps": 3,
        "countdown": 0,
        "wrist_roll_offset_deg": -90.0,
        "wrist_wh": (160, 120),
        "overhead_wh": (160, 120),
        "follower_calibration_dir": tmp_path / "cal",
        "follower_robot_id": "worker_compat",
        "seed": 80,
    }
    worker = RecordingWorker()
    try:
        worker.submit(**kwargs)
        assert worker.wait(30)
        assert state.error is None, state.error
        initial = state.initial_sim_state.copy()
        cameras = [t.wrist_image.copy() for t in state.transitions]
        states = np.stack([t.state for t in state.transitions])
        env_states = np.stack([t.env_state for t in state.transitions])
        worker.submit(**kwargs)
        assert worker.wait(30)
        assert state.error is None, state.error
        assert state.episode_seed == 80
        assert state.initial_sim_state == initial
        np.testing.assert_array_equal([t.state for t in state.transitions], states)
        np.testing.assert_array_equal([t.env_state for t in state.transitions], env_states)
        np.testing.assert_array_equal([t.wrist_image for t in state.transitions], cameras)
        state.episodes_completed = 1
        worker.submit(**kwargs)
        assert worker.wait(30)
        assert state.error is None, state.error
        assert state.episode_seed == 81
        assert len(environments) == 1
        next_initial = state.initial_sim_state.copy()
        assert owner_threads[0] != threading.get_ident()
    finally:
        worker.close()

    fresh = RecordingState(num_episodes=1)
    recording_thread(**{**kwargs, "state": fresh, "seed": 81})
    assert fresh.error is None, fresh.error
    assert fresh.initial_sim_state == next_initial
    assert len(environments) == 2


class _Leader:
    def get_action(self):
        return {f"{name}.pos": 50.0 if name == "gripper" else 0.0 for name in SO101_JOINT_NAMES}
