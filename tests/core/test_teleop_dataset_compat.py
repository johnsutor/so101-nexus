"""End-to-end LeRobot compatibility test for the Gradio teleop recorder."""

from __future__ import annotations

import os
import threading

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


class _FakeLeader:
    def connect(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    def get_action(self) -> dict[str, float]:
        return {
            "shoulder_pan.pos": 0.0,
            "shoulder_lift.pos": 0.0,
            "elbow_flex.pos": 0.0,
            "wrist_flex.pos": 0.0,
            "wrist_roll.pos": 0.0,
            "gripper.pos": 50.0,
        }


@pytest.mark.slow
def test_gradio_recording_reloads_as_lerobot_dataset(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("so101_nexus.mujoco")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_nexus.config import SO101_JOINT_NAMES
    from so101_nexus.teleop.dataset import (
        OVERHEAD_KEY,
        WRIST_KEY,
        FieldSelection,
        build_features,
        build_frame,
    )
    from so101_nexus.teleop.recorder import RecordingState, recording_thread

    state = RecordingState(num_episodes=1)
    leader = _FakeLeader()
    leader.connect()

    thread = threading.Thread(
        target=recording_thread,
        kwargs={
            "state": state,
            "env_id": "MuJoCoTouch-v1",
            "leader": leader,
            "joint_names": SO101_JOINT_NAMES,
            "fps": 30,
            "max_steps": 5,
            "countdown": 0,
            "wrist_roll_offset_deg": -90.0,
            "wrist_wh": (160, 120),
            "overhead_wh": (160, 120),
            "follower_calibration_dir": tmp_path / "cal",
            "follower_robot_id": "teleop_sim_smoke",
        },
        daemon=True,
    )
    thread.start()
    thread.join(timeout=30)

    assert not thread.is_alive()
    assert state.error is None, state.error
    assert len(state.episode_actions) == 5

    selection = FieldSelection()
    action_features = {f"{name}.pos": float for name in SO101_JOINT_NAMES}
    follower_features = {**action_features, "wrist": (120, 160, 3), "overhead": (120, 160, 3)}
    features = build_features(selection, follower_features, action_features)

    dataset_root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/teleop-smoke",
        fps=30,
        features=features,
        robot_type="sim_so_follower",
        root=dataset_root,
        use_videos=True,
    )

    for i in range(len(state.episode_actions)):
        frame = build_frame(
            selection,
            state=state.episode_states[i],
            action=state.episode_actions[i],
            task="reach the target",
            wrist_image=state.episode_wrist_images[i],
            overhead_image=state.episode_overhead_images[i],
        )
        dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()

    reloaded = LeRobotDataset("local/teleop-smoke", root=dataset_root)
    assert set(reloaded.features) >= {
        "action",
        "observation.state",
        WRIST_KEY,
        OVERHEAD_KEY,
    }
    assert reloaded.features["action"]["shape"] == (len(SO101_JOINT_NAMES),)
    assert reloaded.features["observation.state"]["shape"] == (len(SO101_JOINT_NAMES),)
    assert reloaded.features["action"]["names"][0] == "shoulder_pan.pos"
    assert reloaded.features[WRIST_KEY]["dtype"] == "video"
    assert reloaded.features[OVERHEAD_KEY]["dtype"] == "video"
    assert reloaded.fps == 30
    assert len(reloaded) == 5
    assert list(reloaded.meta.tasks.index) == ["reach the target"]

    sample = reloaded[0]
    gripper_idx = SO101_JOINT_NAMES.index("gripper")
    assert sample["action"].shape == (len(SO101_JOINT_NAMES),)
    assert 0.0 <= float(sample["action"][gripper_idx]) <= 100.0
    assert 0.0 <= float(sample["observation.state"][gripper_idx]) <= 100.0
