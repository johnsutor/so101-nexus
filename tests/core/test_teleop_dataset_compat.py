"""End-to-end LeRobot compatibility test for the Gradio teleop recorder."""

from __future__ import annotations

import os
import threading

import numpy as np
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


def _record_episode(tmp_path, *, env_id: str, max_steps: int, robot_id: str):
    """Drive the recording thread against *env_id* and return the populated state."""
    from so101_nexus.config import SO101_JOINT_NAMES
    from so101_nexus.teleop.recorder import RecordingState, recording_thread

    state = RecordingState(num_episodes=1)
    leader = _FakeLeader()
    leader.connect()
    thread = threading.Thread(
        target=recording_thread,
        kwargs={
            "state": state,
            "env_id": env_id,
            "leader": leader,
            "joint_names": SO101_JOINT_NAMES,
            "fps": 30,
            "max_steps": max_steps,
            "countdown": 0,
            "wrist_roll_offset_deg": -90.0,
            "wrist_wh": (160, 120),
            "overhead_wh": (160, 120),
            "follower_calibration_dir": tmp_path / "cal",
            "follower_robot_id": robot_id,
        },
        daemon=True,
    )
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive()
    assert state.error is None, state.error
    assert len(state.episode_actions) == max_steps
    return state


@pytest.mark.slow
def test_gradio_recording_reloads_as_lerobot_dataset(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("so101_nexus.mujoco")

    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_nexus.config import SO101_JOINT_NAMES
    from so101_nexus.observations import privileged_state_feature_names
    from so101_nexus.teleop.dataset import (
        OVERHEAD_KEY,
        REWARD_COMPONENT_FEATURE_KEYS,
        REWARD_KEY,
        WRIST_KEY,
        FieldSelection,
        _make_reward_scalar_dataset_cls,
        build_features,
        build_frame,
    )
    from so101_nexus.teleop.session import resolve_recording_observations

    state = _record_episode(
        tmp_path, env_id="MuJoCoTouch-v1", max_steps=5, robot_id="teleop_sim_smoke"
    )

    selection = FieldSelection()
    action_features = {f"{name}.pos": float for name in SO101_JOINT_NAMES}
    follower_features = {**action_features, "wrist": (120, 160, 3), "overhead": (120, 160, 3)}
    # FieldSelection() defaults environment_state=True, so the privileged low-dim
    # state must be declared (via names resolved from the SAME recording config the
    # follower ran) and supplied to every frame; otherwise build_frame raises.
    env_state_names = privileged_state_feature_names(
        resolve_recording_observations("MuJoCoTouch-v1", (160, 120), (160, 120))
    )
    features = build_features(
        selection, follower_features, action_features, env_state_names=env_state_names
    )

    dataset_root = tmp_path / "dataset"
    dataset = _make_reward_scalar_dataset_cls().create(
        repo_id="local/teleop-smoke",
        fps=30,
        features=features,
        robot_type="sim_so_follower",
        root=dataset_root,
        use_videos=True,
    )

    for i in range(len(state.episode_actions)):
        reward = state.episode_rewards[i] if i < len(state.episode_rewards) else 0.0
        components = (
            state.episode_reward_components[i] if i < len(state.episode_reward_components) else None
        )
        frame = build_frame(
            selection,
            state=state.episode_states[i],
            action=state.episode_actions[i],
            task="reach the target",
            reward=reward,
            reward_components=components,
            env_state=state.episode_env_states[i],
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
        REWARD_KEY,
        WRIST_KEY,
        OVERHEAD_KEY,
        *REWARD_COMPONENT_FEATURE_KEYS,
    }
    assert reloaded.features[REWARD_KEY]["shape"] == (1,)
    assert reloaded.features[REWARD_KEY]["dtype"] == "float32"
    assert reloaded.features["action"]["shape"] == (len(SO101_JOINT_NAMES),)
    assert reloaded.features["observation.state"]["shape"] == (len(SO101_JOINT_NAMES),)
    assert reloaded.features["action"]["names"][0] == "shoulder_pan.pos"
    assert reloaded.features[WRIST_KEY]["dtype"] == "video"
    assert reloaded.features[OVERHEAD_KEY]["dtype"] == "video"
    assert reloaded.fps == 30
    assert len(reloaded) == 5
    assert list(reloaded.meta.tasks.index) == ["reach the target"]

    sample = reloaded[0]
    assert sample["action"].shape == (len(SO101_JOINT_NAMES),)
    # Reward is a (1,) feature in the schema (required by `validate_frame`),
    # which LeRobot maps to a scalar HF Value (like `timestamp`), so it reads
    # back as a 0-d tensor holding the per-step transition reward.
    assert sample[REWARD_KEY].dim() == 0
    assert torch.isfinite(sample[REWARD_KEY]).all()
    assert float(sample[REWARD_KEY]) == pytest.approx(state.episode_rewards[0])

    # LeRobot computes per-feature min/max/mean/std/quantiles in stats.json
    # automatically (compute_episode_stats -> aggregate_stats), so reward
    # normalization bounds are available downstream via dataset.meta.stats["reward"].
    reward_stats = reloaded.meta.stats[REWARD_KEY]
    assert {"min", "max", "mean", "std"}.issubset(reward_stats)
    reward_min = float(np.asarray(reward_stats["min"]).reshape(-1)[0])
    reward_max = float(np.asarray(reward_stats["max"]).reshape(-1)[0])
    assert reward_min <= float(sample[REWARD_KEY]) <= reward_max

    # Each reward facet round-trips as its own 0-d scalar and sums back to the
    # recorded total (TouchEnv's simple_reward puts its sole progress term in
    # "reaching"; "grasping"/"task_objective" are pinned at zero).
    component_total = 0.0
    for name in REWARD_COMPONENT_FEATURE_KEYS:
        assert sample[name].dim() == 0
        component_total += float(sample[name])
    assert component_total == pytest.approx(float(sample[REWARD_KEY]), abs=1e-5)
    np.testing.assert_allclose(
        float(sample["reward_components.reaching"]),
        state.episode_reward_components[0]["reaching"],
        atol=1e-5,
    )


@pytest.mark.slow
def test_camera_free_recording_reloads_env_state_success_done(tmp_path) -> None:
    pytest.importorskip("mujoco")
    pytest.importorskip("so101_nexus.mujoco")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101_nexus.config import SO101_JOINT_NAMES
    from so101_nexus.observations import privileged_state_feature_names
    from so101_nexus.teleop.dataset import (
        DONE_KEY,
        ENV_STATE_KEY,
        SUCCESS_KEY,
        FieldSelection,
        _make_reward_scalar_dataset_cls,
        build_features,
        build_frame,
    )
    from so101_nexus.teleop.session import resolve_recording_observations

    env_id = "MuJoCoPickLift-v1"
    max_steps = 6
    state = _record_episode(
        tmp_path, env_id=env_id, max_steps=max_steps, robot_id="teleop_envstate_smoke"
    )
    # The privileged low-dim state, success, and done are recorded once per step,
    # aligned with the action buffer.
    assert len(state.episode_env_states) == max_steps
    assert len(state.episode_successes) == max_steps
    assert len(state.episode_dones) == max_steps

    # Camera-free selection: no wrist/overhead video, so `finalize` never invokes
    # ffmpeg. This isolates the env_state/success/done channels from video encoding
    # (which can hang in some environments), while still exercising the real
    # LeRobot write + reload path via `_make_reward_scalar_dataset_cls`.
    selection = FieldSelection(
        wrist_image=False,
        overhead_image=False,
        environment_state=True,
        task=True,
    )
    action_features = {f"{name}.pos": float for name in SO101_JOINT_NAMES}
    env_state_names = privileged_state_feature_names(
        resolve_recording_observations(env_id, (160, 120), (160, 120))
    )
    assert env_state_names, "PickLift env must expose privileged state names"
    assert env_state_names[0] == "joint_positions_0"
    n = len(env_state_names)

    # No cameras selected, so the follower features reduce to the joint floats,
    # yielding observation.state + action + the always-on scalar channels.
    features = build_features(
        selection, action_features, action_features, env_state_names=env_state_names
    )

    dataset_root = tmp_path / "dataset"
    dataset = _make_reward_scalar_dataset_cls().create(
        repo_id="local/teleop-envstate",
        fps=30,
        features=features,
        robot_type="sim_so_follower",
        root=dataset_root,
        use_videos=False,
    )

    for i in range(len(state.episode_actions)):
        reward = state.episode_rewards[i] if i < len(state.episode_rewards) else 0.0
        components = (
            state.episode_reward_components[i] if i < len(state.episode_reward_components) else None
        )
        frame = build_frame(
            selection,
            state=state.episode_states[i],
            action=state.episode_actions[i],
            task="pick and lift the object",
            reward=reward,
            reward_components=components,
            success=state.episode_successes[i],
            done=state.episode_dones[i],
            env_state=state.episode_env_states[i],
            wrist_image=None,
            overhead_image=None,
        )
        dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()

    reloaded = LeRobotDataset("local/teleop-envstate", root=dataset_root)

    # Privileged state reloads as a self-describing (n,) float32 vector, with the
    # per-dimension names carried through from privileged_state_feature_names.
    assert reloaded.features[ENV_STATE_KEY]["shape"] == (n,)
    assert reloaded.features[ENV_STATE_KEY]["dtype"] == "float32"
    assert reloaded.features[ENV_STATE_KEY]["names"][0] == "joint_positions_0"
    # success/done are always-on scalar env-step channels, declared like reward.
    for key in (SUCCESS_KEY, DONE_KEY):
        assert reloaded.features[key]["shape"] == (1,)
        assert reloaded.features[key]["dtype"] == "float32"

    assert len(reloaded) == max_steps

    sample = reloaded[0]
    # env_state reloads as an (n,) vector holding the exact recorded frame-0 state.
    assert sample[ENV_STATE_KEY].shape == (n,)
    np.testing.assert_allclose(
        sample[ENV_STATE_KEY].numpy(),
        state.episode_env_states[0],
        rtol=1e-5,
        atol=1e-6,
    )
    # Like reward, the (1,) success/done features map to scalar HF Values, so they
    # read back as 0-d tensors (one per frame), not (1,) vectors.
    assert sample[SUCCESS_KEY].dim() == 0
    assert sample[DONE_KEY].dim() == 0
    assert float(sample[SUCCESS_KEY]) == pytest.approx(state.episode_successes[0])
    assert float(sample[DONE_KEY]) == pytest.approx(state.episode_dones[0])
    # PickEnv's multi-objective breakdown records a live "grasping" facet
    # (unlike TouchEnv/MoveEnv/LookAtEnv, where it is pinned at zero).
    assert sample["reward_components.grasping"].dim() == 0
    assert float(sample["reward_components.grasping"]) == pytest.approx(
        state.episode_reward_components[0]["grasping"], abs=1e-5
    )

    # LeRobot's automatic per-feature stats aggregation covers every recorded
    # channel, so normalization bounds exist downstream for the new channels too.
    for key in (ENV_STATE_KEY, SUCCESS_KEY, DONE_KEY):
        assert key in reloaded.meta.stats
