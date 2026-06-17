"""Consolidated Gymnasium tests for every ManiSkill SO101-Nexus environment.

Replaces the per-task test files (test_pick_env.py, test_reach_env.py, ...)
with a single parametric suite backed by the shared
``so101_nexus_core.testing.run_env_contract`` helper. Backend-specific
assertions that aren't part of the shared contract live at the bottom
of this file.
"""

from __future__ import annotations

import importlib
import math

import gymnasium as gym
import numpy as np
import pytest
import torch
from mani_skill import ASSET_DIR

import so101_nexus_maniskill  # noqa: F401 - registers envs
from so101_nexus_core.config import (
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    ReachConfig,
    RobotConfig,
)
from so101_nexus_core.constants import CUBE_COLOR_MAP, YCB_OBJECTS
from so101_nexus_core.objects import CubeObject, YCBObject
from so101_nexus_core.observations import (
    EndEffectorPose,
    GazeDirection,
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
)
from so101_nexus_core.testing import (
    run_env_contract,
    skip_if_vectorized_runtime_unavailable,
)
from so101_nexus_maniskill.look_at_env import LookAtEnv
from so101_nexus_maniskill.pick_and_place import (
    PickAndPlaceSO100Env,
    PickAndPlaceSO101Env,
)
from so101_nexus_maniskill.pick_env import (
    PickLiftSO100Env,
    PickLiftSO101Env,
)

BASE_KWARGS = {"obs_mode": "state", "num_envs": 1, "render_mode": None}

REACH_ENV_IDS = ["ManiSkillReachSO100-v1", "ManiSkillReachSO101-v1"]
LOOKAT_ENV_IDS = ["ManiSkillLookAtSO100-v1", "ManiSkillLookAtSO101-v1"]
MOVE_ENV_IDS = ["ManiSkillMoveSO100-v1", "ManiSkillMoveSO101-v1"]
PICK_LIFT_ENV_IDS = ["ManiSkillPickLiftSO100-v1", "ManiSkillPickLiftSO101-v1"]
PICK_AND_PLACE_ENV_IDS = [
    "ManiSkillPickAndPlaceSO100-v1",
    "ManiSkillPickAndPlaceSO101-v1",
]

ENV_MATRIX: list[tuple[str, type]] = (
    [(e, ReachConfig) for e in REACH_ENV_IDS]
    + [(e, LookAtConfig) for e in LOOKAT_ENV_IDS]
    + [(e, MoveConfig) for e in MOVE_ENV_IDS]
    + [(e, PickConfig) for e in PICK_LIFT_ENV_IDS]
    + [(e, PickAndPlaceConfig) for e in PICK_AND_PLACE_ENV_IDS]
)
ENV_IDS = [e for e, _ in ENV_MATRIX]

CUBE_COLORS = list(CUBE_COLOR_MAP.keys())
YCB_MODEL_IDS = list(YCB_OBJECTS.keys())
MOVE_DIRECTIONS = ["up", "down", "left", "right", "forward", "backward"]

N_STEPS = 3
RESET_SETTLE_ENV_MATRIX = [
    ("ManiSkillReachSO101-v1", ReachConfig),
    ("ManiSkillLookAtSO101-v1", LookAtConfig),
    ("ManiSkillMoveSO101-v1", MoveConfig),
    ("ManiSkillPickLiftSO101-v1", PickConfig),
    ("ManiSkillPickAndPlaceSO101-v1", PickAndPlaceConfig),
]


def _has_ycb_assets() -> bool:
    """ManiSkill YCB manifest must be downloaded before pick tests can run."""
    return (ASSET_DIR / "assets" / "mani_skill2_ycb" / "info_pick_v0.json").exists()


def _run_episode(env, n_steps: int = N_STEPS):
    """Reset env, take n_steps random actions, and return final (obs, info)."""
    obs, info = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward is not None
    return obs, info


def _make_vector_env_or_skip(env_id: str, **kwargs):
    """Build a vectorized env, skipping only on GPU/runtime-availability errors.

    Delegates to the shared ``skip_if_vectorized_runtime_unavailable`` helper so
    genuine construction errors (a bad link name, a failed patch) surface as
    failures instead of being masked as skips.
    """
    try:
        return gym.make(env_id, **kwargs)
    except Exception as exc:
        skip_if_vectorized_runtime_unavailable(exc)


@pytest.mark.parametrize("env_id,config_cls", ENV_MATRIX)
def test_gymnasium_contract(env_id, config_cls):
    del config_cls  # parametrized for symmetry with other matrix tests.
    run_env_contract(env_id, make_kwargs=BASE_KWARGS)


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
@pytest.mark.parametrize("color", CUBE_COLORS)
def test_pick_cube_color(env_id, color):
    config = PickConfig(objects=[CubeObject(color=color)])  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        assert color in env.unwrapped.task_description
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.skipif(not _has_ycb_assets(), reason="ManiSkill YCB assets not present")
@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
@pytest.mark.parametrize("model_id", YCB_MODEL_IDS)
def test_pick_ycb_object(env_id, model_id):
    config = PickConfig(objects=[YCBObject(model_id=model_id)])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        assert YCB_OBJECTS[model_id] in env.unwrapped.task_description
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_multiple_cubes_with_distractors(env_id):
    """PickLift with a pool of cubes and distractors spawns correctly.

    Also verifies distractors are sampled without replacement: with a 3-cube
    pool and 2 distractors, every spawned object identity (target + distractors)
    is distinct.
    """
    objects: list[CubeObject] = [
        CubeObject(color="red"),
        CubeObject(color="blue"),
        CubeObject(color="green"),
    ]
    config = PickConfig(objects=objects, n_distractors=2)  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        inner = env.unwrapped
        colors = [inner._target_obj.color] + [d.color for d in inner._distractors_spec]
        assert len(set(colors)) == len(colors), f"duplicate distractor identities: {colors}"
        _run_episode(env)
    finally:
        env.close()


def _spawned_identities(inner) -> tuple[str, tuple[str, ...]]:
    """Return (target_color, (distractor_colors,)) for the current episode."""
    return inner._target_obj.color, tuple(d.color for d in inner._distractors_spec)


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_identities_reproducible_by_seed(env_id):
    """reset(seed=S) reproduces the same target/distractor identities."""
    objects = [
        CubeObject(color="red"),
        CubeObject(color="blue"),
        CubeObject(color="green"),
    ]
    config = PickConfig(objects=objects, n_distractors=1)  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        inner = env.unwrapped
        env.reset(seed=123)
        first = _spawned_identities(inner)
        env.reset(seed=123)
        second = _spawned_identities(inner)
        assert first == second, f"seed=123 not reproducible: {first} vs {second}"

        # A different seed is allowed to differ; scan a few to show identities
        # actually vary across seeds (variable-object pool reconfigures).
        seen = set()
        for s in range(8):
            env.reset(seed=s)
            seen.add(_spawned_identities(inner))
        assert len(seen) > 1, f"identities never varied across seeds: {seen}"
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_distractors_unique_when_pool_allows(env_id):
    """Distractors are unique (no duplicates) when the pool is large enough."""
    objects = [
        CubeObject(color="red"),
        CubeObject(color="blue"),
        CubeObject(color="green"),
        CubeObject(color="yellow"),
    ]
    config = PickConfig(objects=objects, n_distractors=3)  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        inner = env.unwrapped
        for s in range(6):
            env.reset(seed=s)
            target, distractors = _spawned_identities(inner)
            all_ids = [target, *distractors]
            assert len(set(all_ids)) == len(all_ids), f"duplicate identity: {all_ids}"
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_object_separation_respected(env_id):
    """Spawned objects satisfy min_object_separation + bounding radii (num_envs=1)."""
    from so101_nexus_maniskill.pick_env import _obj_bounding_radius

    objects = [
        CubeObject(color="red"),
        CubeObject(color="blue"),
        CubeObject(color="green"),
    ]
    min_sep = 0.04
    config = PickConfig(  # type: ignore[arg-type]
        objects=objects, n_distractors=2, min_object_separation=min_sep
    )
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        inner = env.unwrapped
        for s in range(6):
            env.reset(seed=s)
            radii = [_obj_bounding_radius(inner._target_obj)] + [
                _obj_bounding_radius(d) for d in inner._distractors_spec
            ]
            xys = [inner.obj.pose.p[0, :2].cpu()] + [
                d.pose.p[0, :2].cpu() for d in inner.distractors
            ]
            n = len(xys)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = float(torch.linalg.norm(xys[i] - xys[j]))
                    required = min_sep + radii[i] + radii[j]
                    # Settling can nudge objects slightly; allow a small tolerance.
                    assert dist >= required - 0.02, (
                        f"objects {i},{j} too close: {dist} < {required}"
                    )
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_separation_batched_vec(env_id):
    """GPU-gated: batched per-row separation holds for num_envs>1."""
    from so101_nexus_maniskill.pick_env import _obj_bounding_radius

    objects = [
        CubeObject(color="red"),
        CubeObject(color="blue"),
        CubeObject(color="green"),
    ]
    min_sep = 0.04
    config = PickConfig(  # type: ignore[arg-type]
        objects=objects, n_distractors=2, min_object_separation=min_sep
    )
    try:
        env = gym.make(env_id, config=config, obs_mode="state", num_envs=2, render_mode=None)
    except Exception as exc:  # narrowed: only GPU-availability errors become skips
        skip_if_vectorized_runtime_unavailable(exc)
    try:
        inner = env.unwrapped
        env.reset(seed=0)
        radii = [_obj_bounding_radius(inner._target_obj)] + [
            _obj_bounding_radius(d) for d in inner._distractors_spec
        ]
        xys = [inner.obj.pose.p[:, :2].cpu()] + [d.pose.p[:, :2].cpu() for d in inner.distractors]
        n = len(xys)
        for i in range(n):
            for j in range(i + 1, n):
                dist = torch.linalg.norm(xys[i] - xys[j], dim=1)
                required = min_sep + radii[i] + radii[j]
                assert bool((dist >= required - 0.02).all()), (
                    f"objects {i},{j} too close in some env row"
                )
    finally:
        env.close()


@pytest.mark.parametrize("env_id", LOOKAT_ENV_IDS)
@pytest.mark.parametrize("color", CUBE_COLORS)
def test_look_at_cube_color(env_id, color):
    config = LookAtConfig(objects=[CubeObject(color=color)])  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        assert color in env.unwrapped.task_description
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", MOVE_ENV_IDS)
@pytest.mark.parametrize("direction", MOVE_DIRECTIONS)
def test_move_direction(env_id, direction):
    config = MoveConfig(direction=direction)  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        assert direction in env.unwrapped.task_description
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
@pytest.mark.parametrize("cube_color", CUBE_COLORS)
def test_pick_and_place_cube_colors(env_id, cube_color):
    target_color = "blue" if cube_color != "blue" else "red"
    config = PickAndPlaceConfig(cube_colors=cube_color, target_colors=target_color)
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
@pytest.mark.parametrize("target_color", CUBE_COLORS)
def test_pick_and_place_target_colors(env_id, target_color):
    """PickAndPlace works with every target disc colour."""
    cube_color = "red" if target_color != "red" else "blue"
    config = PickAndPlaceConfig(cube_colors=cube_color, target_colors=target_color)
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_default_state_rollout_smoke(env_id):
    """Default state-mode envs expose spaces, reset repeatedly, and keep bounded rewards."""
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        assert env.action_space.shape == (6,)
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

        for _ in range(5):
            obs, _ = env.reset()
            assert obs is not None
            env.step(env.action_space.sample())

        env.reset()
        for _ in range(10):
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            assert (reward >= 0).all(), f"Reward below 0 for {env_id}"
            assert (reward <= 1).all(), f"Reward above 1 for {env_id}"
            if terminated.any() if isinstance(terminated, torch.Tensor) else terminated:
                env.reset()
    finally:
        env.close()


@pytest.mark.parametrize("env_id,config_cls", RESET_SETTLE_ENV_MATRIX)
@pytest.mark.parametrize("reset_settle_frames", [0, 2])
def test_reset_settle_configs_return_observation_in_space(
    env_id: str,
    config_cls: type,
    reset_settle_frames: int,
):
    config = config_cls(reset_settle_frames=reset_settle_frames)
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
    finally:
        env.close()


@pytest.mark.parametrize("env_id,config_cls", RESET_SETTLE_ENV_MATRIX)
def test_reset_settle_frames_step_scene_during_reset(monkeypatch, env_id: str, config_cls: type):
    config = config_cls(reset_settle_frames=2)
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        inner = env.unwrapped
        step_count = 0
        original_step = inner.scene.step

        def counted_step(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        monkeypatch.setattr(inner.scene, "step", counted_step)
        env.reset()

        assert step_count == 2
    finally:
        env.close()


def test_move_reset_settle_zero_preserves_one_kinematics_step(monkeypatch):
    config = MoveConfig(direction="right", reset_settle_frames=0)
    env = gym.make("ManiSkillMoveSO101-v1", config=config, **BASE_KWARGS)
    try:
        env.reset()
        inner = env.unwrapped
        step_count = 0
        original_step = inner.scene.step

        def counted_step(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            return original_step(*args, **kwargs)

        monkeypatch.setattr(inner.scene, "step", counted_step)
        env.reset()

        tcp_to_target = inner.target_site.pose.p - inner.agent.tcp_pose.p
        dist = torch.linalg.norm(tcp_to_target, dim=1)
        assert step_count == 1
        assert dist[0].item() == pytest.approx(config.target_distance, rel=0.05)
    finally:
        env.close()


@pytest.mark.parametrize(
    "env_id,config",
    [
        ("ManiSkillPickLiftSO101-v1", PickConfig(reset_settle_frames=2)),
        ("ManiSkillPickAndPlaceSO101-v1", PickAndPlaceConfig(reset_settle_frames=2)),
    ],
)
def test_initial_object_z_matches_post_settle_actor_pose(env_id: str, config):
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset(seed=0)
        inner = env.unwrapped
        assert inner._initial_obj_z[0].item() == pytest.approx(inner.obj.pose.p[0, 2].item())
    finally:
        env.close()


def test_partial_reset_settle_preserves_inactive_env_state():
    env = _make_vector_env_or_skip(
        "ManiSkillReachSO101-v1",
        config=ReachConfig(reset_settle_frames=2),
        obs_mode="state",
        num_envs=2,
        render_mode=None,
    )
    try:
        env.reset()
        inner = env.unwrapped
        state_before = inner.get_state().clone()
        env_idx = torch.tensor([0], device=inner.device)

        env.reset(options={"env_idx": env_idx})

        state_after = inner.get_state()
        torch.testing.assert_close(state_after[1], state_before[1])
    finally:
        env.close()


@pytest.mark.parametrize("env_id", REACH_ENV_IDS)
def test_reach_evaluate_keys(env_id):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        info = env.unwrapped.evaluate()
        assert {"tcp_to_target_dist", "success"} <= set(info.keys())
    finally:
        env.close()


@pytest.mark.parametrize("env_id", LOOKAT_ENV_IDS)
def test_lookat_evaluate_keys(env_id):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        info = env.unwrapped.evaluate()
        assert {"orientation_error", "success"} <= set(info.keys())
    finally:
        env.close()


def test_lookat_dense_reward_uses_orientation_error_info():
    """Reward must use evaluate() info, not recompute orientation vectors."""
    env = object.__new__(LookAtEnv)
    env.config = LookAtConfig()
    bonus = env.config.reward.completion_bonus
    # get_reward stamps these norms into info before compute_dense_reward runs;
    # the reward path reads them rather than recomputing, so supply them here.
    info = {
        "orientation_error": torch.tensor([0.0, torch.pi], dtype=torch.float32),
        "success": torch.tensor([False, True]),
        "action_delta_norm": torch.zeros(2, dtype=torch.float32),
        "energy_norm": torch.zeros(2, dtype=torch.float32),
    }

    reward = LookAtEnv.compute_dense_reward(
        env,
        obs={},
        action=torch.empty((2, 0), dtype=torch.float32),
        info=info,
    )

    expected = torch.tensor([1.0 - bonus, bonus], dtype=torch.float32)
    torch.testing.assert_close(reward, expected)


@pytest.mark.parametrize("env_id", MOVE_ENV_IDS)
def test_move_evaluate_keys(env_id):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        info = env.unwrapped.evaluate()
        assert {"tcp_to_target_dist", "success"} <= set(info.keys())
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_lift_evaluate_keys_exact(env_id):
    """PickLift evaluate() returns the documented full key set."""
    expected = {"is_grasped", "lift_height", "success", "tcp_to_obj_dist"}
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        info = env.unwrapped.evaluate()
        assert set(info.keys()) == expected
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_evaluate_keys_exact(env_id):
    """PickAndPlace evaluate() returns the documented full key set."""
    expected = {
        "obj_to_target_dist",
        "is_obj_placed",
        "is_grasped",
        "is_robot_static",
        "lift_height",
        "success",
        "tcp_to_obj_dist",
    }
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        info = env.unwrapped.evaluate()
        assert set(info.keys()) == expected
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_task_description_starts_with_capital(env_id):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        desc = env.unwrapped.task_description
        assert desc
        assert desc[0].isupper()
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_task_description_includes_colors(env_id):
    env = gym.make(
        env_id,
        config=PickAndPlaceConfig(cube_colors="green", target_colors="blue"),
        **BASE_KWARGS,
    )
    try:
        desc = env.unwrapped.task_description
        assert "green" in desc
        assert "blue" in desc
    finally:
        env.close()


_reach_obs_params = [
    (JointPositions, "joint_positions"),
    (EndEffectorPose, "tcp_pose"),
    (TargetOffset, "target_offset"),
]


@pytest.mark.parametrize("env_id", REACH_ENV_IDS)
@pytest.mark.parametrize(
    "obs_cls,expected_key",
    _reach_obs_params,
    ids=[cls.__name__ for cls, _ in _reach_obs_params],
)
def test_reach_observation_component(env_id, obs_cls, expected_key):
    """Each valid observation component works on Reach envs."""
    config = ReachConfig(observations=[JointPositions(), obs_cls()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        if obs_cls is not JointPositions:
            extra = env.unwrapped._get_obs_extra(info)
            assert expected_key in extra
        _run_episode(env)
    finally:
        env.close()


_lookat_obs_params = [
    (JointPositions, "joint_positions"),
    (EndEffectorPose, "tcp_pose"),
    (GazeDirection, "gaze_direction"),
]


@pytest.mark.parametrize("env_id", LOOKAT_ENV_IDS)
@pytest.mark.parametrize(
    "obs_cls,expected_key",
    _lookat_obs_params,
    ids=[cls.__name__ for cls, _ in _lookat_obs_params],
)
def test_lookat_observation_component(env_id, obs_cls, expected_key):
    """Each valid observation component works on LookAt envs."""
    config = LookAtConfig(observations=[JointPositions(), obs_cls()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        if obs_cls is not JointPositions:
            extra = env.unwrapped._get_obs_extra(info)
            assert expected_key in extra
        _run_episode(env)
    finally:
        env.close()


_move_obs_params = [
    (JointPositions, "joint_positions"),
    (EndEffectorPose, "tcp_pose"),
    (TargetOffset, "target_offset"),
]


@pytest.mark.parametrize("env_id", MOVE_ENV_IDS)
@pytest.mark.parametrize(
    "obs_cls,expected_key",
    _move_obs_params,
    ids=[cls.__name__ for cls, _ in _move_obs_params],
)
def test_move_observation_component(env_id, obs_cls, expected_key):
    """Each valid observation component works on Move envs."""
    config = MoveConfig(observations=[JointPositions(), obs_cls()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        if obs_cls is not JointPositions:
            extra = env.unwrapped._get_obs_extra(info)
            assert expected_key in extra
        _run_episode(env)
    finally:
        env.close()


_pick_obs_params = [
    (EndEffectorPose, "tcp_pose"),
    (GraspState, "is_grasped"),
    (ObjectPose, "object_pose"),
    (ObjectOffset, "object_offset"),
]


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
@pytest.mark.parametrize(
    "obs_cls,expected_key",
    _pick_obs_params,
    ids=[cls.__name__ for cls, _ in _pick_obs_params],
)
def test_pick_observation_component(env_id, obs_cls, expected_key):
    """Each valid Pick observation component appears in obs_extra when requested."""
    config = PickConfig(observations=[JointPositions(), obs_cls()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        extra = env.unwrapped._get_obs_extra(info)
        assert expected_key in extra
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_obs_extra_honors_config_observations(env_id):
    """Pick obs_extra is component-driven, not a fixed object-state leak.

    Regression: the old _get_obs_extra emitted obj_pose/tcp regardless of
    config.observations. With observations=[JointPositions()] no object
    components are requested, so obs_extra must contain no object state.
    """
    config = PickConfig(observations=[JointPositions()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        extra = env.unwrapped._get_obs_extra(info)
        assert "object_pose" not in extra
        assert "object_offset" not in extra
        assert "tcp_pose" not in extra
        assert "is_grasped" not in extra

        # With object components present, they ARE included.
        config2 = PickConfig(observations=[JointPositions(), ObjectPose(), ObjectOffset()])
        env2 = gym.make(env_id, config=config2, **BASE_KWARGS)
        try:
            env2.reset()
            _, _, _, _, info2 = env2.step(env2.action_space.sample())
            extra2 = env2.unwrapped._get_obs_extra(info2)
            assert "object_pose" in extra2
            assert "object_offset" in extra2
        finally:
            env2.close()
    finally:
        env.close()


_pick_and_place_obs_params = [
    (EndEffectorPose, "tcp_pose"),
    (GraspState, "is_grasped"),
    (ObjectPose, "object_pose"),
    (ObjectOffset, "object_offset"),
    (TargetPosition, "target_position"),
    (TargetOffset, "target_offset"),
]


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
@pytest.mark.parametrize(
    "obs_cls,expected_key",
    _pick_and_place_obs_params,
    ids=[cls.__name__ for cls, _ in _pick_and_place_obs_params],
)
def test_pick_and_place_observation_component(env_id, obs_cls, expected_key):
    """Each valid PickAndPlace observation component appears in obs_extra."""
    config = PickAndPlaceConfig(observations=[JointPositions(), obs_cls()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        extra = env.unwrapped._get_obs_extra(info)
        assert expected_key in extra
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_obs_extra_honors_config_observations(env_id):
    """PickAndPlace obs_extra is component-driven, not a fixed target/object leak."""
    config = PickAndPlaceConfig(observations=[JointPositions()])
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        extra = env.unwrapped._get_obs_extra(info)
        assert "object_pose" not in extra
        assert "object_offset" not in extra
        assert "target_position" not in extra
        assert "target_offset" not in extra

        # With target components present, they ARE included.
        config2 = PickAndPlaceConfig(
            observations=[JointPositions(), TargetPosition(), TargetOffset()]
        )
        env2 = gym.make(env_id, config=config2, **BASE_KWARGS)
        try:
            env2.reset()
            _, _, _, _, info2 = env2.step(env2.action_space.sample())
            extra2 = env2.unwrapped._get_obs_extra(info2)
            assert "target_position" in extra2
            assert "target_offset" in extra2
        finally:
            env2.close()
    finally:
        env.close()


_obs_cam_env_params = [
    (REACH_ENV_IDS[0], ReachConfig),
    (LOOKAT_ENV_IDS[0], LookAtConfig),
    (MOVE_ENV_IDS[0], MoveConfig),
    (PICK_LIFT_ENV_IDS[0], PickConfig),
    (PICK_AND_PLACE_ENV_IDS[0], PickAndPlaceConfig),
]


class TestObservationDrivenCameras:
    """Camera observation components create the expected sensor configs."""

    @pytest.mark.parametrize(
        "env_id,config_cls",
        _obs_cam_env_params,
        ids=[eid for eid, _ in _obs_cam_env_params],
    )
    def test_overhead_camera_sensor(self, env_id, config_cls):
        config = config_cls(observations=[JointPositions(), OverheadCamera(width=64, height=48)])
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        try:
            sensor_names = [cfg.uid for cfg in env.unwrapped._default_sensor_configs]
            assert "overhead_camera" in sensor_names
            _run_episode(env)
        finally:
            env.close()

    @pytest.mark.parametrize(
        "env_id,config_cls",
        _obs_cam_env_params,
        ids=[eid for eid, _ in _obs_cam_env_params],
    )
    def test_wrist_camera_sensor(self, env_id, config_cls):
        config = config_cls(observations=[JointPositions(), WristCamera(width=64, height=48)])
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        try:
            sensor_names = [cfg.uid for cfg in env.unwrapped._default_sensor_configs]
            assert "wrist_camera" in sensor_names
            _run_episode(env)
        finally:
            env.close()

    @pytest.mark.parametrize(
        "env_id,config_cls",
        _obs_cam_env_params,
        ids=[eid for eid, _ in _obs_cam_env_params],
    )
    def test_both_cameras_sensor(self, env_id, config_cls):
        config = config_cls(
            observations=[
                JointPositions(),
                WristCamera(width=64, height=48),
                OverheadCamera(width=32, height=24),
            ]
        )
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        try:
            sensor_names = [cfg.uid for cfg in env.unwrapped._default_sensor_configs]
            assert "wrist_camera" in sensor_names
            assert "overhead_camera" in sensor_names
            _run_episode(env)
        finally:
            env.close()

    @pytest.mark.parametrize(
        "env_id,config_cls",
        _obs_cam_env_params,
        ids=[eid for eid, _ in _obs_cam_env_params],
    )
    def test_no_camera_no_obs_sensors(self, env_id, config_cls):
        config = config_cls(observations=[JointPositions()])
        env = gym.make(env_id, config=config, **BASE_KWARGS)
        try:
            sensor_names = [cfg.uid for cfg in env.unwrapped._default_sensor_configs]
            assert "overhead_camera" not in sensor_names
            assert "wrist_camera" not in sensor_names
            _run_episode(env)
        finally:
            env.close()


@pytest.mark.parametrize(
    "env_id",
    [
        "ManiSkillReachSO101-v1",
        "ManiSkillLookAtSO101-v1",
        "ManiSkillMoveSO101-v1",
        "ManiSkillPickLiftSO101-v1",
        "ManiSkillPickAndPlaceSO101-v1",
    ],
)
def test_actor_builders_set_initial_poses(monkeypatch, env_id):
    """Scene loading should follow ManiSkill's initial-pose guidance."""
    actor_builder = importlib.import_module("mani_skill.utils.building.actor_builder")
    warnings: list[str] = []
    monkeypatch.setattr(actor_builder.logger, "warn", warnings.append)

    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
    finally:
        env.close()

    assert not any("No initial pose set for actor builder" in w for w in warnings)


class TestConstructionValidation:
    def test_pick_invalid_robot_uid(self):
        with pytest.raises(ValueError, match="robot_uids"):
            gym.make("ManiSkillPickLiftSO100-v1", robot_uids="panda", **BASE_KWARGS)

    def test_pick_empty_objects_raises(self):
        with pytest.raises(ValueError, match="objects must not be empty"):
            PickConfig(objects=[])

    def test_pick_mesh_object_raises_not_supported(self):
        """MeshObject is not supported on the ManiSkill backend."""
        from so101_nexus_core.objects import MeshObject

        obj = MeshObject(
            collision_mesh_path="/tmp/fake.obj",
            visual_mesh_path="/tmp/fake.obj",
            mass=0.01,
            name="fake mesh",
        )
        cfg = PickConfig(objects=[obj])
        with pytest.raises(TypeError, match="Unsupported object type"):
            gym.make("ManiSkillPickLiftSO100-v1", config=cfg, **BASE_KWARGS)

    def test_pick_and_place_invalid_cube_colors(self):
        with pytest.raises(ValueError, match="cube_colors"):
            PickAndPlaceConfig(cube_colors="neon")

    def test_pick_and_place_invalid_target_colors(self):
        with pytest.raises(ValueError, match="target_colors"):
            PickAndPlaceConfig(target_colors="neon")

    def test_pick_and_place_same_cube_and_target_color_warns(self):
        with pytest.warns(UserWarning, match="overlap"):
            PickAndPlaceConfig(cube_colors="red", target_colors="red")

    def test_pick_and_place_invalid_robot_uid(self):
        with pytest.raises(ValueError, match="robot_uids"):
            gym.make("ManiSkillPickAndPlaceSO100-v1", robot_uids="panda", **BASE_KWARGS)


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS + PICK_AND_PLACE_ENV_IDS)
def test_robot_base_uses_keyframe_rotation(env_id):
    """Robot base pose must use the keyframe's Z-rotation so it faces the workspace."""
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        inner = env.unwrapped
        expected_q = inner.agent.keyframes["rest"].pose.q
        actual_q = inner.agent.robot.pose.q[0].cpu().numpy()
        np.testing.assert_allclose(actual_q, expected_q, atol=1e-4)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_lift_obj_spawns_in_radius_bounds(env_id):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        inner = env.unwrapped
        min_r = inner._robot_cfg["spawn_min_radius"]
        max_r = inner._robot_cfg["spawn_max_radius"]
        cx, cy = inner._robot_cfg["cube_spawn_center"]
        obj_p = inner.obj.pose.p[0].cpu()
        r = float(torch.sqrt((obj_p[0] - cx) ** 2 + (obj_p[1] - cy) ** 2))
        assert min_r <= r <= max_r
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_lift_spawn_center_offsets_object_positions(env_id):
    """PickLift: objects should be offset by spawn_center, not centered at origin."""
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        xs = []
        for seed in range(10):
            env.reset(seed=seed)
            obj_x = env.unwrapped.obj.pose.p[0, 0].cpu().item()
            xs.append(obj_x)
        # Mean x should be near spawn_center x=0.15, not near 0.
        assert sum(xs) / len(xs) > 0.10
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_target_on_ground(env_id):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        target_z = env.unwrapped.target_site.pose.p[0, 2].cpu().item()
        assert target_z < 0.01
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_target_visible(env_id):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        assert env.unwrapped.target_site not in env.unwrapped._hidden_objects
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_cube_target_separation(env_id):
    min_sep = PickAndPlaceConfig().min_cube_target_separation
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        for _ in range(5):
            env.reset()
            cube_xy = env.unwrapped.obj.pose.p[0, :2].cpu()
            target_xy = env.unwrapped.target_site.pose.p[0, :2].cpu()
            dist = torch.linalg.norm(cube_xy - target_xy).item()
            assert dist >= min_sep - 1e-4
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_cube_target_separation_batched_vec(env_id):
    """GPU-gated: every env row satisfies min_cube_target_separation for num_envs>1."""
    min_sep = PickAndPlaceConfig().min_cube_target_separation
    try:
        env = gym.make(env_id, obs_mode="state", num_envs=16, render_mode=None)
    except Exception as exc:  # narrowed: only GPU-availability errors become skips
        skip_if_vectorized_runtime_unavailable(exc)
    try:
        inner = env.unwrapped
        for s in range(3):
            env.reset(seed=s)
            cube_xy = inner.obj.pose.p[:, :2].cpu()
            target_xy = inner.target_site.pose.p[:, :2].cpu()
            dists = torch.linalg.norm(cube_xy - target_xy, dim=1)
            assert bool((dists >= min_sep - 1e-4).all()), (
                f"some env row violates min_cube_target_separation: min={dists.min().item()}"
            )
    finally:
        env.close()


def test_sample_cube_xy_separated_returns_valid_positions():
    """Sampler returns per-row positions that all satisfy the separation (CPU)."""
    from so101_nexus_maniskill.pick_and_place import sample_cube_xy_separated_from_target

    device = torch.device("cpu")
    min_sep = 0.1
    target_xy = torch.zeros((4, 2), device=device)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)

    out = sample_cube_xy_separated_from_target(
        target_xy=target_xy,
        min_r=0.5,
        max_r=0.6,
        angle_half=0.3,
        min_separation=min_sep,
        center=(0.0, 0.0),
        device=device,
        generator=gen,
    )
    assert out.shape == (4, 2)
    dists = torch.linalg.norm(out - target_xy, dim=1)
    assert bool((dists >= min_sep).all())


def test_sample_cube_xy_separated_only_invalid_rows_change(monkeypatch):
    """Valid rows are left untouched; only invalid rows are resampled."""
    from so101_nexus_maniskill import pick_and_place as pap

    device = torch.device("cpu")
    min_sep = 0.1
    target_xy = torch.zeros((3, 2), device=device)

    # First sampler call returns a full batch with row 0 on top of the target
    # (invalid) and rows 1, 2 far away (valid). The single resample call must
    # request exactly the one invalid row and is handed a valid position.
    requested_rows: list[int] = []

    def fake_sampler(*, rows, min_r, max_r, angle_half, center, device, generator=None):
        requested_rows.append(rows)
        if len(requested_rows) == 1:
            return torch.tensor([[0.0, 0.0], [5.0, 0.0], [6.0, 0.0]], device=device)
        return torch.full((rows, 2), 9.0, device=device)

    monkeypatch.setattr(pap, "_sample_polar_arc_xy", fake_sampler)

    out = pap.sample_cube_xy_separated_from_target(
        target_xy=target_xy,
        min_r=0.5,
        max_r=0.6,
        angle_half=0.3,
        min_separation=min_sep,
        center=(0.0, 0.0),
        device=device,
    )

    # Valid rows are untouched; the invalid row was resampled to a valid spot.
    assert torch.equal(out[1], torch.tensor([5.0, 0.0]))
    assert torch.equal(out[2], torch.tensor([6.0, 0.0]))
    assert bool((torch.linalg.norm(out - target_xy, dim=1) >= min_sep).all())
    # Exactly one resample, of exactly the single invalid row.
    assert requested_rows == [3, 1]


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_target_disc_lies_flat(env_id):
    """Target disc cylinder must be rotated so it lies flat on the ground (axis along Z)."""
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        target_q = env.unwrapped.target_site.pose.q[0].cpu().numpy()
        expected_q = np.array([0.7071068, 0.7071068, 0.0, 0.0])
        np.testing.assert_allclose(np.abs(target_q), np.abs(expected_q), atol=1e-3)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_spawn_center_offsets_cube(env_id):
    """Cube should be offset by spawn_center."""
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        cube_xs = []
        for seed in range(10):
            env.reset(seed=seed)
            cube_x = env.unwrapped.obj.pose.p[0, 0].cpu().item()
            cube_xs.append(cube_x)
        assert sum(cube_xs) / len(cube_xs) > 0.10
    finally:
        env.close()


def test_pick_lift_so100_uses_so100_agent():
    env = gym.make("ManiSkillPickLiftSO100-v1", **BASE_KWARGS)
    try:
        inner = env.unwrapped
        assert isinstance(inner, PickLiftSO100Env)
        assert inner.robot_uids == "so100"
    finally:
        env.close()


def test_pick_lift_so101_uses_so101_agent():
    env = gym.make("ManiSkillPickLiftSO101-v1", **BASE_KWARGS)
    try:
        inner = env.unwrapped
        assert isinstance(inner, PickLiftSO101Env)
        assert inner.robot_uids == "so101"
    finally:
        env.close()


def test_pick_and_place_so100_uses_so100_agent():
    env = gym.make("ManiSkillPickAndPlaceSO100-v1", **BASE_KWARGS)
    try:
        inner = env.unwrapped
        assert isinstance(inner, PickAndPlaceSO100Env)
        assert inner.robot_uids == "so100"
    finally:
        env.close()


def test_pick_and_place_so101_uses_so101_agent():
    env = gym.make("ManiSkillPickAndPlaceSO101-v1", **BASE_KWARGS)
    try:
        inner = env.unwrapped
        assert isinstance(inner, PickAndPlaceSO101Env)
        assert inner.robot_uids == "so101"
    finally:
        env.close()


def test_pick_lift_no_table_scene_builder():
    env = gym.make("ManiSkillPickLiftSO101-v1", **BASE_KWARGS)
    try:
        inner = env.unwrapped
        assert not hasattr(inner, "table_scene"), "Should not have a table_scene attribute"
    finally:
        env.close()


def test_pick_lift_robot_base_at_origin():
    env = gym.make("ManiSkillPickLiftSO101-v1", **BASE_KWARGS)
    try:
        inner = env.unwrapped
        inner.reset()
        base_pos = inner.agent.robot.pose.p[0].cpu()
        assert base_pos[0].item() == pytest.approx(0.0, abs=0.01)
        assert base_pos[1].item() == pytest.approx(0.0, abs=0.01)
        assert base_pos[2].item() == pytest.approx(0.0, abs=0.01)
    finally:
        env.close()


# Non-default thresholds, distinct from the agent defaults (min_force=0.5,
# threshold=0.2), so a captured value can only equal these if the env forwarded
# the configured RobotConfig fields rather than relying on the hardcoded default.
_GRASP_FORCE = 1.7
_STATIC_VEL = 0.0123


@pytest.mark.parametrize("env_id", PICK_LIFT_ENV_IDS)
def test_pick_lift_forwards_grasp_force_threshold(env_id):
    """PickLift.evaluate must pass RobotConfig.grasp_force_threshold to is_grasping."""
    config = PickConfig(robot=RobotConfig(grasp_force_threshold=_GRASP_FORCE))
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        inner = env.unwrapped
        inner.reset()
        captured: dict = {}

        original = inner.agent.is_grasping

        def _capture(object=None, *, min_force=None, **kwargs):
            captured["min_force"] = min_force
            return original(object, min_force=min_force, **kwargs)

        inner.agent.is_grasping = _capture
        inner.evaluate()
        assert captured["min_force"] == pytest.approx(_GRASP_FORCE)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_forwards_thresholds(env_id):
    """PickAndPlace.evaluate must forward both grasp and static RobotConfig fields."""
    config = PickAndPlaceConfig(
        robot=RobotConfig(grasp_force_threshold=_GRASP_FORCE, static_vel_threshold=_STATIC_VEL)
    )
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        inner = env.unwrapped
        inner.reset()
        captured: dict = {}

        orig_grasp = inner.agent.is_grasping
        orig_static = inner.agent.is_static

        def _capture_grasp(object=None, *, min_force=None, **kwargs):
            captured["min_force"] = min_force
            return orig_grasp(object, min_force=min_force, **kwargs)

        def _capture_static(*, threshold=None, **kwargs):
            captured["threshold"] = threshold
            return orig_static(threshold=threshold, **kwargs)

        inner.agent.is_grasping = _capture_grasp
        inner.agent.is_static = _capture_static
        inner.evaluate()
        assert captured["min_force"] == pytest.approx(_GRASP_FORCE)
        assert captured["threshold"] == pytest.approx(_STATIC_VEL)
    finally:
        env.close()


def test_sim_config_matches_mujoco_timestep(so101_reach_env):
    """ManiSkill sim/control freq match the MuJoCo 0.005 s / 0.02 s cadence."""
    sim_cfg = so101_reach_env.sim_config
    assert sim_cfg.sim_freq == 200
    assert sim_cfg.control_freq == 50
    # 200 Hz sim -> 0.005 s step; 50 Hz control -> 0.02 s interval.
    assert 1.0 / sim_cfg.sim_freq == pytest.approx(0.005)
    assert 1.0 / sim_cfg.control_freq == pytest.approx(0.02)


def test_human_render_camera_fov_is_45_deg(so101_reach_env):
    """The human render camera vertical FOV matches MuJoCo's 45 deg (in radians)."""
    cam_cfg = so101_reach_env._default_human_render_camera_configs
    assert cam_cfg.fov == pytest.approx(math.radians(45.0))


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_color_description_agreement(env_id):
    """task_description names the SAME color applied to the cube/target."""
    config = PickAndPlaceConfig(
        cube_colors=["red", "green", "yellow"],
        target_colors=["blue", "purple"],
    )
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        inner = env.unwrapped
        inner.reset(seed=7)
        desc = inner.task_description
        assert inner.cube_color_name in config.cube_colors
        assert inner.target_color_name in config.target_colors
        assert inner.cube_color_name in desc
        assert inner.target_color_name in desc
    finally:
        env.close()


@pytest.mark.parametrize("env_id", PICK_AND_PLACE_ENV_IDS)
def test_pick_and_place_color_reproducible_by_seed(env_id):
    """reset(seed=S) reproduces the same sampled cube/target colors twice."""
    config = PickAndPlaceConfig(
        cube_colors=["red", "green", "yellow"],
        target_colors=["blue", "purple", "orange"],
    )
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        inner = env.unwrapped
        inner.reset(seed=42)
        first = (inner.cube_color_name, inner.target_color_name)
        inner.reset(seed=42)
        second = (inner.cube_color_name, inner.target_color_name)
        assert first == second
    finally:
        env.close()
