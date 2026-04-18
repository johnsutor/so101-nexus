"""Consolidated Gymnasium tests for every ManiSkill SO101-Nexus environment.

Replaces the per-task test files (test_pick_env.py, test_reach_env.py, ...)
with a single parametric suite backed by the shared
``so101_nexus_core.testing.run_env_contract`` helper. Backend-specific
assertions that aren't part of the shared contract live at the bottom
of this file.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
from mani_skill import ASSET_DIR

import so101_nexus_maniskill  # noqa: F401 — registers envs
from so101_nexus_core.config import (
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    ReachConfig,
)
from so101_nexus_core.constants import CUBE_COLOR_MAP, YCB_OBJECTS
from so101_nexus_core.objects import CubeObject, YCBObject
from so101_nexus_core.observations import (
    EndEffectorPose,
    GazeDirection,
    JointPositions,
    OverheadCamera,
    TargetOffset,
    WristCamera,
)
from so101_nexus_core.testing import run_env_contract
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


# ---------------------------------------------------------------------------
# Shared contract.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id,config_cls", ENV_MATRIX)
def test_gymnasium_contract(env_id, config_cls):
    run_env_contract(env_id, config_cls, make_kwargs=BASE_KWARGS)


# ---------------------------------------------------------------------------
# Object parametrics.
# ---------------------------------------------------------------------------


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
    """PickLift with a pool of cubes and distractors spawns correctly."""
    objects: list[CubeObject] = [
        CubeObject(color="red"),
        CubeObject(color="blue"),
        CubeObject(color="green"),
    ]
    config = PickConfig(objects=objects, n_distractors=2)  # type: ignore[arg-type]
    env = gym.make(env_id, config=config, **BASE_KWARGS)
    try:
        env.reset()
        _run_episode(env)
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


# ---------------------------------------------------------------------------
# Action / observation space smoke.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_action_space_shape(env_id):
    """All environments have a 6-DOF action space."""
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        assert env.action_space.shape == (6,)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_observation_in_space(env_id):
    """Observations returned by reset lie in the observation space."""
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        obs, _ = env.reset()
        assert env.observation_space.contains(obs)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_multiple_resets(env_id):
    """Each environment can be reset multiple times without errors."""
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        for _ in range(5):
            obs, _ = env.reset()
            assert obs is not None
            env.step(env.action_space.sample())
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_reward_bounds(env_id):
    """Reward stays in [0, 1] over multiple random steps."""
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        env.reset()
        for _ in range(10):
            _, reward, terminated, _, _ = env.step(env.action_space.sample())
            assert (reward >= 0).all(), f"Reward below 0 for {env_id}"
            assert (reward <= 1).all(), f"Reward above 1 for {env_id}"
            if terminated.any() if isinstance(terminated, torch.Tensor) else terminated:
                env.reset()
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Per-env evaluate() / info key assertions.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Task-description shape checks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_task_description_starts_with_capital(env_id):
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        desc = env.unwrapped.task_description
        assert desc
        assert desc[0].isupper()
    finally:
        env.close()


def test_pick_and_place_task_description_includes_colors():
    env = gym.make(
        "ManiSkillPickAndPlaceSO100-v1",
        config=PickAndPlaceConfig(cube_colors="green", target_colors="blue"),
        **BASE_KWARGS,
    )
    try:
        desc = env.unwrapped.task_description
        assert "green" in desc
        assert "blue" in desc
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Observation-component coverage.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Camera observation integration.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Construction / config validation.
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scene / geometry regression guards.
# ---------------------------------------------------------------------------


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
    cfg = PickAndPlaceConfig()
    env = gym.make(env_id, **BASE_KWARGS)
    try:
        for _ in range(5):
            env.reset()
            cube_xy = env.unwrapped.obj.pose.p[0, :2].cpu()
            target_xy = env.unwrapped.target_site.pose.p[0, :2].cpu()
            dist = torch.linalg.norm(cube_xy - target_xy).item()
            assert dist >= cfg.min_cube_target_separation - 1e-4
    finally:
        env.close()


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


# ---------------------------------------------------------------------------
# Robot subclass / scene structure checks.
# ---------------------------------------------------------------------------


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
