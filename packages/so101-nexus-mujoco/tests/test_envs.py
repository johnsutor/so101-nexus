"""Consolidated Gymnasium tests for every MuJoCo SO101-Nexus environment.

Replaces the per-task test files (test_pick_env.py, test_reach_env.py, ...)
with a single parametric suite backed by the shared
``so101_nexus_core.testing.run_env_contract`` helper. Backend-specific
assertions that aren't part of the shared contract live at the bottom
of this file.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import pytest

import so101_nexus_mujoco  # noqa: F401 — registers envs
from so101_nexus_core.config import (
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    ReachConfig,
)
from so101_nexus_core.constants import CUBE_COLOR_MAP, YCB_OBJECTS
from so101_nexus_core.objects import CubeObject, MeshObject, YCBObject
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
from so101_nexus_core.testing import run_env_contract

ENV_MATRIX: list[tuple[str, type]] = [
    ("MuJoCoReach-v1", ReachConfig),
    ("MuJoCoLookAt-v1", LookAtConfig),
    ("MuJoCoMove-v1", MoveConfig),
    ("MuJoCoPickLift-v1", PickConfig),
    ("MuJoCoPickAndPlace-v1", PickAndPlaceConfig),
]
ENV_IDS = [e for e, _ in ENV_MATRIX]

CUBE_COLORS = list(CUBE_COLOR_MAP.keys())
YCB_MODEL_IDS = list(YCB_OBJECTS.keys())
MOVE_DIRECTIONS = ["up", "down", "left", "right", "forward", "backward"]
CONTROL_MODES = ["pd_joint_pos", "pd_joint_delta_pos", "pd_joint_target_delta_pos"]

OBS_SIZES: dict[type, int] = {
    JointPositions: 6,
    EndEffectorPose: 7,
    TargetOffset: 3,
    GazeDirection: 3,
    GraspState: 1,
    ObjectPose: 7,
    ObjectOffset: 3,
    TargetPosition: 3,
}

N_STEPS = 3


def _run_episode(env, n_steps: int = N_STEPS):
    """Reset env, take n_steps random actions, and return final (obs, info)."""
    obs, info = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
    return obs, info


# ---------------------------------------------------------------------------
# Shared contract — one parametrized call per env id.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("env_id,config_cls", ENV_MATRIX)
def test_gymnasium_contract(env_id: str, config_cls: type):
    """Every env satisfies the shared Gymnasium contract."""
    del config_cls  # parametrized for symmetry with other matrix tests.
    run_env_contract(env_id)


# ---------------------------------------------------------------------------
# Observation-component matrix (backend-generic but not in the shared contract).
# ---------------------------------------------------------------------------


_ENV_OBS_MAP: dict[str, list[type]] = {
    "MuJoCoReach-v1": [JointPositions, EndEffectorPose, TargetOffset],
    "MuJoCoLookAt-v1": [JointPositions, EndEffectorPose, GazeDirection],
    "MuJoCoMove-v1": [JointPositions, EndEffectorPose, TargetOffset],
    "MuJoCoPickLift-v1": [
        JointPositions,
        EndEffectorPose,
        GraspState,
        ObjectPose,
        ObjectOffset,
    ],
    "MuJoCoPickAndPlace-v1": [
        JointPositions,
        EndEffectorPose,
        GraspState,
        TargetPosition,
        ObjectPose,
        ObjectOffset,
        TargetOffset,
    ],
}


def _single_obs_params():
    for env_id, config_cls in ENV_MATRIX:
        for obs_cls in _ENV_OBS_MAP[env_id]:
            yield env_id, config_cls, obs_cls


@pytest.mark.parametrize(
    "env_id,config_cls,obs_cls",
    list(_single_obs_params()),
    ids=lambda p: p.__name__ if isinstance(p, type) else str(p),
)
def test_single_observation_component(env_id, config_cls, obs_cls):
    config = config_cls(observations=[obs_cls()])
    env = gym.make(env_id, config=config)
    try:
        obs, _ = env.reset()
        assert obs.shape == (OBS_SIZES[obs_cls],)
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id,config_cls", ENV_MATRIX)
def test_all_observation_components_combined(env_id, config_cls):
    obs_classes = _ENV_OBS_MAP[env_id]
    config = config_cls(observations=[cls() for cls in obs_classes])
    env = gym.make(env_id, config=config)
    try:
        obs, _ = env.reset()
        expected = sum(OBS_SIZES[cls] for cls in obs_classes)
        assert obs.shape == (expected,)
        _run_episode(env)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Per-env info keys (backend-specific content not in the shared contract).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "env_id,extra_key",
    [
        ("MuJoCoReach-v1", "tcp_to_target_dist"),
        ("MuJoCoMove-v1", "tcp_to_target_dist"),
        ("MuJoCoLookAt-v1", "orientation_error"),
        ("MuJoCoPickLift-v1", "lift_height"),
    ],
)
def test_env_info_has_extra_key(env_id, extra_key):
    env = gym.make(env_id)
    try:
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert extra_key in info
        assert "success" in info
    finally:
        env.close()


def test_pick_and_place_info_keys_exact():
    """PickAndPlaceEnv info keys are the documented full set."""
    expected = {
        "obj_to_target_dist",
        "is_obj_placed",
        "is_grasped",
        "is_robot_static",
        "lift_height",
        "success",
        "tcp_to_obj_dist",
    }
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        _, info = env.reset()
        assert set(info.keys()) == expected
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Object parametrics — cubes, YCB, mixed pools.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color", CUBE_COLORS)
def test_pick_cube_color(color):
    config = PickConfig(objects=[CubeObject(color=color)])  # type: ignore[arg-type]
    env = gym.make("MuJoCoPickLift-v1", config=config)
    try:
        env.reset()
        assert color in env.unwrapped.task_description  # type: ignore[attr-defined]
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("model_id", YCB_MODEL_IDS)
def test_pick_ycb_object(model_id):
    config = PickConfig(objects=[YCBObject(model_id=model_id)])
    env = gym.make("MuJoCoPickLift-v1", config=config)
    try:
        env.reset()
        assert YCB_OBJECTS[model_id] in env.unwrapped.task_description  # type: ignore[attr-defined]
        _run_episode(env)
    finally:
        env.close()


def test_pick_multiple_cubes_with_distractors():
    """PickLift with a homogeneous cube pool and distractors spawns correctly."""
    objects: list[CubeObject] = [
        CubeObject(color="red"),
        CubeObject(color="blue"),
        CubeObject(color="green"),
    ]
    config = PickConfig(objects=objects, n_distractors=2)  # type: ignore[arg-type]
    env = gym.make("MuJoCoPickLift-v1", config=config)
    try:
        obs, _ = env.reset()
        assert obs.shape == (18,)
        _run_episode(env)
    finally:
        env.close()


def test_pick_mixed_pool_with_distractors():
    objects = [
        YCBObject(model_id="011_banana"),
        CubeObject(color="blue"),
        YCBObject(model_id="058_golf_ball"),
    ]
    config = PickConfig(objects=objects, n_distractors=2)
    env = gym.make("MuJoCoPickLift-v1", config=config)
    try:
        obs, _ = env.reset()
        assert obs.shape == (18,)
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("model_id", ["011_banana", "030_fork", "031_spoon", "032_knife"])
def test_pick_ycb_collision_geom_starts_above_floor(model_id):
    from so101_nexus_mujoco.spawn_utils import mesh_geom_world_min_z

    config = PickConfig(objects=[YCBObject(model_id=model_id)], reset_settle_frames=0)
    env = gym.make("MuJoCoPickLift-v1", config=config)
    try:
        env.reset(seed=0)
        inner = env.unwrapped
        slot = inner._slots[inner._target_slot_idx]  # type: ignore[attr-defined]
        min_z = mesh_geom_world_min_z(inner.model, inner.data, slot.geom_id)  # type: ignore[attr-defined]
        assert min_z >= -1e-6
    finally:
        env.close()


def test_mesh_object_xml_uses_hidden_collision_and_visible_visual_groups():
    from so101_nexus_mujoco.pick_env import _mesh_xml_body

    xml = _mesh_xml_body("pick_slot_0", 0, 0.01)

    assert 'name="pick_slot_0_collision"' in xml
    assert 'mesh="pick_coll_0"' in xml
    assert 'group="3"' in xml
    assert 'name="pick_slot_0_visual"' in xml
    assert 'mesh="pick_vis_0"' in xml
    assert 'group="2"' in xml


def test_ycb_scene_xml_binds_cached_texture(monkeypatch, tmp_path):
    from so101_nexus_mujoco import pick_env

    collision_path = tmp_path / "collision.obj"
    visual_path = tmp_path / "visual.obj"
    texture_path = tmp_path / "texture.png"
    texture_path.write_text("texture", encoding="utf-8")
    monkeypatch.setattr(pick_env, "get_ycb_collision_mesh", lambda _model_id: collision_path)
    monkeypatch.setattr(pick_env, "get_ycb_visual_mesh", lambda _model_id: visual_path)
    monkeypatch.setattr(
        pick_env,
        "get_ycb_texture_file",
        lambda _model_id: texture_path,
        raising=False,
    )

    xml = pick_env._build_scene_xml(
        [YCBObject(model_id="011_banana")],
        ["pick_slot_0"],
        [0.1, 0.2, 0.3, 1.0],
    )

    assert f'<texture name="pick_tex_0" type="2d" file="{texture_path}"/>' in xml
    assert '<material name="pick_mat_0" texture="pick_tex_0" texuniform="false"/>' in xml
    assert 'name="pick_slot_0_visual"' in xml
    assert 'material="pick_mat_0"' in xml


def test_ycb_scene_xml_uses_posix_paths_for_cached_assets(monkeypatch):
    from so101_nexus_mujoco import pick_env

    class _FakePath:
        def __init__(self, posix_path: str, native_path: str):
            self._posix_path = posix_path
            self._native_path = native_path

        def __str__(self) -> str:
            return self._native_path

        def as_posix(self) -> str:
            return self._posix_path

        def exists(self) -> bool:
            return True

    collision_path = _FakePath("C:/cache/ycb/collision.obj", r"C:\cache\ycb\collision.obj")
    visual_path = _FakePath("C:/cache/ycb/visual.obj", r"C:\cache\ycb\visual.obj")
    texture_path = _FakePath("C:/cache/ycb/texture.png", r"C:\cache\ycb\texture.png")
    monkeypatch.setattr(pick_env, "get_ycb_collision_mesh", lambda _model_id: collision_path)
    monkeypatch.setattr(pick_env, "get_ycb_visual_mesh", lambda _model_id: visual_path)
    monkeypatch.setattr(
        pick_env,
        "get_ycb_texture_file",
        lambda _model_id: texture_path,
        raising=False,
    )

    xml = pick_env._build_scene_xml(
        [YCBObject(model_id="011_banana")],
        ["pick_slot_0"],
        [0.1, 0.2, 0.3, 1.0],
    )

    assert 'file="C:/cache/ycb/collision.obj"' in xml
    assert 'file="C:/cache/ycb/visual.obj"' in xml
    assert 'file="C:/cache/ycb/texture.png"' in xml
    assert r"C:\cache\ycb" not in xml


def test_scene_xml_leaves_cubes_and_mesh_objects_untextured(tmp_path):
    from so101_nexus_mujoco import pick_env

    mesh = MeshObject(
        collision_mesh_path=str(tmp_path / "collision.obj"),
        visual_mesh_path=str(tmp_path / "visual.obj"),
        mass=0.02,
        name="custom",
    )

    xml = pick_env._build_scene_xml(
        [CubeObject(color="red"), mesh],
        ["pick_slot_0", "pick_slot_1"],
        [0.1, 0.2, 0.3, 1.0],
    )

    assert "<texture " not in xml
    assert 'material="pick_mat_' not in xml


@pytest.mark.parametrize("color", CUBE_COLORS)
def test_look_at_cube_color(color):
    config = LookAtConfig(objects=[CubeObject(color=color)])  # type: ignore[arg-type]
    env = gym.make("MuJoCoLookAt-v1", config=config)
    try:
        env.reset()
        assert color in env.unwrapped.task_description  # type: ignore[attr-defined]
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("direction", MOVE_DIRECTIONS)
def test_move_direction(direction):
    config = MoveConfig(direction=direction)  # type: ignore[arg-type]
    env = gym.make("MuJoCoMove-v1", config=config)
    try:
        env.reset()
        assert direction in env.unwrapped.task_description  # type: ignore[attr-defined]
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("cube_color", CUBE_COLORS)
def test_pick_and_place_cube_colors(cube_color):
    target_color = "blue" if cube_color != "blue" else "red"
    config = PickAndPlaceConfig(cube_colors=cube_color, target_colors=target_color)
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    try:
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("target_color", CUBE_COLORS)
def test_pick_and_place_target_colors(target_color):
    """PickAndPlace works with every target disc colour."""
    cube_color = "red" if target_color != "red" else "blue"
    config = PickAndPlaceConfig(cube_colors=cube_color, target_colors=target_color)
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    try:
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
@pytest.mark.parametrize("control_mode", CONTROL_MODES)
def test_control_mode(env_id, control_mode):
    env = gym.make(env_id, control_mode=control_mode)
    try:
        _run_episode(env)
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Per-env observation-vector shape + geometry assertions.
# ---------------------------------------------------------------------------


def test_pick_and_place_default_obs_shape():
    """PickAndPlace default obs is a 24-dim flat vector."""
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        obs, _ = env.reset()
        assert obs.shape == (24,)
    finally:
        env.close()


def test_pick_and_place_target_z_near_ground():
    """PickAndPlace target Z component in obs is near ground plane."""
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        obs, _ = env.reset()
        target_pos = obs[8:11]
        assert target_pos[2] < 0.01
    finally:
        env.close()


def test_pick_and_place_cube_spawns_in_bounds():
    """Cube spawn radius lies within [spawn_min_radius, spawn_max_radius]."""
    cfg = PickAndPlaceConfig()
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        for _ in range(5):
            env.reset()
            cube_xy = env.unwrapped._get_cube_pose()[:2]  # type: ignore[attr-defined]
            cx, cy = cfg.spawn_center
            r = float(np.sqrt((cube_xy[0] - cx) ** 2 + (cube_xy[1] - cy) ** 2))
            assert cfg.spawn_min_radius <= r <= cfg.spawn_max_radius
    finally:
        env.close()


def test_pick_and_place_min_cube_target_separation():
    cfg = PickAndPlaceConfig()
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        for _ in range(10):
            env.reset()
            cube_xy = env.unwrapped._get_cube_pose()[:2]  # type: ignore[attr-defined]
            target_xy = env.unwrapped._get_target_pos()[:2]  # type: ignore[attr-defined]
            dist = float(np.linalg.norm(cube_xy - target_xy))
            assert dist >= cfg.min_cube_target_separation - 1e-6
    finally:
        env.close()


def test_pick_and_place_success_false_at_reset():
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        _, info = env.reset()
        assert not info["success"]
    finally:
        env.close()


def test_pick_and_place_reward_in_unit_range():
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert 0.0 <= float(reward) <= 1.0
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Reset settling.
# ---------------------------------------------------------------------------


def test_reset_settle_zero_keeps_mujoco_time_at_reset():
    config = ReachConfig(reset_settle_frames=0)
    env = gym.make("MuJoCoReach-v1", config=config)
    try:
        env.reset()
        assert env.unwrapped.data.time == pytest.approx(0.0)  # type: ignore[attr-defined]
    finally:
        env.close()


def test_reset_settle_frames_advance_mujoco_time_by_environment_frames():
    config = ReachConfig(reset_settle_frames=2)
    env = gym.make("MuJoCoReach-v1", config=config)
    try:
        env.reset()
        inner = env.unwrapped
        expected = 2 * inner._N_SUBSTEPS * inner.model.opt.timestep  # type: ignore[attr-defined]
        assert inner.data.time == pytest.approx(expected)  # type: ignore[attr-defined]
    finally:
        env.close()


def test_pick_initial_object_z_matches_post_settle_pose():
    config = PickConfig(reset_settle_frames=2)
    env = gym.make("MuJoCoPickLift-v1", config=config)
    try:
        env.reset(seed=0)
        inner = env.unwrapped
        target_z = float(inner._get_target_pose()[2])  # type: ignore[attr-defined]
        assert inner._initial_obj_z == pytest.approx(target_z)  # type: ignore[attr-defined]
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Task-description shape checks.
# ---------------------------------------------------------------------------


def test_pick_and_place_task_description_mentions_colors():
    config = PickAndPlaceConfig(cube_colors="red", target_colors="blue")
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    try:
        desc = env.unwrapped.task_description  # type: ignore[attr-defined]
        assert isinstance(desc, str)
        assert "red" in desc
        assert "blue" in desc
    finally:
        env.close()


def test_pick_and_place_task_description_starts_capital():
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        desc = env.unwrapped.task_description  # type: ignore[attr-defined]
        assert desc[0].isupper()
    finally:
        env.close()


def test_pick_and_place_task_description_is_instance_attr():
    """Task description is an instance attribute, not a class attribute."""
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        assert "task_description" in env.unwrapped.__dict__  # type: ignore[attr-defined]
    finally:
        env.close()


# ---------------------------------------------------------------------------
# MuJoCo-specific geometry / scene checks.
# ---------------------------------------------------------------------------


def test_spawn_center_offsets_object_positions():
    """PickLift: spawned object positions are offset by spawn_center, not origin."""
    config = PickConfig(spawn_angle_half_range_deg=30.0)
    env = gym.make("MuJoCoPickLift-v1", config=config)
    positions = []
    try:
        for seed in range(20):
            env.reset(seed=seed)
            slot = env.unwrapped._slots[env.unwrapped._target_slot_idx]  # type: ignore[attr-defined]
            obj_pos = env.unwrapped.data.qpos[slot.qpos_addr : slot.qpos_addr + 2].copy()  # type: ignore[attr-defined]
            positions.append(obj_pos)
    finally:
        env.close()
    positions = np.array(positions)
    assert positions[:, 0].mean() > 0.10


def test_spawn_center_offsets_pick_and_place_cube_and_target():
    config = PickAndPlaceConfig(spawn_angle_half_range_deg=30.0)
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    cube_xs = []
    try:
        for seed in range(20):
            env.reset(seed=seed)
            cube_pos = env.unwrapped._get_cube_pose()[:2]  # type: ignore[attr-defined]
            cube_xs.append(cube_pos[0])
    finally:
        env.close()
    assert np.mean(cube_xs) > 0.10


def test_reach_target_on_ground_and_offset():
    """Reach target spawns on the ground plane and is offset by spawn_center."""
    env = gym.make("MuJoCoReach-v1")
    xs: list[float] = []
    try:
        for seed in range(20):
            env.reset(seed=seed)
            target_pos = env.unwrapped._target_pos  # type: ignore[attr-defined]
            assert target_pos[2] < 0.05, f"Target z={target_pos[2]} too high (seed={seed})"
            xs.append(float(target_pos[0]))
    finally:
        env.close()
    assert np.mean(xs) > 0.10


def test_pick_and_place_target_disc_geom_exists():
    import mujoco

    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        env.reset()
        geom_id = mujoco.mj_name2id(
            env.unwrapped.model,  # type: ignore[attr-defined]
            mujoco.mjtObj.mjOBJ_GEOM,
            "target_disc",
        )
        assert geom_id >= 0
    finally:
        env.close()


def test_pick_and_place_no_mocap_goal_body():
    import mujoco

    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        env.reset()
        body_id = mujoco.mj_name2id(
            env.unwrapped.model,  # type: ignore[attr-defined]
            mujoco.mjtObj.mjOBJ_BODY,
            "goal",
        )
        assert body_id == -1
    finally:
        env.close()


# ---------------------------------------------------------------------------
# LookAt regression: TCP forward must match gripper direction.
# ---------------------------------------------------------------------------


def test_lookat_tcp_forward_matches_gripper_direction():
    """Regression guard: TCP forward must equal -Z of gripper's parent body.

    The gripperframe site has a 180° Y rotation so its local Z points from
    the wrist toward the fingertips. An earlier bug used a 90° Y rotation,
    which made the look-at reward peak when the arm pointed away from the
    target.
    """
    env = gym.make("MuJoCoLookAt-v1")
    try:
        inner = env.unwrapped
        inner.reset()
        tcp_forward = inner._get_tcp_forward()  # type: ignore[attr-defined]
        body_id = inner.model.site_bodyid[inner._tcp_site_id]  # type: ignore[attr-defined]
        body_z = inner.data.xmat[body_id].reshape(3, 3)[:, 2]  # type: ignore[attr-defined]
        np.testing.assert_allclose(
            tcp_forward,
            -body_z,
            atol=1e-6,
            err_msg="TCP forward should equal -Z of parent body (gripper direction).",
        )
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Render mode smoke.
# ---------------------------------------------------------------------------


def test_pick_and_place_rgb_array_render():
    env = gym.make("MuJoCoPickAndPlace-v1", render_mode="rgb_array")
    try:
        env.reset()
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Camera observation integration — not part of the shared contract.
# ---------------------------------------------------------------------------


_CAMERA_ENVS: list[tuple[str, type]] = ENV_MATRIX


@pytest.mark.parametrize("env_id,config_cls", _CAMERA_ENVS)
def test_overhead_camera_obs(env_id, config_cls):
    cfg = config_cls(observations=[JointPositions(), OverheadCamera(width=64, height=48)])
    env = gym.make(env_id, config=cfg)
    try:
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert obs["overhead_camera"].shape == (48, 64, 3)
        assert obs["overhead_camera"].dtype == np.uint8
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        assert "overhead_camera" in obs2
    finally:
        env.close()


@pytest.mark.parametrize("env_id,config_cls", _CAMERA_ENVS)
def test_wrist_camera_obs(env_id, config_cls):
    cfg = config_cls(observations=[JointPositions(), WristCamera(width=64, height=48)])
    env = gym.make(env_id, config=cfg)
    try:
        obs, _ = env.reset()
        assert isinstance(obs, dict)
        assert obs["wrist_camera"].shape == (48, 64, 3)
        assert obs["wrist_camera"].dtype == np.uint8
    finally:
        env.close()


@pytest.mark.parametrize("env_id,config_cls", _CAMERA_ENVS)
def test_both_cameras_obs(env_id, config_cls):
    cfg = config_cls(
        observations=[
            JointPositions(),
            WristCamera(width=64, height=48),
            OverheadCamera(width=32, height=24),
        ]
    )
    env = gym.make(env_id, config=cfg)
    try:
        obs, _ = env.reset()
        assert set(obs.keys()) == {"state", "wrist_camera", "overhead_camera"}
        assert obs["wrist_camera"].shape == (48, 64, 3)
        assert obs["overhead_camera"].shape == (24, 32, 3)
    finally:
        env.close()


@pytest.mark.parametrize("env_id,config_cls", _CAMERA_ENVS)
def test_no_camera_flat_obs(env_id, config_cls):
    """When no cameras are configured, obs is a flat numpy array (not a dict)."""
    cfg = config_cls(observations=[JointPositions()])
    env = gym.make(env_id, config=cfg)
    try:
        obs, _ = env.reset()
        assert isinstance(obs, np.ndarray)
        assert obs.shape == (6,)
    finally:
        env.close()


@pytest.mark.parametrize("env_id,config_cls", _CAMERA_ENVS)
def test_visual_obs_mode(env_id, config_cls):
    cfg = config_cls(
        obs_mode="visual",
        observations=[JointPositions(), OverheadCamera(width=64, height=48)],
    )
    env = gym.make(env_id, config=cfg)
    try:
        obs, info = env.reset()
        assert isinstance(obs, dict)
        assert obs["state"].shape == (6,)
        assert "privileged_state" in info
        assert "overhead_camera" in obs
    finally:
        env.close()


@pytest.mark.parametrize(
    "env_id,config_cls",
    [("MuJoCoReach-v1", ReachConfig), ("MuJoCoPickLift-v1", PickConfig)],
)
def test_visual_obs_mode_with_both_cameras(env_id, config_cls):
    cfg = config_cls(
        obs_mode="visual",
        observations=[
            JointPositions(),
            WristCamera(width=64, height=48),
            OverheadCamera(width=32, height=24),
        ],
    )
    env = gym.make(env_id, config=cfg)
    try:
        obs, info = env.reset()
        assert obs["state"].shape == (6,)
        assert "privileged_state" in info
        assert "wrist_camera" in obs
        assert "overhead_camera" in obs
    finally:
        env.close()


def test_render_independent_of_overhead_obs():
    """Render should work even when OverheadCamera is also used as an obs component."""
    cfg = ReachConfig(observations=[JointPositions(), OverheadCamera(width=64, height=48)])
    env = gym.make("MuJoCoReach-v1", config=cfg, render_mode="rgb_array")
    try:
        env.reset()
        frame = env.render()
        assert frame is not None
        assert frame.dtype == np.uint8
    finally:
        env.close()


def test_pick_hidden_cube_slots_are_inert_below_floor():
    """Hidden pool slots must have zero contact bits and stay below the floor.

    Reproduces the pre-fix bug where a stack of hidden bodies at (0, 0, -10)
    with collision enabled was deterministically exploded above the floor by
    the constraint solver during _settle_after_reset.
    """
    objects: list[CubeObject] = [
        CubeObject(color=c) for c in ("red", "blue", "green", "yellow", "purple")
    ]
    config = PickConfig(objects=objects, n_distractors=0)  # type: ignore[arg-type]
    env = gym.make("MuJoCoPickLift-v1", config=config)
    try:
        env.reset(seed=42)
        inner = env.unwrapped
        target_idx = inner._target_slot_idx  # type: ignore[attr-defined]
        for i, slot in enumerate(inner._slots):  # type: ignore[attr-defined]
            geom_id = slot.geom_id
            qpos_addr = slot.qpos_addr
            z = float(inner.data.qpos[qpos_addr + 2])  # type: ignore[attr-defined]
            if i == target_idx:
                assert inner.model.geom_contype[geom_id] == 1  # type: ignore[attr-defined]
                assert inner.model.geom_conaffinity[geom_id] == 1  # type: ignore[attr-defined]
                assert z > 0.0, f"target slot {i} should be above the floor, got z={z}"
            else:
                assert inner.model.geom_contype[geom_id] == 0  # type: ignore[attr-defined]
                assert inner.model.geom_conaffinity[geom_id] == 0  # type: ignore[attr-defined]
                # Pre-fix this z was ~+14.6; with collisions disabled it must
                # stay below the floor. Small gravitational drift over the
                # default reset_settle_frames is tolerated.
                assert z < 0.0, f"hidden slot {i} drifted above the floor: z={z}"
                assert z < -5.0, (
                    f"hidden slot {i} suspiciously close to the floor: z={z}; "
                    "the hide mechanism may have regressed"
                )
    finally:
        env.close()


def test_pick_hidden_slot_collisions_restore_when_slot_becomes_target():
    """A slot that was hidden last episode must be collidable when it becomes target."""
    objects: list[CubeObject] = [
        CubeObject(color=c) for c in ("red", "blue", "green", "yellow", "purple")
    ]
    config = PickConfig(objects=objects, n_distractors=0)  # type: ignore[arg-type]
    env = gym.make("MuJoCoPickLift-v1", config=config)
    try:
        env.reset(seed=42)
        inner = env.unwrapped
        target_a = inner._target_slot_idx  # type: ignore[attr-defined]

        attempts = 0
        while inner._target_slot_idx == target_a and attempts < 16:  # type: ignore[attr-defined]
            env.reset(seed=attempts + 100)
            attempts += 1
        target_b = inner._target_slot_idx  # type: ignore[attr-defined]
        assert target_b != target_a, (
            "test setup failed: could not find a reseat that changed the target"
        )

        new_geom = inner._slots[target_b].geom_id  # type: ignore[attr-defined]
        old_geom = inner._slots[target_a].geom_id  # type: ignore[attr-defined]
        assert inner.model.geom_contype[new_geom] == 1  # type: ignore[attr-defined]
        assert inner.model.geom_conaffinity[new_geom] == 1  # type: ignore[attr-defined]
        assert inner.model.geom_contype[old_geom] == 0  # type: ignore[attr-defined]
        assert inner.model.geom_conaffinity[old_geom] == 0  # type: ignore[attr-defined]
    finally:
        env.close()


def test_pick_hidden_ycb_slots_are_inert_below_floor():
    """Same fix must hold for YCB pools with collision and visual geoms."""
    objects = [
        YCBObject(model_id="011_banana"),
        YCBObject(model_id="030_fork"),
        YCBObject(model_id="058_golf_ball"),
    ]
    config = PickConfig(objects=objects, n_distractors=0)
    env = gym.make("MuJoCoPickLift-v1", config=config)
    try:
        env.reset(seed=42)
        inner = env.unwrapped
        target_idx = inner._target_slot_idx  # type: ignore[attr-defined]
        for i, slot in enumerate(inner._slots):  # type: ignore[attr-defined]
            geom_id = slot.geom_id
            z = float(inner.data.qpos[slot.qpos_addr + 2])  # type: ignore[attr-defined]
            body_id = int(inner.model.geom_bodyid[geom_id])  # type: ignore[attr-defined]
            attached_geoms = [
                gi
                for gi in range(inner.model.ngeom)  # type: ignore[attr-defined]
                if int(inner.model.geom_bodyid[gi]) == body_id  # type: ignore[attr-defined]
            ]
            assert len(attached_geoms) >= 2, (
                f"slot {i} unexpectedly has only {len(attached_geoms)} geom(s); "
                "test assumption (mesh body has >=2 geoms) is wrong"
            )
            if i == target_idx:
                assert inner.model.geom_contype[geom_id] == 1  # type: ignore[attr-defined]
                assert inner.model.geom_conaffinity[geom_id] == 1  # type: ignore[attr-defined]
            else:
                for gi in attached_geoms:
                    assert inner.model.geom_contype[gi] == 0, (  # type: ignore[attr-defined]
                        f"hidden YCB slot {i} geom {gi} has contype != 0"
                    )
                    assert inner.model.geom_conaffinity[gi] == 0, (  # type: ignore[attr-defined]
                        f"hidden YCB slot {i} geom {gi} has conaffinity != 0"
                    )
                assert z < 0.0, f"hidden YCB slot {i} drifted above the floor: z={z}"
                assert z < -5.0, (
                    f"hidden YCB slot {i} suspiciously close to the floor: z={z}"
                )
    finally:
        env.close()
