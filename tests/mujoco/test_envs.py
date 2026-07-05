"""Consolidated Gymnasium tests for every MuJoCo SO101-Nexus environment.

Replaces the per-task test files (test_pick_env.py, test_touch_env.py, ...)
with a single parametric suite backed by the shared
``so101_nexus.testing.run_env_contract`` helper. Backend-specific
assertions that aren't part of the shared contract live at the bottom
of this file.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import numpy as np
import pytest

import so101_nexus.mujoco  # noqa: F401 - registers envs
from so101_nexus.config import (
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    TouchConfig,
)
from so101_nexus.constants import CUBE_COLOR_MAP, YCB_OBJECTS
from so101_nexus.objects import CubeObject, YCBObject
from so101_nexus.observations import (
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
from so101_nexus.testing import run_env_contract

ENV_MATRIX: list[tuple[str, type]] = [
    ("MuJoCoTouch-v1", TouchConfig),
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


@pytest.mark.parametrize("env_id,config_cls", ENV_MATRIX)
def test_gymnasium_contract(env_id: str, config_cls: type):
    """Every env satisfies the shared Gymnasium contract."""
    del config_cls  # parametrized for symmetry with other matrix tests.
    run_env_contract(env_id)


_ENV_OBS_MAP: dict[str, list[type]] = {
    "MuJoCoTouch-v1": [JointPositions, EndEffectorPose, ObjectOffset],
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


@pytest.mark.parametrize(
    "env_id,extra_key",
    [
        ("MuJoCoTouch-v1", "tcp_to_obj_dist"),
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
        assert obs.shape == (24,)
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
        assert obs.shape == (24,)
        _run_episode(env)
    finally:
        env.close()


@pytest.mark.parametrize("model_id", ["011_banana", "030_fork", "031_spoon", "032_knife"])
def test_pick_ycb_collision_geom_starts_above_floor(model_id):
    from so101_nexus.mujoco.spawn_utils import mesh_geom_world_min_z

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


def test_look_at_task_description_delegates_to_config():
    config = LookAtConfig(objects=[CubeObject(color="red")])  # type: ignore[arg-type]
    env = gym.make("MuJoCoLookAt-v1", config=config)
    try:
        assert env.unwrapped.task_description == config.task_description  # type: ignore[attr-defined]
        assert "_task_description" not in env.unwrapped.__dict__
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


DELTA_CONTROL_MODES = ["pd_joint_delta_pos", "pd_joint_target_delta_pos"]

# Physical per-joint delta scale a normalized +1 action maps to (radians):
# +/-0.05 for the five arm joints, +/-0.2 for the gripper. This mirrors
# so101_nexus.mujoco.base_env._DELTA_ACTION_SCALE and is the cross-backend
# delta action contract.
_EXPECTED_DELTA_SCALE = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.2], dtype=np.float64)


@pytest.mark.parametrize("control_mode", DELTA_CONTROL_MODES)
def test_delta_action_space_is_normalized(control_mode):
    """Both delta modes expose a normalized [-1, 1] action space (six joints),
    matching the normalized delta action contract."""
    env = gym.make("MuJoCoTouch-v1", control_mode=control_mode)
    try:
        space = env.action_space
        assert space.shape == (6,)
        np.testing.assert_allclose(space.low, [-1.0] * 6, atol=1e-6)
        np.testing.assert_allclose(space.high, [1.0] * 6, atol=1e-6)
    finally:
        env.close()


@pytest.mark.parametrize("control_mode", DELTA_CONTROL_MODES)
def test_normalized_plus_one_matches_physical_max_delta(control_mode):
    """An all +1 normalized action moves joint targets by exactly the physical
    delta scale, i.e. the internal scaling reproduces the old physical-max
    behavior."""
    env = gym.make("MuJoCoTouch-v1", control_mode=control_mode)
    try:
        env.reset(seed=0)
        unwrapped = env.unwrapped
        actuator_ids = unwrapped._actuator_ids  # type: ignore[attr-defined]
        target_high = unwrapped._target_high  # type: ignore[attr-defined]

        # The base the delta is added to differs by mode: pd_joint_delta_pos
        # integrates from the measured joint positions, pd_joint_target_delta_pos
        # integrates from the held target. Read each mode's base at step time.
        if control_mode == "pd_joint_delta_pos":
            base = unwrapped._get_current_qpos()  # type: ignore[attr-defined]
        else:
            base = unwrapped._prev_target.copy()  # type: ignore[attr-defined]

        action = np.ones(6, dtype=np.float32)
        env.step(action)

        after = unwrapped.data.ctrl[actuator_ids].copy()  # type: ignore[attr-defined]
        # Per joint, the commanded target moved by the physical scale, unless it
        # was clamped at the upper target bound.
        expected = np.minimum(base + _EXPECTED_DELTA_SCALE, target_high)
        np.testing.assert_allclose(after, expected, atol=1e-6)
    finally:
        env.close()


@pytest.mark.parametrize("control_mode", DELTA_CONTROL_MODES)
def test_delta_penalty_norms_use_normalized_action(control_mode):
    """Penalty norms (energy_norm) are computed on the normalized public action,
    so an all +1 action yields energy_norm == sqrt(6)."""
    env = gym.make("MuJoCoTouch-v1", control_mode=control_mode)
    try:
        env.reset(seed=0)
        action = np.ones(6, dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert info["energy_norm"] == pytest.approx(np.sqrt(6.0), abs=1e-6)
    finally:
        env.close()


def test_pick_and_place_default_obs_shape():
    """PickAndPlace default obs is a 30-dim flat vector."""
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        obs, _ = env.reset()
        assert obs.shape == (30,)
    finally:
        env.close()


def test_pick_and_place_target_z_near_ground():
    """PickAndPlace target Z component in obs is near ground plane."""
    env = gym.make("MuJoCoPickAndPlace-v1")
    try:
        obs, _ = env.reset()
        target_pos = obs[14:17]
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
            cube_xy = env.unwrapped._get_object_pose()[:2]  # type: ignore[attr-defined]
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
            cube_xy = env.unwrapped._get_object_pose()[:2]  # type: ignore[attr-defined]
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


def test_reset_settle_zero_keeps_mujoco_time_at_reset():
    config = TouchConfig(reset_settle_frames=0)
    env = gym.make("MuJoCoTouch-v1", config=config)
    try:
        env.reset()
        assert env.unwrapped.data.time == pytest.approx(0.0)  # type: ignore[attr-defined]
    finally:
        env.close()


def test_reset_settle_frames_advance_mujoco_time_by_environment_frames():
    config = TouchConfig(reset_settle_frames=2)
    env = gym.make("MuJoCoTouch-v1", config=config)
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
            cube_pos = env.unwrapped._get_object_pose()[:2]  # type: ignore[attr-defined]
            cube_xs.append(cube_pos[0])
    finally:
        env.close()
    assert np.mean(cube_xs) > 0.10


def test_touch_object_grounded_on_table():
    """The touch target rests on the table (z ~ object half-size) within the spawn arc.

    Unlike the old reach marker that floated in a 3-D cube, the touch target is a
    real object grounded on the table, so its depth is unambiguous in the camera.
    """
    env = gym.make("MuJoCoTouch-v1")
    u = env.unwrapped
    cx, cy = u.config.spawn_center
    min_r, max_r = u.config.spawn_min_radius, u.config.spawn_max_radius
    try:
        for seed in range(20):
            env.reset(seed=seed)
            obj = u._get_target_pose()[:3]
            cube_half = u._slots[u._target_slot_idx].obj.half_size
            assert obj[2] == pytest.approx(cube_half, abs=5e-3), (
                f"object not grounded on the table (seed={seed}, z={obj[2]})"
            )
            r = float(np.hypot(obj[0] - cx, obj[1] - cy))
            assert min_r - 1e-2 <= r <= max_r + 1e-2, (
                f"object outside spawn arc (seed={seed}, r={r})"
            )
    finally:
        env.close()


def test_touch_margin_affects_success():
    """A larger touch_margin flips a borderline TCP-object distance to success.

    Guards that the public touch_margin knob changes runtime behavior.
    """

    def _success(margin: float) -> bool:
        env = gym.make("MuJoCoTouch-v1", config=TouchConfig(touch_margin=margin)).unwrapped
        try:
            env.reset(seed=0)
            slot = env._slots[env._target_slot_idx]
            r = env._target_bounding_radius()
            tcp = env._get_tcp_pose()[:3]
            # Place the cube center a fixed distance (r + 0.04) from the TCP.
            env.data.qpos[slot.qpos_addr : slot.qpos_addr + 3] = tcp + np.array(
                [r + 0.04, 0.0, 0.0]
            )
            return bool(env._get_info()["success"])
        finally:
            env.close()

    assert not _success(0.02)  # threshold r+0.02 < distance r+0.04 -> no touch
    assert _success(0.06)  # threshold r+0.06 > distance r+0.04 -> touch


def test_move_initial_distance_equals_target_distance():
    """After reset the TCP-to-target distance equals config.target_distance.

    Regression for the pre-settle vs post-settle target drift: the target must be
    computed from the settled TCP so the initial distance matches the requested
    move distance, matching the cross-backend contract.
    """
    cfg = MoveConfig(target_distance=0.10)
    env = gym.make("MuJoCoMove-v1", config=cfg)
    try:
        for seed in range(5):
            _, info = env.reset(seed=seed)
            assert info["tcp_to_target_dist"] == pytest.approx(cfg.target_distance, abs=1e-3)
    finally:
        env.close()


def test_lookat_target_pose_is_dynamics_immune():
    """LookAt target must not move from contact or gravity (kinematic-equivalent).

    Mirrors a kinematic body_type: the arm passes through the marker
    and the marker never drifts under dynamics.
    """
    env = gym.make("MuJoCoLookAt-v1")
    try:
        env.reset(seed=0)
        pos_before = env.unwrapped._get_target_pos().copy()  # type: ignore[attr-defined]
        # Drive the arm hard for several steps; with a colliding/dynamic target this
        # would bump and shift it. Kinematic target must stay fixed.
        for _ in range(30):
            action = env.action_space.sample()
            env.step(action)
        pos_after = env.unwrapped._get_target_pos().copy()  # type: ignore[attr-defined]
        np.testing.assert_allclose(pos_after, pos_before, atol=1e-9)
    finally:
        env.close()


def test_lookat_success_means_object_in_camera_fov():
    """LookAt succeeds when the object is within the wrist camera's field of view."""

    def _success(fov_deg, aligned: bool) -> bool:
        import mujoco

        env = gym.make("MuJoCoLookAt-v1", config=LookAtConfig(fov_deg=fov_deg)).unwrapped
        try:
            env.reset(seed=0)
            axis = env._gaze_axis()
            tcp = env._get_tcp_pose()[:3]
            mid = env._look_target_mocap_id
            # Place the target along the camera optical axis (in frame) or opposite
            # (behind the camera, out of frame).
            pos = tcp + axis * 0.10 if aligned else tcp - axis * 0.10
            env.data.mocap_pos[mid] = pos
            mujoco.mj_forward(env.model, env.data)
            return bool(env._get_info()["success"])
        finally:
            env.close()

    assert _success(48.5, aligned=True) is True  # on-axis -> within FOV
    assert _success(48.5, aligned=False) is False  # behind camera -> out of FOV


def test_lookat_fov_deg_is_a_live_knob():
    """A borderline gaze angle flips with the FOV (success = angle <= fov_deg/2)."""

    def _success(fov_deg: float) -> bool:
        import mujoco

        env = gym.make("MuJoCoLookAt-v1", config=LookAtConfig(fov_deg=fov_deg)).unwrapped
        try:
            env.reset(seed=0)
            axis = env._gaze_axis()
            tcp = env._get_tcp_pose()[:3]
            perp = np.array([axis[1], -axis[0], 0.0])
            perp = perp / (np.linalg.norm(perp) + 1e-9)
            angle = float(np.deg2rad(18.0))
            env.data.mocap_pos[env._look_target_mocap_id] = (
                tcp + (np.cos(angle) * axis + np.sin(angle) * perp) * 0.10
            )
            mujoco.mj_forward(env.model, env.data)
            return bool(env._get_info()["success"])
        finally:
            env.close()

    assert not _success(10.0)  # half-FOV 5 deg < achieved 18 deg angle -> out of frame
    assert _success(50.0)  # half-FOV 25 deg > achieved 18 deg angle -> in frame


def test_lookat_success_threshold_defaults_to_live_camera_fov():
    """With fov_deg=None the threshold reads the actual wrist camera FOV."""
    env = gym.make("MuJoCoLookAt-v1").unwrapped
    try:
        env.reset(seed=0)
        expected = float(np.radians(env.model.cam_fovy[env._wrist_cam_id].item()) / 2.0)
        assert env._success_half_fov_rad() == pytest.approx(expected)
    finally:
        env.close()


def test_move_success_is_directional_displacement():
    """Move succeeds on forward travel along the move direction, ignoring sideways drift."""
    env = gym.make("MuJoCoMove-v1", config=MoveConfig(direction="left", target_distance=0.10))
    u = env.unwrapped  # type: ignore[attr-defined]
    try:
        env.reset(seed=0)
        tcp = u._get_tcp_pose()[:3]
        # Travel exactly target_distance left, plus a large sideways drift.
        u._start_pos = tcp.copy()
        u.data.site_xpos[u._tcp_site_id] = tcp + u._dir_vec * u.config.target_distance
        assert bool(u._get_info()["success"]) is True
        # Short of target_distance: no success.
        u.data.site_xpos[u._tcp_site_id] = tcp + u._dir_vec * (
            u.config.target_distance - u.config.success_threshold - 0.005
        )
        assert bool(u._get_info()["success"]) is False
    finally:
        env.close()


def test_move_clamped_downward_target_can_succeed():
    """A downward move whose target is clamped above the floor can still succeed.

    Regression: success must compare against the clamped (reachable) target
    displacement, not the raw configured target_distance.
    """
    env = gym.make("MuJoCoMove-v1", config=MoveConfig(direction="down", target_distance=1.0))
    u = env.unwrapped  # type: ignore[attr-defined]
    try:
        env.reset(seed=0)
        # The clamped target sits above the floor, so the reachable displacement
        # is less than target_distance. Reaching the clamped target must succeed.
        u.data.site_xpos[u._tcp_site_id] = u._target_pos.copy()
        assert bool(u._get_info()["success"]) is True
        assert u._target_displacement < u.config.target_distance  # confirm it was clamped
    finally:
        env.close()


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


def test_lookat_reward_uses_orientation_error_info(monkeypatch):
    """Reward must use info from _get_info(), not recompute orientation vectors."""
    env = gym.make("MuJoCoLookAt-v1")
    try:
        inner = env.unwrapped
        inner.reset()

        def fail_if_recomputed(*args, **kwargs):
            raise AssertionError("reward recomputed orientation vectors")

        monkeypatch.setattr(inner, "_gaze_axis", fail_if_recomputed)
        monkeypatch.setattr(inner, "_get_target_pos", fail_if_recomputed)
        monkeypatch.setattr(inner, "_get_tcp_pose", fail_if_recomputed)

        reward = inner._compute_reward({"orientation_error": 0.0, "success": False})

        expected = 1.0 - inner.config.reward.completion_bonus
        assert reward == pytest.approx(expected)
    finally:
        env.close()


def test_pick_and_place_rgb_array_render():
    env = gym.make("MuJoCoPickAndPlace-v1", render_mode="rgb_array")
    try:
        env.reset()
        frame = env.render()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 3
    finally:
        env.close()


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
def test_state_obs_is_float32(env_id, config_cls):
    """State observations are float32, matching the Warp backend and torch models."""
    cfg = config_cls(observations=[JointPositions()])
    env = gym.make(env_id, config=cfg)
    try:
        assert env.observation_space.dtype == np.float32
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        obs2, _, _, _, _ = env.step(env.action_space.sample())
        assert obs2.dtype == np.float32
    finally:
        env.close()


@pytest.mark.parametrize("env_id,config_cls", _CAMERA_ENVS)
def test_visual_state_and_privileged_state_are_float32(env_id, config_cls):
    """Visual-mode flat 'state' key and the privileged_state info are float32."""
    cfg = config_cls(
        obs_mode="visual",
        observations=[JointPositions(), OverheadCamera(width=64, height=48)],
    )
    env = gym.make(env_id, config=cfg)
    try:
        obs, info = env.reset()
        assert obs["state"].dtype == np.float32
        assert info["privileged_state"].dtype == np.float32
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ENV_IDS)
def test_max_episode_steps_override_truncates(env_id):
    """gym.make max_episode_steps overrides the registered horizon for every env."""
    n = 3
    env = gym.make(env_id, max_episode_steps=n)
    try:
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        truncations = [bool(env.step(action)[3]) for _ in range(n)]
        assert truncations == [False, False, True]
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
    [("MuJoCoTouch-v1", TouchConfig), ("MuJoCoPickLift-v1", PickConfig)],
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
    cfg = TouchConfig(observations=[JointPositions(), OverheadCamera(width=64, height=48)])
    env = gym.make("MuJoCoTouch-v1", config=cfg, render_mode="rgb_array")
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
                assert z < -5.0, f"hidden YCB slot {i} suspiciously close to the floor: z={z}"
    finally:
        env.close()


def test_pick_and_place_color_description_agreement():
    """task_description names the SAME color applied to the cube/target geoms."""
    import numpy as np

    from so101_nexus.constants import COLOR_MAP

    config = PickAndPlaceConfig(
        cube_colors=["red", "green", "yellow"],
        target_colors=["blue", "purple"],
    )
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    try:
        inner = env.unwrapped
        inner.reset(seed=7)
        desc = inner.task_description
        assert inner.cube_color_name in config.cube_colors
        assert inner.target_color_name in config.target_colors
        assert inner.cube_color_name in desc
        assert inner.target_color_name in desc
        # Rendered geom rgba matches the named color.
        cube_rgba = inner.model.geom_rgba[inner._obj_geom_id]
        target_rgba = inner.model.geom_rgba[inner._target_geom_id]
        np.testing.assert_allclose(cube_rgba, COLOR_MAP[inner.cube_color_name], atol=1e-6)
        np.testing.assert_allclose(target_rgba, COLOR_MAP[inner.target_color_name], atol=1e-6)
    finally:
        env.close()


def test_pick_and_place_color_reproducible_by_seed():
    """reset(seed=S) reproduces the same sampled cube/target colors twice."""
    config = PickAndPlaceConfig(
        cube_colors=["red", "green", "yellow"],
        target_colors=["blue", "purple", "orange"],
    )
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    try:
        inner = env.unwrapped
        inner.reset(seed=42)
        first = (inner.cube_color_name, inner.target_color_name)
        inner.reset(seed=42)
        second = (inner.cube_color_name, inner.target_color_name)
        assert first == second
    finally:
        env.close()


def test_pick_and_place_ycb_object_task_description():
    """An explicit object pool names the carried object in the task description."""
    config = PickAndPlaceConfig(objects=[YCBObject(model_id="011_banana")])
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    try:
        env.reset(seed=0)
        desc = env.unwrapped.task_description  # type: ignore[attr-defined]
        assert "banana" in desc
        assert "circle" in desc
        _run_episode(env)
    finally:
        env.close()


def test_pick_and_place_object_pose_tracks_selected_slot():
    """ObjectPose/TargetOffset route to the per-episode selected object slot."""
    config = PickAndPlaceConfig(cube_colors=["red", "green", "yellow"])
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    try:
        inner = env.unwrapped
        obs, _ = inner.reset(seed=3)
        selected = inner._slots[inner._target_slot_idx]  # type: ignore[attr-defined]
        obj_pose = inner.data.qpos[selected.qpos_addr : selected.qpos_addr + 7]  # type: ignore[attr-defined]
        # Default obs layout: joints(6)+ee(7)+grasp(1)+target_pos(3)+obj_pose(7)+...
        np.testing.assert_allclose(obs[17:24], obj_pose, atol=1e-6)
    finally:
        env.close()


def test_pick_and_place_info_keys_with_object_pool():
    """Placement info keys stay stable for a non-cube carried object."""
    config = PickAndPlaceConfig(objects=[YCBObject(model_id="011_banana")])
    env = gym.make("MuJoCoPickAndPlace-v1", config=config)
    expected = {
        "obj_to_target_dist",
        "is_obj_placed",
        "is_grasped",
        "is_robot_static",
        "lift_height",
        "success",
        "tcp_to_obj_dist",
    }
    try:
        _, info = env.reset(seed=0)
        assert set(info.keys()) == expected
    finally:
        env.close()


# ---------------------------------------------------------------------------
# Observation-ordering contract.
#
# The flat state vector is the concatenation of each non-camera component's
# value in exactly ``config.observations`` order. The observation list does not
# touch the seeded reset RNG or the physics, so at a fixed seed a component's
# value is identical regardless of what else is in the list. Therefore a
# permuted list, sliced at cumulative component sizes, must reproduce -- segment
# by segment -- the single-component configs reset at the same seed. A builder
# that sorted or otherwise reordered the list would misalign the slices and
# fail these assertions.
# ---------------------------------------------------------------------------


def _single_component_obs(env_id, config_cls, obs_cls, seed: int = 0):
    """Flat obs vector for a config holding only ``obs_cls``, reset at ``seed``."""
    env = gym.make(env_id, config=config_cls(observations=[obs_cls()]))
    try:
        obs, _ = env.reset(seed=seed)
    finally:
        env.close()
    return obs


def _assert_flat_vector_matches_ordered_singles(env_id, config_cls, perm, seed: int = 0):
    env = gym.make(env_id, config=config_cls(observations=[cls() for cls in perm]))
    try:
        full, _ = env.reset(seed=seed)
    finally:
        env.close()
    assert isinstance(full, np.ndarray)
    offset = 0
    for cls in perm:
        size = OBS_SIZES[cls]
        single = _single_component_obs(env_id, config_cls, cls, seed=seed)
        segment = full[offset : offset + size]
        assert np.allclose(segment, single, atol=1e-6), (
            f"{env_id}: {cls.__name__} at flat slice [{offset}:{offset + size}] does not "
            f"match its single-component obs -- components not concatenated in list order"
        )
        offset += size
    assert offset == full.shape[0], "flat vector length != sum of component sizes"


_ORDERING_CASES = [
    (
        "MuJoCoTouch-v1",
        TouchConfig,
        [ObjectOffset, JointPositions, GraspState, EndEffectorPose, ObjectPose],
    ),
    (
        "MuJoCoPickAndPlace-v1",
        PickAndPlaceConfig,
        [
            TargetOffset,
            ObjectPose,
            JointPositions,
            TargetPosition,
            GraspState,
            ObjectOffset,
            EndEffectorPose,
        ],
    ),
]


@pytest.mark.parametrize("env_id,config_cls,components", _ORDERING_CASES)
@pytest.mark.parametrize("order", ["as_given", "reversed"])
def test_flat_obs_preserves_component_list_order(env_id, config_cls, components, order):
    """Non-camera components appear in the flat state vector in ``config.observations``
    order. Exercising both a non-sorted permutation and its reverse means any builder
    that sorts or reorders the list must misalign at least one segment and fail."""
    perm = list(components) if order == "as_given" else list(reversed(components))
    _assert_flat_vector_matches_ordered_singles(env_id, config_cls, perm)


def test_camera_interleaved_is_skipped_but_state_order_preserved():
    """obs_mode='state' with a camera between two state components: the flat 'state'
    key concatenates ONLY the non-camera components (JointPositions then ObjectOffset)
    in list order, and the camera image is a separate key. Proves the camera is skipped
    while the relative order of the surrounding state components is preserved."""
    cfg = TouchConfig(
        observations=[JointPositions(), WristCamera(width=64, height=48), ObjectOffset()]
    )
    env = gym.make("MuJoCoTouch-v1", config=cfg)
    try:
        obs, _ = env.reset(seed=0)
    finally:
        env.close()
    assert isinstance(obs, dict)
    assert set(obs.keys()) == {"state", "wrist_camera"}
    assert obs["wrist_camera"].shape == (48, 64, 3)
    state = obs["state"]
    n_joint = OBS_SIZES[JointPositions]
    assert state.shape == (n_joint + OBS_SIZES[ObjectOffset],)
    jp = _single_component_obs("MuJoCoTouch-v1", TouchConfig, JointPositions)
    oo = _single_component_obs("MuJoCoTouch-v1", TouchConfig, ObjectOffset)
    assert np.allclose(state[:n_joint], jp, atol=1e-6)
    assert np.allclose(state[n_joint:], oo, atol=1e-6)
