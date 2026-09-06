"""Real-physics regressions for the versioned center-placement contract."""

from __future__ import annotations

import json

import mujoco
import numpy as np
import pytest

from so101_nexus.config import PickAndPlaceConfig, PickAndPlaceV2Config
from so101_nexus.mujoco.pick_and_place import PickAndPlaceEnv, PickAndPlaceV2Env
from so101_nexus.objects import CubeObject, MeshObject
from so101_nexus.observations import JointPositions

_GOAL = np.array([0.45, 0.3])
_IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def make_env():
    envs = []

    def make(**kwargs):
        kwargs.setdefault("objects", [CubeObject(half_size=0.0127, color="red")])
        config = PickAndPlaceV2Config(
            target_disc_radius=0.04,
            observations=[JointPositions()],
            **kwargs,
        )
        env = PickAndPlaceV2Env(
            config=config, control_mode="pd_joint_delta_pos", robot_init_qpos_noise=0.0
        )
        envs.append(env)
        env.reset(seed=3)
        env.model.body_pos[env._target_body_id, :2] = _GOAL
        mujoco.mj_forward(env.model, env.data)
        return env

    yield make
    for env in envs:
        env.close()


@pytest.fixture
def mesh_box(tmp_path):
    def make(half_size, shift=(0.0, 0.0, 0.0)):
        half_size = np.asarray(half_size)
        shift = np.asarray(shift)
        vertices = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=float,
            )
            * half_size
            + shift
        )
        faces = [
            (1, 3, 2),
            (1, 4, 3),
            (5, 6, 7),
            (5, 7, 8),
            (1, 2, 6),
            (1, 6, 5),
            (2, 3, 7),
            (2, 7, 6),
            (3, 4, 8),
            (3, 8, 7),
            (4, 1, 5),
            (4, 5, 8),
        ]
        path = tmp_path / f"box_{len(list(tmp_path.iterdir()))}.obj"
        path.write_text(
            "".join(f"v {x} {y} {z}\n" for x, y, z in vertices)
            + "".join(f"f {a} {b} {c}\n" for a, b, c in faces)
        )
        return MeshObject(str(path), str(path), mass=0.03, name="elongated box")

    return make


def _pose(env, position, quat=_IDENTITY, velocity=None):
    slot = env._slots[env._target_slot_idx]
    env.data.qpos[slot.qpos_addr : slot.qpos_addr + 3] = position
    env.data.qpos[slot.qpos_addr + 3 : slot.qpos_addr + 7] = quat
    env.data.qvel[slot.dof_addr : slot.dof_addr + 6] = 0.0 if velocity is None else velocity
    mujoco.mj_forward(env.model, env.data)


def _advance(env, seconds=0.5):
    count = int(np.ceil(seconds / (env._N_SUBSTEPS * env.model.opt.timestep)))
    for _ in range(count):
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    return info


def _place_cube(env, offset=(0.0, 0.0)):
    slot = env._slots[env._target_slot_idx]
    _pose(env, [*(_GOAL + offset), slot.obj.half_size])


def test_stable_cube_centroid_inside_visible_disc_succeeds(make_env):
    env = make_env()
    # Part of this cube lies outside the disc. Center mode still accepts it.
    _place_cube(env, offset=(0.035, 0.0))
    info = _advance(env)
    assert info["success"] is True
    assert info["is_obj_placed"] is True
    assert info["geometric_placement_ok"] is True
    assert info["supported_by_table"] is True
    assert info["supported_by_robot"] is False
    assert info["supported_by_other"] is False
    assert info["placement_centroid_xy"] == pytest.approx(_GOAL + np.array([0.035, 0.0]), abs=1e-4)
    assert info["placement_dwell"] >= env.config.placement_dwell_time
    assert info["table_support_force"] == pytest.approx(0.01 * 9.81, rel=0.03)
    json.dumps(info)
    obs = env._get_obs()
    assert env.observation_space.contains(obs)
    np.testing.assert_allclose(obs, env.data.qpos[env._qpos_addrs], atol=1e-7)


def test_centroid_outside_disc_fails_after_sustained_support(make_env):
    env = make_env()
    _place_cube(env, offset=(0.045, 0.0))
    info = _advance(env)
    assert info["supported_by_table"] is True
    assert info["placement_dwell"] >= env.config.placement_dwell_time
    assert info["geometric_placement_ok"] is False
    assert info["is_obj_placed"] is False
    assert info["success"] is False


def test_flying_and_sliding_objects_fail_and_clear_dwell(make_env):
    env = make_env()
    _place_cube(env)
    assert _advance(env)["success"] is True
    _pose(env, [*_GOAL, 0.2], velocity=[0.1, 0.0, 0.2, 0.0, 0.0, 0.0])
    info = env._get_info()
    assert info["geometric_placement_ok"] is True
    assert info["supported_by_table"] is False
    assert info["success"] is False
    assert _advance(env, seconds=0.02)["placement_dwell"] == 0.0

    _place_cube(env)
    assert _advance(env)["success"] is True
    slot = env._slots[env._target_slot_idx]
    env.data.qvel[slot.dof_addr] = 0.4
    mujoco.mj_forward(env.model, env.data)
    info = env._get_info()
    assert info["object_linear_speed"] > env.config.object_static_lin_threshold
    assert info["is_obj_static"] is False
    assert info["success"] is False
    assert _advance(env, seconds=0.02)["placement_dwell"] == 0.0


def test_one_finger_force_rejects_object_without_a_grasp(make_env):
    env = make_env()
    slot = env._slots[env._target_slot_idx]
    geom = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "fixed_jaw_sph_tip1")
    position = env.data.geom_xpos[geom].copy()
    position[0] += slot.obj.half_size + env.model.geom_size[geom, 0] - 0.002
    _pose(env, position)
    loaded_sides = set()
    force = np.zeros(6)
    for row in range(env.data.ncon):
        contact = env.data.contact[row]
        if contact.geom1 in slot.geom_ids:
            other = contact.geom2
        elif contact.geom2 in slot.geom_ids:
            other = contact.geom1
        else:
            continue
        mujoco.mj_contactForce(env.model, env.data, row, force)
        if np.linalg.norm(force[:3]) <= env.config.support_force_tolerance:
            continue
        if other in env._gripper_geom_ids:
            loaded_sides.add("fixed")
        elif other in env._jaw_geom_ids:
            loaded_sides.add("moving")
    assert loaded_sides == {"moving"}
    env.model.body_pos[env._target_body_id, :2] = env._get_object_pose()[:2]
    mujoco.mj_forward(env.model, env.data)
    info = env._get_info()
    assert info["geometric_placement_ok"] is True
    assert info["robot_support_force"] > env.config.support_force_tolerance
    assert info["is_grasped"] == 0.0
    assert info["supported_by_robot"] is True
    assert info["is_obj_placed"] is False
    assert info["success"] is False


def test_table_support_does_not_override_unopposed_gripper_force(make_env):
    half = 0.05
    env = make_env(objects=[CubeObject(half_size=half, color="red")])
    position = np.array([0.18, -0.06, half - 0.0001])
    env.model.body_pos[env._target_body_id, :2] = position[:2]
    _pose(env, position)
    info = env._get_info()
    assert info["supported_by_table"] is True
    assert info["supported_by_robot"] is True
    assert info["is_grasped"] == 0.0
    assert info["geometric_placement_ok"] is True
    assert info["is_obj_static"] is True
    assert info["robot_support_force"] > env.config.support_force_tolerance
    assert info["is_obj_placed"] is False
    assert info["success"] is False


def test_another_object_cannot_replace_table_support(make_env):
    env = make_env(distractors=[CubeObject(half_size=0.025, color="green")], n_distractors=1)
    distractor = env._distractor_slots[0]
    env.data.qpos[distractor.qpos_addr : distractor.qpos_addr + 7] = [
        *_GOAL,
        0.025,
        1.0,
        0.0,
        0.0,
        0.0,
    ]
    env.data.qvel[distractor.dof_addr : distractor.dof_addr + 6] = 0.0
    _pose(env, [*_GOAL, 0.05 + 0.0127])
    info = _advance(env)
    assert info["geometric_placement_ok"] is True
    assert info["supported_by_other"] is True
    assert info["supported_by_table"] is False
    assert info["success"] is False
    assert info["placement_dwell"] == 0.0


@pytest.mark.parametrize(
    "quat",
    [
        _IDENTITY,
        np.array([1.0, 1.0, 0.0, 0.0]) / np.sqrt(2),
        np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2),
    ],
)
def test_elongated_mesh_alternative_rest_orientations(make_env, mesh_box, quat):
    half_size = np.array([0.055, 0.018, 0.012])
    env = make_env(objects=[mesh_box(half_size)])
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    height = np.abs(rotation.reshape(3, 3)[2]) @ half_size
    _pose(env, [*_GOAL, height], quat)
    info = _advance(env)
    assert info["placement_centroid_xy"] == pytest.approx(_GOAL, abs=2e-4)
    assert info["supported_by_table"] is True
    assert info["success"] is True


def test_shifted_mesh_frame_preserves_geometry_reward_and_com_speed(make_env, mesh_box):
    half_size = np.array([0.055, 0.018, 0.012])
    shift = np.array([0.15, -0.08, 0.035])
    plain = make_env(objects=[mesh_box(half_size)])
    shifted = make_env(objects=[mesh_box(half_size, shift)])
    quat = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, quat)
    rotation = rotation.reshape(3, 3)
    center = np.array([*_GOAL, half_size[2]])
    _pose(plain, center, quat)
    _pose(shifted, center - rotation @ shift, quat)
    for env in (plain, shifted):
        env._refresh_reset_reference_state()
        assert _advance(env)["success"] is True
    infos = [env._get_info() for env in (plain, shifted)]
    for key in ("placement_centroid_xy", "obj_to_target_dist", "tcp_to_obj_dist", "task_potential"):
        assert infos[0][key] == pytest.approx(infos[1][key], abs=3e-4)

    # A fixed free-joint origin does not imply a fixed COM for a translated mesh.
    for env, offset in ((plain, np.zeros(3)), (shifted, shift)):
        _pose(env, center - rotation @ offset, quat, velocity=[0, 0, 0, 0, 0, 0.1])
    assert plain._get_info()["object_linear_speed"] == pytest.approx(0.0, abs=1e-8)
    shifted_info = shifted._get_info()
    assert shifted_info["object_linear_speed"] > shifted.config.object_static_lin_threshold
    assert shifted_info["is_obj_static"] is False
    assert shifted_info["success"] is False


def test_info_reads_do_not_accrue_dwell_and_reset_clears_it(make_env):
    env = make_env(reset_settle_frames=30)
    assert env._get_info()["placement_dwell"] == 0.0
    _place_cube(env)
    for _ in range(10):
        assert env._get_info()["placement_dwell"] == 0.0
        assert env._get_info()["success"] is False
    info = _advance(env, seconds=0.02)
    assert info["placement_dwell"] <= env._N_SUBSTEPS * env.model.opt.timestep
    before = info["placement_dwell"]
    for _ in range(10):
        assert env._get_info()["placement_dwell"] == before
    assert _advance(env)["success"] is True
    _, reset_info = env.reset(seed=3)
    assert reset_info["placement_dwell"] == 0.0
    assert reset_info["success"] is False


def test_a_motion_break_within_one_control_step_restarts_dwell(make_env, monkeypatch):
    env = make_env()
    _place_cube(env)
    assert _advance(env)["success"] is True
    slot = env._slots[env._target_slot_idx]
    settled_pose = env.data.qpos[slot.qpos_addr : slot.qpos_addr + 7].copy()
    real_step = mujoco.mj_step
    substep = 0

    def step_with_motion_break(model, data):
        nonlocal substep
        substep += 1
        if substep == 2:
            data.qvel[slot.dof_addr] = 0.4
        elif substep == 3:
            data.qpos[slot.qpos_addr : slot.qpos_addr + 7] = settled_pose
            data.qvel[slot.dof_addr : slot.dof_addr + 6] = 0.0
        real_step(model, data)

    # Only the physical pose/velocity changes. Every substep still uses the real solver.
    monkeypatch.setattr(mujoco, "mj_step", step_with_motion_break)
    _, _, _, _, info = env.step(np.zeros(env.action_space.shape, dtype=np.float32))
    assert info["supported_by_table"] is True
    assert info["is_obj_static"] is True
    assert info["is_obj_placed"] is True
    assert info["success"] is False
    assert info["placement_dwell"] == pytest.approx((env._N_SUBSTEPS - 2) * env.model.opt.timestep)


def test_config_versions_cannot_cross_environment_versions():
    with pytest.raises(TypeError):
        PickAndPlaceEnv(config=PickAndPlaceV2Config())
    with pytest.raises(TypeError):
        PickAndPlaceV2Env(config=PickAndPlaceConfig())


def test_contract_tracks_active_object_thresholds_and_timestep(make_env):
    env = make_env(objects=[CubeObject(half_size=0.01), CubeObject(half_size=0.02)])
    env.reset(seed=3, options={"target_index": 1})
    old = env._get_info()["placement_contract"]
    env.config.support_min_weight_fraction = 0.95
    env.model.opt.timestep = 0.01
    current = json.loads(json.dumps(env._get_info()["placement_contract"]))
    assert current["target_index"] == 1
    assert current["physics_timestep"] == 0.01
    active = current["objects"][current["target_index"]]
    assert active["object"] == repr(env._slots[1].obj)
    assert active["required_upward_force_N"] == pytest.approx(0.95 * active["weight_N"])
    assert old["support_min_weight_fraction"] == 0.9
    current["objects"][1]["weight_N"] = -1
    assert env._get_info()["placement_contract"]["objects"][1]["weight_N"] > 0
