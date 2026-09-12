"""Physical support, dwell, and batch isolation for center-placement v2."""

import numpy as np
import pytest

pytestmark = pytest.mark.warp


def test_cuda_graph_replays_physics_and_substep_dwell(env_factory):
    import torch
    import warp as wp

    from so101_nexus import PickAndPlaceV2Config

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    env = env_factory(
        backend="warp",
        task="PickAndPlace",
        version=2,
        device="cuda",
        config=PickAndPlaceV2Config(reset_settle_frames=0),
    )
    assert env._step_graph is not None
    env.reset(seed=3)
    _put_object(env, 0)
    _put_object(env, 1, height=2.0)
    _forward(env)
    graph = env._step_graph
    with wp.ScopedDevice(env._wp_device):
        for _ in range(3):
            state = {
                name: wp.clone(value)
                for name, value in vars(env.data).items()
                if isinstance(value, wp.array)
            }
            dwell = env._placement_dwell.clone()
            env._advance_physics()
            expected_qpos = env.qpos.clone()
            expected_dwell = env._placement_dwell.clone()
            for name, value in state.items():
                wp.copy(getattr(env.data, name), value)
            env._placement_dwell.copy_(dwell)
            env._step_graph = None
            env._advance_physics()
            torch.testing.assert_close(env.qpos, expected_qpos)
            torch.testing.assert_close(env._placement_dwell, expected_dwell)
            assert env._placement_dwell[0] > dwell[0]
            assert env._placement_dwell[1] == 0
            env._step_graph = graph
        env.config.support_min_weight_fraction = 100.0
        env._advance_physics()
        assert not env._placement_dwell.any()


@pytest.fixture
def env():
    from so101_nexus.config import PickAndPlaceV2Config
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceV2VectorEnv

    instance = WarpPickAndPlaceV2VectorEnv(
        num_envs=2,
        config=PickAndPlaceV2Config(reset_settle_frames=0, terminate_on_success=False),
        device="cpu",
        seed=3,
    )
    instance.reset(seed=3)
    yield instance
    instance.close()


def _put_object(env, world, *, height=None, xy=(0.7, 0.0)):
    import torch

    qadr = int(env._target_qadr[world])
    dadr = int(env._target_dadr[world])
    if height is None:
        height = env.config.cube_half_size - 1e-5
    env.qpos[world, qadr : qadr + 3] = torch.tensor((*xy, height), device=env.device)
    env.qpos[world, qadr + 3 : qadr + 7] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device)
    env.qvel[world, dadr : dadr + 6] = 0.0
    env._mocap_pos[world, env._target_mocap_id, :2] = torch.tensor(xy, device=env.device)


def _forward(env):
    import mujoco_warp as mjw
    import warp as wp

    with wp.ScopedDevice(env._wp_device):
        mjw.forward(env.model, env.data)


def test_resting_placement_requires_physics_dwell_and_queries_do_not_add_time(env):
    import torch

    _put_object(env, 0)
    _put_object(env, 1, height=2.0)
    _forward(env)
    first = env._placement_info()
    assert first["geometric_placement_ok"].tolist() == [True, True]
    assert first["supported_by_table"].tolist() == [True, False]
    assert not first["success"].any()
    for _ in range(3):
        assert torch.equal(env._placement_info()["placement_dwell"], first["placement_dwell"])
    action = env._joint_qpos().clone()
    for _ in range(30):
        _, _, terminated, _, info = env.step(action)
        assert not terminated.any()
        assert not info["success"][1]
        if info["success"][0]:
            break
    assert info["success"].tolist() == [True, False]
    assert info["placement_dwell"][0] >= env.config.placement_dwell_time
    assert info["placement_dwell"][1] == 0
    assert info["table_support_force"][0] > 0


def test_other_object_support_does_not_count_as_table_support():
    import torch

    from so101_nexus.config import PickAndPlaceV2Config
    from so101_nexus.objects import CubeObject
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceV2VectorEnv

    cfg = PickAndPlaceV2Config(
        n_distractors=1,
        distractors=[CubeObject(color="green")],
        reset_settle_frames=0,
        terminate_on_success=False,
    )
    instance = WarpPickAndPlaceV2VectorEnv(num_envs=1, config=cfg, device="cpu", seed=1)
    try:
        instance.reset(seed=1)
        half = cfg.cube_half_size
        _put_object(instance, 0, height=3 * half - 2e-5)
        qa = instance._slot_qadr_host[instance._d_offset]
        instance.qpos[0, qa : qa + 7] = torch.tensor((0.7, 0.0, half - 1e-5, 1.0, 0.0, 0.0, 0.0))
        for _ in range(20):
            _, _, _, _, info = instance.step(instance._joint_qpos().clone())
        assert info["geometric_placement_ok"][0]
        assert info["supported_by_other"][0]
        assert not info["supported_by_table"][0]
        assert not info["is_obj_placed"][0]
        assert not info["success"][0]
    finally:
        instance.close()


def test_contact_force_reduction_handles_order_lateral_robot_force_and_stale_rows(env, monkeypatch):
    import mujoco
    import torch

    floor = mujoco.mj_name2id(env._mjm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    robot = int(env._placement_robot_geom_mask.nonzero()[0, 0])
    target = int(env._obj_geom_mask[0].nonzero()[0, 0])
    weight = float(env._placement_weights[env._target_slot[0]])
    # Both table contact orders contribute upward force. A lateral robot force
    # invalidates world 0, even when no two-sided grasp exists.
    geoms = torch.tensor([[floor, target], [target, floor], [robot, target], [-1, -1]])
    frames = torch.eye(3).repeat(4, 1, 1)
    forces = torch.tensor(
        [
            [0.0, 0.0, weight, 0.0, 0.0, 0.0],
            [0.0, 0.0, -weight, 0.0, 0.0, 0.0],
            [weight, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1e6, 1e6, 1e6, 0.0, 0.0, 0.0],
        ]
    )
    monkeypatch.setattr(env, "_contact_geom_view", geoms)
    monkeypatch.setattr(env, "_contact_frame_view", frames)
    monkeypatch.setattr(env, "_contact_world_view", torch.tensor([0, 1, 0, 99999]))
    monkeypatch.setattr(env, "_nacon_view", torch.tensor([3]))
    monkeypatch.setattr(env, "_placement_contact_rows", torch.arange(4))
    monkeypatch.setattr(env, "_placement_contact_forces", lambda: forces)
    _put_object(env, 0)
    _put_object(env, 1)
    _forward(env)
    info = env._placement_info()
    assert info["supported_by_table"].tolist() == [True, True]
    assert info["supported_by_robot"].tolist() == [True, False]
    assert info["is_obj_placed"].tolist() == [False, True]
    torch.testing.assert_close(info["table_support_force"], torch.full((2,), weight))


def test_substep_support_loss_restarts_dwell(env, monkeypatch):
    import mujoco_warp as mjw
    import torch
    import warp as wp

    _put_object(env, 0)
    _put_object(env, 1)
    _forward(env)
    env._placement_dwell.fill_(env.config.placement_dwell_time)
    # The second substep loses support only in world 0. A final-frame-only
    # evaluator incorrectly retains the full dwell duration.
    qualified = iter([[True, True], [False, True], [True, True], [True, True]])
    original = env._support_info

    def support():
        info = original()
        info["qualified"] = torch.tensor(next(qualified))
        return info

    monkeypatch.setattr(env, "_support_info", support)
    monkeypatch.setattr(mjw, "step", lambda *args: None)
    # step() normally supplies the device context for this internal method.
    with wp.ScopedDevice(env._wp_device):
        env._advance_physics()
    assert env._placement_dwell[0] == pytest.approx(2 * env._mjm.opt.timestep)
    assert env._placement_dwell[1] >= env.config.placement_dwell_time


def test_same_step_autoreset_preserves_transition_diagnostics_and_other_world_dwell(env):
    import torch

    _put_object(env, 0)
    _put_object(env, 1)
    _forward(env)
    for _ in range(30):
        env.step(env._joint_qpos().clone())
    env._elapsed[0] = env.max_episode_steps - 1
    before = env._placement_dwell.clone()
    _, _, _, truncated, info = env.step(env._joint_qpos().clone())
    assert truncated.tolist() == [True, False]
    assert info["placement_dwell"][0] >= before[0]
    assert env._placement_dwell[0] == 0
    assert env._placement_dwell[1] >= before[1]
    _, reset_info = env.reset(seed=8)
    assert torch.equal(reset_info["placement_dwell"], torch.zeros(2))
    assert not reset_info["success"].any()
    assert reset_info["placement_contract"]["placement_mode"] == "center"


def test_reset_settle_does_not_count_as_placement_dwell():
    import torch

    from so101_nexus.config import PickAndPlaceV2Config
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceV2VectorEnv

    instance = WarpPickAndPlaceV2VectorEnv(
        num_envs=1, config=PickAndPlaceV2Config(reset_settle_frames=15), device="cpu"
    )
    try:
        obs, info = instance.reset(seed=0)
        assert torch.equal(info["placement_dwell"], torch.zeros(1))
        assert not info["success"].any()
        assert obs.shape == instance.observation_space.shape
    finally:
        instance.close()


def test_scalar_and_warp_agree_on_equivalent_supported_and_airborne_scenes(env):
    import mujoco

    from so101_nexus.config import PickAndPlaceV2Config
    from so101_nexus.mujoco.pick_and_place import PickAndPlaceV2Env

    scalar = PickAndPlaceV2Env(config=PickAndPlaceV2Config(reset_settle_frames=0))
    try:
        scalar.reset(seed=0)
        for world, height in enumerate((env.config.cube_half_size - 1e-5, 0.5)):
            _put_object(env, world, height=height)
            # Match compiled object and robot coordinates by joint name. The
            # backends compile their object pools with different body names.
            for joint in range(scalar.model.njnt):
                name = mujoco.mj_id2name(scalar.model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                other = mujoco.mj_name2id(env._mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
                if other >= 0 and scalar.model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                    scalar.data.qpos[scalar.model.jnt_qposadr[joint]] = float(
                        env.qpos[world, env._mjm.jnt_qposadr[other]]
                    )
            free = np.flatnonzero(scalar.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0]
            qa = scalar.model.jnt_qposadr[free]
            scalar.data.qpos[qa : qa + 7] = (0.7, 0.0, height, 1.0, 0.0, 0.0, 0.0)
            scalar.data.qvel[:] = 0
            target_bid = mujoco.mj_name2id(scalar.model, mujoco.mjtObj.mjOBJ_BODY, "target")
            scalar.model.body_pos[target_bid, :2] = (0.7, 0.0)
            mujoco.mj_forward(scalar.model, scalar.data)
            _forward(env)
            scalar_info = scalar._placement_info()
            vector_info = env._placement_info()
            for key in (
                "geometric_placement_ok",
                "supported_by_table",
                "supported_by_robot",
                "supported_by_other",
                "is_obj_static",
                "is_obj_placed",
                "success",
            ):
                assert bool(vector_info[key][world]) == bool(scalar_info[key]), key
            np.testing.assert_allclose(
                vector_info["placement_centroid_xy"][world].numpy(),
                scalar_info["placement_centroid_xy"],
                atol=1e-6,
            )
    finally:
        scalar.close()


def test_shifted_mesh_frame_preserves_reference_and_measures_com_motion(tmp_path):
    import mujoco
    import torch

    from so101_nexus.config import PickAndPlaceV2Config
    from so101_nexus.objects import MeshObject
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceV2VectorEnv

    half = np.array([0.055, 0.018, 0.012])
    shift = np.array([0.15, -0.08, 0.035])
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
            ]
        )
        * half
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
    objects = []
    for index, offset in enumerate((np.zeros(3), shift)):
        path = tmp_path / f"box_{index}.obj"
        path.write_text(
            "".join(f"v {x} {y} {z}\n" for x, y, z in vertices + offset)
            + "".join(f"f {a} {b} {c}\n" for a, b, c in faces)
        )
        objects.append(MeshObject(str(path), str(path), mass=0.03, name="box"))
    instance = WarpPickAndPlaceV2VectorEnv(
        num_envs=2,
        config=PickAndPlaceV2Config(objects=objects, reset_settle_frames=0),
        device="cpu",
    )
    try:
        instance.reset(seed=0, options={"target_index": [0, 1]})
        for world, offset in enumerate((np.zeros(3), shift)):
            _put_object(
                instance,
                world,
                xy=np.array([0.7, 0.0]) - offset[:2],
                height=half[2] - offset[2],
            )
        instance._mocap_pos[:, instance._target_mocap_id, :2] = torch.tensor([0.6, 0.0])
        _forward(instance)
        reference = instance._placement_reference_pos()
        torch.testing.assert_close(reference[0], reference[1], atol=1e-6, rtol=0)
        # Reach and transport geometry use the visual reference, not the two
        # free-joint origins that describe this same visible placement.
        instance._initial_obj_z[:] = reference[:, 2]
        potential = instance._task_potential(
            reference, instance._target_disc_pos(), torch.ones(2), torch.zeros(2, dtype=torch.bool)
        )
        torch.testing.assert_close(potential[0], potential[1], atol=1e-6, rtol=0)
        for world in range(2):
            instance.qvel[world, int(instance._target_dadr[world]) + 5] = 0.4
        _forward(instance)
        info = instance._placement_info()
        assert info["is_obj_static"].tolist() == [True, False]
        assert info["object_linear_speed"][0] < 1e-6
        assert info["object_linear_speed"][1] > instance.config.object_static_lin_threshold
        scalar_data = mujoco.MjData(instance._mjm)
        for world in range(2):
            scalar_data.qpos[:] = instance.qpos[world].numpy()
            scalar_data.qvel[:] = instance.qvel[world].numpy()
            mujoco.mj_forward(instance._mjm, scalar_data)
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                instance._mjm,
                scalar_data,
                mujoco.mjtObj.mjOBJ_BODY,
                int(instance._placement_body_ids[world]),
                velocity,
                0,
            )
            assert float(info["object_linear_speed"][world]) == pytest.approx(
                np.linalg.norm(velocity[3:]), abs=1e-6
            )
    finally:
        instance.close()


@pytest.mark.parametrize("motion", ["slide", "throw"])
def test_real_object_motion_clears_only_its_world_dwell(env, motion):
    _put_object(env, 0, height=0.2 if motion == "throw" else None)
    _put_object(env, 1)
    dadr = int(env._target_dadr[0])
    env.qvel[0, dadr + (2 if motion == "throw" else 0)] = 1.0
    env._placement_dwell.fill_(env.config.placement_dwell_time)
    _forward(env)
    initial = env._placement_info()
    assert initial["geometric_placement_ok"][0]
    assert initial["object_linear_speed"][0] > env.config.object_static_lin_threshold
    assert not initial["is_obj_static"][0]
    assert not initial["success"][0]
    _, _, _, _, info = env.step(env._joint_qpos().clone())
    assert info["placement_dwell"][0] == 0
    assert not info["success"][0]
    assert info["placement_dwell"][1] >= env.config.placement_dwell_time


def test_contract_uses_active_device_timestep_and_independent_snapshots(env):
    import warp as wp

    old = env._placement_info()["placement_contract"]
    wp.to_torch(env.model.opt.timestep).fill_(0.01)
    env.config.support_min_weight_fraction = 0.95
    current = env._placement_info()["placement_contract"]
    assert current["physics_timestep"] == pytest.approx([0.01])
    assert current["target_index"] == env._target_slot.tolist()
    assert old["physics_timestep"] == pytest.approx([0.005])
    assert old["support_min_weight_fraction"] == 0.9
    current["objects"][0]["weight_N"] = -1
    assert env._placement_info()["placement_contract"]["objects"][0]["weight_N"] > 0
