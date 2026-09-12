"""Unit tests for the Warp base-env helpers: mat->quat and grasp reduction."""

import pytest

pytestmark = pytest.mark.warp


def test_mat_to_quat_matches_mujoco_up_to_sign():
    import mujoco
    import numpy as np
    import torch

    from so101_nexus.warp.base_env import _mat_to_quat

    rng = np.random.default_rng(0)
    mats, refs = [], []
    for _ in range(64):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        m = np.zeros(9)
        mujoco.mju_quat2Mat(m, q)
        mats.append(m.reshape(3, 3))
        ref = np.zeros(4)
        mujoco.mju_mat2Quat(ref, m)
        refs.append(ref)
    got = _mat_to_quat(torch.tensor(np.stack(mats), dtype=torch.float64)).numpy()
    ref = np.stack(refs)
    # Compare up to sign (quaternion double cover): |dot| == 1.
    dots = np.abs((got * ref).sum(axis=1))
    np.testing.assert_allclose(dots, np.ones(len(refs)), atol=1e-9)


def _frames(normals):
    """Pack inward normals into contact frames whose first row is the normal."""
    import torch

    frame = torch.zeros((len(normals), 3, 3))
    frame[:, 0, :] = torch.tensor(normals, dtype=torch.float32)
    return frame


def _mask(per_world_geoms, ngeom):
    """Pack per-world target geom ids into the ``(num_envs, ngeom)`` boolean lookup."""
    import torch

    mask = torch.zeros((len(per_world_geoms), ngeom), dtype=torch.bool)
    for world, geoms in enumerate(per_world_geoms):
        mask[world, list(geoms)] = True
    return mask


def test_grasp_from_contacts_two_sided_and_isolation():
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    obj = _mask([[49], [49], [49]], ngeom=60)
    gripper = torch.zeros(60, dtype=torch.bool)
    gripper[30] = True
    jaw = torch.zeros(60, dtype=torch.bool)
    jaw[41] = True
    # world0: both fingers, strong, opposing -> grasp; world1: gripper only -> no;
    # world2: gripper strong but jaw sub-threshold -> no.
    contact_geom = torch.tensor([[49, 30], [49, 41], [49, 30], [49, 30], [49, 41]])
    contact_world = torch.tensor([0, 0, 1, 2, 2])
    normal_force = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.1])
    # The object is geom1 everywhere, so the stored frame normal is flipped by
    # the reduction; store the outward normal for each finger.
    frames = _frames(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    grasp = _grasp_from_contacts(
        contact_geom=contact_geom,
        contact_world=contact_world,
        contact_frame=frames,
        normal_force=normal_force,
        nacon=5,
        obj_mask=obj,
        gripper_mask=gripper,
        jaw_mask=jaw,
        threshold=0.5,
        opposing_threshold=0.3,
        num_envs=3,
    )
    assert grasp.tolist() == [1.0, 0.0, 0.0]


def test_grasp_from_contacts_rejects_same_side_straddle():
    """Both finger sets pressing the same face is not a grasp.

    This is the load-bearing half of the predicate: an object too wide for the
    jaw to close on is touched bilaterally while it rests on the table, and
    contact count alone cannot tell that apart from a pinch.
    """
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    obj = _mask([[49], [49]], ngeom=60)
    gripper = torch.zeros(60, dtype=torch.bool)
    gripper[30] = True
    jaw = torch.zeros(60, dtype=torch.bool)
    jaw[41] = True
    contact_geom = torch.tensor([[49, 30], [49, 41], [49, 30], [49, 41]])
    contact_world = torch.tensor([0, 0, 1, 1])
    normal_force = torch.ones(4)
    # world0: both fingers push the same face (parallel inward normals).
    # world1: a genuine pinch, for contrast under one identical call.
    frames = _frames([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    grasp = _grasp_from_contacts(
        contact_geom=contact_geom,
        contact_world=contact_world,
        contact_frame=frames,
        normal_force=normal_force,
        nacon=4,
        obj_mask=obj,
        gripper_mask=gripper,
        jaw_mask=jaw,
        threshold=0.5,
        opposing_threshold=0.3,
        num_envs=2,
    )
    assert grasp.tolist() == [0.0, 1.0]


def test_grasp_from_contacts_one_sided_contact_is_never_a_grasp():
    """The both-sides guard is what carries this at ``opposing_threshold <= 0``.

    At the default threshold a single side is already rejected by the dot test
    (a zero resultant gives dot 0), so only a non-positive threshold exercises
    the guard: without it, one finger touching would read as grasped.
    """
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    gripper = torch.zeros(60, dtype=torch.bool)
    gripper[30] = True
    jaw = torch.zeros(60, dtype=torch.bool)
    jaw[41] = True
    grasp = _grasp_from_contacts(
        contact_geom=torch.tensor([[49, 30]]),
        contact_world=torch.tensor([0]),
        contact_frame=_frames([[0.0, 0.0, 1.0]]),
        normal_force=torch.ones(1),
        nacon=1,
        obj_mask=_mask([[49]], ngeom=60),
        gripper_mask=gripper,
        jaw_mask=jaw,
        threshold=0.5,
        opposing_threshold=-1.0,
        num_envs=1,
    )
    assert grasp.tolist() == [0.0]


def test_grasp_from_contacts_opposing_threshold_minus_one_is_contact_only():
    """``-1.0`` restores the pre-0.4.14 bilateral-contact-only predicate."""
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    obj = _mask([[49]], ngeom=60)
    gripper = torch.zeros(60, dtype=torch.bool)
    gripper[30] = True
    jaw = torch.zeros(60, dtype=torch.bool)
    jaw[41] = True
    grasp = _grasp_from_contacts(
        contact_geom=torch.tensor([[49, 30], [49, 41]]),
        contact_world=torch.tensor([0, 0]),
        contact_frame=_frames([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        normal_force=torch.ones(2),
        nacon=2,
        obj_mask=obj,
        gripper_mask=gripper,
        jaw_mask=jaw,
        threshold=0.5,
        opposing_threshold=-1.0,
        num_envs=1,
    )
    assert grasp.tolist() == [1.0]


def test_grasp_from_contacts_empty_is_zero():
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    obj = _mask([[5], [5]], ngeom=10)
    mask = torch.zeros(10, dtype=torch.bool)
    grasp = _grasp_from_contacts(
        contact_geom=torch.zeros((4, 2), dtype=torch.long),
        contact_world=torch.zeros(4, dtype=torch.long),
        contact_frame=torch.zeros((4, 3, 3)),
        normal_force=torch.zeros(4),
        nacon=0,
        obj_mask=obj,
        gripper_mask=mask,
        jaw_mask=mask,
        threshold=0.5,
        opposing_threshold=0.3,
        num_envs=2,
    )
    assert grasp.tolist() == [0.0, 0.0]


def test_grasp_ignores_invalid_and_padded_contacts():
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    grasp = _grasp_from_contacts(
        contact_geom=torch.tensor([[2, 0], [2, 1], [999, -1], [2, 0]]),
        contact_world=torch.tensor([0, 0, 999, 0]),
        contact_frame=_frames([[-1.0, 0, 0], [1.0, 0, 0], [0, 0, 1], [1.0, 0, 0]]),
        normal_force=torch.tensor([1.0, 1.0, float("nan"), float("nan")]),
        nacon=torch.tensor(3),
        obj_mask=_mask([[2]], ngeom=3),
        gripper_mask=torch.tensor([True, False, False]),
        jaw_mask=torch.tensor([False, True, False]),
        threshold=0.5,
        opposing_threshold=0.3,
        num_envs=1,
    )
    assert grasp.tolist() == [1.0]


def test_step_reuses_contact_forces_but_external_queries_read_live_state(env_factory, monkeypatch):
    import mujoco_warp as mjw

    env = env_factory(backend="warp", task="PickLift")
    env.reset(seed=0)
    calls = 0
    original = mjw.contact_force

    def contact_force(*args):
        nonlocal calls
        calls += 1
        return original(*args)

    monkeypatch.setattr(mjw, "contact_force", contact_force)
    for expected in (1, 2):
        env.step(env._joint_qpos().clone())
        assert calls == expected
    env._is_grasping()
    env._is_grasping()
    assert calls == 4
    env._elapsed[0] = env.max_episode_steps
    _, _, _, truncated, _ = env.step(env._joint_qpos().clone())
    assert truncated.tolist() == [True, False]
    assert calls == 6
    assert env._contact_cache is None


def test_contact_query_can_be_captured_without_reading_count_on_cpu(env_factory):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    env = env_factory(backend="warp", task="PickLift", device="cuda")
    env.reset(seed=0)
    result = torch.empty(env.num_envs, device=env.device)
    graph = env._capture_torch_graph(lambda: result.copy_(env._is_grasping()), "Grasp")
    assert graph is not None
    for _ in range(2):
        env.step(env._joint_qpos().clone())
        graph.replay()
        torch.testing.assert_close(result, env._is_grasping())


def test_grasp_from_contacts_aggregates_across_object_parts():
    """Fingers landing on different convex parts of one object still oppose.

    A decomposed mesh spreads the two finger contacts over separate geoms of the
    same body. Reducing per geom would leave each part one-sided and score 0.
    """
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    gripper = torch.zeros(60, dtype=torch.bool)
    gripper[30] = True
    jaw = torch.zeros(60, dtype=torch.bool)
    jaw[41] = True
    grasp = _grasp_from_contacts(
        contact_geom=torch.tensor([[49, 30], [52, 41]]),
        contact_world=torch.tensor([0, 0]),
        contact_frame=_frames([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        normal_force=torch.ones(2),
        nacon=2,
        obj_mask=_mask([[49, 52]], ngeom=60),
        gripper_mask=gripper,
        jaw_mask=jaw,
        threshold=0.5,
        opposing_threshold=0.3,
        num_envs=1,
    )
    assert grasp.tolist() == [1.0]
