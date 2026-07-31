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


def test_grasp_from_contacts_two_sided_and_isolation():
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    obj = torch.tensor([49, 49, 49])
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
        obj_geom=obj,
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

    obj = torch.tensor([49, 49])
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
        obj_geom=obj,
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
        obj_geom=torch.tensor([49]),
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

    obj = torch.tensor([49])
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
        obj_geom=obj,
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

    obj = torch.tensor([5, 5])
    mask = torch.zeros(10, dtype=torch.bool)
    grasp = _grasp_from_contacts(
        contact_geom=torch.zeros((4, 2), dtype=torch.long),
        contact_world=torch.zeros(4, dtype=torch.long),
        contact_frame=torch.zeros((4, 3, 3)),
        normal_force=torch.zeros(4),
        nacon=0,
        obj_geom=obj,
        gripper_mask=mask,
        jaw_mask=mask,
        threshold=0.5,
        opposing_threshold=0.3,
        num_envs=2,
    )
    assert grasp.tolist() == [0.0, 0.0]
