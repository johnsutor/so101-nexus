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


def test_grasp_from_contacts_two_sided_and_isolation():
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    obj = torch.tensor([49, 49, 49])
    gripper = torch.zeros(60, dtype=torch.bool)
    gripper[30] = True
    jaw = torch.zeros(60, dtype=torch.bool)
    jaw[41] = True
    # world0: both fingers, strong -> grasp; world1: gripper only -> no;
    # world2: gripper strong but jaw sub-threshold -> no.
    contact_geom = torch.tensor([[49, 30], [49, 41], [49, 30], [49, 30], [49, 41]])
    contact_world = torch.tensor([0, 0, 1, 2, 2])
    normal_force = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.1])
    grasp = _grasp_from_contacts(
        contact_geom=contact_geom,
        contact_world=contact_world,
        normal_force=normal_force,
        nacon=5,
        obj_geom=obj,
        gripper_mask=gripper,
        jaw_mask=jaw,
        threshold=0.5,
        num_envs=3,
    )
    assert grasp.tolist() == [1.0, 0.0, 0.0]


def test_grasp_from_contacts_empty_is_zero():
    import torch

    from so101_nexus.warp.base_env import _grasp_from_contacts

    obj = torch.tensor([5, 5])
    mask = torch.zeros(10, dtype=torch.bool)
    grasp = _grasp_from_contacts(
        contact_geom=torch.zeros((4, 2), dtype=torch.long),
        contact_world=torch.zeros(4, dtype=torch.long),
        normal_force=torch.zeros(4),
        nacon=0,
        obj_geom=obj,
        gripper_mask=mask,
        jaw_mask=mask,
        threshold=0.5,
        num_envs=2,
    )
    assert grasp.tolist() == [0.0, 0.0]
