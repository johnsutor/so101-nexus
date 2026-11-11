"""Upstream camera and joint corrections preserve the validated local grasp model."""

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from so101_nexus import get_so101_mujoco_model_path


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(get_so101_mujoco_model_path()))


def test_upstream_wrist_camera_pose_and_fov(model):
    camera = model.camera("wrist_cam")
    np.testing.assert_allclose(camera.pos, [-0.00296, 0.05436, -0.05310])
    orientation = np.array([0.958280, -0.285792, -0.001556, 0.004547])
    np.testing.assert_allclose(camera.quat, orientation / np.linalg.norm(orientation))
    assert camera.fovy[0] == 61
    np.testing.assert_array_equal(model.cam_sensorsize[camera.id], 0)


@pytest.mark.parametrize(
    "name,kind,size,position,group",
    [
        (
            "wrist_camera_pcb",
            mujoco.mjtGeom.mjGEOM_BOX,
            [0.019, 0.019, 0.0005],
            [-0.00101, 0.05958, -0.04515],
            2,
        ),
        (
            "wrist_camera_lens",
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [0.007, 0.00475, 0],
            [-0.00298, 0.05669, -0.04954],
            1,
        ),
    ],
)
def test_upstream_camera_geometry_is_noncolliding(model, name, kind, size, position, group):
    geom = model.geom(name)
    assert geom.type[0] == kind
    assert geom.bodyid[0] == model.body("camera_mount").id
    np.testing.assert_allclose(geom.size, size)
    np.testing.assert_allclose(geom.pos, position)
    assert geom.group[0] == group
    assert geom.contype[0] == 0
    assert geom.conaffinity[0] == 0


def test_upstream_wrist_roll_range_matches_actuator(model):
    np.testing.assert_allclose(model.joint("wrist_roll").range, [-2.7438473, 2.8412063])
    np.testing.assert_allclose(
        model.joint("wrist_roll").range,
        model.actuator("wrist_roll").ctrlrange,
        atol=1e-5,
    )


def test_local_pad_contacts_tcp_and_numerics_are_preserved(model):
    for jaw in ("fixed_jaw", "moving_jaw"):
        for part in ("low", "high", "tip"):
            geom = model.geom(f"{jaw}_pad_{part}")
            assert geom.type[0] == mujoco.mjtGeom.mjGEOM_MESH
            assert geom.contype[0] != 0
            assert geom.condim[0] == 6
            np.testing.assert_allclose(geom.solref, [0.005, 1])
            np.testing.assert_allclose(geom.solimp, [0.95, 0.999, 0.001, 0.5, 2])
            np.testing.assert_allclose(geom.friction, [1, 5e-3, 5e-4])
    np.testing.assert_allclose(model.site("gripperframe").quat, [0, 0, 1, 0])
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT
    assert model.opt.timestep == 0.0025
    assert model.opt.impratio == 100
    assert model.opt.iterations == 20


def test_vendored_mesh_assets_and_revision_provenance_are_preserved():
    directory = Path(get_so101_mujoco_model_path()).parent
    assets = sorted((directory / "assets").glob("*.stl"))
    digest = hashlib.sha256()
    for path in assets:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    assert len(assets) == 19
    assert digest.hexdigest() == "81184e90bfcf1fdae7b48606ae184a7d83ae473ac88822e1731e149679e43681"
    assert "98a70c569c4c1b6da467a20012f766ee1f8dc521" in (directory / "PROVENANCE.md").read_text()
