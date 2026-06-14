"""Drift guards: menagerie_constants must match the vendored SO101 MJCF."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from so101_nexus_core import get_so101_mujoco_model_path
from so101_nexus_maniskill import menagerie_constants as mc


def _bodies() -> dict[str, ET.Element]:
    root = ET.parse(get_so101_mujoco_model_path()).getroot()
    return {b.get("name"): b for b in root.iter("body")}


def _principal(fullinertia: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return (eigenvalues ascending, eigenvectors) of a MuJoCo fullinertia."""
    ixx, iyy, izz, ixy, ixz, iyz = fullinertia
    mat = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
    evals, evecs = np.linalg.eigh(mat)
    return evals, evecs


@pytest.mark.parametrize("body_name", list(mc.LINK_INERTIALS))
def test_inertial_mass_and_com_match_xml(body_name):
    body = _bodies()[body_name]
    inertial = body.find("inertial")
    xml_mass = float(inertial.get("mass"))
    xml_com = np.array([float(v) for v in inertial.get("pos").split()])
    const = mc.LINK_INERTIALS[body_name]
    assert const.mass == pytest.approx(xml_mass, rel=0, abs=1e-9)
    np.testing.assert_allclose(const.com_pos, xml_com, atol=1e-9)


@pytest.mark.parametrize("body_name", list(mc.LINK_INERTIALS))
def test_inertial_principal_moments_match_xml(body_name):
    body = _bodies()[body_name]
    inertial = body.find("inertial")
    fullinertia = [float(v) for v in inertial.get("fullinertia").split()]
    evals, _ = _principal(fullinertia)
    np.testing.assert_allclose(
        sorted(mc.LINK_INERTIALS[body_name].principal_moments), sorted(evals), atol=1e-12
    )


@pytest.mark.parametrize("body_name", list(mc.LINK_INERTIALS))
def test_inertial_full_tensor_reconstructs_xml(body_name):
    """principal_quat is load-bearing: _apply_menagerie_patches feeds it to
    set_cmass_local_pose, so a wrong orientation gives the wrong full inertia
    tensor while passing the mass/COM/sorted-moment checks. Reconstruct the full
    tensor from (principal_moments, principal_quat) and compare to the XML
    fullinertia matrix - this is invariant to eigenvector sign/order ambiguity.
    """
    from transforms3d.quaternions import quat2mat

    body = _bodies()[body_name]
    ixx, iyy, izz, ixy, ixz, iyz = [
        float(v) for v in body.find("inertial").get("fullinertia").split()
    ]
    expected = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]], dtype=np.float64)
    inertial = mc.LINK_INERTIALS[body_name]
    rot = quat2mat(inertial.principal_quat)
    reconstructed = rot @ np.diag(inertial.principal_moments) @ rot.T
    np.testing.assert_allclose(reconstructed, expected, atol=1e-10)


def test_tcp_offset_matches_xml():
    body = _bodies()["gripper"]
    site = next(s for s in body.findall("site") if s.get("name") == "gripperframe")
    xml_pos = np.array([float(v) for v in site.get("pos").split()])
    xml_quat = np.array([float(v) for v in site.get("quat").split()])
    np.testing.assert_allclose(mc.TCP_OFFSET_POS, xml_pos, atol=1e-9)
    np.testing.assert_allclose(mc.TCP_OFFSET_QUAT, xml_quat, atol=1e-9)


def test_joint_dynamics_match_sts3215_class():
    root = ET.parse(get_so101_mujoco_model_path()).getroot()
    sts = next(d for d in root.iter("default") if d.get("class") == "sts3215")
    joint = sts.find("joint")
    position = sts.find("position")
    assert pytest.approx(float(joint.get("frictionloss"))) == mc.JOINT_FRICTIONLOSS
    assert pytest.approx(float(joint.get("armature"))) == mc.JOINT_ARMATURE
    assert pytest.approx(float(joint.get("damping"))) == mc.PASSIVE_JOINT_DAMPING
    assert pytest.approx(float(position.get("kp"))) == mc.DRIVE_STIFFNESS
    kv = float(position.get("kv"))
    forcerange = [float(v) for v in position.get("forcerange").split()]
    assert pytest.approx(kv) == mc.STS3215_KV
    assert pytest.approx(forcerange[1]) == mc.FORCE_LIMIT
    # Drive damping folds passive joint damping into the controller drive.
    assert pytest.approx(kv + float(joint.get("damping"))) == mc.DRIVE_DAMPING


def test_joint_ranges_match_xml():
    root = ET.parse(get_so101_mujoco_model_path()).getroot()
    ranges = {
        j.get("name"): [float(v) for v in j.get("range").split()]
        for j in root.iter("joint")
        if j.get("range")
    }
    for name, (lo, hi) in mc.JOINT_LIMITS.items():
        np.testing.assert_allclose([lo, hi], ranges[name], atol=1e-9)


def test_gripper_friction_links_are_the_three_finger_bodies():
    assert mc.GRIPPER_FRICTION_LINKS == ("gripper", "moving_jaw_so101_v1", "camera_mount")
    assert mc.GRIPPER_STATIC_FRICTION == 1.0
    assert mc.GRIPPER_DYNAMIC_FRICTION == 1.0
    assert mc.GRIPPER_PATCH_RADIUS == 0.1
    assert mc.GRIPPER_MIN_PATCH_RADIUS == 0.1
