"""Locks the MJCF simulator TCP site to the URDF tool frame used by LeRobot.

End-effector actions are resolved against the MJCF ``gripperframe`` site, while
LeRobot's kinematics tooling resolves them against ``so101_new_calib.urdf``. If
the two frames drift apart, the same commanded pose means two different things in
simulation and on hardware, silently. These tests are the guard for that.

The URDF is parsed as XML rather than loaded through MuJoCo's URDF importer: the
importer rejects or reshapes the zero-inertia dummy links that carry the tool
frames, which would fail for reasons unrelated to frame agreement.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from so101_nexus import get_so101_mujoco_model_path, get_so101_urdf_path
from so101_nexus.config import SO101_JOINT_NAMES, SO101_TCP_FRAME_NAME, SO101_TCP_SITE_NAME

mujoco = pytest.importorskip("mujoco")

# The MJCF body that corresponds to the URDF ``gripper_link``; both tool frames
# hang off it as fixed offsets.
MJCF_GRIPPER_BODY = "gripper"
URDF_GRIPPER_LINK = "gripper_link"
TCP_FRAME_JOINT = "tcp_frame_joint"
UPSTREAM_FRAME_LINK = "gripper_frame_link"
UPSTREAM_FRAME_JOINT = "gripper_frame_joint"

# Upstream LeRobot puts its tool frame at the fixed fingertip, 19.9 mm behind the
# simulator site, which sits between the jaws. The gap is a convention difference,
# not a bug, and moving either frame would shift every recorded dataset.
UPSTREAM_FRAME_OFFSET_M = 0.0199
UPSTREAM_FRAME_OFFSET_TOL_M = 5e-5

POSITION_TOL_M = 1e-6
ORIENTATION_TOL_DEG = 1e-3
CONFIG_COUNT = 64
SEED = 20260724


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw to a rotation matrix (``Rz @ Ry @ Rx``)."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return (
        np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        @ np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        @ np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    )


def urdf_fixed_joint_origin(joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(xyz, rotation)`` origin of a fixed URDF joint off the gripper link."""
    root = ET.parse(get_so101_urdf_path()).getroot()
    for joint in root.iter("joint"):
        if joint.get("name") != joint_name:
            continue
        assert joint.get("type") == "fixed", (
            f"URDF joint {joint_name!r} must stay fixed; a movable tool frame would "
            "break the constant offset this test relies on."
        )
        parent = joint.find("parent")
        assert parent is not None, f"URDF joint {joint_name!r} is missing its <parent>."
        assert parent.get("link") == URDF_GRIPPER_LINK, (
            f"URDF joint {joint_name!r} must hang off {URDF_GRIPPER_LINK!r}, which is the "
            f"link the MJCF body {MJCF_GRIPPER_BODY!r} mirrors."
        )
        origin = joint.find("origin")
        assert origin is not None, f"URDF joint {joint_name!r} is missing its <origin>."
        xyz = np.fromstring(origin.get("xyz", ""), sep=" ")
        rpy = np.fromstring(origin.get("rpy", ""), sep=" ")
        return xyz, rpy_to_matrix(rpy)
    raise AssertionError(
        f"URDF joint {joint_name!r} is missing from {get_so101_urdf_path()}. The simulator "
        "end-effector modes need it to place the tool frame."
    )


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Geodesic angle in degrees between two rotation matrices."""
    trace = np.trace(a.T @ b)
    return float(np.degrees(np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))))


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(str(get_so101_mujoco_model_path()))


def sample_qpos(model, rng: np.random.Generator) -> np.ndarray:
    """Draw one joint configuration uniformly inside the SO101 joint limits."""
    values = np.zeros(model.nq)
    for name in SO101_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert joint_id >= 0, f"MJCF model is missing joint {name!r}."
        low, high = model.jnt_range[joint_id]
        values[model.jnt_qposadr[joint_id]] = rng.uniform(low, high)
    return values


def gripper_body_frames(model) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(body_pos, body_mat, site_pos, site_mat)`` over random configurations."""
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, MJCF_GRIPPER_BODY)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, SO101_TCP_SITE_NAME)
    assert body_id >= 0, f"MJCF model is missing body {MJCF_GRIPPER_BODY!r}."
    assert site_id >= 0, (
        f"MJCF model is missing site {SO101_TCP_SITE_NAME!r}, which the end-effector "
        "control modes read the tool pose from."
    )

    rng = np.random.default_rng(SEED)
    frames = []
    for _ in range(CONFIG_COUNT):
        data.qpos[:] = sample_qpos(model, rng)
        mujoco.mj_forward(model, data)
        frames.append(
            (
                data.xpos[body_id].copy(),
                data.xmat[body_id].reshape(3, 3).copy(),
                data.site_xpos[site_id].copy(),
                data.site_xmat[site_id].reshape(3, 3).copy(),
            )
        )
    return frames


def test_urdf_tcp_frame_matches_the_mjcf_simulator_site(model) -> None:
    tcp_xyz, tcp_rot = urdf_fixed_joint_origin(TCP_FRAME_JOINT)

    worst_pos_m = 0.0
    worst_ang_deg = 0.0
    for body_pos, body_mat, site_pos, site_mat in gripper_body_frames(model):
        urdf_pos = body_pos + body_mat @ tcp_xyz
        urdf_mat = body_mat @ tcp_rot
        worst_pos_m = max(worst_pos_m, float(np.linalg.norm(urdf_pos - site_pos)))
        worst_ang_deg = max(worst_ang_deg, rotation_angle_deg(urdf_mat, site_mat))

    detail = (
        f"worst position error {worst_pos_m * 1e3:.6f} mm (limit "
        f"{POSITION_TOL_M * 1e3:g} mm), worst orientation error {worst_ang_deg:.6f} deg "
        f"(limit {ORIENTATION_TOL_DEG:g} deg) over {CONFIG_COUNT} random configurations."
    )
    cause = (
        f"The MJCF site {SO101_TCP_SITE_NAME!r} and the URDF {SO101_TCP_FRAME_NAME!r} frame no "
        f"longer describe the same point. Either the site moved in so101.xml, the "
        f"{TCP_FRAME_JOINT!r} origin changed in so101_new_calib.urdf, or the kinematic chain "
        f"above {URDF_GRIPPER_LINK!r} diverged between the two models. Until they agree, an "
        "end-effector pose command means one thing in simulation and another on hardware."
    )
    assert worst_pos_m < POSITION_TOL_M, f"{cause} {detail}"
    assert worst_ang_deg < ORIENTATION_TOL_DEG, f"{cause} {detail}"


def test_upstream_gripper_frame_offset_is_left_untouched() -> None:
    """The upstream LeRobot tool frame must keep its 19.9 mm fingertip offset.

    ``tcp_frame_link`` was added precisely so nobody has to "fix" this one; moving
    it would change what every existing URDF-based LeRobot pipeline computes.
    """
    tcp_xyz, tcp_rot = urdf_fixed_joint_origin(TCP_FRAME_JOINT)
    upstream_xyz, upstream_rot = urdf_fixed_joint_origin(UPSTREAM_FRAME_JOINT)
    offset = tcp_xyz - upstream_xyz
    offset_m = float(np.linalg.norm(offset))

    assert abs(offset_m - UPSTREAM_FRAME_OFFSET_M) < UPSTREAM_FRAME_OFFSET_TOL_M, (
        f"The URDF {UPSTREAM_FRAME_LINK!r} frame is now {offset_m * 1e3:.4f} mm from "
        f"{SO101_TCP_FRAME_NAME!r}, not the expected {UPSTREAM_FRAME_OFFSET_M * 1e3:.1f} mm. "
        f"{UPSTREAM_FRAME_LINK!r} is upstream LeRobot's fingertip convention and must stay "
        f"where it is; {SO101_TCP_FRAME_NAME!r} is the frame that tracks the simulator site."
    )

    # The offset is documented as a pure translation along the gripper-link x axis.
    # A lateral component or a relative rotation means a frame was re-authored
    # rather than offset, which the scalar distance above would not catch.
    assert abs(offset[0]) == pytest.approx(offset_m, abs=1e-5), (
        f"The {UPSTREAM_FRAME_LINK!r} offset {offset} is no longer along the gripper-link "
        "x axis, so the two tool frames differ by more than the documented fingertip gap."
    )
    assert rotation_angle_deg(tcp_rot, upstream_rot) < ORIENTATION_TOL_DEG, (
        f"The URDF frames {UPSTREAM_FRAME_LINK!r} and {SO101_TCP_FRAME_NAME!r} no longer share "
        "an orientation, so the fingertip gap is not a pure translation any more."
    )
