"""Tests binding the MuJoCo backend to the vendored MuJoCo Menagerie SO-101 model.

Covers model bindings, tuned-physics options, grasp-geom selection, TCP pose,
and wrist-camera FOV randomization after the menagerie swap.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import tempfile

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import so101_nexus_mujoco  # noqa: F401 — registers envs
from so101_nexus_core import (
    get_so101_mujoco_model_dir,
    get_so101_mujoco_model_path,
)
from so101_nexus_core.config import (
    LookAtConfig,
    MoveConfig,
    PickAndPlaceConfig,
    PickConfig,
    ReachConfig,
)
from so101_nexus_core.observations import JointPositions, WristCamera
from so101_nexus_mujoco.base_env import SCENE_OPTION_XML, SO101NexusMuJoCoBaseEnv

_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

_ENVS = [
    ("MuJoCoReach-v1", ReachConfig),
    ("MuJoCoMove-v1", MoveConfig),
    ("MuJoCoLookAt-v1", LookAtConfig),
    ("MuJoCoPickLift-v1", PickConfig),
    ("MuJoCoPickAndPlace-v1", PickAndPlaceConfig),
]


def _raw_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(get_so101_mujoco_model_path()))


# --- vendored model file ---------------------------------------------------


def test_wrist_cam_has_no_physical_intrinsics():
    m = _raw_model()
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
    assert cam_id >= 0
    # Physical intrinsics removed so fovy-based randomization is honored.
    assert np.allclose(m.cam_sensorsize[cam_id], 0.0)
    assert m.cam_fovy[cam_id] > 0.0


def test_model_diff_bindings():
    m = _raw_model()

    def aid(n: str) -> int:
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n)

    def jid(n: str) -> int:
        return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)

    for n in _JOINTS:
        assert np.allclose(m.actuator_forcerange[aid(n)], [-2.94, 2.94])
    # wrist_roll joint upper is tighter than its advertised ctrlrange upper.
    assert m.jnt_range[jid("wrist_roll")][1] < 2.80
    assert m.actuator_ctrlrange[aid("wrist_roll")][1] > 2.80


# --- env construction / physics binding ------------------------------------


@pytest.mark.parametrize("env_id,cfg_cls", _ENVS)
def test_env_builds_and_binds_menagerie(env_id, cfg_cls):
    env = gym.make(env_id, config=cfg_cls()).unwrapped
    try:
        m = env.model
        for n in _JOINTS:
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0
            assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) >= 0
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "gripperframe") >= 0
        assert mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam") >= 0
        env.reset(seed=0)
        env.step(env.action_space.sample())
    finally:
        env.close()


def test_control_step_invariant():
    env = gym.make("MuJoCoReach-v1", config=ReachConfig()).unwrapped
    try:
        # Wrapper option (after include) wins: timestep is the menagerie 0.005.
        assert abs(env.model.opt.timestep - 0.005) < 1e-9
        assert abs(env.model.opt.timestep * SO101NexusMuJoCoBaseEnv._N_SUBSTEPS - 0.02) < 1e-9
        # Tuned physics adopted.
        assert env.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        assert env.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    finally:
        env.close()


def test_wrapper_must_be_colocated_with_model(tmp_path):
    # Co-location is the load-bearing fact for mesh resolution. A wrapper beside
    # the model loads; the same wrapper in a foreign dir fails to find meshes.
    xml = (
        f'<mujoco model="probe"><compiler angle="radian"/>'
        f'<include file="{get_so101_mujoco_model_path()}"/>{SCENE_OPTION_XML}</mujoco>'
    )
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".xml", dir=get_so101_mujoco_model_dir(), delete=True
    ) as f:
        f.write(xml)
        f.flush()
        mujoco.MjModel.from_xml_path(f.name)  # loads

    bad = tmp_path / "probe.xml"
    bad.write_text(xml)
    with pytest.raises(Exception):
        mujoco.MjModel.from_xml_path(str(bad))


# --- grasp geom selection --------------------------------------------------


def test_grasp_geom_sets_select_condim6_fingers():
    env = gym.make("MuJoCoPickLift-v1", config=PickConfig()).unwrapped
    try:
        m = env.model

        def names(ids):
            return {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) for i in ids}

        g = names(env._gripper_geom_ids)
        j = names(env._jaw_geom_ids)
        assert any(n and n.startswith("fixed_jaw_") for n in g)
        assert any(n and n.startswith("moving_jaw_") for n in j)
        # Every selected finger geom uses condim 6 (no condim-3 wrist-roll box,
        # no camera-mount boxes).
        for i in env._gripper_geom_ids | env._jaw_geom_ids:
            assert m.geom_condim[i] == 6
    finally:
        env.close()


# --- TCP pose (task-semantic change) ---------------------------------------


def test_rest_tcp_pose_matches_menagerie():
    env = gym.make("MuJoCoReach-v1", config=ReachConfig(), robot_init_qpos_noise=0.0).unwrapped
    try:
        env.reset(seed=0, options={"init_qpos": np.zeros(6)})
        tcp = env._get_tcp_pose()
        # Pinned compiled menagerie TCP at zero qpos (differs from the old model's
        # ~[0.39136, -0.00001, 0.22647], a deliberate ~2 cm Z shift).
        assert np.allclose(
            tcp[:3], [0.3914432501, -0.0009794699, 0.2460073072], atol=1e-6
        )
        # Orientation reflects the library-convention gripperframe quat (0 0 1 0).
        assert np.allclose(
            tcp[3:], [0.7064841888, 0.0172191552, 0.7073102466, 0.0171990353], atol=1e-6
        )
    finally:
        env.close()


# --- wrist camera FOV randomization ----------------------------------------


def test_wrist_fov_randomization_changes_projection():
    cfg = ReachConfig(
        observations=[
            JointPositions(),
            WristCamera(width=64, height=64, fov_deg_range=(40.0, 90.0)),
        ]
    )
    env = gym.make("MuJoCoReach-v1", config=cfg).unwrapped
    try:
        env.reset(seed=1)
        fov_a = float(env.model.cam_fovy[env._wrist_cam_id])
        env.reset(seed=7)
        fov_b = float(env.model.cam_fovy[env._wrist_cam_id])
        # fovy is actually written (intrinsics stripped) and varies by seed.
        assert fov_a != fov_b
        assert 40.0 <= fov_a <= 90.0
        assert 40.0 <= fov_b <= 90.0
    finally:
        env.close()
