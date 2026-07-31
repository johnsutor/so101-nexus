"""MuJoCo tests for the grasp predicate, target pinning, and the force observations.

The grasp cases drive real physics rather than synthetic contacts: the point of
the predicate change is what MuJoCo reports for two specific geometries, which a
mocked contact list cannot show.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import gymnasium as gym
import mujoco
import numpy as np
import pytest

import so101_nexus.mujoco  # noqa: F401 - registers envs
from so101_nexus.config import PickAndPlaceConfig, PickConfig, RobotConfig
from so101_nexus.objects import CubeObject
from so101_nexus.observations import (
    GraspState,
    GripperContactForce,
    JointEfforts,
    JointPositions,
)
from so101_nexus.testing import component_slice

_OPEN = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
_CLOSE = np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
_LIFT = np.array([0, 0, 0.4, 0, 0, 0, -1], dtype=np.float32)


def _pick_env(*, half_size=0.012, robot=None, objects=None, observations=None, n_distractors=0):
    config = PickConfig(
        objects=objects if objects is not None else [CubeObject(half_size=half_size, color="red")],
        n_distractors=n_distractors,
        observations=observations,
        robot=robot if robot is not None else RobotConfig(),
    )
    return gym.make("MuJoCoPickLift-v1", config=config, control_mode="pd_ee_delta_pose").unwrapped


def _fingertip_geoms(env):
    return [
        g
        for g in list(env._gripper_geom_ids) + list(env._jaw_geom_ids)
        if env.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE
    ]


def _pinch_the_target(env, steps=60):
    """Close the jaws on the target held between the fingertips."""
    for _ in range(60):
        env.step(_OPEN)
    tips = _fingertip_geoms(env)
    slot = env._slots[env._target_slot_idx]
    for _ in range(steps):
        mid = env.data.geom_xpos[tips].mean(0)
        env.data.qpos[slot.qpos_addr : slot.qpos_addr + 3] = mid
        env.data.qpos[slot.qpos_addr + 3 : slot.qpos_addr + 7] = [1, 0, 0, 0]
        env.data.qvel[slot.dof_addr : slot.dof_addr + 6] = 0.0
        env.step(_CLOSE)


#: Cube half-size (m) far wider than the jaw can close on, so the fingers can
#: only ever press its near face. Compiled at this size rather than resized at
#: runtime: geom_rbound and the body BVH are compile-time broadphase data.
_OVER_WIDE_HALF_SIZE = 0.05


def _straddle_the_target(env, penetration=0.04):
    """Press both finger sets into the same face of an over-wide target.

    Reproduces the reported defect geometry: an object the jaw cannot close on
    is contacted bilaterally, but the two sides push it the same way rather than
    pinching it, so their force-weighted mean normals fail to oppose. (They are
    not all parallel: at this depth one jaw geom reaches the cube's top face and
    dominates that side's mean, which is still nowhere near opposing the
    gripper's.) The penetration is deep because MuJoCo's solve concentrates the
    reaction on a single contact until both finger sets are firmly loaded; below
    about 0.03 m only one side carries force and the predicate is uninteresting.
    """
    half = _OVER_WIDE_HALF_SIZE
    fingers = list(env._gripper_geom_ids) + list(env._jaw_geom_ids)
    x_face = max(env.data.geom_xpos[g][0] for g in fingers) - penetration
    slot = env._slots[env._target_slot_idx]
    env.data.qpos[slot.qpos_addr : slot.qpos_addr + 3] = [x_face + half, 0.0, half]
    env.data.qpos[slot.qpos_addr + 3 : slot.qpos_addr + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(env.model, env.data)


def _loaded_finger_sides(env):
    """Return which finger sets currently bear force against the target geom."""
    sides = set()
    force = np.zeros(6)
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        pair = (contact.geom1, contact.geom2)
        if env._obj_geom_id not in pair:
            continue
        other = pair[1] if pair[0] == env._obj_geom_id else pair[0]
        mujoco.mj_contactForce(env.model, env.data, i, force)
        if abs(force[0]) < env.config.robot.grasp_force_threshold:
            continue
        if other in env._gripper_geom_ids:
            sides.add("gripper")
        elif other in env._jaw_geom_ids:
            sides.add("jaw")
    return sides


def test_straddling_an_over_wide_object_is_not_a_grasp():
    """Bilateral loaded contact with same-side normals must read 0.0.

    Regression for the reported defect: the fingers press one face of an object
    too wide to close on, which the old contact-count predicate scored as a
    grasp while the object rested on the table bearing no load.
    """
    env = _pick_env(half_size=_OVER_WIDE_HALF_SIZE)
    try:
        env.reset(seed=0)
        _straddle_the_target(env)
        assert _loaded_finger_sides(env) == {"gripper", "jaw"}
        assert env._is_grasping() == 0.0
    finally:
        env.close()


def test_opposing_normal_threshold_of_minus_one_restores_contact_only_behaviour():
    """The config knob is load bearing: -1.0 scores the same straddle as a grasp."""
    env = _pick_env(
        half_size=_OVER_WIDE_HALF_SIZE,
        robot=RobotConfig(grasp_opposing_normal_threshold=-1.0),
    )
    try:
        env.reset(seed=0)
        _straddle_the_target(env)
        assert _loaded_finger_sides(env) == {"gripper", "jaw"}
        assert env._is_grasping() == 1.0
    finally:
        env.close()


def test_a_real_pinch_still_grasps_and_carries_the_object():
    """The stricter predicate must not cost a genuine, load-bearing grasp."""
    env = _pick_env()
    try:
        env.reset(seed=3)
        _pinch_the_target(env)
        assert env._is_grasping() == 1.0
        z0 = float(env._get_target_pose()[2])
        for _ in range(80):
            env.step(_LIFT)
        assert env._is_grasping() == 1.0
        assert float(env._get_target_pose()[2]) - z0 > env.config.lift_threshold
    finally:
        env.close()


def test_gripper_contact_force_tracks_the_squeeze():
    """The force observation is zero in free space and non-zero while pinching."""
    env = _pick_env(
        observations=[JointPositions(), GripperContactForce(), GraspState()],
    )
    try:
        obs, _ = env.reset(seed=3)
        force_slice = component_slice(env, GripperContactForce)
        assert np.allclose(obs[force_slice], 0.0)
        _pinch_the_target(env)
        obs = env._get_obs()
        assert np.linalg.norm(obs[force_slice]) > 0.0
    finally:
        env.close()


def test_gripper_contact_force_points_away_from_what_the_fingers_push():
    """Magnitude alone would pass a flipped sign or a transposed contact frame.

    The over-wide cube sits on the +x side of the fingers, so Newton's third law
    puts the resultant *on the gripper* along -x.
    """
    env = _pick_env(
        half_size=_OVER_WIDE_HALF_SIZE,
        observations=[JointPositions(), GripperContactForce()],
    )
    try:
        env.reset(seed=0)
        _straddle_the_target(env)
        force = env._get_gripper_contact_force()
        assert force[0] < -1.0, force
    finally:
        env.close()


def test_joint_efforts_are_the_live_actuator_forces():
    """The effort slice tracks data.qfrc_actuator, not a constant."""
    env = _pick_env(observations=[JointPositions(), JointEfforts()])
    try:
        env.reset(seed=0)
        effort_slice = component_slice(env, JointEfforts)
        seen = []
        for _ in range(10):
            obs, *_ = env.step(_CLOSE)
            np.testing.assert_allclose(
                obs[effort_slice],
                env.data.qfrc_actuator[env._qvel_addrs].astype(np.float32),
                rtol=1e-6,
            )
            seen.append(obs[effort_slice].copy())
        assert not np.allclose(seen[0], seen[-1])
    finally:
        env.close()


_POOL = [CubeObject(half_size=0.02, color=c) for c in ("red", "blue", "green")]


def test_target_index_option_selects_the_target():
    """reset(options={'target_index': k}) makes slot k the target and reports it."""
    env = _pick_env(objects=_POOL, n_distractors=1)
    try:
        for k in range(len(_POOL)):
            _, info = env.reset(seed=11, options={"target_index": k})
            assert info["target_index"] == k
            assert info["target_object"] == repr(_POOL[k])
            assert env._obj_geom_id == env._slots[k].geom_id
    finally:
        env.close()


def test_target_index_pins_the_target_without_moving_the_scene():
    """Counterfactual pairs: same seed, same object placements, different target.

    This is the recipe the option exists for - one seeded draw entangles layout
    and target, so a language-conditioned policy can never be shown two episodes
    that differ only in which object is named. The unpinned reset is the
    baseline on purpose: comparing two pinned resets to each other would still
    pass if the pin shifted the RNG stream by the same amount every time.
    """
    env = _pick_env(objects=_POOL, n_distractors=2)
    try:
        _, info = env.reset(seed=5)
        baseline = env.data.qpos.copy()
        drawn = info["target_index"]

        _, pinned = env.reset(seed=5, options={"target_index": drawn})
        assert pinned["target_index"] == drawn
        np.testing.assert_array_equal(env.data.qpos, baseline)

        other = (drawn + 1) % len(_POOL)
        _, relabelled = env.reset(seed=5, options={"target_index": other})
        assert relabelled["target_index"] == other
        np.testing.assert_array_equal(env.data.qpos, baseline)
    finally:
        env.close()


def test_pick_and_place_target_index_pins_without_moving_the_scene():
    """The carried-object pin must leave the disc pose and colour untouched too.

    Pick-and-place draws the disc colour, the disc pose, and the object pose
    from the same generator after the target draw, so a pin that skips the draw
    shifts every one of them.
    """
    config = PickAndPlaceConfig(objects=_POOL)
    env = gym.make("MuJoCoPickAndPlace-v1", config=config).unwrapped
    try:
        _, info = env.reset(seed=7)
        baseline = env.data.qpos.copy()
        disc = env.model.body_pos[env._target_body_id].copy()
        colour = env.target_color_name
        drawn = info["target_index"]

        for k in range(len(_POOL)):
            _, pinned = env.reset(seed=7, options={"target_index": k})
            assert pinned["target_index"] == k
            np.testing.assert_array_equal(env.model.body_pos[env._target_body_id], disc)
            assert env.target_color_name == colour
            if k == drawn:
                np.testing.assert_array_equal(env.data.qpos, baseline)
    finally:
        env.close()


def test_target_index_out_of_range_raises():
    env = _pick_env(objects=_POOL, n_distractors=1)
    try:
        with pytest.raises(ValueError, match="target_index"):
            env.reset(seed=0, options={"target_index": len(_POOL)})
    finally:
        env.close()


def test_omitting_target_index_keeps_the_seeded_random_target():
    """The option is opt-in; without it the pool draw is unchanged and seeded."""
    env = _pick_env(objects=_POOL, n_distractors=1)
    try:
        _, first = env.reset(seed=9)
        _, again = env.reset(seed=9)
        assert first["target_index"] == again["target_index"]
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ["MuJoCoStackCube-v1", "MuJoCoMove-v1", "MuJoCoLookAt-v1"])
def test_target_index_on_a_poolless_task_raises(env_id):
    """Matches the Warp backend: a pin no task consumes is an error, not a no-op.

    Silently ignoring it would let a data-collection script record every episode
    under the wrong target label with nothing to signal the mistake.
    """
    env = gym.make(env_id)
    try:
        with pytest.raises(ValueError, match="no object pool"):
            env.reset(seed=0, options={"target_index": 0})
    finally:
        env.close()
