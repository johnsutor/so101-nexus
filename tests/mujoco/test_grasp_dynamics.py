"""Supported acquisition, loaded movement, and release under real physics."""

import mujoco
import numpy as np
import pytest

from so101_nexus.kinematics import quat_conjugate, quat_multiply, quat_to_rotvec


def _step_to(env, position, quaternion, *, gripper=-1):
    tcp = env._get_tcp_pose()
    action = np.zeros(7, dtype=np.float32)
    action[:3] = (position - tcp[:3]) / env._ee_delta_scale[:3]
    rotation = quat_to_rotvec(quat_multiply(quaternion, quat_conjugate(tcp[3:])))
    action[3:6] = rotation / env._ee_delta_scale[3:6]
    action[-1] = gripper
    env.step(action)


def _relative_object_position(env):
    rotation = env.data.site_xmat[env._tcp_site_id].reshape(3, 3)
    return rotation.T @ (env._get_target_pose()[:3] - env._get_tcp_pose()[:3])


@pytest.mark.parametrize("mass", [0.005, 0.01, 0.1])
@pytest.mark.parametrize("half_size,angle_deg", [(0.009525, 0), (0.012, 0), (0.02, 0), (0.012, 45)])
def test_supported_pick_survives_hold_reversals_and_releases(
    supported_pick_factory, half_size, angle_deg, mass
):
    """A freely acquired cube must stay in a shallow pinch until released."""
    env, pose = supported_pick_factory(half_size=half_size, mass=mass, angle_deg=angle_deg)
    env.reset(seed=3)
    slot = env._slots[env._target_slot_idx]
    env.data.qpos[slot.qpos_addr : slot.qpos_addr + 7] = pose
    env.data.qvel[slot.dof_addr : slot.dof_addr + 6] = 0
    mujoco.mj_forward(env.model, env.data)
    desired = env._get_tcp_pose()
    # Place the cube once; its physical support carries it while the jaws close.
    for step in range(100):
        _step_to(env, desired[:3], desired[3:])
        if step >= 90:
            assert env._is_grasping() == 1
    start = env._get_tcp_pose()[:3]
    initial_height = float(env._get_target_pose()[2])
    relative = _relative_object_position(env)
    fingers = env._gripper_geom_ids | env._jaw_geom_ids
    target_geoms = set(env._obj_geom_ids)
    observations = []
    waypoints = (
        ("lift", (0, 0, 0.10), 50),
        ("hold", (0, 0, 0.10), round(3 / env.control_dt)),
        ("forward", (0.04, 0, 0.10), 30),
        ("backward", (-0.04, 0, 0.10), 60),
        ("center", (0, 0, 0.10), 30),
        ("right", (0, 0.04, 0.10), 30),
        ("left", (0, -0.04, 0.10), 60),
        ("return", (0, 0, 0.10), 30),
    )
    previous = start
    for phase, offset, count in waypoints:
        destination = start + np.array(offset)
        for fraction in np.linspace(1 / count, 1, count):
            position = previous + fraction * (destination - previous)
            _step_to(env, position, desired[3:])
            penetration = max(
                (
                    -contact.dist
                    for contact in env.data.contact
                    if (contact.geom1 in target_geoms and contact.geom2 in fingers)
                    or (contact.geom2 in target_geoms and contact.geom1 in fingers)
                ),
                default=0,
            )
            observations.append(
                (
                    phase,
                    float(np.linalg.norm(_relative_object_position(env) - relative)),
                    penetration,
                    env._is_grasping(),
                    float(env._get_target_pose()[2]),
                )
            )
        previous = destination
    slip = max(row[1] for row in observations)
    penetration = max(row[2] for row in observations)
    lost = {row[0] for row in observations if row[3] != 1}
    minimum_height = min(row[4] for row in observations if row[0] != "lift")
    metrics = f"slip={slip:.6f}, penetration={penetration:.6f}, lost={lost}"
    assert minimum_height - initial_height > 0.05, metrics
    assert not lost, metrics
    assert slip < 0.005, metrics
    assert penetration < 0.0015, metrics
    held_height = float(env._get_target_pose()[2])
    for _ in range(60):
        _step_to(env, previous, desired[3:], gripper=1)
    for _ in range(40):
        _step_to(env, previous - np.array([0, 0, 0.08]), desired[3:], gripper=1)
    assert env._is_grasping() == 0
    assert held_height - float(env._get_target_pose()[2]) > 0.05
