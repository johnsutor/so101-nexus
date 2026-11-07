"""Free-object grasp retention under motion on Warp CPU and CUDA."""

import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp

from so101_nexus.kinematics import quat_conjugate, quat_multiply, quat_to_rotvec

pytestmark = pytest.mark.warp


def _step_to(env, position, quaternion, *, gripper=-1):
    tcp = env._get_tcp_pose7()
    action = torch.zeros((env.num_envs, 7), device=env.device)
    action[:, :3] = (position - tcp[:, :3]) / env._ee_delta_scale[:3]
    rotation = quat_to_rotvec(quat_multiply(quaternion, quat_conjugate(tcp[:, 3:])))
    action[:, 3:6] = rotation / env._ee_delta_scale[3:6]
    action[:, -1] = gripper
    return env.step(action)


def _local_offset(env):
    offset = env._target_pos() - env.site_xpos[:, env._tcp_site_id]
    rotation = env.site_xmat[:, env._tcp_site_id]
    return torch.bmm(rotation.transpose(1, 2), offset.unsqueeze(-1)).squeeze(-1)


def _penetration(env):
    count = int(env._nacon_view[0])
    geom = env._contact_geom_view[:count].long()
    world = env._contact_world_view[:count].long()
    finger = env._gripper_mask | env._jaw_mask
    target = env._obj_geom_mask
    selected = (target[world, geom[:, 0]] & finger[geom[:, 1]]) | (
        target[world, geom[:, 1]] & finger[geom[:, 0]]
    )
    distances = wp.to_torch(env.data.contact.dist)[:count][selected]
    return max(0.0, -float(distances.min())) if distances.numel() else 0.0


@pytest.mark.parametrize("device,graph", [("cpu", False), ("cuda", False), ("cuda", True)])
@pytest.mark.parametrize("mass", [0.005, 0.01, 0.1])
@pytest.mark.parametrize("half_size,angle_deg", [(0.009525, 0), (0.012, 0), (0.02, 0), (0.012, 45)])
def test_supported_pick_survives_hold_translation_and_reversal(
    supported_pick_factory, device, graph, half_size, angle_deg, mass
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    env, pose = supported_pick_factory(
        "warp",
        half_size=half_size,
        angle_deg=angle_deg,
        mass=mass,
        device=device,
        max_episode_steps=1024,
    )
    if graph:
        assert env._step_graph is not None
    else:
        env._step_graph = None
    env.reset(seed=3)
    cols = env._target_qadr[:, None] + torch.arange(7, device=env.device)
    dofs = env._target_dadr[:, None] + torch.arange(6, device=env.device)
    env.qpos[env._world_rows[:, None], cols] = torch.as_tensor(
        pose, device=env.device, dtype=torch.float32
    )
    env.qvel[env._world_rows[:, None], dofs] = 0
    with wp.ScopedDevice(env._wp_device):
        mjw.forward(env.model, env.data)
    desired = env._get_tcp_pose7().clone()
    for step in range(100):
        _step_to(env, desired[:, :3], desired[:, 3:])
        if step >= 90:
            assert (env._is_grasping() == 1.0).all()
    z0 = env._target_pos()[:, 2].clone()
    start = env.site_xpos[:, env._tcp_site_id].clone()
    initial = _local_offset(env).clone()
    waypoints = [
        ("lift", (0, 0, 0.10), 50),
        ("hold", (0, 0, 0.10), 150),
        ("forward", (0.04, 0, 0.10), 30),
        ("backward", (-0.04, 0, 0.10), 60),
        ("center", (0, 0, 0.10), 30),
        ("right", (0, 0.04, 0.10), 30),
        ("left", (0, -0.04, 0.10), 60),
        ("return", (0, 0, 0.10), 30),
    ]
    previous = start
    metrics = []
    for phase, offset, count in waypoints:
        destination = start + torch.tensor(offset, device=env.device)
        for fraction in np.linspace(1 / count, 1, count):
            position = previous + fraction * (destination - previous)
            _, _, terminated, truncated, _ = _step_to(env, position, desired[:, 3:])
            assert not (terminated | truncated).any()
            assert torch.isfinite(env.qpos).all()
            metrics.append(
                (
                    phase,
                    float(torch.linalg.vector_norm(_local_offset(env) - initial, dim=1).max()),
                    _penetration(env),
                    bool((env._is_grasping() == 1.0).all()),
                    float((env._target_pos()[:, 2] - z0).min()),
                )
            )
        previous = destination
    slip = max(row[1] for row in metrics)
    penetration = max(row[2] for row in metrics)
    lost = {row[0] for row in metrics if not row[3]}
    minimum_height = min(row[4] for row in metrics if row[0] != "lift")
    detail = f"slip={slip:.6f}, penetration={penetration:.6f}, lost={lost}"
    assert minimum_height > env.config.lift_threshold, detail
    assert not lost, detail
    assert slip < 0.005, detail
    assert penetration < 0.0015, detail
    held_z = env._target_pos()[:, 2].clone()
    for _ in range(60):
        _step_to(env, previous, desired[:, 3:], gripper=1)
    for _ in range(40):
        _step_to(
            env,
            previous - torch.tensor([0, 0, 0.08], device=env.device),
            desired[:, 3:],
            gripper=1,
        )
    assert (env._is_grasping() == 0.0).all(), "opening the jaws did not release the cube"
    assert (held_z - env._target_pos()[:, 2] > env.config.lift_threshold).all(), (
        "the released cube did not fall after withdrawing the open jaws"
    )
