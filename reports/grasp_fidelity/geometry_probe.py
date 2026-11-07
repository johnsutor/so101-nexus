"""Reproduce supplemental supported-grasp geometry probes from the repository root.

Run ``uv run python reports/grasp_fidelity/geometry_probe.py`` for all 53 cases.
These exploratory probes mutate support dimensions after compilation without
refreshing compiled bounding radii. Large support offsets require confirmation
with a rebuilt scene. Translation includes lift transients, not just hold creep.
No hardware validation, randomized pose coverage, or reversal motion is included.
"""

import importlib.util
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from so101_nexus import PhysicsConfig
from so101_nexus.physics import apply_physics_config

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent


def module(path, name):
    """Load the existing fixture or helper module without copying its behavior."""
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cf = module(Path("tests/conftest.py"), "grip_cf")
test = module(Path("tests/mujoco/test_grasp_dynamics.py"), "grip_test")
variants = {
    "baseline": PhysicsConfig(),
    "impratio1000": PhysicsConfig(impratio=1000),
    "noslip5": PhysicsConfig(noslip_iterations=5),
    "dt1.25ms": PhysicsConfig(timestep_s=0.00125),
    "d0.99": PhysicsConfig(gripper_solimp=(0.99, 0.999, 0.001, 0.5, 2)),
}


def run(variant, depth, mass, angle):  # noqa: PLR0915
    """Acquire, lift, hold, and open one supported cube."""
    gen = cf.env_factory.__wrapped__()
    make_env = next(gen)
    mp = pytest.MonkeyPatch()
    try:
        make = cf.supported_pick_factory.__wrapped__(make_env, mp)
        env, pose = make(mass=mass, angle_deg=angle)
        env._N_SUBSTEPS = apply_physics_config(env.model, variants[variant], backend="mujoco")
        env.reset(seed=3)
        fingers = env._gripper_geom_ids | env._jaw_geom_ids
        if variant == "friction0":
            env.model.geom_friction[list(fingers)] = 0
        if variant == "torque10pct":
            env.model.actuator_forcerange[env.model.actuator("gripper").id] *= 0.1
        rotation = env.data.xmat[env._gripper_body_id].reshape(3, 3)
        shift = rotation[:, 2] * (depth - 0.008)
        pose[:3] += shift
        gid = env.model.geom("pickup_support").id
        top = pose[2] - 0.012
        env.model.geom_pos[gid] = [pose[0], pose[1], top / 2]
        env.model.geom_size[gid, 2] = top / 2
        slot = env._slots[env._target_slot_idx]
        env.data.qpos[slot.qpos_addr : slot.qpos_addr + 7] = pose
        env.data.qvel[slot.dof_addr : slot.dof_addr + 6] = 0
        mujoco.mj_forward(env.model, env.data)
        desired = env._get_tcp_pose()
        for _ in range(100):
            test._step_to(env, desired[:3], desired[3:])
        acquired = env._is_grasping() == 1
        start = env._get_tcp_pose()[:3].copy()
        initial_height = float(env._get_target_pose()[2])
        relative = test._relative_object_position(env).copy()
        slip = 0.0
        penetration = 0.0
        lost = 0
        fingers = env._gripper_geom_ids | env._jaw_geom_ids
        targets = set(env._obj_geom_ids)
        for step in range(250):
            desired_pos = start + np.array([0, 0, 0.10 * min((step + 1) / 50, 1)])
            test._step_to(env, desired_pos, desired[3:])
            slip = max(slip, float(np.linalg.norm(test._relative_object_position(env) - relative)))
            lost += int(env._is_grasping() != 1)
            penetration = max(
                penetration,
                max(
                    (
                        -c.dist
                        for c in env.data.contact
                        if (c.geom1 in targets and c.geom2 in fingers)
                        or (c.geom2 in targets and c.geom1 in fingers)
                    ),
                    default=0,
                ),
            )
        lift = float(env._get_target_pose()[2]) - initial_height
        held_height = float(env._get_target_pose()[2])
        for _ in range(100):
            test._step_to(env, desired_pos, desired[3:], gripper=1)
        drop = held_height - float(env._get_target_pose()[2])
        if variant == "baseline" and depth == 0.008 and mass == 0.1:
            info = {
                "qpos_deg": np.degrees(env.data.qpos[:6]).tolist(),
                "ctrl_deg": np.degrees(env.data.ctrl).tolist(),
                "target_pos": env._get_target_pose()[:3].tolist(),
                "tcp": env._get_tcp_pose()[:3].tolist(),
                "contacts": [
                    (
                        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom1)),
                        mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, int(c.geom2)),
                        float(c.dist),
                    )
                    for c in env.data.contact
                ],
            }
            (OUTPUT / "release_debug.json").write_text(json.dumps(info, indent=2))
        return {
            "variant": variant,
            "depth_mm": depth * 1000,
            "mass_g": mass * 1000,
            "angle_deg": angle,
            "acquired": bool(acquired),
            "slip_mm": slip * 1000,
            "penetration_mm": penetration * 1000,
            "lost_steps": lost,
            "lift_mm": lift * 1000,
            "release_drop_mm": drop * 1000,
        }
    finally:
        mp.undo()
        gen.close()


def main():
    """Write the original numerical sweep and its edge-case controls."""
    rows = []
    for depth, mass, angle in [
        (0, 0.01, 0),
        (0, 0.1, 0),
        (0.004, 0.1, 0),
        (0.008, 0.1, 45),
        (-0.004, 0.01, 0),
    ]:
        for variant in variants:
            start = time.perf_counter()
            row = run(variant, depth, mass, angle)
            row["wall_s"] = time.perf_counter() - start
            rows.append(row)
            print(json.dumps(row), flush=True)
    (OUTPUT / "geometry_results.json").write_text(json.dumps(rows, indent=2) + "\n")
    variants.update({"friction0": PhysicsConfig(), "torque10pct": PhysicsConfig()})
    rows = []
    for depth, mass in [(-0.012, 0.1), (-0.02, 0.1), (0.008, 0.5), (0.008, 0.1)]:
        for variant in variants:
            row = run(variant, depth, mass, 0)
            rows.append(row)
            print(json.dumps(row), flush=True)
    (OUTPUT / "geometry_edges.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
