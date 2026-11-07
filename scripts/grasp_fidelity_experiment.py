"""Run a CPU grasp sensitivity study using the supported-pick regression fixture.

Run with the development environment from the repository root. This intentionally
uses test helpers so the acquisition geometry cannot drift from the regression.
Results are a sensitivity study, not a hardware calibration or success benchmark.
"""

import argparse
import json
import platform
import runpy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TypedDict

import mujoco
import numpy as np
import pytest

from so101_nexus import PhysicsConfig
from so101_nexus.physics import apply_physics_config

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = runpy.run_path(str(ROOT / "tests/conftest.py"))
HELPERS = runpy.run_path(str(ROOT / "tests/mujoco/test_grasp_dynamics.py"))
PROFILES = {
    "baseline": PhysicsConfig(),
    "impratio_1000": PhysicsConfig(impratio=1000),
    "noslip_3": PhysicsConfig(noslip_iterations=3),
    "half_timestep": PhysicsConfig(timestep_s=0.00125),
    "impratio_1000_half_timestep": PhysicsConfig(impratio=1000, timestep_s=0.00125),
    "tight_solver": PhysicsConfig(iterations=100, tolerance=1e-10),
    "impedance_099": PhysicsConfig(gripper_solimp=(0.99, 0.999, 0.001, 0.5, 2)),
    **dict.fromkeys(
        ("condim_3", "condim_4", "half_torque", "friction_03", "object_friction_01"),
        PhysicsConfig(),
    ),
}
CASES = ((0.009525, 0, 0.01), (0.012, 0, 0.1), (0.02, 0, 0.1), (0.012, 45, 0.1))


@dataclass
class Sample:
    """One control-boundary observation of a loaded grasp."""

    phase: str
    slip_m: float
    relative_position: np.ndarray
    height_m: float
    grasping: bool
    penetration_m: float
    normal_force_sum_n: float


def _configure(env, profile):
    env._N_SUBSTEPS = apply_physics_config(env.model, PROFILES[profile], backend="mujoco")
    fingers = sorted(env._gripper_geom_ids | env._jaw_geom_ids)
    if profile.startswith("condim_"):
        env.model.geom_condim[fingers] = int(profile[-1])
    if profile == "half_torque":
        env.model.actuator_forcerange[env.model.actuator("gripper").id] *= 0.5
    if profile == "friction_03":
        env.model.geom_friction[fingers, 0] = 0.3
    if profile == "object_friction_01":
        env.model.geom_friction[env._obj_geom_ids, 0] = 0.1


class ContactMetrics(TypedDict):
    """Contact diagnostics sampled after a control step."""

    penetration_m: float
    normal_force_sum_n: float
    contact_count: int
    effective_mu: list[float]


def _measure(env) -> ContactMetrics:
    fingers = env._gripper_geom_ids | env._jaw_geom_ids
    target = set(env._obj_geom_ids)
    contacts = [
        contact
        for contact in env.data.contact
        if (contact.geom1 in fingers and contact.geom2 in target)
        or (contact.geom2 in fingers and contact.geom1 in target)
    ]
    force = np.zeros(6)
    normal = 0.0
    for index, contact in enumerate(env.data.contact):
        if (contact.geom1 in fingers and contact.geom2 in target) or (
            contact.geom2 in fingers and contact.geom1 in target
        ):
            mujoco.mj_contactForce(env.model, env.data, index, force)
            normal += max(0.0, float(force[0]))
    return {
        "penetration_m": max((max(0.0, -c.dist) for c in contacts), default=0.0),
        "normal_force_sum_n": normal,
        "contact_count": len(contacts),
        "effective_mu": sorted({float(c.friction[0]) for c in contacts}),
    }


def _trajectory(env, hold_s):
    step_to = HELPERS["_step_to"]
    relative_position = HELPERS["_relative_object_position"]
    desired = env._get_tcp_pose().copy()
    for _ in range(100):
        step_to(env, desired[:3], desired[3:])
    acquisition = _measure(env)
    acquired = bool(env._is_grasping())
    relative = relative_position(env).copy()
    initial_height = float(env._get_target_pose()[2])
    start = env._get_tcp_pose()[:3].copy()
    previous = start
    rows = []
    phases = (
        ("lift", (0, 0, 0.10), 50),
        ("hold", (0, 0, 0.10), max(1, round(hold_s / env.control_dt))),
        ("forward", (0.04, 0, 0.10), 30),
        ("backward", (-0.04, 0, 0.10), 60),
        ("center", (0, 0, 0.10), 30),
        ("right", (0, 0.04, 0.10), 30),
        ("left", (0, -0.04, 0.10), 60),
        ("return", (0, 0, 0.10), 30),
    )
    for phase, offset, count in phases:
        destination = start + np.array(offset)
        for fraction in np.linspace(1 / count, 1, count):
            step_to(env, previous + fraction * (destination - previous), desired[3:])
            contact = _measure(env)
            rows.append(
                Sample(
                    phase=phase,
                    slip_m=float(np.linalg.norm(relative_position(env) - relative)),
                    relative_position=relative_position(env).copy(),
                    height_m=float(env._get_target_pose()[2]),
                    grasping=bool(env._is_grasping()),
                    penetration_m=contact["penetration_m"],
                    normal_force_sum_n=contact["normal_force_sum_n"],
                )
            )
        previous = destination
    held_height = float(env._get_target_pose()[2])
    for _ in range(60):
        step_to(env, previous, desired[3:], gripper=1)
    for _ in range(40):
        step_to(env, previous - np.array([0, 0, 0.08]), desired[3:], gripper=1)
    released = not env._is_grasping() and held_height - float(env._get_target_pose()[2]) > 0.05
    hold = [row for row in rows if row.phase == "hold"]
    return {
        "hold_drift_m": float(
            np.linalg.norm(hold[-1].relative_position - hold[0].relative_position)
        ),
        "acquired": acquired,
        "acquisition": acquisition,
        "max_slip_m": max(row.slip_m for row in rows),
        "hold_slip_end_m": [row.slip_m for row in rows if row.phase == "hold"][-1],
        "max_penetration_m": max(row.penetration_m for row in rows),
        "min_lift_m": min(row.height_m for row in rows if row.phase != "lift") - initial_height,
        "lost_samples": sum(not row.grasping for row in rows),
        "released": bool(released),
        "max_normal_force_sum_n": max(row.normal_force_sum_n for row in rows),
    }


def _run(profile, case, hold_s):
    generator = FIXTURES["env_factory"].__wrapped__()
    factory = next(generator)
    try:
        with pytest.MonkeyPatch.context() as patch:
            make = FIXTURES["supported_pick_factory"].__wrapped__(factory, patch)
            half_size, angle_deg, mass = case
            env, pose = make(half_size=half_size, angle_deg=angle_deg, mass=mass)
            _configure(env, profile)
            env.reset(seed=3)
            slot = env._slots[env._target_slot_idx]
            env.data.qpos[slot.qpos_addr : slot.qpos_addr + 7] = pose
            env.data.qvel[slot.dof_addr : slot.dof_addr + 6] = 0
            mujoco.mj_forward(env.model, env.data)
            started = perf_counter()
            result = _trajectory(env, hold_s)
            result["wall_s"] = perf_counter() - started
            result["warnings"] = [int(w.number) for w in env.data.warning]
            return {
                "profile": profile,
                "half_size_m": half_size,
                "angle_deg": angle_deg,
                "mass_kg": mass,
                **result,
            }
    finally:
        generator.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hold-s", type=float, default=30)
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    args = parser.parse_args()
    if not np.isfinite(args.hold_s) or args.hold_s <= 0:
        parser.error("--hold-s must be finite and positive")
    results = []
    for profile in args.profiles:
        for case in CASES:
            result = _run(profile, case, args.hold_s)
            results.append(result)
            print(json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "mujoco": mujoco.__version__,
                "platform": platform.platform(),
                "hold_s": args.hold_s,
                "seed": 3,
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
