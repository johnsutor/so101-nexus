"""Apply numerical options before simulation or device conversion."""

from typing import Literal

import mujoco
import numpy as np

from so101_nexus.config import EnvironmentConfig, PhysicsConfig


def apply_robot_config(model: mujoco.MjModel, config: EnvironmentConfig) -> None:
    """Apply hardware dynamics only when the simulation uses its recorded clock."""
    profile = config.robot.profile
    if profile is None:
        return
    from so101_nexus.profiles import apply_robot_profile

    period = (config.physics or PhysicsConfig()).control_period_s
    if not np.isclose(period, profile.control_period_s, rtol=0, atol=1e-12):
        raise ValueError("RobotProfile.control_period_s must match PhysicsConfig.control_period_s")
    apply_robot_profile(model, profile)


def apply_physics_config(
    model: mujoco.MjModel, config: PhysicsConfig, *, backend: Literal["mujoco", "warp"]
) -> int:
    """Apply supported options and return the number of substeps per command."""
    if backend == "warp":
        if config.noslip_iterations:
            raise ValueError("MuJoCo Warp does not support NoSlip; set noslip_iterations=0")
        if config.tolerance < float(np.float32(1e-6)):
            raise ValueError(
                "MuJoCo Warp requires tolerance >= 1e-6 to avoid a silent conversion clamp"
            )
    gripper_geoms = None
    if config.gripper_solimp is not None:
        gripper_geoms = np.zeros(model.ngeom, dtype=bool)
        for name in ("gripper", "moving_jaw_so101_v1"):
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            selected = (
                (model.geom_bodyid == body_id)
                & (model.geom_contype != 0)
                & (model.geom_condim == 6)
            )
            if body_id < 0 or not selected.any():
                raise ValueError(
                    f"gripper_solimp requires collidable six-dimensional jaw geoms on {name!r}"
                )
            gripper_geoms |= selected
    opt = model.opt
    opt.timestep = config.timestep_s
    opt.integrator = {
        "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
        "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
    }[config.integrator]
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    opt.impratio = config.impratio
    opt.iterations = config.iterations
    opt.ls_iterations = config.ls_iterations
    opt.tolerance = config.tolerance
    opt.noslip_iterations = config.noslip_iterations
    if gripper_geoms is not None:
        model.geom_solimp[gripper_geoms] = config.gripper_solimp
    return config.substeps
