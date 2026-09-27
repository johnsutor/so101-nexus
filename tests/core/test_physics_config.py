"""Explicit physics options preserve the control clock and reach both backends."""

from dataclasses import FrozenInstanceError

import mujoco
import numpy as np
import pytest

from so101_nexus import MoveConfig, PhysicsConfig, PickConfig


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timestep_s": 0},
        {"timestep_s": True},
        {"timestep_s": 0.04},
        {"timestep_s": 1e-320},
        {"timestep_s": float("nan")},
        {"timestep_s": 0.003},
        {"control_period_s": float("inf")},
        {"control_period_s": "0.02"},
        {"impratio": 0},
        {"iterations": 1.5},
        {"iterations": True},
        {"ls_iterations": 0},
        {"noslip_iterations": -1},
        {"integrator": "RK4"},
        {"solver": "PGS"},
        {"cone": "pyramidal"},
        {"tolerance": -1},
    ],
)
def test_invalid_physics_rejected(kwargs):
    with pytest.raises(ValueError, match="must"):
        PhysicsConfig(**kwargs)


def test_physics_config_is_immutable():
    physics = PhysicsConfig()
    assert physics.substeps == 4
    with pytest.raises(FrozenInstanceError):
        physics.timestep_s = 0.001


def test_environment_rejects_untyped_physics():
    with pytest.raises(TypeError, match="PhysicsConfig"):
        MoveConfig(physics={"timestep_s": 0.001})


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
@pytest.mark.parametrize("timestep,substeps", [(0.0025, 8), (0.001, 20)])
def test_config_reaches_runtime_and_preserves_control_clock(
    env_factory, backend, timestep, substeps
):
    physics = PhysicsConfig(timestep_s=timestep, impratio=1000, iterations=100, ls_iterations=100)
    env = env_factory(
        backend=backend,
        task="Move",
        config=MoveConfig(
            physics=physics,
            reset_settle_frames=2,
            terminate_on_success=False,
        ),
    ).unwrapped
    model = env.model if backend == "mujoco" else env.mjm
    assert substeps == env._N_SUBSTEPS
    assert env.control_dt == pytest.approx(0.02)
    assert model.opt.timestep == timestep
    assert model.opt.impratio == 1000
    assert model.opt.iterations == model.opt.ls_iterations == 100
    assert model.opt.noslip_iterations == 0
    assert model.opt.tolerance == physics.tolerance
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICIT
    env.reset(seed=123)
    if backend == "mujoco":
        before = env.data.time
        action = env.data.ctrl.copy()
    else:
        before = env.data.time.numpy().copy()
        action = env.ctrl.clone()
        assert env.model.opt.timestep.numpy()[0] == np.float32(timestep)
        assert env.model.opt.impratio_invsqrt.numpy()[0] == np.float32(1 / np.sqrt(1000))
        assert env.model.opt.tolerance.numpy()[0] == np.float32(physics.tolerance)
    np.testing.assert_allclose(before, 0.04, rtol=1e-5, atol=1e-8)
    env.step(action)
    after = env.data.time if backend == "mujoco" else env.data.time.numpy()
    np.testing.assert_allclose(after - before, 0.02, rtol=1e-5, atol=1e-8)


def test_warp_rejects_unsupported_noslip_before_transfer(env_factory):
    with pytest.raises(ValueError, match="NoSlip"):
        env_factory(
            backend="warp",
            task="Move",
            config=MoveConfig(
                physics=PhysicsConfig(noslip_iterations=3),
            ),
        )


@pytest.mark.parametrize("backend,noslip", [("mujoco", 3), ("warp", 0)])
def test_omitted_config_preserves_native_options(env_factory, backend, noslip):
    env = env_factory(
        backend=backend, task="Move", config=MoveConfig(reset_settle_frames=0)
    ).unwrapped
    model = env.model if backend == "mujoco" else env.mjm
    assert model.opt.timestep == 0.005
    assert model.opt.impratio == 10
    assert model.opt.noslip_iterations == noslip
    assert env._N_SUBSTEPS == 4
    assert env.control_dt == 0.02


def test_warp_rejects_tolerance_clamp_before_transfer(env_factory):
    with pytest.raises(ValueError, match="tolerance"):
        env_factory(
            backend="warp",
            task="Move",
            config=MoveConfig(
                physics=PhysicsConfig(tolerance=1e-8),
            ),
        )


@pytest.mark.parametrize(
    "task,version",
    [
        ("PickLift", 1),
        ("PickAndPlace", 1),
        ("PickAndPlace", 2),
        ("PickReturn", 1),
        ("StackCube", 1),
        ("Touch", 1),
        ("LookAt", 1),
    ],
)
@pytest.mark.parametrize("backend", ["mujoco", "warp"])
def test_all_tasks_receive_options(env_factory, task, version, backend):
    from so101_nexus import (
        LookAtConfig,
        PickAndPlaceConfig,
        PickAndPlaceV2Config,
        PickConfig,
        PickReturnConfig,
        StackCubeConfig,
        TouchConfig,
    )

    config_type = {
        "PickLift": PickConfig,
        "PickAndPlace": PickAndPlaceConfig if version == 1 else PickAndPlaceV2Config,
        "PickReturn": PickReturnConfig,
        "StackCube": StackCubeConfig,
        "Touch": TouchConfig,
        "LookAt": LookAtConfig,
    }[task]
    config = config_type(
        physics=PhysicsConfig(
            timestep_s=0.0025,
            integrator="implicitfast",
            noslip_iterations=3 if backend == "mujoco" else 0,
        )
    )
    env = env_factory(backend=backend, task=task, version=version, config=config).unwrapped
    model = env.model if backend == "mujoco" else env.mjm
    assert model.opt.timestep == 0.0025
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert model.opt.noslip_iterations == (3 if backend == "mujoco" else 0)
    assert env._N_SUBSTEPS == 8


def test_float32_minimum_tolerance_is_accepted_on_warp(env_factory):
    tolerance = float(np.float32(1e-6))
    env = env_factory(
        backend="warp",
        task="Move",
        config=MoveConfig(
            physics=PhysicsConfig(tolerance=tolerance),
            reset_settle_frames=0,
        ),
    ).unwrapped
    assert env.mjm.opt.tolerance == tolerance
    assert float(env.model.opt.tolerance.numpy()[0]) == tolerance


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
def test_explicit_control_period_changes_runtime_clock(env_factory, backend):
    env = env_factory(
        backend=backend,
        task="Move",
        config=MoveConfig(
            physics=PhysicsConfig(timestep_s=0.0025, control_period_s=0.04),
            reset_settle_frames=0,
            terminate_on_success=False,
        ),
    ).unwrapped
    env.reset(seed=456)
    action = env.data.ctrl.copy() if backend == "mujoco" else env.ctrl.clone()
    env.step(action)
    elapsed = env.data.time if backend == "mujoco" else env.data.time.numpy()
    np.testing.assert_allclose(elapsed, 0.04, rtol=1e-5, atol=1e-8)
    assert env.control_dt == 0.04
    assert env._N_SUBSTEPS == 16


@pytest.mark.parametrize(
    "solimp",
    [
        (0.0, 0.999, 0.001, 0.5, 2.0),
        (0.99, 1.0, 0.001, 0.5, 2.0),
        (0.999, 0.99, 0.001, 0.5, 2.0),
        (0.99, 0.999, 0.0, 0.5, 2.0),
        (0.99, 0.999, 0.001, 0.0, 2.0),
        (0.99, 0.999, 0.001, 1.0, 2.0),
        (0.99, 0.999, 0.001, 0.5, 0.5),
        (0.99, 0.999, 0.001, 0.5, float("nan")),
        (0.99, 0.999),
        [0.99, 0.999, 0.001, 0.5, 2.0],
    ],
)
def test_invalid_gripper_impedance_rejected(solimp):
    with pytest.raises(ValueError, match="gripper_solimp"):
        PhysicsConfig(gripper_solimp=solimp)


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
@pytest.mark.parametrize("solimp", [None, (0.99, 0.999, 0.001, 0.5, 2.0)])
def test_gripper_compliance_override_is_scoped_and_transferred(env_factory, backend, solimp):
    baseline = env_factory(
        backend=backend,
        task="Move",
        config=MoveConfig(
            reset_settle_frames=0,
        ),
    ).unwrapped
    env = env_factory(
        backend=backend,
        task="Move",
        config=MoveConfig(
            physics=PhysicsConfig(
                timestep_s=0.0025,
                impratio=1000,
                iterations=100,
                ls_iterations=100,
                gripper_solimp=solimp,
            ),
            reset_settle_frames=0,
        ),
    ).unwrapped
    model = env.model if backend == "mujoco" else env.mjm
    original = baseline.model if backend == "mujoco" else baseline.mjm
    jaw_bodies = [model.body(name).id for name in ("gripper", "moving_jaw_so101_v1")]
    selected = (
        np.isin(model.geom_bodyid, jaw_bodies)
        & (model.geom_contype != 0)
        & (model.geom_condim == 6)
    )
    assert selected.any()
    np.testing.assert_array_equal(
        model.geom_solimp[selected],
        original.geom_solimp[selected]
        if solimp is None
        else np.broadcast_to(solimp, (selected.sum(), 5)),
    )
    np.testing.assert_array_equal(model.geom_solimp[~selected], original.geom_solimp[~selected])
    for field in (
        "geom_friction",
        "geom_solref",
        "geom_size",
        "geom_contype",
        "geom_conaffinity",
        "body_mass",
        "actuator_forcerange",
    ):
        np.testing.assert_array_equal(getattr(model, field), getattr(original, field))
    if backend == "warp":
        np.testing.assert_array_equal(
            env.model.geom_solimp.numpy()[0, selected],
            model.geom_solimp[selected].astype(np.float32),
        )


def test_gripper_override_fails_if_jaw_geometry_is_absent():
    from so101_nexus.physics import apply_physics_config

    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    original_timestep = model.opt.timestep
    with pytest.raises(ValueError, match="jaw"):
        apply_physics_config(
            model, PhysicsConfig(gripper_solimp=(0.99, 0.999, 0.001, 0.5, 2.0)), backend="mujoco"
        )
    assert model.opt.timestep == original_timestep


@pytest.mark.parametrize("control_period_s", [0.01, 0.04, 0.03])
@pytest.mark.parametrize("task,config_type", [("Move", MoveConfig), ("PickLift", PickConfig)])
def test_render_cadence_tracks_each_cpu_environment(
    env_factory, control_period_s, task, config_type
):
    baseline = env_factory(task=task, config=config_type(reset_settle_frames=0)).unwrapped
    original_metadata = type(baseline).metadata.copy()
    configured = env_factory(
        task=task,
        config=config_type(
            physics=PhysicsConfig(control_period_s=control_period_s), reset_settle_frames=0
        ),
    ).unwrapped
    assert configured.metadata["render_fps"] == pytest.approx(1 / control_period_s)
    assert baseline.metadata["render_fps"] == 50
    assert type(configured).metadata == original_metadata
