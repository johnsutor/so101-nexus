"""Headless MuJoCo rendering for the test suite."""

import os
from contextlib import ExitStack

import pytest

os.environ.setdefault("MUJOCO_GL", "egl")


@pytest.fixture
def env_factory():
    """Construct registered environments and close them after the test."""
    import gymnasium as gym

    with ExitStack() as stack:

        def make(backend="mujoco", task="Touch", version=1, **kwargs):
            pytest.importorskip(f"so101_nexus.{backend}")
            if backend == "mujoco" and kwargs.pop("vector", False):
                num_envs = kwargs.pop("num_envs", 2)
                env = gym.make_vec(f"MuJoCo{task}-v{version}", num_envs=num_envs, **kwargs)
            elif backend == "warp":
                pytest.importorskip("mujoco_warp")
                pytest.importorskip("torch")
                device = kwargs.pop("device", "cpu")
                env = gym.make_vec(f"Warp{task}-v{version}", num_envs=2, device=device, **kwargs)
            else:
                env = gym.make(f"MuJoCo{task}-v{version}", **kwargs)
            stack.callback(env.close)
            return env

        yield make


@pytest.fixture
def supported_pick_factory(env_factory, monkeypatch):
    """Build a pickup scene with a cube supported near the open finger pads."""
    import numpy as np

    from so101_nexus import CubeObject, PickConfig, RobotConfig, object_slots

    build_scene = object_slots.build_object_scene_xml

    def make(backend="mujoco", *, half_size=0.012, mass=0.01, angle_deg=0, **kwargs):
        config = PickConfig(
            objects=[CubeObject(half_size=half_size, mass=mass)],
            robot=RobotConfig(
                rest_qpos_deg=(0, -np.degrees(1), np.degrees(1), 90, 0, np.degrees(1.3))
            ),
            terminate_on_success=False,
            robot_init_qpos_noise=0,
        )
        common = {"config": config, "control_mode": "pd_ee_delta_pose", "robot_init_qpos_noise": 0}
        probe = env_factory(task="PickLift", **common).unwrapped
        probe.reset(seed=3)
        rotation = probe.data.xmat[probe._gripper_body_id].reshape(3, 3)
        normal = rotation[:, 0]
        angle_rad = np.deg2rad(angle_deg)
        extent = half_size * (abs(np.cos(angle_rad)) + abs(np.sin(angle_rad)))
        # An 8 mm proximal pickup clears the thin tip and uses the broader pads.
        center = (
            probe.data.geom_xpos[probe.model.geom("fixed_jaw_box4").id]
            + normal * (extent + 0.0012)
            + rotation[:, 2] * 0.008
        )
        yaw_rad = np.arctan2(normal[1], normal[0]) + angle_rad
        pose = np.array([*center, np.cos(yaw_rad / 2), 0, 0, np.sin(yaw_rad / 2)])
        top = center[2] - half_size
        assert top > 0
        radius = half_size * 0.3
        support = (
            f'<geom name="pickup_support" type="box" size="{radius} {radius} {top / 2}" '
            f'pos="{center[0]} {center[1]} {top / 2}" friction="1 0.005 0.0005"/>'
        )

        def supported_scene(*args, **scene_kwargs):
            scene_kwargs["extra_bodies"] = scene_kwargs.get("extra_bodies", "") + support
            return build_scene(*args, **scene_kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(object_slots, "build_object_scene_xml", supported_scene)
            if backend == "warp":
                common.pop("robot_init_qpos_noise")
            env = env_factory(backend, task="PickLift", **common, **kwargs).unwrapped
        return env, pose

    return make
