"""Visible distal jaw surfaces must remain inside their collision boundaries."""

import mujoco
import numpy as np
import pytest

from so101_nexus import get_so101_mujoco_model_path


@pytest.mark.parametrize(
    "body,mesh_name,sign,y_range,z_range",
    [
        (
            "gripper",
            "wrist_roll_follower_so101_v1",
            -1,
            (-0.008, 0.008),
            (-0.104, -0.080),
        ),
        (
            "moving_jaw_so101_v1",
            "moving_jaw_so101_v1",
            1,
            (-0.082, -0.060),
            (0.011, 0.027),
        ),
    ],
)
def test_distal_jaw_colliders_cover_visible_gripping_surfaces(
    body, mesh_name, sign, y_range, z_range
):
    """Probe actual visible triangles against MuJoCo's compiled convex colliders."""
    spec = mujoco.MjSpec.from_file(str(get_so101_mujoco_model_path()))
    radius = 1e-7
    spec.worldbody.add_geom(
        name="surface_probe",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        contype=0,
        conaffinity=0,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = model.body(body).id
    geoms = np.flatnonzero(model.geom_bodyid == body_id)
    visual = next(
        int(g)
        for g in geoms
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH
        and model.mesh(model.geom_dataid[g]).name == mesh_name
    )
    colliders = [int(g) for g in geoms if model.geom_contype[g]]
    probe = model.geom("surface_probe").id
    rotation = data.xmat[body_id].reshape(3, 3)
    direction = rotation @ np.array([sign, 0, 0])
    gaps = []
    for y in np.arange(*y_range, 0.0005):
        for z in np.arange(*z_range, 0.0005):
            origin = data.xpos[body_id] + rotation @ np.array([-sign * 0.06, y, z])
            distance = mujoco.mj_rayMesh(model, data, visual, origin, direction)
            if distance < 0:
                continue
            data.geom_xpos[probe] = origin + distance * direction
            gaps.append(
                min(mujoco.mj_geomDistance(model, data, g, probe, 0.1, None) for g in colliders)
                + radius
            )
    assert len(gaps) > 500, "the ray grid must cover the visible gripping surface"
    assert max(gaps) <= 0.0003, f"visible {body} surface is {max(gaps) * 1000:.3f} mm uncovered"
