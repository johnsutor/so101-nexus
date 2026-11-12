"""Interactive model-coordinate overlays for physical reference alignment."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

import numpy as np

from so101_nexus.config import SO101_JOINT_NAMES

if TYPE_CHECKING:
    from collections.abc import Sequence

    import mujoco
    import plotly.graph_objects as go


def _angles(values: Sequence[float] | np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (6,) or not np.isfinite(result).all():
        raise ValueError("The overlay requires six finite model angles in degrees")
    return result


def _cluster_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # A display-only 3 mm mesh avoids streaming the full collision-resolution assets.
    _, inverse = np.unique(np.round(vertices / 0.003), axis=0, return_inverse=True)
    count = np.bincount(inverse)
    reduced = np.column_stack(
        [np.bincount(inverse, weights=vertices[:, axis]) / count for axis in range(3)]
    )
    mapped = inverse[faces]
    valid = (mapped[:, 0] != mapped[:, 1]) & (mapped[:, 1] != mapped[:, 2])
    valid &= mapped[:, 0] != mapped[:, 2]
    mapped = mapped[valid]
    _, unique = np.unique(np.sort(mapped, axis=1), axis=0, return_index=True)
    return reduced.astype(np.float32), mapped[unique].astype(np.int32)


@cache
def _geometry() -> tuple[mujoco.MjModel, tuple[tuple[int, np.ndarray], ...], np.ndarray]:
    import mujoco

    from so101_nexus import get_so101_mujoco_model_path

    model = mujoco.MjModel.from_xml_path(str(get_so101_mujoco_model_path()))
    parts, triangles, offset = [], [], 0
    for geom_id in range(model.ngeom):
        if model.geom_group[geom_id] != 2 or model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mesh_id = model.geom_dataid[geom_id]
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        face_start, face_count = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
        vertices, faces = _cluster_mesh(
            model.mesh_vert[start : start + count],
            model.mesh_face[face_start : face_start + face_count],
        )
        vertices.flags.writeable = False
        parts.append((geom_id, vertices))
        triangles.append(faces + offset)
        offset += len(vertices)
    indices = np.concatenate(triangles)
    if offset <= np.iinfo(np.uint16).max:
        indices = indices.astype(np.uint16)
    indices.flags.writeable = False
    return model, tuple(parts), indices


def _pose_vertices(angles_deg: np.ndarray) -> tuple[np.ndarray, mujoco.MjData]:
    import mujoco

    model, parts, _ = _geometry()
    # MjData is per call so concurrent browser updates never share mutable kinematics.
    data = mujoco.MjData(model)
    for name, angle_deg in zip(SO101_JOINT_NAMES, angles_deg, strict=True):
        data.qpos[model.joint(name).qposadr[0]] = np.deg2rad(angle_deg)
    mujoco.mj_forward(model, data)
    vertices = np.concatenate(
        [
            vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
            for geom_id, vertices in parts
        ]
    ).astype(np.float32)
    return vertices, data


def render_overlay(
    reference_deg: Sequence[float] | np.ndarray,
    measured_deg: Sequence[float] | np.ndarray | None = None,
    *,
    provisional: bool = True,
    active_joint: str | None = None,
) -> go.Figure:
    """Render the reference ghost and live pose in one model coordinate frame.

    Parameters
    ----------
    reference_deg : sequence of float
        Six model joint angles in degrees, including the gripper.
    measured_deg : sequence of float, optional
        Hardware values transformed to model degrees. Unverified offsets require
        ``provisional=True``; visual overlap cannot validate physical alignment.
    provisional : bool
        Whether measured model coordinates still depend on unverified alignment.
    active_joint : str, optional
        Joint whose reference rotation axis should be highlighted.
    """
    import plotly.graph_objects as go

    if active_joint is not None and active_joint not in SO101_JOINT_NAMES:
        raise ValueError(f"Unknown active joint: {active_joint}")
    model, _, faces = _geometry()
    figure = go.Figure()
    reference_data = None
    for angles_deg, name, color, opacity in (
        (reference_deg, "Reference (ghost)", "#0891b2", 0.28),
        (
            measured_deg,
            "Measured (provisional)" if provisional else "Measured",
            "#b45309",
            1.0,
        ),
    ):
        if angles_deg is None:
            continue
        vertices, data = _pose_vertices(_angles(angles_deg))
        if reference_data is None:
            reference_data = data
        figure.add_trace(
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                name=name,
                color=color,
                opacity=opacity,
                showlegend=True,
                hoverinfo="name",
                flatshading=False,
            )
        )
    if active_joint is not None and reference_data is not None:
        joint_id = model.joint(active_joint).id
        anchor = reference_data.xanchor[joint_id]
        axis = reference_data.xaxis[joint_id] * 0.045
        endpoints = np.array([anchor - axis, anchor + axis])
        figure.add_trace(
            go.Scatter3d(
                x=endpoints[:, 0],
                y=endpoints[:, 1],
                z=endpoints[:, 2],
                mode="lines+markers",
                line={"color": "#0e7490", "width": 9},
                marker={"color": "#0e7490", "size": 3},
                name=f"Active joint: {active_joint.replace('_', ' ')}",
                hoverinfo="name",
            )
        )
    figure.update_layout(
        template="plotly_white",
        height=450,
        margin={"l": 0, "r": 0, "t": 35, "b": 0},
        uirevision="calibration-overlay",
        legend={"orientation": "h", "y": 1.04, "x": 0},
        scene={
            "xaxis": {"title": "X (m)", "range": [-0.35, 0.4]},
            "yaxis": {"title": "Y (m)", "range": [-0.35, 0.35]},
            "zaxis": {"title": "Z (m)", "range": [-0.05, 0.5]},
            "aspectmode": "data",
            "camera": {
                "eye": {"x": 1.8, "y": -2.2, "z": 1.1},
                "center": {"x": 0, "y": 0, "z": -0.2},
            },
        },
    )
    if measured_deg is None:
        figure.add_annotation(
            text="No live measurement.<br>Connect read-only to display the arm.",
            x=0.5,
            y=0.02,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    return figure
