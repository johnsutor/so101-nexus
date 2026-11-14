"""Interactive model-coordinate overlays for physical reference alignment."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal

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


def _fit_bounds(vertices: np.ndarray) -> tuple[tuple[float, float], ...]:
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    margin = np.maximum(upper - lower, 0.02) * 0.08 + 0.002
    return tuple(
        (float(lo), float(hi)) for lo, hi in zip(lower - margin, upper + margin, strict=True)
    )


def _framing_vertices(
    vertices: np.ndarray, data: mujoco.MjData, focus_joint: str | None
) -> np.ndarray:
    if focus_joint is None:
        return vertices
    if focus_joint not in SO101_JOINT_NAMES:
        raise ValueError(f"Unknown focus_joint: {focus_joint}")
    model, parts, _ = _geometry()
    joint_id = model.joint(focus_joint).id
    context_body = int(model.body_parentid[model.jnt_bodyid[joint_id]])
    included = np.zeros(model.nbody, dtype=bool)
    included[context_body] = True
    for body_id in range(context_body + 1, model.nbody):
        included[body_id] = included[model.body_parentid[body_id]]
    anchor = data.xanchor[joint_id]
    context_radius_m = 0.0
    if focus_joint == "wrist_roll":
        context_anchor = data.xanchor[model.body_jntadr[context_body]]
        context_radius_m = 1.75 * np.linalg.norm(anchor - context_anchor)
    selected, offset = [], 0
    for geom_id, local in parts:
        points = vertices[offset : offset + len(local)]
        offset += len(local)
        body_id = model.geom_bodyid[geom_id]
        if included[body_id]:
            selected.append(points)
        elif focus_joint == "wrist_roll" and body_id == model.body_parentid[context_body]:
            # Distal forearm context distinguishes wrist rotation from jaw motion.
            selected.append(points[np.linalg.norm(points - anchor, axis=1) <= context_radius_m])
    return np.concatenate(selected)


def overlay_bounds(
    reference_deg: Sequence[float] | np.ndarray,
    measured_deg: Sequence[float] | np.ndarray | None = None,
    *,
    focus_joint: str | None = None,
) -> tuple[tuple[float, float], ...]:
    """Return padded X, Y, and Z limits enclosing both posed robot meshes in meters.

    Save these limits in the UI session and reuse them between explicit fit actions
    to prevent small encoder changes from moving the viewport during alignment.
    ``focus_joint`` frames that joint and downstream links with upstream context.
    It changes only the view limits; all mesh coordinates remain in world meters.
    """
    poses = (reference_deg,) if measured_deg is None else (reference_deg, measured_deg)
    vertices = [_framing_vertices(*_pose_vertices(_angles(pose)), focus_joint) for pose in poses]
    return _fit_bounds(np.concatenate(vertices))


def _camera(view: str) -> dict:
    cameras = {
        "front": ({"x": 0, "y": -0.85, "z": 0}, {"x": 0, "y": 0, "z": 1}),
        "side": ({"x": 0.85, "y": 0, "z": 0}, {"x": 0, "y": 0, "z": 1}),
        "top": ({"x": 0, "y": 0, "z": 0.85}, {"x": 0, "y": 1, "z": 0}),
    }
    if view not in cameras:
        raise ValueError("view must be front, side, or top")
    eye, up = cameras[view]
    return {"eye": eye, "up": up, "projection": {"type": "orthographic"}}


def _scene(bounds_m: Sequence[Sequence[float]], view: str, viewport_aspect: float) -> dict:
    bounds = np.asarray(bounds_m, dtype=np.float64)
    if (
        bounds.shape != (3, 2)
        or not np.isfinite(bounds).all()
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("bounds_m must contain three finite increasing meter ranges")
    axes = {
        f"{name}axis": {
            "title": f"{name.upper()} (m)",
            "range": limits,
            "showgrid": False,
            "zeroline": False,
            "showbackground": False,
            "showspikes": False,
            "nticks": 3,
            "tickfont": {"size": 10},
        }
        for name, limits in zip(("x", "y", "z"), bounds, strict=True)
    }
    ratios = np.ptp(bounds, axis=1)
    ratios /= np.cbrt(np.prod(ratios))
    horizontal, vertical = {"front": (0, 2), "side": (1, 2), "top": (0, 1)}[view]
    # Uniform scaling enlarges the viewport without changing the scale between axes.
    scale = 1.85 / max(ratios[horizontal] / viewport_aspect, ratios[vertical])
    return {
        **axes,
        "aspectmode": "manual",
        "aspectratio": dict(zip(("x", "y", "z"), ratios * scale, strict=True)),
        "camera": _camera(view),
    }


def render_overlay(
    reference_deg: Sequence[float] | np.ndarray,
    measured_deg: Sequence[float] | np.ndarray | None = None,
    *,
    provisional: bool = True,
    active_joint: str | None = None,
    view: Literal["front", "side", "top"] = "front",
    bounds_m: Sequence[Sequence[float]] | None = None,
    viewport_aspect: float = 1.5,
    focus_joint: str | None = None,
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
    view : {"front", "side", "top"}
        Orthographic camera preset. Front and side keep world Z upright. Switching
        presets resets the camera; repeated updates preserve the operator's orbit.
    bounds_m : sequence of pairs of float, optional
        Fixed X, Y, and Z meter ranges from :func:`overlay_bounds`. Without fixed
        bounds, fit the actual union of both meshes. No mesh coordinates change.
    viewport_aspect : float
        Available plot width divided by height. Fits the limiting dimension while
        retaining equal meter scale on all axes. Defaults to a wide desktop plot.
    focus_joint : str, optional
        Frame this joint and its downstream links with adjacent upstream context.
        The full robot remains drawn without modifying its pose or geometry.
    """
    import plotly.graph_objects as go

    if active_joint is not None and active_joint not in SO101_JOINT_NAMES:
        raise ValueError(f"Unknown active joint: {active_joint}")
    _camera(view)
    if (
        isinstance(viewport_aspect, bool)
        or not isinstance(viewport_aspect, (int, float))
        or not np.isfinite(viewport_aspect)
        or viewport_aspect <= 0
    ):
        raise ValueError("viewport_aspect must be finite and positive")
    model, _, faces = _geometry()
    vertices_by_pose = []
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
        vertices_by_pose.append(_framing_vertices(vertices, data, focus_joint))
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
    bounds_m = _fit_bounds(np.concatenate(vertices_by_pose)) if bounds_m is None else bounds_m
    figure.update_layout(
        template="plotly_white",
        height=650,
        margin={"l": 12, "r": 12, "t": 38, "b": 20},
        uirevision=f"calibration-overlay:{view}" + (f":{focus_joint}" if focus_joint else ""),
        legend={"orientation": "h", "y": 1.04, "x": 0},
        scene=_scene(bounds_m, view, viewport_aspect),
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
