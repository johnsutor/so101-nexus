"""Visual-footprint and force predicates for center placement.

The footprint is the union of projected visual triangles, not their summed
areas or their convex hull. Midpoint quadrature is exact along each scanline
and approximate along world Y. Torch is optional and imports only on its path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    import mujoco

    from so101_nexus.object_slots import ObjectSlot

# Bound each interval workspace, independent of the number of worlds or lines.
# A single mesh with more triangles than this limit needs one larger scanline.
_MAX_INTERVAL_VALUES = 1 << 20
_PRIMITIVE_SLICES = 64
_PRIMITIVE_STACKS = 32


@dataclass(frozen=True)
class SlotGeometry[ArrayT]:
    """Compiled visual surface and an exact center for a single symmetric primitive."""

    triangles: ArrayT
    symmetry_center: ArrayT | None = None


def visual_reference(geometry, rotation, position, *, scanlines=256):
    """Return footprint centroid XY and lowest visual Z in the root's world pose."""
    vertices = geometry.triangles.reshape(-1, 3)
    tensor = hasattr(vertices, "clamp")
    reference = position.clone() if tensor else position.copy()
    if geometry.symmetry_center is not None:
        center = (rotation @ geometry.symmetry_center[..., None])[..., 0] + position
        reference[..., :2] = center[..., :2]
        heights = (rotation[..., 2, None, :] * vertices).sum(-1) + position[..., 2, None]
    else:
        world = vertices @ rotation.swapaxes(-1, -2) + position[..., None, :]
        triangles = world.reshape(*position.shape[:-1], -1, 3, 3)
        reference[..., :2] = footprint_centroid(triangles, scanlines=scanlines)
        heights = world[..., 2]
    reference[..., 2] = heights.amin(dim=-1) if tensor else heights.min(axis=-1)
    return reference


def _geom_triangles(model, geom_id: int) -> np.ndarray:
    """Return triangles in the compiled geom frame."""
    import mujoco

    kind = int(model.geom_type[geom_id])
    size = model.geom_size[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        vertex_start = int(model.mesh_vertadr[mesh_id])
        vertex_count = int(model.mesh_vertnum[mesh_id])
        face_start = int(model.mesh_faceadr[mesh_id])
        face_count = int(model.mesh_facenum[mesh_id])
        vertices = model.mesh_vert[vertex_start : vertex_start + vertex_count]
        faces = model.mesh_face[face_start : face_start + face_count]
        return vertices[faces].astype(np.float64)
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        vertices = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=np.float64,
            )
            * size
        )
        faces = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [1, 2, 6],
                [1, 6, 5],
                [2, 3, 7],
                [2, 7, 6],
                [3, 0, 4],
                [3, 4, 7],
            ]
        )
        return vertices[faces]
    if kind not in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_SPHERE):
        raise ValueError(
            f"Placement geometry does not support geom {geom_id} with type {int(kind)}."
        )

    # These inscribed surfaces preserve primitive symmetry under every pose.
    # Cylinder radial chord error is at most r * (1 - cos(pi / 64)).
    # Sphere latitude and longitude intervals are both pi / 32 radians.
    angles = np.arange(_PRIMITIVE_SLICES) * (2 * np.pi / _PRIMITIVE_SLICES)
    circle = np.column_stack((np.cos(angles), np.sin(angles)))
    triangles = []
    if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
        lower = np.column_stack((size[0] * circle, np.full(len(circle), -size[1])))
        upper = lower.copy()
        upper[:, 2] = size[1]
        lower_center = np.array([0.0, 0.0, -size[1]])
        upper_center = -lower_center
        for i in range(len(circle)):
            j = (i + 1) % len(circle)
            triangles.extend(
                (
                    [lower[i], lower[j], upper[j]],
                    [lower[i], upper[j], upper[i]],
                    [lower_center, lower[j], lower[i]],
                    [upper_center, upper[i], upper[j]],
                )
            )
    else:
        latitudes = np.arange(1, _PRIMITIVE_STACKS) * (np.pi / _PRIMITIVE_STACKS)
        rings = np.empty((len(latitudes), len(circle), 3))
        rings[:, :, :2] = np.sin(latitudes)[:, None, None] * circle[None]
        rings[:, :, 2] = np.cos(latitudes)[:, None]
        rings *= size[0]
        north = np.array([0.0, 0.0, size[0]])
        south = -north
        for i in range(len(circle)):
            j = (i + 1) % len(circle)
            triangles.extend(
                ([north, rings[0, i], rings[0, j]], [south, rings[-1, j], rings[-1, i]])
            )
            for k in range(len(rings) - 1):
                triangles.extend(
                    (
                        [rings[k, i], rings[k + 1, i], rings[k + 1, j]],
                        [rings[k, i], rings[k + 1, j], rings[k, j]],
                    )
                )
    return np.asarray(triangles)


def compiled_slot_geometry(
    model: mujoco.MjModel, slots: Sequence[ObjectSlot]
) -> list[SlotGeometry[np.ndarray]]:
    """Return visual geometry in each slot's root-body local frame.

    Group 2 geoms take precedence within the rigid slot subtree. Otherwise,
    visible geoms outside collision groups 3 and 4 define primitive surfaces.
    Mesh faces retain compiler topology. The compiled geom transform restores
    the asset offset and orientation exactly once. Rigid child bodies also
    contribute their local transforms. Articulated subtrees are not supported.

    Boxes and mesh faces are exact. Cylinders use 64 sides. Spheres use 64
    longitudes and 32 latitude intervals. Both curved surfaces are inscribed
    approximations, independent of render settings. No collision hull replaces
    a visual surface. Empty, nonfinite, and zero-area surfaces raise ValueError.
    """
    import mujoco

    result = []
    rotation_buffer = np.empty(9)
    for slot in slots:
        root = int(model.dof_bodyid[slot.dof_addr])
        body_frames = {root: (np.eye(3), np.zeros(3))}
        for body_id in range(root + 1, model.nbody):
            parent = int(model.body_parentid[body_id])
            if parent not in body_frames:
                continue
            if model.body_jntnum[body_id]:
                raise ValueError("Placement geometry requires a rigid object subtree.")
            parent_rotation, parent_position = body_frames[parent]
            mujoco.mju_quat2Mat(rotation_buffer, model.body_quat[body_id])
            body_frames[body_id] = (
                parent_rotation @ rotation_buffer.reshape(3, 3),
                parent_position + parent_rotation @ model.body_pos[body_id],
            )
        geom_ids = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in body_frames]
        visual_ids = [i for i in geom_ids if model.geom_group[i] == 2]
        if not visual_ids:
            visual_ids = [i for i in geom_ids if model.geom_group[i] not in (3, 4)]
        visual_ids = [i for i in visual_ids if model.geom_rgba[i, 3] > 0]
        chunks = []
        symmetry_center = None
        for geom_id in visual_ids:
            local = _geom_triangles(model, geom_id)
            body_rotation, body_position = body_frames[int(model.geom_bodyid[geom_id])]
            mujoco.mju_quat2Mat(rotation_buffer, model.geom_quat[geom_id])
            rotation = body_rotation @ rotation_buffer.reshape(3, 3)
            position = body_position + body_rotation @ model.geom_pos[geom_id]
            chunks.append(local @ rotation.T + position)
            if len(visual_ids) == 1 and int(model.geom_type[geom_id]) in (
                mujoco.mjtGeom.mjGEOM_BOX,
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                mujoco.mjtGeom.mjGEOM_SPHERE,
            ):
                symmetry_center = position.copy()
        if not chunks:
            raise ValueError(f"Placement object body {root} has no visible surface.")
        triangles = np.concatenate(chunks)
        if not np.isfinite(triangles).all():
            raise ValueError(f"Placement object body {root} has nonfinite visual geometry.")
        normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        triangles = triangles[np.any(normal != 0, axis=-1)]
        if not len(triangles):
            raise ValueError(f"Placement object body {root} has a zero-area visual surface.")
        result.append(SlotGeometry(triangles, symmetry_center))
    return result


def _footprint_inputs(triangles, scanlines):
    if (
        isinstance(scanlines, (bool, np.bool_))
        or not isinstance(scanlines, (int, np.integer))
        or scanlines < 1
    ):
        raise ValueError("Placement scanlines must be a positive integer.")
    if hasattr(triangles, "clamp"):
        import torch

        xp = torch
        floating = triangles.is_floating_point()
    else:
        xp = np
        triangles = np.asarray(triangles)
        floating = np.issubdtype(triangles.dtype, np.floating)
    if not floating:
        raise TypeError("Placement triangles must use a floating dtype.")
    if triangles.ndim < 3 or triangles.shape[-2:] != (3, 3):
        raise ValueError("Placement triangles must have shape (..., T, 3, 3).")
    if triangles.shape[-3] == 0:
        raise ValueError("Placement triangles must contain at least one triangle.")
    return triangles, xp


def _scanline_moments(xy, y, xp):
    """Integrate interval unions exactly along X at each sampled Y."""
    tensor = hasattr(xy, "clamp")
    shape = (len(xy), y.shape[1], xy.shape[1])
    if tensor:
        starts = xy.new_full(shape, float("inf"))
        ends = xy.new_full(shape, -float("inf"))
    else:
        starts = np.full(shape, np.inf, dtype=xy.dtype)
        ends = np.full(shape, -np.inf, dtype=xy.dtype)
    for edge in range(3):
        a = xy[:, None, :, edge]
        b = xy[:, None, :, (edge + 1) % 3]
        dy = b[..., 1] - a[..., 1]
        crosses = (y[..., None] >= xp.minimum(a[..., 1], b[..., 1])) & (
            y[..., None] < xp.maximum(a[..., 1], b[..., 1])
        )
        denominator = xp.where(dy != 0, dy, 1.0)
        x = a[..., 0] + (y[..., None] - a[..., 1]) * (b[..., 0] - a[..., 0]) / denominator
        starts = xp.minimum(starts, xp.where(crosses, x, float("inf")))
        ends = xp.maximum(ends, xp.where(crosses, x, -float("inf")))
    # Empty intervals sort after all real intervals and contribute no length.
    valid = starts < ends
    starts = xp.where(valid, starts, float("inf"))
    ends = xp.where(valid, ends, 0.0)
    if tensor:
        starts, order = starts.sort(dim=-1)
        ends = ends.gather(-1, order)
        maximum = ends.cummax(dim=-1).values
        # Autograd retains the unmodified interval starts.
        left = starts.clone() if starts.requires_grad else starts
    else:
        order = np.argsort(starts, axis=-1)
        starts = np.take_along_axis(starts, order, axis=-1)
        ends = np.take_along_axis(ends, order, axis=-1)
        maximum = np.maximum.accumulate(ends, axis=-1)
        left = starts
    left[..., 1:] = xp.maximum(starts[..., 1:], maximum[..., :-1])
    occupied = left < ends
    left = xp.where(occupied, left, 0.0)
    right = xp.where(occupied, ends, 0.0)
    length = right - left
    lengths = length.sum(-1)
    return (
        lengths.sum(-1),
        (length * (left + right) * 0.5).sum(-1).sum(-1),
        (lengths * y).sum(-1),
    )


def footprint_centroid(triangles, *, scanlines=256):
    """Return the XY centroid of a projected visual-triangle union.

    Input has shape ``(..., T, 3, 3)`` in world coordinates. Output has shape
    ``(..., 2)`` and retains the input NumPy or torch floating dtype and device.
    Winding, hidden faces, duplicate faces, and overlaps do not alter the union.

    Each of ``scanlines`` midpoint samples spans the projected Y bounds. Each
    triangle supplies an X interval. Sorted intervals merge through cumulative
    maximum endpoints. Union lengths and first moments give the centroid.
    Coordinates use a local reference to limit translation roundoff.

    This is Y quadrature, not a polygon-union solver. Features thinner than one
    Y step can escape all samples. Error depends on shape and orientation, so
    no shape-independent centroid bound exists for almost-degenerate surfaces.
    A projection with no sampled area raises ValueError instead of returning
    an origin. Batch and scanline chunks bound interval workspace memory.
    """
    triangles, xp = _footprint_inputs(triangles, scanlines)
    tensor = hasattr(triangles, "clamp")

    batch_shape = triangles.shape[:-3]
    count = triangles.shape[-3]
    flat = triangles.reshape(-1, count, 3, 3)
    if tensor:
        centroids = triangles.new_empty((len(flat), 2))
        array_options = {"dtype": triangles.dtype, "device": triangles.device}
    else:
        centroids = np.empty((len(flat), 2), dtype=triangles.dtype)
        array_options = {"dtype": triangles.dtype}
    batch_chunk = max(1, _MAX_INTERVAL_VALUES // (count * scanlines))
    for first in range(0, len(flat), batch_chunk):
        part = flat[first : first + batch_chunk]
        if not bool(xp.isfinite(part).all()):
            raise ValueError("Placement triangles must contain finite coordinates.")
        reference = part[:, 0, 0, :2]
        xy = part[..., :2] - reference[:, None, None, :]
        if tensor:
            low = xy[..., 1].amin(dim=(-2, -1))
            high = xy[..., 1].amax(dim=(-2, -1))
            total = part.new_zeros((len(part), 3))
        else:
            low = xy[..., 1].min(axis=(-2, -1))
            high = xy[..., 1].max(axis=(-2, -1))
            total = np.zeros((len(part), 3), dtype=part.dtype)
        line_chunk = max(1, _MAX_INTERVAL_VALUES // (len(part) * count))
        for line in range(0, scanlines, line_chunk):
            offsets = xp.arange(line, min(line + line_chunk, scanlines), **array_options)
            offsets = (offsets + 0.5) / scanlines
            y = low[:, None] + (high - low)[:, None] * offsets[None, :]
            area, moment_x, moment_y = _scanline_moments(xy, y, xp)
            total[:, 0] += area
            total[:, 1] += moment_x
            total[:, 2] += moment_y
        if not bool(((total[:, 0] > 0) & xp.isfinite(total).all(-1)).all()):
            raise ValueError("Placement projection has no finite sampled area.")
        centroids[first : first + len(part)] = total[:, 1:] / total[:, :1] + reference
    return centroids.reshape(*batch_shape, 2)


def resolved_placement_contract(config, objects, weights, timestep, target_index):
    """Snapshot active evaluator settings and each object's resolved force limits."""
    return {
        **config.placement_contract,
        "physics_timestep": timestep,
        "target_index": target_index,
        "objects": [
            {
                "slot_index": index,
                "object": repr(obj),
                "weight_N": weight,
                "required_upward_force_N": config.support_min_weight_fraction * weight,
                "force_limit_N": max(
                    config.support_force_tolerance,
                    config.support_relative_force_tolerance * weight,
                ),
            }
            for index, (obj, weight) in enumerate(zip(objects, weights, strict=True))
        ],
    }


def support_state(
    table_force,
    robot_force,
    other_force,
    weight,
    linear_speed,
    angular_speed,
    *,
    min_weight_fraction,
    force_tolerance,
    relative_force_tolerance,
    lin_threshold,
    ang_threshold,
):
    """Return table, robot, other, static, and qualified support predicates.

    Forces use newtons. Table force is the upward signed resultant. Robot and
    other forces are sums of contact-force magnitudes. Weight is positive.
    Linear speed is COM speed in m/s. Angular speed uses rad/s. Inputs support
    NumPy or torch broadcasting. Force equality is permitted, as are motion
    threshold equality and table support equal to the required weight fraction.
    """
    relative_limit = relative_force_tolerance * weight
    if hasattr(relative_limit, "clamp"):
        limit = relative_limit.clamp(min=force_tolerance)
    else:
        limit = np.maximum(force_tolerance, relative_limit)
    table_ok = table_force >= min_weight_fraction * weight
    robot_supported = robot_force > limit
    other_supported = other_force > limit
    object_static = (linear_speed <= lin_threshold) & (angular_speed <= ang_threshold)
    qualified = table_ok & ~robot_supported & ~other_supported & object_static
    return table_ok, robot_supported, other_supported, object_static, qualified
