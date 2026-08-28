"""Spawning utilities for MuJoCo SO101-Nexus environments.

Provides helpers for placing objects in the workspace without overlap
and for sampling random object orientations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

    from so101_nexus.object_slots import ObjectSlot

MESH_FLOOR_MARGIN = 0.002
"""Small floor clearance for mesh object spawns, in metres."""


def sample_separated_positions(
    rng: np.random.Generator,
    count: int,
    min_r: float,
    max_r: float,
    angle_half: float,
    min_clearance: float,
    bounding_radii: list[float],
    max_attempts: int = 100,
    center: tuple[float, float] = (0.0, 0.0),
) -> list[tuple[float, float]]:
    """Sample 2D positions in a polar arc with bounding-radius-aware separation.

    Tries up to `max_attempts` times to place each object such that its bounding
    circle (radius = bounding_radii[i]) plus `min_clearance` does not overlap any
    already-placed bounding circle. Falls back to the last sampled position if no
    valid placement is found within the attempt budget.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.
    count : int
        Number of positions to sample.
    min_r : float
        Minimum radial distance from ``center``.
    max_r : float
        Maximum radial distance from ``center``.
    angle_half : float
        Half-angle of the arc in radians; samples are drawn from
        ``[-angle_half, angle_half]``.
    min_clearance : float
        Minimum gap between bounding circles of any two placed objects.
    bounding_radii : list[float]
        Per-object bounding radius used for overlap checks.
    max_attempts : int, optional
        Maximum placement retries per object before falling back.
    center : tuple[float, float], optional
        XY offset applied to all sampled positions. Defaults to ``(0.0, 0.0)``.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    cx, cy = center
    positions: list[tuple[float, float]] = []
    for idx in range(count):
        for _ in range(max_attempts):
            r = rng.uniform(min_r, max_r)
            theta = rng.uniform(-angle_half, angle_half)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            if all(
                np.sqrt((x - px) ** 2 + (y - py) ** 2)
                >= bounding_radii[idx] + bounding_radii[j] + min_clearance
                for j, (px, py) in enumerate(positions)
            ):
                positions.append((x, y))
                break
        else:
            positions.append((x, y))
    return positions


def random_yaw_quat(rng: np.random.Generator) -> np.ndarray:
    """Sample a unit quaternion with uniformly random yaw about the Z axis.

    Parameters
    ----------
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    np.ndarray
        Shape ``(4,)`` quaternion ``[w, x, y, z]`` representing a rotation
        purely about the Z axis by an angle drawn uniformly from ``[0, 2π)``.
    """
    angle_rad = rng.uniform(0.0, 2.0 * np.pi)
    return np.array([np.cos(angle_rad / 2.0), 0.0, 0.0, np.sin(angle_rad / 2.0)])


def mesh_geom_world_min_z(model, data, geom_id: int) -> float:
    """Return the minimum world Z of one compiled mesh collision geom.

    A mesh object body carries one collision geom per convex part of its
    decomposition, so the body's clearance is the minimum over ``slot.geom_ids``
    (see :func:`slot_world_min_z`), not over a single part.
    """
    import mujoco

    if int(model.geom_type[geom_id]) != int(mujoco.mjtGeom.mjGEOM_MESH):
        raise ValueError("mesh_geom_world_min_z requires a mesh geom")
    mesh_id = int(model.geom_dataid[geom_id])
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_count = int(model.mesh_vertnum[mesh_id])
    verts = model.mesh_vert[vert_start : vert_start + vert_count]
    xmat = data.geom_xmat[geom_id].reshape(3, 3)
    xpos = data.geom_xpos[geom_id]
    world = verts @ xmat.T + xpos
    return float(world[:, 2].min())


def slot_world_min_z(model, data, geom_ids: Sequence[int]) -> float:
    """Return the minimum world Z over every collision geom of a slot."""
    return min(mesh_geom_world_min_z(model, data, geom_id) for geom_id in geom_ids)


def align_freejoint_geom_to_floor(
    model,
    data,
    *,
    qpos_addr: int,
    geom_ids: Sequence[int],
    xy: tuple[float, float],
    quat: np.ndarray,
    margin: float = MESH_FLOOR_MARGIN,
) -> float:
    """Set a freejoint mesh pose so its compiled collision geoms clear the floor."""
    import mujoco

    data.qpos[qpos_addr : qpos_addr + 3] = [xy[0], xy[1], 0.0]
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat
    mujoco.mj_forward(model, data)
    min_z = slot_world_min_z(model, data, geom_ids)
    spawn_z = -min_z + margin
    data.qpos[qpos_addr + 2] = spawn_z
    mujoco.mj_forward(model, data)
    return float(spawn_z)


def place_freejoint_slot(
    model, data, slot, rng: np.random.Generator, xy: tuple[float, float]
) -> None:
    """Place an active object slot at ``xy`` with random yaw, resting on the floor.

    ``slot`` is an ``so101_nexus.object_slots.ObjectSlot``; geometric primitives
    use analytic spawn heights while mesh-backed objects clear the floor via their
    compiled collision geom.
    """
    from so101_nexus.objects import PrimitiveObject

    x, y = xy
    if isinstance(slot.obj, PrimitiveObject):
        data.qpos[slot.qpos_addr : slot.qpos_addr + 3] = [x, y, slot.spawn_z]
        data.qpos[slot.qpos_addr + 3 : slot.qpos_addr + 7] = random_yaw_quat(rng)
        return
    import mujoco

    obj_quat = np.zeros(4)
    mujoco.mju_mulQuat(obj_quat, random_yaw_quat(rng), slot.rest_quat)
    align_freejoint_geom_to_floor(
        model, data, qpos_addr=slot.qpos_addr, geom_ids=slot.geom_ids, xy=(x, y), quat=obj_quat
    )


def set_slot_contacts(model, slot, enabled: bool) -> None:
    """Enable or disable collisions on every collision geom of a slot.

    ``geom_ids`` is listed rather than passed as-is because NumPy reads a tuple
    index as a multi-dimensional index.
    """
    geom_ids = list(slot.geom_ids)
    model.geom_contype[geom_ids] = int(enabled)
    model.geom_conaffinity[geom_ids] = int(enabled)


def hide_freejoint_slot(model, data, slot) -> None:
    """Disable collisions and park an inactive object slot far below the floor.

    Zeroing the contact bits keeps the constraint solver from exploding stacked
    hidden bodies during the post-reset settle.
    """
    set_slot_contacts(model, slot, False)
    data.qpos[slot.qpos_addr : slot.qpos_addr + 3] = [0.0, 0.0, -10.0]
    data.qpos[slot.qpos_addr + 3 : slot.qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]


def activate_distractor_slots(
    model, data, slots: list[ObjectSlot], count: int, rng: np.random.Generator
) -> list[ObjectSlot]:
    """Activate ``count`` of ``slots`` and park the rest off-world.

    Restores collisions on the whole distractor pool first, so a slot that was
    hidden by an earlier episode collides again once it is chosen. Returns the
    chosen slots in pool order; the caller places them.

    Parameters
    ----------
    model : mujoco.MjModel
        Compiled model whose per-geom contact flags are toggled.
    data : mujoco.MjData
        State buffer whose freejoint qpos is parked for inactive slots.
    slots : list
        Compiled distractor pool (``so101_nexus.object_slots.ObjectSlot``).
    count : int
        Number of slots to activate, drawn without replacement.
    rng : numpy.random.Generator
        Seeded generator (``self.np_random``) backing the draw, so activation is
        reproducible under ``reset(seed=...)``.
    """
    if not slots:
        return []
    for slot in slots:
        set_slot_contacts(model, slot, True)
    chosen = {int(i) for i in rng.choice(len(slots), size=count, replace=False)}
    for idx, slot in enumerate(slots):
        if idx not in chosen:
            hide_freejoint_slot(model, data, slot)
    return [slots[idx] for idx in sorted(chosen)]
