"""Torch-only reset/sampling helpers for the batched Warp object-slot backend.

The Warp backend shares one compiled model across all worlds, so heterogeneous
target selection is realized by placing the chosen slot per world and parking the
rest off-world (Warp cannot mutate model-global contact bits per world). These
helpers operate on zero-copy ``qpos`` tensor views.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from so101_nexus.object_slots import ObjectSlot
    from so101_nexus.warp._random import WorldEpisodeRNG

HIDE_CLEARANCE = 0.1
"""Clearance (m) between the off-world parking band and the reachable spawn annulus."""


def slot_geom_masks(slots: Sequence[ObjectSlot], ngeom: int, device: torch.device) -> torch.Tensor:
    """Return an ``(n_slots, ngeom)`` boolean lookup of each slot's collision geoms.

    A per-geom mask rather than a padded id matrix: the batched grasp reduction
    then costs two contact-sized gathers instead of a comparison against every
    convex part, so its cost does not grow with the decomposition. Mirrors the
    MuJoCo base env's ``_obj_geom_mask``.
    """
    masks = torch.zeros((len(slots), ngeom), dtype=torch.bool, device=device)
    for row, slot in enumerate(slots):
        masks[row, list(slot.geom_ids)] = True
    return masks


def hidden_slot_band_xy(
    device: torch.device,
    n_slots: int,
    max_bounding_radius: float,
    spawn_max_radius: float,
    spawn_center: tuple[float, float],
) -> torch.Tensor:
    """Return ``(n_slots, 2)`` off-world parking positions for inactive slots.

    Positions form a band beyond the reachable spawn annulus, spaced by object
    diameter so neither active samples nor adjacent hidden bodies overlap (Warp
    contact bits are model-global, so hidden slots stay collidable and must be
    parked, not disabled).
    """
    cx, cy = spawn_center
    step = 2.0 * max_bounding_radius + HIDE_CLEARANCE
    base = spawn_max_radius + 2.0 * max_bounding_radius + HIDE_CLEARANCE
    hide_x = cx - base - step * torch.arange(n_slots, device=device)
    return torch.stack([hide_x, torch.full((n_slots,), cy, device=device)], dim=1)


def sample_polar(
    rng: WorldEpisodeRNG,
    worlds: torch.Tensor,
    min_r: float,
    max_r: float,
    angle_half_rad: float,
    center: tuple[float, float],
) -> torch.Tensor:
    """Sample ``(n, 2)`` XY positions uniformly in a polar arc about ``center``."""
    draws = rng.rand("task", worlds, 2)
    r = draws[:, 0] * (max_r - min_r) + min_r
    theta = (draws[:, 1] * 2.0 - 1.0) * angle_half_rad
    cx, cy = center
    xy = torch.empty((worlds.numel(), 2), device=rng.device)
    xy[:, 0] = cx + r * torch.cos(theta)
    xy[:, 1] = cy + r * torch.sin(theta)
    return xy


def sample_separated_polar(
    rng: WorldEpisodeRNG,
    worlds: torch.Tensor,
    radii: torch.Tensor,
    min_clearance: float,
    min_r: float,
    max_r: float,
    angle_half_rad: float,
    center: tuple[float, float],
    max_attempts: int = 100,
) -> torch.Tensor:
    """Batched bounding-radius-aware polar sampler.

    Mirrors ``so101_nexus.mujoco.spawn_utils.sample_separated_positions`` (two
    objects clear when their centre distance is at least ``r_i + r_j +
    min_clearance``) but vectorized over worlds: each active slot is placed in
    turn, then worlds violating separation against an already-placed slot are
    resampled together.

    Parameters
    ----------
    radii : torch.Tensor
        ``(n_worlds, n_active)`` bounding radius of each active slot per world.

    Returns
    -------
    torch.Tensor
        ``(n_worlds, n_active, 2)`` sampled XY positions.
    """
    n_worlds, n_active = radii.shape
    positions = torch.empty((n_worlds, n_active, 2), device=rng.device)
    for k in range(n_active):
        positions[:, k] = sample_polar(rng, worlds, min_r, max_r, angle_half_rad, center)
        if k == 0:
            continue
        for _ in range(max_attempts):
            placed = positions[:, :k]  # (n_worlds, k, 2)
            cur = positions[:, k : k + 1]  # (n_worlds, 1, 2)
            dist = torch.linalg.norm(placed - cur, dim=2)  # (n_worlds, k)
            required = min_clearance + radii[:, :k] + radii[:, k : k + 1]
            bad = (dist < required).any(dim=1)  # (n_worlds,)
            n_bad = int(bad.sum())
            if n_bad == 0:
                break
            positions[bad, k] = sample_polar(rng, worlds[bad], min_r, max_r, angle_half_rad, center)
    return positions


def random_yaw_quat_batch(rng: WorldEpisodeRNG, worlds: torch.Tensor) -> torch.Tensor:
    """Return ``(n, 4)`` ``wxyz`` quaternions with uniformly random yaw about Z."""
    yaw = rng.rand("task", worlds) * 2.0 * torch.pi
    quat = torch.zeros((worlds.numel(), 4), device=rng.device)
    quat[:, 0] = torch.cos(yaw / 2.0)
    quat[:, 3] = torch.sin(yaw / 2.0)
    return quat


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of batched ``wxyz`` quaternions ``a * b`` -> ``(n, 4)``."""
    aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    out = torch.empty_like(a)
    out[:, 0] = aw * bw - ax * bx - ay * by - az * bz
    out[:, 1] = aw * bx + ax * bw + ay * bz - az * by
    out[:, 2] = aw * by - ax * bz + ay * bw + az * bx
    out[:, 3] = aw * bz + ax * by - ay * bx + az * bw
    return out


def _contact_budget(n_slots: int, n_collision_geoms: int) -> tuple[int, int]:
    """Reserve contacts for grasping, parked slots, and extra convex parts.

    Parts of one decomposed mesh cannot contact each other, so each extra part
    costs much less storage than an independent slot.
    """
    nconmax = 192 + 16 * n_slots + 2 * (n_collision_geoms - n_slots)
    return nconmax, nconmax * 2


class _ObjectSlotRuntime:
    """Share tensor metadata and reset writes for compiled object slots."""

    device: torch.device
    qpos: torch.Tensor
    qvel: torch.Tensor
    _rng: WorldEpisodeRNG

    if TYPE_CHECKING:

        def _setup_slot_visibility(self, slots: Sequence[ObjectSlot]) -> None: ...
        def _set_slot_visibility(self, idx: torch.Tensor, selected: torch.Tensor) -> None: ...

    def _init_slot_tensors(
        self,
        slots: Sequence[ObjectSlot],
        ngeom: int,
        spawn_max_radius: float,
        spawn_center: tuple[float, float],
    ) -> None:
        self._slot_qadr = torch.tensor([s.qpos_addr for s in slots], device=self.device)
        self._slot_dadr = torch.tensor([s.dof_addr for s in slots], device=self.device)
        # Host caches avoid a device synchronization for every parked slot.
        self._slot_qadr_host = [s.qpos_addr for s in slots]
        self._slot_dadr_host = [s.dof_addr for s in slots]
        self._qpos_offsets = torch.arange(7, device=self.device)
        self._slot_geom_masks = slot_geom_masks(slots, ngeom, self.device)
        self._setup_slot_visibility(slots)
        self._slot_spawn_z = torch.tensor(
            [s.spawn_z for s in slots], dtype=torch.float32, device=self.device
        )
        self._slot_bradius = torch.tensor(
            [s.bounding_radius for s in slots], dtype=torch.float32, device=self.device
        )
        self._slot_rest_quat = torch.tensor(
            np.stack([s.rest_quat for s in slots]), dtype=torch.float32, device=self.device
        )
        self._hide_xy = hidden_slot_band_xy(
            self.device,
            len(slots),
            float(self._slot_bradius.max()),
            spawn_max_radius,
            spawn_center,
        )

    def _hide_all_slots(self, idx: torch.Tensor) -> None:
        """Park every slot at its hidden resting pose for the reset worlds.

        Slot qpos columns are shared across worlds, so hiding uses fixed slices;
        active slots are overwritten afterward by ``_place_active_slots``.
        """
        for j, qa in enumerate(self._slot_qadr_host):
            self.qpos[idx, qa] = self._hide_xy[j, 0]
            self.qpos[idx, qa + 1] = self._hide_xy[j, 1]
            self.qpos[idx, qa + 2] = self._slot_spawn_z[j]
            self.qpos[idx, qa + 3 : qa + 7] = self._slot_rest_quat[j]
            da = self._slot_dadr_host[j]
            self.qvel[idx, da : da + 6] = 0.0

    def _place_active_slots(
        self, idx: torch.Tensor, sel: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """Place each rank's selected slot at ``positions[:, k]`` with random yaw."""
        self._set_slot_visibility(idx, sel)
        rows = idx[:, None]
        for k in range(sel.shape[1]):
            sel_k = sel[:, k]  # (n,) pool idx per reset world
            base = self._slot_qadr[sel_k]
            yaw = random_yaw_quat_batch(self._rng, idx)
            quat = quat_mul_wxyz(yaw, self._slot_rest_quat[sel_k])
            self.qpos[rows, base[:, None] + self._qpos_offsets[:3]] = torch.cat(
                [positions[:, k], self._slot_spawn_z[sel_k][:, None]], dim=1
            )
            self.qpos[rows, base[:, None] + self._qpos_offsets[3:]] = quat
