"""Torch-only reset/sampling helpers for the batched Warp object-slot backend.

The Warp backend shares one compiled model across all worlds, so heterogeneous
target selection is realized by placing the chosen slot per world and parking the
rest off-world (Warp cannot mutate model-global contact bits per world). These
helpers operate on zero-copy ``qpos`` tensor views.
"""

from __future__ import annotations

import torch

HIDE_CLEARANCE = 0.1
"""Clearance (m) between the off-world parking band and the reachable spawn annulus."""


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
    generator: torch.Generator,
    device: torch.device,
    n: int,
    min_r: float,
    max_r: float,
    angle_half_rad: float,
    center: tuple[float, float],
) -> torch.Tensor:
    """Sample ``(n, 2)`` XY positions uniformly in a polar arc about ``center``."""
    r = torch.rand(n, generator=generator, device=device) * (max_r - min_r) + min_r
    theta = (torch.rand(n, generator=generator, device=device) * 2.0 - 1.0) * angle_half_rad
    cx, cy = center
    xy = torch.empty((n, 2), device=device)
    xy[:, 0] = cx + r * torch.cos(theta)
    xy[:, 1] = cy + r * torch.sin(theta)
    return xy


def sample_separated_polar(
    generator: torch.Generator,
    device: torch.device,
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
    positions = torch.empty((n_worlds, n_active, 2), device=device)
    for k in range(n_active):
        positions[:, k] = sample_polar(
            generator, device, n_worlds, min_r, max_r, angle_half_rad, center
        )
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
            positions[bad, k] = sample_polar(
                generator, device, n_bad, min_r, max_r, angle_half_rad, center
            )
    return positions


def random_yaw_quat_batch(generator: torch.Generator, device: torch.device, n: int) -> torch.Tensor:
    """Return ``(n, 4)`` ``wxyz`` quaternions with uniformly random yaw about Z."""
    yaw = torch.rand(n, generator=generator, device=device) * 2.0 * torch.pi
    quat = torch.zeros((n, 4), device=device)
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
