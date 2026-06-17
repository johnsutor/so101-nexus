"""Tensor-friendly spawning utilities for the ManiSkill backend.

Mirrors the semantics of ``so101_nexus_mujoco.spawn_utils.sample_separated_positions``
but in a batched torch implementation so it works on the GPU and (planned) MuJoCo
Warp backends, not a NumPy-only path.
"""

from __future__ import annotations

import torch


def sample_separated_positions_torch(
    *,
    num_envs: int,
    bounding_radii: torch.Tensor,
    min_r: float,
    max_r: float,
    angle_half: float,
    min_clearance: float,
    center: tuple[float, float],
    device: torch.device | str,
    max_attempts: int = 100,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample per-env 2D positions in a polar arc with bounding-radius separation.

    Objects are placed one at a time. For each new object, positions are sampled
    in the polar arc and rejected (per env row) if the object's bounding circle
    (radius ``bounding_radii[k]``) plus ``min_clearance`` overlaps any
    already-placed object's bounding circle. Only rows that still violate the
    separation constraint are resampled, up to ``max_attempts`` times, after
    which the last sample is accepted as best effort. This mirrors the semantics
    of ``so101_nexus_mujoco.spawn_utils.sample_separated_positions`` but is
    batched over ``num_envs``.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments (rows).
    bounding_radii : torch.Tensor
        Shape ``(n_objects,)`` per-object bounding radius used for overlap checks.
    min_r, max_r : float
        Radial bounds from ``center``, in metres.
    angle_half : float
        Half-angle of the arc in radians; samples are drawn from
        ``[-angle_half, angle_half]``.
    min_clearance : float
        Minimum gap between bounding circles of any two placed objects.
    center : tuple[float, float]
        XY offset applied to all sampled positions.
    device : torch.device or str
        Device on which to allocate tensors.
    max_attempts : int, optional
        Maximum resampling attempts per object before accepting best effort.
    generator : torch.Generator, optional
        Optional torch RNG for reproducible sampling.

    Returns
    -------
    torch.Tensor
        Shape ``(num_envs, n_objects, 2)`` XY positions per environment.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")

    n_objects = int(bounding_radii.shape[0])
    cx, cy = center
    positions = torch.zeros((num_envs, n_objects, 2), device=device)

    def _sample(rows: int) -> torch.Tensor:
        r = min_r + torch.rand(rows, device=device, generator=generator) * (max_r - min_r)
        theta = (torch.rand(rows, device=device, generator=generator) * 2 - 1) * angle_half
        xy = torch.zeros((rows, 2), device=device)
        xy[:, 0] = cx + r * torch.cos(theta)
        xy[:, 1] = cy + r * torch.sin(theta)
        return xy

    for k in range(n_objects):
        positions[:, k, :] = _sample(num_envs)
        if k == 0:
            continue
        # Required separation to each placed object j: r_k + r_j + clearance.
        min_dist = bounding_radii[k] + bounding_radii[:k] + min_clearance  # (k,)
        for _ in range(max_attempts):
            # (num_envs, k) distance from candidate k to each placed object j.
            dists = torch.linalg.norm(positions[:, k, :].unsqueeze(1) - positions[:, :k, :], dim=2)
            violating = (dists < min_dist.unsqueeze(0)).any(dim=1)  # (num_envs,)
            # bool(violating.any()) and int(violating.sum()) each force a
            # device-to-host sync per iteration (bounded by max_attempts), an
            # accepted tradeoff for this rejection loop. A future GPU-hot-path
            # optimization could use a fixed-iteration variant to avoid syncs.
            if not bool(violating.any()):
                # Loop-exit on success; on exhausting max_attempts the loop falls
                # through and the last sample is returned even if still
                # overlapping (silent best-effort, matching the MuJoCo
                # sample_separated_positions for/else fallback).
                break
            n_bad = int(violating.sum())
            positions[violating, k, :] = _sample(n_bad)

    return positions
