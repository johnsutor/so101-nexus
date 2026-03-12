"""Spawning utilities for MuJoCo SO101-Nexus environments.

Provides helpers for placing objects in the workspace without overlap
and for sampling random object orientations.
"""

from __future__ import annotations

import numpy as np


def sample_separated_positions(
    rng: np.random.Generator,
    count: int,
    min_r: float,
    max_r: float,
    angle_half: float,
    min_clearance: float,
    bounding_radii: list[float],
    max_attempts: int = 100,
) -> list[tuple[float, float]]:
    """Sample 2D positions in a polar arc with bounding-radius-aware separation.

    Tries up to `max_attempts` times to place each object such that its bounding
    circle (radius = bounding_radii[i]) plus `min_clearance` does not overlap any
    already-placed bounding circle. Falls back to the last sampled position if no
    valid placement is found within the attempt budget.
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    positions: list[tuple[float, float]] = []
    for idx in range(count):
        for _ in range(max_attempts):
            r = rng.uniform(min_r, max_r)
            theta = rng.uniform(-angle_half, angle_half)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
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
