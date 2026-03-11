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
