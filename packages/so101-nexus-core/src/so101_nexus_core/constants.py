"""Pure-data constants shared across config and object modules.

This module exists to break the circular dependency between ``config``
and ``objects``: both need color maps and YCB metadata, so the data
lives here where either module can import it without a cycle.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

ColorName = Literal["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "gray"]
ColorConfig = ColorName | list[ColorName]

COLOR_MAP: dict[str, list[float]] = {
    "red": [1.0, 0.0, 0.0, 1.0],
    "orange": [1.0, 0.5, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0, 1.0],
    "green": [0.0, 1.0, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
    "purple": [0.5, 0.0, 0.5, 1.0],
    "black": [0.0, 0.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0, 1.0],
    "gray": [0.5, 0.5, 0.5, 1.0],
}

# CUBE_COLOR_MAP omits "gray" (reserved for ground) — otherwise identical to COLOR_MAP.
CUBE_COLOR_MAP: dict[str, list[float]] = {k: v for k, v in COLOR_MAP.items() if k != "gray"}

TARGET_COLOR_MAP: dict[str, list[float]] = CUBE_COLOR_MAP

YCB_OBJECTS: dict[str, str] = {
    "009_gelatin_box": "gelatin box",
    "011_banana": "banana",
    "030_fork": "fork",
    "031_spoon": "spoon",
    "032_knife": "knife",
    "033_spatula": "spatula",
    "037_scissors": "scissors",
    "040_large_marker": "large marker",
    "043_phillips_screwdriver": "phillips screwdriver",
    "058_golf_ball": "golf ball",
}


def validate_color_config(colors: ColorConfig, field_name: str) -> None:
    """Raise ``ValueError`` if any color name is not in ``COLOR_MAP``."""
    names = [colors] if isinstance(colors, str) else colors
    for name in names:
        if name not in COLOR_MAP:
            raise ValueError(f"{field_name} must be one of {list(COLOR_MAP)}, got {name!r}")


def sample_color(colors: ColorConfig, rng: np.random.Generator | None = None) -> list[float]:
    """Resolve a ColorConfig to an RGBA list. Samples uniformly if given a list."""
    if isinstance(colors, str):
        return COLOR_MAP[colors]
    if rng is None:
        rng = np.random.default_rng()
    chosen = rng.choice(colors)
    return COLOR_MAP[chosen]
