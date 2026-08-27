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

# CUBE_COLOR_MAP omits "gray" (reserved for ground) - otherwise identical to COLOR_MAP.
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

GSO_OBJECTS: dict[str, str] = {
    "Pony_C_Clamp_1440": "C-clamp",
    "Cole_Hardware_Mini_Honey_Dipper": "honey dipper",
    "OXO_Soft_Works_Can_Opener_SnapLock": "can opener",
    "3M_Vinyl_Tape_Green_1_x_36_yd": "tape roll",
    "Shurtape_Gaffers_Tape_Silver_2_x_60_yd": "gaffer tape roll",
    "Big_O_Sponges_Assorted_Cellulose_12_pack": "sponge pack",
    "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup": "ramekin",
    "CoQ10": "supplement bottle",
    "Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold": "sprinkles canister",
    "Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle": "lotion bottle",
    "Black_Elderberry_Syrup_54_oz_Gaia_Herbs": "syrup bottle",
    "Nestle_Raisinets_Milk_Chocolate_35_oz_992_g": "candy box",
}

GSO_MASSES: dict[str, float] = {
    # GSO ships no benchmark masses (unlike YCB); each value is convex-hull
    # volume times an assumed effective density for the packaged object.
    # 448 cm^3 hull, ~0.78 g/cm^3 effective (open steel frame)
    "Pony_C_Clamp_1440": 0.350,
    # 26 cm^3 hull, ~0.70 g/cm^3 effective (solid wood)
    "Cole_Hardware_Mini_Honey_Dipper": 0.018,
    # 318 cm^3 hull, ~0.57 g/cm^3 effective (plastic + metal gears)
    "OXO_Soft_Works_Can_Opener_SnapLock": 0.180,
    # 244 cm^3 hull, ~0.41 g/cm^3 effective (vinyl roll + card)
    "3M_Vinyl_Tape_Green_1_x_36_yd": 0.100,
    # 895 cm^3 hull, ~0.56 g/cm^3 effective (cloth tape roll)
    "Shurtape_Gaffers_Tape_Silver_2_x_60_yd": 0.500,
    # 149 cm^3 hull, ~1.01 g/cm^3 effective (packed sponges)
    "Big_O_Sponges_Assorted_Cellulose_12_pack": 0.150,
    # 233 cm^3 hull, ~0.52 g/cm^3 effective (hollow bowl)
    "BIA_Porcelain_Ramekin_With_Glazed_Rim_35_45_oz_cup": 0.120,
    # 139 cm^3 hull, ~0.54 g/cm^3 effective (bottle + pills)
    "CoQ10": 0.075,
    # 214 cm^3 hull, ~0.84 g/cm^3 effective (sugar-filled canister)
    "Wilton_Pearlized_Sugar_Sprinkles_525_oz_Gold": 0.180,
    # 941 cm^3 hull, ~0.24 g/cm^3 effective (squeeze bottle, mostly cream)
    "Marc_Anthony_Strictly_Curls_Curl_Envy_Perfect_Curl_Cream_6_fl_oz_bottle": 0.230,
    # 459 cm^3 hull, ~0.59 g/cm^3 effective (glass bottle + syrup)
    "Black_Elderberry_Syrup_54_oz_Gaia_Herbs": 0.270,
    # 248 cm^3 hull, ~0.46 g/cm^3 effective (cardboard box + candy)
    "Nestle_Raisinets_Milk_Chocolate_35_oz_992_g": 0.115,
}


def validate_color_config(colors: ColorConfig, field_name: str) -> None:
    """Raise ``ValueError`` on an empty list or a color name not in ``COLOR_MAP``."""
    names = [colors] if isinstance(colors, str) else colors
    if not names:
        raise ValueError(f"{field_name} must contain at least one color, got []")
    for name in names:
        if name not in COLOR_MAP:
            raise ValueError(f"{field_name} must be one of {list(COLOR_MAP)}, got {name!r}")


def sample_color_name(colors: ColorConfig, rng: np.random.Generator | None = None) -> str:
    """Resolve a ColorConfig to a single color name. Samples uniformly if given a list.

    Parameters
    ----------
    colors : ColorConfig
        A single color name or a list of color names.
    rng : numpy.random.Generator, optional
        Seeded RNG used to sample from a list. When ``None`` a fresh
        unseeded generator is used (backward-compatible global-random behavior).

    Returns
    -------
    str
        The chosen color name.
    """
    if isinstance(colors, str):
        return colors
    if rng is None:
        rng = np.random.default_rng()
    return str(rng.choice(colors))


def sample_color(colors: ColorConfig, rng: np.random.Generator | None = None) -> list[float]:
    """Resolve a ColorConfig to an RGBA list. Samples uniformly if given a list.

    Pass a seeded ``rng`` for reproducible color selection under
    ``reset(seed=...)``; ``None`` keeps the unseeded global-random behavior.
    """
    return COLOR_MAP[sample_color_name(colors, rng)]
