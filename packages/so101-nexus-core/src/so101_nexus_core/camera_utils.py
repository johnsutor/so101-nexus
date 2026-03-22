"""Shared camera parameter computation for overhead and privileged views.

Derives camera distance, lookat, and orientation from spawn configuration
so the camera always bounds the full scene.
"""

from __future__ import annotations

import numpy as np

# Default vertical FOV in degrees (MuJoCo default for free cameras).
_DEFAULT_VFOV_DEG: float = 45.0


def _distance_for_radius(radius: float, fov_deg: float = _DEFAULT_VFOV_DEG) -> float:
    """Compute minimum camera distance to see a circle of given radius.

    Uses the half-angle of the field of view to compute the distance
    from which a circle of ``radius`` fits entirely in frame.
    """
    half_fov_rad = np.radians(fov_deg / 2.0)
    return float(radius / np.tan(half_fov_rad))


def compute_overhead_camera_params(
    spawn_center: tuple[float, float] = (0.15, 0.0),
    spawn_max_radius: float = 0.40,
    margin: float = 0.15,
    fov_deg: float = _DEFAULT_VFOV_DEG,
) -> dict[str, object]:
    """Compute overhead (top-down) camera parameters that bound the spawn area.

    Parameters
    ----------
    spawn_center:
        XY center of the spawn region.
    spawn_max_radius:
        Maximum radial distance objects can spawn from the origin.
    margin:
        Extra padding in metres beyond the spawn boundary.
    fov_deg:
        Vertical field of view of the camera in degrees.

    Returns
    -------
    dict with keys: lookat (3,), distance, elevation, azimuth.
    """
    cx, cy = spawn_center
    visible_radius = (
        np.sqrt(cx**2 + cy**2) + spawn_max_radius + margin
    )
    distance = _distance_for_radius(visible_radius, fov_deg)
    return {
        "lookat": np.array([cx, cy, 0.0]),
        "distance": float(distance),
        "elevation": -90,
        "azimuth": 0,
    }


def compute_angled_camera_params(
    spawn_center: tuple[float, float] = (0.15, 0.0),
    spawn_max_radius: float = 0.40,
    margin: float = 0.15,
    elevation: float = -30.0,
    azimuth: float = 160.0,
    fov_deg: float = _DEFAULT_VFOV_DEG,
) -> dict[str, object]:
    """Compute angled (privileged) camera parameters that view the full scene.

    Parameters
    ----------
    spawn_center:
        XY center of the spawn region.
    spawn_max_radius:
        Maximum radial distance objects can spawn from the origin.
    margin:
        Extra padding in metres beyond the spawn boundary.
    elevation:
        Camera elevation angle in degrees (negative = looking down).
    azimuth:
        Camera azimuth angle in degrees.
    fov_deg:
        Vertical field of view in degrees.

    Returns
    -------
    dict with keys: lookat (3,), distance, elevation, azimuth.
    """
    cx, cy = spawn_center
    visible_radius = (
        np.sqrt(cx**2 + cy**2) + spawn_max_radius + margin
    )
    distance = _distance_for_radius(visible_radius, fov_deg) * 1.2
    return {
        "lookat": np.array([cx, cy, 0.0]),
        "distance": float(distance),
        "elevation": elevation,
        "azimuth": azimuth,
    }
