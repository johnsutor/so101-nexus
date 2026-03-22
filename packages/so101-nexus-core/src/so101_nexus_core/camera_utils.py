"""Shared camera parameter computation for overhead and privileged views.

Derives camera distance, lookat, and orientation from spawn configuration
so the camera always bounds the full scene.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Default vertical FOV in degrees (MuJoCo default for free cameras).
_DEFAULT_VFOV_DEG: float = 45.0

# Default render resolution (landscape, matching the forward-arc scene shape).
# Used as fallback when no CameraConfig is available.
DEFAULT_RENDER_WIDTH: int = 640
DEFAULT_RENDER_HEIGHT: int = 480


def _scene_bounds(
    spawn_center: tuple[float, float],
    spawn_max_radius: float,
    margin: float,
) -> tuple[float, float, float, float]:
    """Compute the axis-aligned bounding box of the visible scene.

    The scene includes the robot at the origin and the forward spawn arc.

    Returns
    -------
    (x_min, x_max, y_min, y_max) in world coordinates.
    """
    cx, cy = spawn_center
    # Farthest forward spawn point + margin.
    x_max = cx + spawn_max_radius + margin
    # Robot base is at origin; include a small margin behind it.
    x_min = -margin
    # Sideways extent: spawn arc can reach ±(max_r) from cy.
    y_max = cy + spawn_max_radius + margin
    y_min = cy - spawn_max_radius - margin
    return x_min, x_max, y_min, y_max


def compute_overhead_camera_params(
    spawn_center: tuple[float, float] = (0.15, 0.0),
    spawn_max_radius: float = 0.40,
    margin: float = 0.10,
    fov_deg: float = _DEFAULT_VFOV_DEG,
    aspect: float = DEFAULT_RENDER_WIDTH / DEFAULT_RENDER_HEIGHT,
) -> dict[str, Any]:
    """Compute overhead (top-down) camera parameters that tightly bound the scene.

    The lookat is centered on the bounding box of the robot + spawn area.
    The camera distance is chosen so the full scene fits in frame given
    the vertical FOV and image aspect ratio.

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
    aspect:
        Image width / height ratio (used to check horizontal fit).

    Returns
    -------
    dict with keys: lookat (3,), distance, elevation, azimuth.
    """
    x_min, x_max, y_min, y_max = _scene_bounds(spawn_center, spawn_max_radius, margin)

    # Center the camera on the bounding box.
    look_x = (x_min + x_max) / 2.0
    look_y = (y_min + y_max) / 2.0

    # Half-extents the camera must cover.
    # In the overhead view with azimuth=0 and robot facing +X:
    #   image vertical  -> world X axis
    #   image horizontal -> world Y axis
    half_x = (x_max - x_min) / 2.0
    half_y = (y_max - y_min) / 2.0

    half_vfov_rad = np.radians(fov_deg / 2.0)
    # Distance needed to fit the vertical (X) extent.
    dist_for_x = half_x / np.tan(half_vfov_rad)
    # Distance needed to fit the horizontal (Y) extent given the aspect ratio.
    half_hfov_rad = np.arctan(np.tan(half_vfov_rad) * aspect)
    dist_for_y = half_y / np.tan(half_hfov_rad)

    distance = float(max(dist_for_x, dist_for_y))

    return {
        "lookat": np.array([look_x, look_y, 0.0]),
        "distance": distance,
        "elevation": -90,
        "azimuth": 0,
    }


def compute_overhead_eye_target(
    spawn_center: tuple[float, float] = (0.15, 0.0),
    spawn_max_radius: float = 0.40,
    margin: float = 0.10,
    fov_deg: float = _DEFAULT_VFOV_DEG,
    aspect: float = DEFAULT_RENDER_WIDTH / DEFAULT_RENDER_HEIGHT,
) -> tuple[list[float], list[float]]:
    """Compute eye and target positions for an overhead look-at camera.

    Returns positions suitable for ``sapien_utils.look_at(eye, target)``
    or any eye/target camera API.  The camera looks straight down with
    the robot's forward direction (+X) pointing up in the image.

    Returns
    -------
    (eye, target) — each a 3-element list [x, y, z].
    """
    params = compute_overhead_camera_params(
        spawn_center=spawn_center,
        spawn_max_radius=spawn_max_radius,
        margin=margin,
        fov_deg=fov_deg,
        aspect=aspect,
    )
    lookat = params["lookat"]
    distance = params["distance"]
    eye = [float(lookat[0]), float(lookat[1]), float(distance)]
    target = [float(lookat[0]), float(lookat[1]), 0.0]
    return eye, target


def compute_angled_camera_params(
    spawn_center: tuple[float, float] = (0.15, 0.0),
    spawn_max_radius: float = 0.40,
    margin: float = 0.10,
    elevation: float = -30.0,
    azimuth: float = 160.0,
    fov_deg: float = _DEFAULT_VFOV_DEG,
    aspect: float = DEFAULT_RENDER_WIDTH / DEFAULT_RENDER_HEIGHT,
) -> dict[str, Any]:
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
    aspect:
        Image width / height ratio.

    Returns
    -------
    dict with keys: lookat (3,), distance, elevation, azimuth.
    """
    # Use the same scene bounds but pull the camera back a bit for the angle.
    params = compute_overhead_camera_params(
        spawn_center=spawn_center,
        spawn_max_radius=spawn_max_radius,
        margin=margin,
        fov_deg=fov_deg,
        aspect=aspect,
    )
    return {
        "lookat": params["lookat"],
        "distance": params["distance"] * 1.2,
        "elevation": elevation,
        "azimuth": azimuth,
    }
