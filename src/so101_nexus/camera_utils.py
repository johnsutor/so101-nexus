"""Shared camera parameter computation for overhead and privileged views.

Derives camera distance, lookat, and orientation from spawn configuration
so the camera always bounds the full scene.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

# Default vertical FOV in degrees (MuJoCo default for free cameras).
_DEFAULT_VFOV_DEG: float = 45.0

# Default render resolution (landscape, matching the forward-arc scene shape).
# Used as fallback when no RenderConfig is available.
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

    Returns positions suitable for any eye/target camera API.  The camera
    looks straight down with the robot's forward direction (+X) pointing up
    in the image.

    Returns
    -------
    (eye, target) - each a 3-element list [x, y, z].
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
    margin: float = 0.05,
    elevation: Any = -30.0,
    azimuth: Any = 160.0,
    fov_deg: float = _DEFAULT_VFOV_DEG,
    aspect: float = DEFAULT_RENDER_WIDTH / DEFAULT_RENDER_HEIGHT,
    *,
    spawn_angle_half_range_deg: float = 90.0,
    spawn_half_size: float | None = None,
) -> dict[str, Any]:
    """Fit a side view to the spawn arc and the robot's working volume.

    Perspective depth, view angles, and image aspect determine the distance.
    The target is above the table to avoid wasting pixels on empty foreground.
    Angles can be scalar degrees, NumPy arrays, or torch tensors; distances
    retain the same array backend for batched reset-time camera sampling.

    Parameters
    ----------
    spawn_center, spawn_max_radius, spawn_angle_half_range_deg:
        Center, maximum radius, and half angular range of the spawn arc.
    spawn_half_size:
        Square spawn half-size, replacing the arc when provided.
    margin:
        Padding in meters around the workspace and robot.
    elevation, azimuth:
        Camera orientation in degrees, negative elevation looks down.
    fov_deg, aspect:
        Vertical field of view in degrees and image width / height ratio.

    Returns
    -------
    dict
        Camera lookat, distance, elevation, and azimuth.
    """
    half_range_rad = np.radians(spawn_angle_half_range_deg)
    cx, cy = spawn_center
    x_min = min(0.0, cx + spawn_max_radius * min(0.0, np.cos(half_range_rad)))
    x_max = max(0.0, cx + spawn_max_radius)
    if spawn_half_size is not None:
        x_min, x_max = min(0.0, cx - spawn_half_size), max(0.0, cx + spawn_half_size)
    lookat = np.array([(x_min + x_max) / 2, cy / 2, 0.12])
    tensor_angle = next((angle for angle in (azimuth, elevation) if hasattr(angle, "clamp")), None)
    xp = np if tensor_angle is None else sys.modules["torch"]
    if tensor_angle is not None:
        azimuth = xp.as_tensor(azimuth, dtype=tensor_angle.dtype, device=tensor_angle.device)
        elevation = xp.as_tensor(elevation, dtype=tensor_angle.dtype, device=tensor_angle.device)
    azimuth_rad = azimuth * (np.pi / 180)
    elevation_rad = elevation * (np.pi / 180)
    ca, sa = xp.cos(azimuth_rad), xp.sin(azimuth_rad)
    ce, se = xp.cos(elevation_rad), xp.sin(elevation_rad)
    forward = (ca * ce, sa * ce, se)
    right = (sa, -ca, ca * 0)
    up = (-ca * se, -sa * se, ce)
    tan_vfov = np.tan(np.radians(fov_deg / 2))
    distance = ca * 0
    for axis, tangent in ((right, tan_vfov * aspect), (up, tan_vfov)):
        for sign in (-1, 1):
            vx, vy, vz = (sign * a / tangent - f for a, f in zip(axis, forward, strict=True))
            angle = xp.clip(xp.arctan2(vy, vx), -half_range_rad, half_range_rad)
            arc_extent = xp.clip(vx * xp.cos(angle) + vy * xp.sin(angle), 0, None)
            workspace = cx * vx + cy * vy + spawn_max_radius * arc_extent
            if spawn_half_size is not None:
                workspace = cx * vx + cy * vy + spawn_half_size * (abs(vx) + abs(vy))
            workspace += margin * (vx * vx + vy * vy + vz * vz) ** 0.5
            # A sloped envelope covers the upright arm and its forward reach.
            robot = margin * (abs(vx) + abs(vy)) + 0.4 * xp.clip(vz, 0, None)
            reach = 0.25 * vx + 0.25 * vz + margin * (abs(vx) + abs(vy) + abs(vz))
            bound = xp.maximum(xp.maximum(workspace, robot), reach)
            bound -= sum(v * target for v, target in zip((vx, vy, vz), lookat, strict=True))
            distance = xp.maximum(distance, bound)
    return {"lookat": lookat, "distance": distance, "elevation": elevation, "azimuth": azimuth}


def build_overhead_camera_mjcf(
    spawn_center: tuple[float, float],
    spawn_max_radius: float,
    fov_deg: float,
    width: int,
    height: int,
    *,
    margin: float = 0.10,
    name: str = "overhead_cam",
) -> str:
    """Return a world-fixed overhead ``<camera>`` MJCF element string.

    The Warp renderer rasterizes only model cameras, so the overhead observation
    camera must be a compiled ``<camera>`` (unlike the MuJoCo backend, which uses
    a free ``MjvCamera``). Pose and field of view reuse the same scene framing as
    :func:`compute_overhead_eye_target`, so both backends frame the workspace
    identically. ``xyaxes="0 -1 0 1 0 0"`` orients the camera straight down with
    the robot's forward (+X) axis pointing up in the image.

    Parameters
    ----------
    spawn_center, spawn_max_radius, margin:
        Scene-bounds inputs shared with the overhead framing helpers.
    fov_deg:
        Vertical field of view in degrees.
    width, height:
        Observation resolution (sets the aspect used for framing; the render
        resolution is set separately via the render context).
    name:
        Camera name used for ``mj_name2id`` lookup by the backend.
    """
    eye, _ = compute_overhead_eye_target(
        spawn_center=spawn_center,
        spawn_max_radius=spawn_max_radius,
        margin=margin,
        fov_deg=fov_deg,
        aspect=width / height,
    )
    ex, ey, ez = eye
    return (
        f'    <camera name="{name}" mode="fixed" pos="{ex} {ey} {ez}" '
        f'xyaxes="0 -1 0 1 0 0" fovy="{fov_deg}" resolution="{width} {height}"/>\n'
    )
