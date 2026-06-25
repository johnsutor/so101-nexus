"""Batched RGB readout for the MuJoCo Warp renderer.

``mujoco_warp.render`` packs each rendered pixel into a single ``uint32`` (ABGR)
inside ``RenderContext.rgb_data``. ``mujoco_warp.get_rgb`` unpacks that into
normalized ``float32`` ``[0, 1]``; the SO101-Nexus observation contract is
``uint8`` images matching the MuJoCo backend, so this module unpacks straight to
``uint8`` in one kernel launch (no float round-trip).
"""

from __future__ import annotations

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _unpack_rgb_uint8(
    packed: wp.array2d[wp.uint32],
    rgb_adr: wp.array[int],
    camera_index: int,
    out: wp.array4d[wp.uint8],
):
    """Unpack one active camera's ABGR ``uint32`` pixels into ``(N, H, W, 3)`` RGB."""
    worldid, pixelid = wp.tid()
    width = out.shape[2]
    xid = pixelid % width
    yid = pixelid // width
    val = packed[worldid, rgb_adr[camera_index] + pixelid]
    b = wp.uint8(val & wp.uint32(0xFF))
    g = wp.uint8((val >> wp.uint32(8)) & wp.uint32(0xFF))
    r = wp.uint8((val >> wp.uint32(16)) & wp.uint32(0xFF))
    out[worldid, yid, xid, 0] = r
    out[worldid, yid, xid, 1] = g
    out[worldid, yid, xid, 2] = b


def unpack_rgb_uint8(rc, camera_index: int, out: wp.array) -> None:
    """Write active camera ``camera_index`` RGB into ``out`` (``(N, H, W, 3)`` uint8).

    Parameters
    ----------
    rc:
        A ``mujoco_warp.RenderContext`` after ``mujoco_warp.render`` has run.
    camera_index:
        Active render index (position in the render context's active-camera list),
        not the MuJoCo camera id.
    out:
        Pre-allocated ``wp.array`` of shape ``(num_worlds, height, width, 3)`` and
        dtype ``wp.uint8`` on the render device.
    """
    n_worlds, height, width, _ = out.shape
    wp.launch(
        _unpack_rgb_uint8,
        dim=(n_worlds, height * width),
        inputs=[rc.rgb_data, rc.rgb_adr, camera_index],
        outputs=[out],
    )
