"""LeRobot camera config for simulator-rendered frames."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.cameras import CameraConfig


@CameraConfig.register_subclass("sim")
@dataclass(kw_only=True)
class SimCameraConfig(CameraConfig):
    """Configuration for a camera rendered from the active simulator env.

    ``source`` names the observation key to read (e.g. ``"wrist_camera"``).
    The sentinel ``"render"`` reads ``env.render()`` instead, exposing the
    visualization-only render view (see ``RenderConfig.camera``); it requires
    the env to be constructed with ``render_mode="rgb_array"``.
    """

    source: str
