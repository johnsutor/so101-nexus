"""Render camera resolution config (visualization only, not observations)."""

from __future__ import annotations


class RenderConfig:
    """Render camera resolution settings (visualization only, not observations).

    Args:
        width: Render image width in pixels.
        height: Render image height in pixels.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"render dimensions must be > 0, got {self.width}x{self.height}")

    def __repr__(self) -> str:  # noqa: D105
        return f"RenderConfig(width={self.width}, height={self.height})"
