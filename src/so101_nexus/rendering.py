"""Backend-independent rendering quality profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, init=False)
class RenderingQuality:
    """Resolve a rendering profile with optional explicit overrides.

    Parameters
    ----------
    profile : {"default", "reproducible"}
        ``"default"`` preserves eight MSAA samples and shadows.
        ``"reproducible"`` uses four MSAA samples and disables shadows, a
        workaround tested for repeated renders in a local MuJoCo OpenGL setup.
        It does not guarantee identical pixels across drivers or hardware.
    msaa_samples : int, optional
        Nonnegative multisample count. Zero disables antialiasing. The graphics
        driver may clamp the requested count to its supported maximum.
    shadows : bool, optional
        Enable shadow rendering, overriding the profile.
    shadow_map_size : int, optional
        Positive shadow texture size in pixels, defaulting to 8192.
    """

    profile: Literal["default", "reproducible"]
    msaa_samples: int
    shadows: bool
    shadow_map_size: int

    def __init__(
        self,
        profile: Literal["default", "reproducible"] = "default",
        *,
        msaa_samples: int | None = None,
        shadows: bool | None = None,
        shadow_map_size: int | None = None,
    ) -> None:
        if profile not in ("default", "reproducible"):
            raise ValueError("profile must be 'default' or 'reproducible'")
        samples = (8 if profile == "default" else 4) if msaa_samples is None else msaa_samples
        enabled = profile == "default" if shadows is None else shadows
        shadow_size = 8192 if shadow_map_size is None else shadow_map_size
        for name, value, minimum in (
            ("msaa_samples", samples, 0),
            ("shadow_map_size", shadow_size, 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
        if not isinstance(enabled, bool):
            raise ValueError(f"shadows must be a bool, got {enabled!r}")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "msaa_samples", samples)
        object.__setattr__(self, "shadows", enabled)
        object.__setattr__(self, "shadow_map_size", shadow_size)
