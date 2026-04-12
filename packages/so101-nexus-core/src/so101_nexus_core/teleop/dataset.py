"""Dataset field selection, feature schema, and per-frame builders.

The selection surfaces in the Gradio UI as checkboxes. ``observation.state``
and ``action`` are always written; they are not user-toggleable. Everything
else can be opted out of, in which case it is excluded from both the declared
LeRobot feature schema and every recorded frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

WRIST_KEY = "observation.images.wrist_cam"
OVERHEAD_KEY = "observation.images.overhead_cam"


@dataclass(frozen=True)
class FieldSelection:
    """Which fields the user opted to persist into the LeRobot dataset."""

    wrist_image: bool = True
    overhead_image: bool = True
    task: bool = True

    @property
    def state(self) -> bool:
        """``observation.state`` is always persisted."""
        return True

    @property
    def action(self) -> bool:
        """``action`` is always persisted."""
        return True


def build_features(
    selection: FieldSelection,
    joint_names: tuple[str, ...],
    wrist_wh: tuple[int, int],
    overhead_wh: tuple[int, int],
) -> dict[str, dict[str, Any]]:
    """Return the LeRobot feature dict for the selected fields.

    Parameters
    ----------
    selection
        Which fields the user chose to persist.
    joint_names
        Names of the robot joints, used for state/action axis labels.
    wrist_wh
        ``(width, height)`` of the wrist camera image, in pixels.
    overhead_wh
        ``(width, height)`` of the overhead camera image, in pixels.
    """
    axes = list(joint_names)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(axes),),
            "names": {"axes": axes},
        },
        "action": {
            "dtype": "float32",
            "shape": (len(axes),),
            "names": {"axes": axes},
        },
    }
    if selection.wrist_image:
        w, h = wrist_wh
        features[WRIST_KEY] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": {"axes": ["height", "width", "channels"]},
        }
    if selection.overhead_image:
        w, h = overhead_wh
        features[OVERHEAD_KEY] = {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": {"axes": ["height", "width", "channels"]},
        }
    return features


def build_frame(
    selection: FieldSelection,
    *,
    state: np.ndarray,
    action: np.ndarray,
    task: str,
    wrist_image: np.ndarray | None,
    overhead_image: np.ndarray | None,
) -> dict[str, Any]:
    """Assemble one LeRobot frame dict using only the selected fields."""
    frame: dict[str, Any] = {
        "observation.state": state.astype(np.float32),
        "action": action.astype(np.float32),
    }
    if selection.task:
        frame["task"] = task
    if selection.wrist_image:
        if wrist_image is None:
            raise ValueError(
                "wrist_image selected but no wrist frame was recorded;"
                "check that the env exposes a WristCamera observation."
            )
        frame[WRIST_KEY] = wrist_image
    if selection.overhead_image:
        if overhead_image is None:
            raise ValueError(
                "overhead_image selected but no overhead frame was recorded;"
                "check that the env exposes an OverheadCamera observation."
            )
        frame[OVERHEAD_KEY] = overhead_image
    return frame
