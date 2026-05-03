"""LookAtConfig: config for the look-at primitive task."""

from __future__ import annotations

import numpy as np

from so101_nexus_core.config.base import EnvironmentConfig, _normalize_objects
from so101_nexus_core.objects import CubeObject, SceneObject
from so101_nexus_core.observations import JointPositions


class LookAtConfig(EnvironmentConfig):
    """Config for the look-at primitive task.

    Args:
        objects: Object(s) to sample as the look-at target. Accepts a single
            SceneObject, a list, or None (defaults to [CubeObject()]).
            Only CubeObject targets are currently supported.
        orientation_success_threshold_deg: Max angular error in degrees for success.
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        objects: list[SceneObject] | SceneObject | None = None,
        orientation_success_threshold_deg: float = 5.73,
        **kwargs,
    ) -> None:
        kwargs.setdefault("max_episode_steps", 256)
        super().__init__(**kwargs)
        self.objects: list[SceneObject] = _normalize_objects(objects, CubeObject())
        self.orientation_success_threshold_deg = orientation_success_threshold_deg
        if self.observations is None:
            self.observations = [JointPositions()]
        for obj in self.objects:
            if not isinstance(obj, CubeObject):
                raise TypeError(
                    f"LookAtConfig only supports CubeObject targets, got {type(obj).__name__}"
                )

    @property
    def _orientation_success_threshold_rad(self) -> float:
        """Orientation success threshold converted to radians (internal use only)."""
        return float(np.radians(self.orientation_success_threshold_deg))
