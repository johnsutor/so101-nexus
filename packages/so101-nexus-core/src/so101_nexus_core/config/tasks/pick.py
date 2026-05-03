"""PickConfig: unified pick environment config."""

from __future__ import annotations

from so101_nexus_core.config.base import EnvironmentConfig, _normalize_objects
from so101_nexus_core.objects import CubeObject, SceneObject
from so101_nexus_core.observations import (
    EndEffectorPose,
    GraspState,
    ObjectOffset,
    ObjectPose,
)


class PickConfig(EnvironmentConfig):
    """Config for the unified pick environment.

    The ``objects`` list defines the pool of scene objects to sample from each
    episode. One object is chosen as the target; ``n_distractors`` additional
    objects are sampled from the remaining pool and placed as distractors.
    Task descriptions are auto-generated from each object's ``__repr__``.

    Args:
        objects: Pool of scene objects to sample from. Accepts a single ``SceneObject``,
            a list of ``SceneObject``, or ``None`` (defaults to ``[CubeObject()]``).
            A single object is automatically wrapped in a list.
        n_distractors: Number of distractor objects to place. 0 means single-object scene.
        lift_threshold: Minimum height above initial z to count as lifted.
        max_goal_height: Height cap used to normalize lift progress to [0, 1].
        min_object_separation: Minimum distance between spawned objects (metres).
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        objects: list[SceneObject] | SceneObject | None = None,
        n_distractors: int = 0,
        lift_threshold: float = 0.05,
        max_goal_height: float = 0.08,
        min_object_separation: float = 0.04,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.objects: list[SceneObject] = _normalize_objects(objects, CubeObject())
        self.n_distractors = n_distractors
        self.lift_threshold = lift_threshold
        self.max_goal_height = max_goal_height
        self.min_object_separation = min_object_separation
        if self.n_distractors < 0:
            raise ValueError(f"n_distractors must be >= 0, got {self.n_distractors}")
        if self.n_distractors > 0 and len(self.objects) < self.n_distractors + 1:
            raise ValueError(
                f"objects pool must have at least n_distractors+1={self.n_distractors + 1} "
                f"entries to support {self.n_distractors} distractors, got {len(self.objects)}"
            )
        if self.min_object_separation < 0:
            raise ValueError(
                f"min_object_separation must be >= 0, got {self.min_object_separation}"
            )
        if self.observations is None:
            self.observations = [EndEffectorPose(), GraspState(), ObjectPose(), ObjectOffset()]

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PickConfig(objects={self.objects!r}, n_distractors={self.n_distractors}, "
            f"lift_threshold={self.lift_threshold}, max_goal_height={self.max_goal_height})"
        )
