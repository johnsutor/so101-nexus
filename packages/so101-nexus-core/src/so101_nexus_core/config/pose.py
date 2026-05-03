"""Robot arm pose definitions with fixed and free joints."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from so101_nexus_core.config._types import JointSpec


class Pose:
    """A named robot arm pose with fixed and free joints.

    Each joint is either a fixed angle (``float``) or a uniform sampling
    range (``tuple[float, float]``). Fixed joints return the same value
    every call; free joints are sampled uniformly.

    All angles are in degrees for the public API.

    Parameters
    ----------
    name : str
        Human-readable identifier for this pose.
    shoulder_pan_deg : JointSpec
        Shoulder pan angle or range in degrees.
    shoulder_lift_deg : JointSpec
        Shoulder lift angle or range in degrees.
    elbow_flex_deg : JointSpec
        Elbow flex angle or range in degrees.
    wrist_flex_deg : JointSpec
        Wrist flex angle or range in degrees.
    wrist_roll_deg : JointSpec
        Wrist roll angle or range in degrees.
    gripper_deg : JointSpec
        Gripper angle or range in degrees.
    """

    def __init__(
        self,
        *,
        name: str,
        shoulder_pan_deg: JointSpec,
        shoulder_lift_deg: JointSpec,
        elbow_flex_deg: JointSpec,
        wrist_flex_deg: JointSpec,
        wrist_roll_deg: JointSpec,
        gripper_deg: JointSpec,
    ) -> None:
        self.name = name
        self._specs: tuple[JointSpec, ...] = (
            shoulder_pan_deg,
            shoulder_lift_deg,
            elbow_flex_deg,
            wrist_flex_deg,
            wrist_roll_deg,
            gripper_deg,
        )
        for spec in self._specs:
            if isinstance(spec, tuple) and spec[0] > spec[1]:
                raise ValueError(f"Joint range min must be <= max, got ({spec[0]}, {spec[1]})")

    def sample(self, rng: np.random.Generator) -> tuple[float, ...]:
        """Return a concrete 6-tuple of joint angles in degrees."""
        values: list[float] = []
        for spec in self._specs:
            if isinstance(spec, tuple):
                values.append(float(rng.uniform(spec[0], spec[1])))
            else:
                values.append(float(spec))
        return tuple(values)

    def sample_rad(self, rng: np.random.Generator) -> tuple[float, ...]:
        """Return a concrete 6-tuple of joint angles in radians."""
        return tuple(float(np.radians(v)) for v in self.sample(rng))

    def __repr__(self) -> str:  # noqa: D105
        return f"Pose(name={self.name!r})"


REST_POSE = Pose(
    name="rest",
    shoulder_pan_deg=(-110.0, 110.0),
    shoulder_lift_deg=-90.0,
    elbow_flex_deg=90.0,
    wrist_flex_deg=37.8152144786,
    wrist_roll_deg=(-157.0, 163.0),
    gripper_deg=(-10.0, 100.0),
)

EXTENDED_POSE = Pose(
    name="extended",
    shoulder_pan_deg=(-110.0, 110.0),
    shoulder_lift_deg=-30.0,
    elbow_flex_deg=20.0,
    wrist_flex_deg=10.0,
    wrist_roll_deg=(-157.0, 163.0),
    gripper_deg=(-10.0, 100.0),
)

POSES: dict[str, Pose] = {
    "rest": REST_POSE,
    "extended": EXTENDED_POSE,
}
