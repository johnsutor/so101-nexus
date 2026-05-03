"""Config for pick-and-place environments."""

from __future__ import annotations

import warnings

from so101_nexus_core.config.base import EnvironmentConfig
from so101_nexus_core.constants import ColorConfig, validate_color_config
from so101_nexus_core.observations import (
    EndEffectorPose,
    GraspState,
    ObjectOffset,
    ObjectPose,
    TargetOffset,
    TargetPosition,
)


class PickAndPlaceConfig(EnvironmentConfig):
    """Config for pick-and-place environments.

    Args:
        cube_colors: Cube color(s).
        target_colors: Target disc color(s).
        cube_half_size: Half-size of the cube in metres.
        cube_mass: Mass of the cube in kg.
        target_disc_radius: Radius of the target disc.
        min_cube_target_separation: Minimum separation between cube and target.
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        cube_colors: ColorConfig = "red",
        target_colors: ColorConfig = "blue",
        cube_half_size: float = 0.0125,
        cube_mass: float = 0.01,
        target_disc_radius: float = 0.05,
        min_cube_target_separation: float = 0.0375,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.cube_colors = cube_colors
        self.target_colors = target_colors
        self.cube_half_size = cube_half_size
        self.cube_mass = cube_mass
        self.target_disc_radius = target_disc_radius
        self.min_cube_target_separation = min_cube_target_separation
        validate_color_config(self.cube_colors, "cube_colors")
        validate_color_config(self.target_colors, "target_colors")
        cube_set = (
            {self.cube_colors} if isinstance(self.cube_colors, str) else set(self.cube_colors)
        )
        target_set = (
            {self.target_colors} if isinstance(self.target_colors, str) else set(self.target_colors)
        )
        overlap = cube_set & target_set
        if overlap:
            warnings.warn(
                f"cube_colors and target_colors overlap on {overlap}; "
                "the cube and target may be the same color in some episodes",
                stacklevel=2,
            )
        if not (0.01 <= self.cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {self.cube_half_size}")
        if self.target_disc_radius <= 0:
            raise ValueError(f"target_disc_radius must be > 0, got {self.target_disc_radius}")
        if self.min_cube_target_separation < 0:
            raise ValueError(
                f"min_cube_target_separation must be >= 0, got {self.min_cube_target_separation}"
            )
        if self.observations is None:
            self.observations = [
                EndEffectorPose(),
                GraspState(),
                TargetPosition(),
                ObjectPose(),
                ObjectOffset(),
                TargetOffset(),
            ]

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PickAndPlaceConfig(cube_colors={self.cube_colors!r}, "
            f"target_colors={self.target_colors!r}, cube_half_size={self.cube_half_size})"
        )
