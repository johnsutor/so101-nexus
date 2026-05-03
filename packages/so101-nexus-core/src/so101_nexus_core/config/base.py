"""EnvironmentConfig base class plus shared ``_normalize_objects`` helper."""

from __future__ import annotations

from typing import TYPE_CHECKING

from so101_nexus_core.config.render import RenderConfig
from so101_nexus_core.config.reward import RewardConfig
from so101_nexus_core.config.robot import RobotConfig
from so101_nexus_core.constants import ColorConfig, validate_color_config
from so101_nexus_core.objects import SceneObject

if TYPE_CHECKING:
    from so101_nexus_core.config._types import ObsMode
    from so101_nexus_core.observations import Observation


def _normalize_objects(
    objects: list[SceneObject] | SceneObject | None,
    default: SceneObject,
) -> list[SceneObject]:
    """Normalize an objects argument to a non-empty list of SceneObject.

    Used by PickConfig and LookAtConfig. Each call site keeps its own
    bespoke post-normalization validation (n_distractors checks for Pick,
    isinstance(CubeObject) enforcement for LookAt). The helper handles only
    the None / single-instance / iterable branches.
    """
    if objects is None:
        return [default]
    if isinstance(objects, SceneObject):
        return [objects]
    objs = list(objects)
    if not objs:
        raise ValueError("objects must not be empty")
    return objs


class EnvironmentConfig:
    """Base config shared by all environments.

    Contains only parameters that every environment needs. Task-specific
    parameters live in subclass configs (PickConfig, etc.).

    Args:
        render: Render camera resolution settings (visualization only).
        reward: Reward configuration.
        robot: Robot configuration.
        ground_colors: Ground plane color(s).
        max_episode_steps: Maximum steps per episode.
        goal_thresh: Distance threshold for goal achievement.
        spawn_half_size: Half-size of spawn region.
        spawn_center: Center of spawn region (x, y).
        spawn_min_radius: Minimum spawn radius.
        spawn_max_radius: Maximum spawn radius.
        spawn_angle_half_range_deg: Half angular range for spawn angle in degrees.
        obs_mode: Observation mode (state or visual).
        robot_colors: Robot arm color(s).
        robot_init_qpos_noise: Initial joint position noise.
        observations: Observation components to include in the state vector.
    """

    def __init__(
        self,
        render: RenderConfig | None = None,
        reward: RewardConfig | None = None,
        robot: RobotConfig | None = None,
        ground_colors: ColorConfig = "gray",
        max_episode_steps: int = 1024,
        goal_thresh: float = 0.025,
        spawn_half_size: float = 0.05,
        spawn_center: tuple[float, float] = (0.15, 0.0),
        spawn_min_radius: float = 0.10,
        spawn_max_radius: float = 0.30,
        spawn_angle_half_range_deg: float = 90.0,
        obs_mode: ObsMode = "state",
        robot_colors: ColorConfig = "yellow",
        robot_init_qpos_noise: float = 0.02,
        observations: list[Observation] | None = None,
    ) -> None:
        self.render = render if render is not None else RenderConfig()
        self.reward = reward if reward is not None else RewardConfig()
        self.robot = robot if robot is not None else RobotConfig()
        self.ground_colors = ground_colors
        self.max_episode_steps = max_episode_steps
        self.goal_thresh = goal_thresh
        self.spawn_half_size = spawn_half_size
        self.spawn_center = spawn_center
        self.spawn_min_radius = spawn_min_radius
        self.spawn_max_radius = spawn_max_radius
        self.spawn_angle_half_range_deg = spawn_angle_half_range_deg
        self.obs_mode = obs_mode
        self.robot_colors = robot_colors
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.observations = observations
        if self.obs_mode not in ("state", "visual"):
            raise ValueError(f"obs_mode must be state|visual, got {self.obs_mode!r}")
        if self.obs_mode == "visual":
            from so101_nexus_core.observations import _CameraObservation as _CamObs

            has_camera_component = self.observations is not None and any(
                isinstance(c, _CamObs) for c in self.observations
            )
            if not has_camera_component:
                raise ValueError(
                    "obs_mode='visual' requires at least one camera observation "
                    "component (e.g. WristCamera() or OverheadCamera()) in observations"
                )
        validate_color_config(self.ground_colors, "ground_colors")
        validate_color_config(self.robot_colors, "robot_colors")
        if self.spawn_min_radius < 0:
            raise ValueError(f"spawn_min_radius must be >= 0, got {self.spawn_min_radius}")
        if self.spawn_max_radius <= self.spawn_min_radius:
            raise ValueError(
                f"spawn_max_radius ({self.spawn_max_radius}) must be > "
                f"spawn_min_radius ({self.spawn_min_radius})"
            )
        if not (0.0 <= self.spawn_angle_half_range_deg <= 180.0):
            raise ValueError(
                f"spawn_angle_half_range_deg must be in [0, 180], "
                f"got {self.spawn_angle_half_range_deg}"
            )
        if self.observations is not None:
            from so101_nexus_core.observations import _CameraObservation

            cam_types = [type(c) for c in self.observations if isinstance(c, _CameraObservation)]
            if len(cam_types) != len(set(cam_types)):
                raise ValueError("Duplicate camera observation components are not allowed")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"{type(self).__name__}(max_episode_steps={self.max_episode_steps}, "
            f"goal_thresh={self.goal_thresh})"
        )
