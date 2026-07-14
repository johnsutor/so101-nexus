"""Canonical, typed configuration objects for SO101-Nexus.

This module provides a HuggingFace-style configuration surface.
Each environment type gets its own config that inherits from a shared base.
Configs stay backend-agnostic: the MuJoCo backend consumes them today and the
planned MuJoCo Warp backend will reuse them unchanged.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from so101_nexus.constants import (
    ColorConfig,
    validate_color_config,
)
from so101_nexus.objects import CubeObject, SceneObject
from so101_nexus.observations import (
    EndEffectorPose,
    GazeDirection,
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    TargetOffset,
    TargetPosition,
)

if TYPE_CHECKING:
    from so101_nexus.observations import Observation
ControlMode = Literal["pd_joint_pos", "pd_joint_delta_pos", "pd_joint_target_delta_pos"]
ObsMode = Literal["state", "visual"]

YcbModelId = Literal[
    "009_gelatin_box",
    "011_banana",
    "030_fork",
    "031_spoon",
    "032_knife",
    "033_spatula",
    "037_scissors",
    "040_large_marker",
    "043_phillips_screwdriver",
    "058_golf_ball",
]

SO101_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

MoveDirection = Literal["up", "down", "left", "right", "forward", "backward"]

DIRECTION_VECTORS: dict[MoveDirection, tuple[float, float, float]] = {
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
}

_validate_color_config = validate_color_config


class RenderConfig:
    """Render camera resolution settings (visualization only, not observations).

    Parameters
    ----------
    width : int
        Render image width in pixels.
    height : int
        Render image height in pixels.
    """

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self.width = width
        self.height = height
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"render dimensions must be > 0, got {self.width}x{self.height}")

    def __repr__(self) -> str:  # noqa: D105
        return f"RenderConfig(width={self.width}, height={self.height})"


JointSpec = float | tuple[float, float]


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

    def bounds_rad(self) -> tuple[np.ndarray, np.ndarray]:
        """Return per-joint ``(low, high)`` radian bounds for batched sampling.

        Fixed joints yield ``low == high``; range joints yield their endpoints.
        Lets a batched, seeded sampler draw ``low + u * (high - low)`` per world
        from the same spec ``sample_rad`` uses for the scalar path.
        """
        low = np.array([s[0] if isinstance(s, tuple) else s for s in self._specs], dtype=np.float64)
        high = np.array(
            [s[1] if isinstance(s, tuple) else s for s in self._specs], dtype=np.float64
        )
        return np.radians(low), np.radians(high)

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


class RobotConfig:
    """Configurable robot parameters.

    Joint names are intentionally not included here - they are structural
    identifiers that must match the URDF/MJCF and should not be overridden.

    Parameters
    ----------
    rest_qpos_deg : tuple[float, ...]
        Rest joint positions in degrees.
    init_pose : str | Pose | None
        Initial pose for resets. A string looks up from ``POSES``,
        a ``Pose`` instance is used directly, ``None`` uses the legacy
        ``rest_qpos_deg`` + noise path.
    grasp_force_threshold : float
        Force threshold for grasp detection.
    static_vel_threshold : float
        Velocity threshold for static detection.
    """

    def __init__(
        self,
        rest_qpos_deg: tuple[float, ...] = (0.0, -90.0, 90.0, 37.8152144786, 0.0, -63.0253574644),
        init_pose: str | Pose | None = None,
        grasp_force_threshold: float = 0.5,
        static_vel_threshold: float = 0.2,
    ) -> None:
        self.rest_qpos_deg = rest_qpos_deg
        self.grasp_force_threshold = grasp_force_threshold
        self.static_vel_threshold = static_vel_threshold
        if len(self.rest_qpos_deg) != 6:
            raise ValueError(
                f"rest_qpos_deg must have exactly 6 elements, got {len(self.rest_qpos_deg)}"
            )
        if isinstance(init_pose, str) and init_pose not in POSES:
            raise ValueError(f"Unknown pose name {init_pose!r}. Available: {list(POSES)}")
        self.init_pose: str | Pose | None = init_pose

    def resolve_pose(self) -> Pose | None:
        """Return the resolved Pose object, or None if not set."""
        if self.init_pose is None:
            return None
        if isinstance(self.init_pose, Pose):
            return self.init_pose
        return POSES[self.init_pose]

    @property
    def rest_qpos_rad(self) -> tuple[float, ...]:
        """Rest joint positions in radians."""
        return tuple(float(np.radians(v)) for v in self.rest_qpos_deg)

    @property
    def rest_qpos(self) -> tuple[float, ...]:
        """Backward-compatible alias returning radians."""
        return self.rest_qpos_rad

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RobotConfig(init_pose={self.init_pose!r}, "
            f"grasp_force_threshold={self.grasp_force_threshold}, "
            f"static_vel_threshold={self.static_vel_threshold})"
        )


class RobotCameraPreset:
    """Robot-specific camera and mounting parameters.

    Parameters
    ----------
    base_quat : tuple[float, float, float, float]
        Base quaternion (w, x, y, z).
    sensor_cam_eye_pos : tuple[float, float, float]
        Sensor camera eye position.
    sensor_cam_target_pos : tuple[float, float, float]
        Sensor camera target position.
    human_cam_eye_pos : tuple[float, float, float]
        Human camera eye position.
    human_cam_target_pos : tuple[float, float, float]
        Human camera target position.
    wrist_camera_mount_link : str
        Link name for wrist camera mounting.
    wrist_cam_pos_center : tuple[float, float, float]
        Center position for wrist camera.
    wrist_cam_pos_noise : tuple[float, float, float]
        Position noise for wrist camera.
    wrist_cam_euler_center_deg : tuple[float, float, float]
        Center Euler angles in degrees.
    wrist_cam_euler_noise_deg : tuple[float, float, float]
        Euler angle noise in degrees.
    """

    def __init__(
        self,
        base_quat: tuple[float, float, float, float],
        sensor_cam_eye_pos: tuple[float, float, float],
        sensor_cam_target_pos: tuple[float, float, float],
        human_cam_eye_pos: tuple[float, float, float],
        human_cam_target_pos: tuple[float, float, float],
        wrist_camera_mount_link: str,
        wrist_cam_pos_center: tuple[float, float, float],
        wrist_cam_pos_noise: tuple[float, float, float],
        wrist_cam_euler_center_deg: tuple[float, float, float],
        wrist_cam_euler_noise_deg: tuple[float, float, float],
    ) -> None:
        self.base_quat = base_quat
        self.sensor_cam_eye_pos = sensor_cam_eye_pos
        self.sensor_cam_target_pos = sensor_cam_target_pos
        self.human_cam_eye_pos = human_cam_eye_pos
        self.human_cam_target_pos = human_cam_target_pos
        self.wrist_camera_mount_link = wrist_camera_mount_link
        self.wrist_cam_pos_center = wrist_cam_pos_center
        self.wrist_cam_pos_noise = wrist_cam_pos_noise
        self.wrist_cam_euler_center_deg = wrist_cam_euler_center_deg
        self.wrist_cam_euler_noise_deg = wrist_cam_euler_noise_deg

    @property
    def wrist_cam_euler_center_rad(self) -> tuple[float, float, float]:
        """Wrist camera Euler center angles converted to radians."""
        x, y, z = self.wrist_cam_euler_center_deg
        return (float(np.radians(x)), float(np.radians(y)), float(np.radians(z)))

    @property
    def wrist_cam_euler_noise_rad(self) -> tuple[float, float, float]:
        """Wrist camera Euler noise angles converted to radians."""
        x, y, z = self.wrist_cam_euler_noise_deg
        return (float(np.radians(x)), float(np.radians(y)), float(np.radians(z)))

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RobotCameraPreset(wrist_camera_mount_link={self.wrist_camera_mount_link!r}, "
            f"wrist_cam_euler_center_deg={self.wrist_cam_euler_center_deg})"
        )


REWARD_COMPONENT_KEYS: tuple[str, ...] = (
    "reaching",
    "grasping",
    "task_objective",
    "completion_bonus",
    "action_delta_penalty",
    "energy_penalty",
)
"""Named reward facets recorded alongside the scalar total.

Mirrors ``RewardConfig``'s own weight/penalty names. Every task backend
decomposes its per-step reward onto this fixed vocabulary (see
``RewardConfig.compute_components`` / ``compute_simple_components``), so the
recorded LeRobot dataset schema does not depend on which task is running.
"""


class RewardConfig:
    """Normalized reward budget.

    The four component weights must sum to 1.0. Penalty terms are applied
    additively and subtracted from the base reward.

    Parameters
    ----------
    reaching : float
        Weight for TCP-to-object distance shaping.
    grasping : float
        Weight for grasp binary signal.
    task_objective : float
        Weight for task-specific progress.
    completion_bonus : float
        Weight for episode completion signal.
    action_delta_penalty : float
        Penalty coefficient on L2 norm of consecutive action deltas.
    energy_penalty : float
        Penalty coefficient on L2 norm of the action vector (energy cost).
    tanh_shaping_scale : float
        Scale factor for tanh distance shaping.
    velocity_shaping_scale : float
        Scale factor for tanh velocity shaping (used by task potentials that
        include a staticness factor, e.g. pick-and-place; see
        ``PickAndPlaceEnv._task_potential`` in ``so101_nexus.mujoco.pick_and_place``).
    """

    def __init__(
        self,
        reaching: float = 0.25,
        grasping: float = 0.25,
        task_objective: float = 0.25,
        completion_bonus: float = 0.25,
        action_delta_penalty: float = 0.0,
        energy_penalty: float = 0.0,
        tanh_shaping_scale: float = 5.0,
        velocity_shaping_scale: float = 15.0,
    ) -> None:
        total = reaching + grasping + task_objective + completion_bonus
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(f"Reward weights must sum to 1.0, got {total:.6f}")
        self.reaching = reaching
        self.grasping = grasping
        self.task_objective = task_objective
        self.completion_bonus = completion_bonus
        self.action_delta_penalty = action_delta_penalty
        self.energy_penalty = energy_penalty
        self.tanh_shaping_scale = tanh_shaping_scale
        self.velocity_shaping_scale = velocity_shaping_scale

    def compute(
        self,
        reach_progress: Any,
        is_grasped: Any,
        task_progress: Any,
        is_complete: Any,
        action_delta_norm: Any = 0.0,
        energy_norm: Any = 0.0,
    ) -> Any:
        """Compute a normalized reward using this config's weights.

        Tensor-agnostic like ``apply_penalties``: ``reach_progress`` /
        ``task_progress`` and the ``is_grasped`` / ``is_complete`` flags may be
        Python scalars, NumPy arrays, or torch tensors. Plain multiplication
        promotes Python/NumPy/torch booleans to numeric, so both simulation
        backends call this one combiner instead of inlining the weighted sum.
        The scalar path returns a plain ``float``.

        On completion the reward is lifted to the full budget (the four weights
        sum to 1.0), so a successful terminal step is always the global maximum
        with ``completion_bonus`` as the guaranteed margin over any non-terminal
        state. This mirrors ManiSkill PickCube's ``reward[success] = max`` and
        keeps success rewarding even when a task (pick-and-place) must release the
        grasp to finish, which would otherwise zero the grasping and task terms.
        ``is_complete`` is forwarded to ``apply_penalties`` so a penalized success
        stays floored at the guaranteed margin instead of falling below the best
        reward any non-terminal state can reach.
        """
        shaped = (
            self.reaching * reach_progress
            + self.grasping * is_grasped
            + self.task_objective * task_progress
        )
        base = shaped + (1.0 - shaped) * is_complete
        return self.apply_penalties(
            base,
            action_delta_norm=action_delta_norm,
            energy_norm=energy_norm,
            is_complete=is_complete,
        )

    def compute_components(
        self,
        reach_progress: Any,
        is_grasped: Any,
        task_progress: Any,
        is_complete: Any,
        action_delta_norm: Any = 0.0,
        energy_norm: Any = 0.0,
    ) -> dict[str, Any]:
        """Weighted reward terms that sum to ``compute()``'s return value.

        Same tensor-agnostic dispatch as ``compute()``. Each of the six
        ``REWARD_COMPONENT_KEYS`` is a named facet of the reward (the four
        budget weights plus the two penalties) so callers can log or record
        them individually, e.g. per-step in a LeRobot dataset. ``completion_bonus``
        is defined as the residual against ``compute()``'s total, so it absorbs
        both the terminal budget lift and any penalty-floor rescue from
        ``apply_penalties`` -- the six terms always sum exactly to what
        ``compute()`` returns.
        """
        reaching_term = self.reaching * reach_progress
        grasping_term = self.grasping * is_grasped
        task_term = self.task_objective * task_progress
        action_penalty_term = -self.action_delta_penalty * action_delta_norm
        energy_penalty_term = -self.energy_penalty * energy_norm
        total = self.compute(
            reach_progress,
            is_grasped,
            task_progress,
            is_complete,
            action_delta_norm=action_delta_norm,
            energy_norm=energy_norm,
        )
        completion_term = (
            total
            - reaching_term
            - grasping_term
            - task_term
            - action_penalty_term
            - energy_penalty_term
        )
        values = (
            reaching_term,
            grasping_term,
            task_term,
            completion_term,
            action_penalty_term,
            energy_penalty_term,
        )
        return dict(zip(REWARD_COMPONENT_KEYS, values, strict=True))

    def compute_simple_components(
        self,
        progress: Any,
        success: Any,
        *,
        progress_key: str = "reaching",
        action_delta_norm: Any = 0.0,
        energy_norm: Any = 0.0,
    ) -> dict[str, Any]:
        """Component breakdown for single-objective ``simple_reward`` tasks.

        Mirrors ``so101_nexus.rewards.simple_reward`` + ``apply_penalties``,
        the formula ``TouchEnv``/``MoveEnv``/``LookAtEnv`` use instead of
        ``compute()`` (their ``reaching``/``grasping``/``task_objective``
        weights are inert, see ``TestInertRewardWeightWarning``). ``progress_key``
        selects which of ``"reaching"``/``"task_objective"`` receives the
        shaped progress term -- ``"grasping"`` and the other progress bucket
        are always zero. Terms sum exactly to what ``simple_reward()`` +
        ``apply_penalties()`` return.
        """
        if progress_key not in ("reaching", "task_objective"):
            raise ValueError(
                f"progress_key must be 'reaching' or 'task_objective', got {progress_key!r}"
            )
        shaped = (1.0 - self.completion_bonus) * progress
        base = shaped + (1.0 - shaped) * success
        action_penalty_term = -self.action_delta_penalty * action_delta_norm
        energy_penalty_term = -self.energy_penalty * energy_norm
        total = self.apply_penalties(
            base,
            action_delta_norm=action_delta_norm,
            energy_norm=energy_norm,
            is_complete=success,
        )
        completion_term = total - shaped - action_penalty_term - energy_penalty_term
        components: dict[str, Any] = dict.fromkeys(REWARD_COMPONENT_KEYS, 0.0)
        components["completion_bonus"] = completion_term
        components["action_delta_penalty"] = action_penalty_term
        components["energy_penalty"] = energy_penalty_term
        components[progress_key] = shaped
        return components

    def apply_penalties(
        self,
        base: Any,
        action_delta_norm: Any = 0.0,
        energy_norm: Any = 0.0,
        is_complete: Any = False,
    ) -> Any:
        """Subtract action-smoothness and energy penalties from a base reward.

        Works for Python scalars, NumPy arrays, and torch tensors because it uses
        only arithmetic operators that all three types overload. A tensor backend
        may inline the equivalent ops, but should cite this method.

        Parameters
        ----------
        base : float or array_like
            Base reward before penalties.
        action_delta_norm : float or array_like, optional
            L2 norm of the difference between consecutive policy actions.
        energy_norm : float or array_like, optional
            L2 norm of the current action vector.
        is_complete : bool or array_like, optional
            Whether the task is complete on this step. When true, the penalized
            reward is floored at ``1 - completion_bonus``, so a penalized success
            never falls below the best reward any non-terminal state can reach.
            Defaults to ``False`` (no flooring), matching the historical
            penalty-only behaviour when the caller has no completion signal.

        Returns
        -------
        Same type as ``base``
            ``base - action_delta_penalty * action_delta_norm - energy_penalty *
            energy_norm``, floored at ``1 - completion_bonus`` where
            ``is_complete``.
        """
        penalized = (
            base - self.action_delta_penalty * action_delta_norm - self.energy_penalty * energy_norm
        )
        floor = 1.0 - self.completion_bonus
        shortfall = floor - penalized
        needs_floor = is_complete * (shortfall > 0)
        return penalized + shortfall * needs_floor

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"RewardConfig(reaching={self.reaching}, grasping={self.grasping}, "
            f"task_objective={self.task_objective}, completion_bonus={self.completion_bonus}, "
            f"action_delta_penalty={self.action_delta_penalty}, "
            f"energy_penalty={self.energy_penalty})"
        )


_REWARD_DEFAULTS = RewardConfig()


def _warn_inert_reward_weights(
    reward: RewardConfig, task_name: str, *, uses_tanh_scale: bool
) -> None:
    """Warn when a single-objective task's ``RewardConfig`` sets fields it ignores.

    ``TouchConfig``/``MoveConfig``/``LookAtConfig`` reward via
    ``so101_nexus.rewards.simple_reward``, not ``RewardConfig.compute``, so
    ``reaching``/``grasping``/``task_objective`` never affect the reward.
    ``LookAtConfig`` additionally ignores ``tanh_shaping_scale`` because
    ``orientation_progress`` has no distance term.
    """
    inert = {
        "reaching": reward.reaching,
        "grasping": reward.grasping,
        "task_objective": reward.task_objective,
    }
    if not uses_tanh_scale:
        inert["tanh_shaping_scale"] = reward.tanh_shaping_scale
    customized = [name for name, value in inert.items() if value != getattr(_REWARD_DEFAULTS, name)]
    if customized:
        live_fields = "completion_bonus" + (", tanh_shaping_scale" if uses_tanh_scale else "")
        warnings.warn(
            f"{task_name} reward is single-objective and ignores RewardConfig field(s) "
            f"{customized}; only {live_fields}, action_delta_penalty, and energy_penalty "
            "affect it (see so101_nexus.rewards.simple_reward).",
            stacklevel=3,
        )


def _warn_inert_velocity_scale(reward: RewardConfig, task_name: str) -> None:
    """Warn when ``velocity_shaping_scale`` is customized on a task that ignores it.

    Only ``PickAndPlaceEnv``'s task potential (``Phi_place``) reads
    ``velocity_shaping_scale`` (see ``_task_potential``); every other task
    (``PickConfig``/``PickLiftEnv``, ``TouchConfig``, ``MoveConfig``,
    ``LookAtConfig``) never constructs a velocity-gated potential, so
    customizing it there is a silent no-op.
    """
    if reward.velocity_shaping_scale != _REWARD_DEFAULTS.velocity_shaping_scale:
        warnings.warn(
            f"{task_name} reward ignores RewardConfig.velocity_shaping_scale (only "
            "PickAndPlaceConfig's task potential uses it; see "
            "so101_nexus.mujoco.pick_and_place.PickAndPlaceEnv._task_potential).",
            stacklevel=3,
        )


class EnvironmentConfig:
    """Base config shared by all environments.

    Contains only parameters that every environment needs. Task-specific
    parameters live in subclass configs (PickConfig, etc.).

    Parameters
    ----------
    render : RenderConfig, optional
        Render camera resolution settings (visualization only).
    reward : RewardConfig, optional
        Reward configuration.
    robot : RobotConfig, optional
        Robot configuration.
    ground_colors : ColorConfig
        Ground plane color(s).
    reset_settle_frames : int
        No-op environment frames advanced after reset before observation.
    goal_thresh : float
        Distance threshold for goal achievement.
    spawn_half_size : float
        Half-size of spawn region.
    spawn_center : tuple[float, float]
        Center of spawn region (x, y).
    spawn_min_radius : float
        Minimum spawn radius.
    spawn_max_radius : float
        Maximum spawn radius.
    spawn_angle_half_range_deg : float
        Half angular range for spawn angle in degrees.
    obs_mode : ObsMode
        Observation mode (state or visual).
    robot_colors : ColorConfig
        Robot arm color(s).
    robot_init_qpos_noise : float
        Initial joint position noise.
    observations : list, optional
        Observation components to include in the state vector. When ``None``
        (the default), each config populates every observation applicable to
        its task, i.e. all privileged state components the backend can compute.
    """

    def __init__(
        self,
        render: RenderConfig | None = None,
        reward: RewardConfig | None = None,
        robot: RobotConfig | None = None,
        ground_colors: ColorConfig = "gray",
        reset_settle_frames: int = 5,
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
        self.reset_settle_frames = reset_settle_frames
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
            from so101_nexus.observations import CameraObservation

            has_camera_component = self.observations is not None and any(
                isinstance(c, CameraObservation) for c in self.observations
            )
            if not has_camera_component:
                raise ValueError(
                    "obs_mode='visual' requires at least one camera observation "
                    "component (e.g. WristCamera() or OverheadCamera()) in observations"
                )
        _validate_color_config(self.ground_colors, "ground_colors")
        _validate_color_config(self.robot_colors, "robot_colors")
        if not isinstance(self.reset_settle_frames, int):
            raise ValueError(
                "reset_settle_frames must be an integer, got "
                f"{type(self.reset_settle_frames).__name__}"
            )
        if self.reset_settle_frames < 0:
            raise ValueError(f"reset_settle_frames must be >= 0, got {self.reset_settle_frames}")
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
            from so101_nexus.observations import CameraObservation

            cam_types = [type(c) for c in self.observations if isinstance(c, CameraObservation)]
            if len(cam_types) != len(set(cam_types)):
                raise ValueError("Duplicate camera observation components are not allowed")

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}(goal_thresh={self.goal_thresh})"


def _normalize_objects(
    objects: list[SceneObject] | SceneObject | None,
    default: SceneObject,
) -> list[SceneObject]:
    """Return a non-empty scene object list from a flexible object input."""
    if objects is None:
        return [default]
    if isinstance(objects, SceneObject):
        return [objects]
    normalized = list(objects)
    if not normalized:
        raise ValueError("objects must not be empty")
    non_objects = [o for o in normalized if not isinstance(o, SceneObject)]
    if non_objects:
        names = [type(o).__name__ for o in non_objects]
        raise TypeError(f"objects must all be SceneObject instances, got {names}")
    return normalized


def describe_pick_target(target: object) -> str:
    """Return the canonical task description for a pick target.

    Backends call this instead of inlining the
    f-string, so the template lives in exactly one place.
    """
    return f"Pick up the {target!r}."


def describe_touch_target(target: object) -> str:
    """Return the canonical task description for a touch target."""
    return f"Touch the {target!r}."


def describe_place_target(obj: object, target_name: str) -> str:
    """Return the canonical task description for a pick-and-place target.

    Shared by both backends so the object-generic template lives in one place.
    """
    return f"Pick up the {obj!r} and place it on the {target_name} circle."


class PickConfig(EnvironmentConfig):
    """Config for the unified pick environment.

    The ``objects`` list defines the pool of scene objects to sample from each
    episode. One object is chosen as the target; ``n_distractors`` additional
    objects are sampled from the remaining pool and placed as distractors.
    Task descriptions are auto-generated from each object's ``__repr__``.

    Parameters
    ----------
    objects : list[SceneObject] or SceneObject, optional
        Pool of scene objects to sample from. Accepts a single ``SceneObject``,
        a list of ``SceneObject``, or ``None`` (defaults to ``[CubeObject()]``).
        A single object is automatically wrapped in a list.
    n_distractors : int
        Number of distractor objects to place. 0 means single-object scene.
    lift_threshold : float
        Minimum height above initial z to count as lifted.
    max_goal_height : float
        Height cap used to normalize lift progress to [0, 1].
    min_object_separation : float
        Minimum distance between spawned objects (metres).
    **kwargs
        Forwarded to EnvironmentConfig.
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
        _warn_inert_velocity_scale(self.reward, type(self).__name__)
        self.objects = _normalize_objects(objects, CubeObject())
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
            self.observations = [
                JointPositions(),
                EndEffectorPose(),
                GraspState(),
                ObjectPose(),
                ObjectOffset(),
            ]

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PickConfig(objects={self.objects!r}, n_distractors={self.n_distractors}, "
            f"lift_threshold={self.lift_threshold}, max_goal_height={self.max_goal_height})"
        )


class PickAndPlaceConfig(EnvironmentConfig):
    """Config for pick-and-place environments.

    The carried object is chosen per episode from an object pool. By default the
    pool is one cube per colour in ``cube_colors`` (so selecting the target slot
    reproduces the legacy per-episode cube-colour variation); pass ``objects`` to
    carry ``YCBObject`` / ``MeshObject`` instead. The goal disc colour is sampled
    from ``target_colors``.

    Parameters
    ----------
    objects : list[SceneObject] or SceneObject, optional
        Carried-object pool. ``None`` (default) derives a cube pool from
        ``cube_colors`` / ``cube_half_size`` / ``cube_mass``. Providing this with
        any non-default cube-sugar argument raises ``ValueError`` to avoid
        ambiguous configs.
    target_colors : ColorConfig
        Goal disc color(s).
    target_disc_radius : float
        Radius of the goal disc.
    min_object_target_separation : float, optional
        Minimum spawn separation between the object and the goal disc. ``None``
        falls back to ``min_cube_target_separation``.
    cube_colors : ColorConfig
        Cube color(s) for the default cube pool (compatibility sugar).
    cube_half_size : float
        Half-size of the default cube(s) in metres (compatibility sugar).
    cube_mass : float
        Mass of the default cube(s) in kg (compatibility sugar).
    min_cube_target_separation : float
        Deprecated alias for ``min_object_target_separation``.
    **kwargs
        Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        cube_colors: ColorConfig = "red",
        target_colors: ColorConfig = "blue",
        cube_half_size: float = 0.0125,
        cube_mass: float = 0.01,
        target_disc_radius: float = 0.05,
        min_cube_target_separation: float = 0.0375,
        *,
        objects: list[SceneObject] | SceneObject | None = None,
        min_object_target_separation: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        non_default_cube_sugar = [
            name
            for name, value, default in (
                ("cube_colors", cube_colors, "red"),
                ("cube_half_size", cube_half_size, 0.0125),
                ("cube_mass", cube_mass, 0.01),
            )
            if value != default
        ]
        if objects is not None and non_default_cube_sugar:
            raise ValueError(
                "PickAndPlaceConfig got both an explicit object pool and non-default "
                f"cube sugar {non_default_cube_sugar}; set colour/size/mass on the "
                "CubeObject pool entries instead"
            )

        self.objects = objects if objects is None else _normalize_objects(objects, CubeObject())
        self.cube_colors = cube_colors
        self.target_colors = target_colors
        self.cube_half_size = cube_half_size
        self.cube_mass = cube_mass
        self.target_disc_radius = target_disc_radius
        self.min_object_target_separation = (
            min_object_target_separation
            if min_object_target_separation is not None
            else min_cube_target_separation
        )

        _validate_color_config(self.cube_colors, "cube_colors")
        _validate_color_config(self.target_colors, "target_colors")
        cube_set = (
            {self.cube_colors} if isinstance(self.cube_colors, str) else set(self.cube_colors)
        )
        target_set = (
            {self.target_colors} if isinstance(self.target_colors, str) else set(self.target_colors)
        )
        overlap = cube_set & target_set
        if overlap and objects is None:
            warnings.warn(
                f"cube_colors and target_colors overlap on {overlap}; "
                "the cube and target may be the same color in some episodes",
                stacklevel=2,
            )
        if not (0.01 <= self.cube_half_size <= 0.05):
            raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {self.cube_half_size}")
        if self.target_disc_radius <= 0:
            raise ValueError(f"target_disc_radius must be > 0, got {self.target_disc_radius}")
        if min_cube_target_separation < 0:
            raise ValueError(
                f"min_cube_target_separation must be >= 0, got {min_cube_target_separation}"
            )
        if min_object_target_separation is not None and min_object_target_separation < 0:
            raise ValueError(
                f"min_object_target_separation must be >= 0, got {min_object_target_separation}"
            )
        if self.observations is None:
            self.observations = [
                JointPositions(),
                EndEffectorPose(),
                GraspState(),
                TargetPosition(),
                ObjectPose(),
                ObjectOffset(),
                TargetOffset(),
            ]

    def object_pool(self) -> list[SceneObject]:
        """Return the resolved carried-object pool (cube pool when ``objects`` is None)."""
        if self.objects is None:
            colors = (
                [self.cube_colors] if isinstance(self.cube_colors, str) else list(self.cube_colors)
            )
            return [
                CubeObject(half_size=self.cube_half_size, mass=self.cube_mass, color=c)
                for c in colors
            ]
        return _normalize_objects(self.objects, CubeObject())

    @property
    def min_cube_target_separation(self) -> float:
        """Deprecated alias for ``min_object_target_separation``."""
        return self.min_object_target_separation

    @min_cube_target_separation.setter
    def min_cube_target_separation(self, value: float) -> None:
        self.min_object_target_separation = value

    @staticmethod
    def describe(cube_name: str, target_name: str) -> str:
        """Build a cube task description (deprecated; use ``describe_place_target``).

        Retained for backward compatibility; produces the object-generic template
        for a cube of the given colour.
        """
        return f"Pick up the {cube_name} cube and place it on the {target_name} circle."

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"PickAndPlaceConfig(objects={self.objects!r}, "
            f"target_colors={self.target_colors!r}, cube_half_size={self.cube_half_size})"
        )

    @property
    def task_description(self) -> str:
        """Canonical task description for the first pool object and target colour."""
        target_name = (
            self.target_colors if isinstance(self.target_colors, str) else self.target_colors[0]
        )
        return describe_place_target(self.object_pool()[0], target_name)


class TouchConfig(PickConfig):
    """Config for the touch-an-object primitive task.

    Reuses PickConfig's object pool (``objects``, ``n_distractors``,
    ``min_object_separation``), so the touch target can be any cube, YCB object
    (for example a spoon), or mesh, optionally among distractors, and the teleop
    UI exposes the same object customization. The inherited ``lift_threshold`` and
    ``max_goal_height`` are unused by the touch task.

    Parameters
    ----------
    touch_margin : float
        Clearance (m) added to the target object's bounding radius; the task
        succeeds when the TCP is within ``bounding_radius + touch_margin`` of the
        object centre, so success coincides with the gripper reaching objects of
        any size.
    **kwargs
        Forwarded to PickConfig.
    """

    def __init__(self, touch_margin: float = 0.03, **kwargs) -> None:
        super().__init__(**kwargs)
        _warn_inert_reward_weights(self.reward, "TouchConfig", uses_tanh_scale=True)
        self.touch_margin = touch_margin
        if self.touch_margin < 0:
            raise ValueError(f"touch_margin must be >= 0, got {self.touch_margin}")

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"TouchConfig(objects={self.objects!r}, n_distractors={self.n_distractors}, "
            f"touch_margin={self.touch_margin})"
        )


class LookAtConfig(EnvironmentConfig):
    """Config for the look-at primitive task.

    Parameters
    ----------
    objects : list[SceneObject] or SceneObject, optional
        Object(s) to sample as the look-at target. Accepts a single
        SceneObject, a list, or None (defaults to [CubeObject()]).
        Only CubeObject targets are currently supported.
    fov_deg : float or None
        Vertical field of view (degrees) of the wrist camera used for the gaze
        axis. The task succeeds when the target lies within the camera's field
        of view, i.e. the angle between the wrist-camera optical axis and the
        direction to the object is at most ``fov_deg / 2``. When ``None``
        (default), the FOV is read from the actual wrist camera at runtime
        (``model.cam_fovy``), so the success band auto-adapts to any camera
        configuration, a wider FOV, or per-episode FOV randomization. Pin it
        only to override the live camera value.
    **kwargs
        Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        objects: list[SceneObject] | SceneObject | None = None,
        fov_deg: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        _warn_inert_reward_weights(self.reward, "LookAtConfig", uses_tanh_scale=False)
        _warn_inert_velocity_scale(self.reward, "LookAtConfig")
        self.objects = _normalize_objects(objects, CubeObject())
        self.fov_deg = fov_deg
        if self.fov_deg is not None and self.fov_deg <= 0:
            raise ValueError(f"fov_deg must be > 0 or None, got {self.fov_deg}")
        if self.observations is None:
            self.observations = [JointPositions(), EndEffectorPose(), GazeDirection()]
        for obj in self.objects:
            if not isinstance(obj, CubeObject):
                raise TypeError(
                    f"LookAtConfig only supports CubeObject targets, got {type(obj).__name__}"
                )

    @property
    def task_description(self) -> str:
        """Canonical task description derived from the configured target object."""
        return f"Look at the {self.objects[0]!r}."


class MoveConfig(EnvironmentConfig):
    """Config for the directional move primitive task.

    Parameters
    ----------
    direction : MoveDirection
        Cardinal direction to move the TCP.
    target_distance : float
        Distance in metres to travel from the initial TCP position.
    success_threshold : float
        Projection shortfall tolerance (m): success when displacement along the
        move direction is at least ``target_distance - success_threshold``.
    **kwargs
        Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        direction: MoveDirection = "up",
        target_distance: float = 0.10,
        success_threshold: float = 0.01,
        **kwargs,
    ) -> None:
        if direction not in DIRECTION_VECTORS:
            raise ValueError(
                f"direction must be one of {list(DIRECTION_VECTORS)}, got {direction!r}"
            )
        super().__init__(**kwargs)
        _warn_inert_reward_weights(self.reward, "MoveConfig", uses_tanh_scale=True)
        _warn_inert_velocity_scale(self.reward, "MoveConfig")
        self.direction = direction
        self.target_distance = target_distance
        self.success_threshold = success_threshold
        if self.observations is None:
            self.observations = [JointPositions(), EndEffectorPose(), TargetOffset()]

    @property
    def task_description(self) -> str:
        """Canonical task description derived from direction and distance."""
        return f"Move the end-effector {self.direction} by {self.target_distance:.2f} m."


# sqrt(2)/2 - used for 90-degree rotation quaternions in camera presets.
SQRT_HALF = float(np.sqrt(0.5))

ROBOT_CAMERA_PRESETS: dict[str, RobotCameraPreset] = {
    # SO-100: base rotated 90° around Z (faces +X). Wrist cam on Fixed_Jaw link.
    "so100": RobotCameraPreset(
        base_quat=(SQRT_HALF, 0.0, 0.0, SQRT_HALF),
        sensor_cam_eye_pos=(0.0, 0.3, 0.3),
        sensor_cam_target_pos=(0.15, 0.0, 0.02),
        human_cam_eye_pos=(0.0, 0.4, 0.4),
        human_cam_target_pos=(0.15, 0.0, 0.05),
        wrist_camera_mount_link="Fixed_Jaw",
        wrist_cam_pos_center=(0.0, -0.045, -0.045),
        wrist_cam_pos_noise=(0.0, 0.015, 0.015),
        wrist_cam_euler_center_deg=(-180.0, -37.5, -90.0),
        wrist_cam_euler_noise_deg=(0.0, 7.5, 0.0),
    ),
    # SO-101: base identity quaternion (faces +X natively). Wrist cam on
    # camera_mount (the menagerie body where the wrist camera lives).
    "so101": RobotCameraPreset(
        base_quat=(1.0, 0.0, 0.0, 0.0),
        sensor_cam_eye_pos=(0.0, 0.3, 0.3),
        sensor_cam_target_pos=(0.15, 0.0, 0.02),
        human_cam_eye_pos=(0.0, 0.4, 0.4),
        human_cam_target_pos=(0.15, 0.0, 0.05),
        wrist_camera_mount_link="camera_mount",
        wrist_cam_pos_center=(0.0, 0.04, -0.04),
        wrist_cam_pos_noise=(0.005, 0.01, 0.01),
        wrist_cam_euler_center_deg=(-180.0, 37.5, -90.0),
        wrist_cam_euler_noise_deg=(0.0, 11.4591559026, 0.0),
    ),
}
