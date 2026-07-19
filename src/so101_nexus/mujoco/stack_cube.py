"""MuJoCo stack-cube environment.

Two dedicated freejoint cube bodies are compiled into the scene: cube A (the
movable cube) and cube B (the stacking base). Colors for both are resampled
every reset via ``model.geom_rgba`` -- the same per-geom recolor trick
``PickAndPlaceEnv`` uses for its goal disc -- so both cubes vary color
independently and per episode without rebuilding the model.
"""

from __future__ import annotations

import tempfile
from typing import ClassVar, cast

import mujoco
import numpy as np

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, StackCubeConfig, describe_stack_target
from so101_nexus.constants import COLOR_MAP, ColorName, sample_color_name
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus.mujoco.spawn_utils import place_freejoint_slot, sample_separated_positions
from so101_nexus.object_slots import ObjectSlot, build_object_scene_xml, extract_object_slots
from so101_nexus.objects import CubeObject
from so101_nexus.rewards import (
    cube_stack_offset_ok,
    cube_static_ok,
    place_grasp_potential,
    place_reach_potential,
    place_task_potential,
    potential_shaping,
)
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()


class StackCubeEnv(SO101NexusMuJoCoBaseEnv):
    """Stack-cube environment: pick up cube A and stack it on top of cube B.

    Success requires cube A to rest directly on cube B (within
    ``config.stack_alignment_margin``), the arm to be static, cube A itself to
    be static (below ``config.cube_static_lin_threshold`` /
    ``config.cube_static_ang_threshold``, ManiSkill's ``is_cubeA_static``
    thresholds), and cube A to no longer be grasped -- a strict superset of
    ManiSkill's ``StackCubeEnv.evaluate`` predicate (which does not check arm
    staticness). A cube released in-band while still descending does not count
    as success until it settles.
    """

    config: StackCubeConfig
    default_config_cls: ClassVar[type[StackCubeConfig]] = StackCubeConfig

    def __init__(
        self,
        config: StackCubeConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = StackCubeConfig()
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        self.cube_half_size = config.cube_half_size
        # First configured colours seed the compiled model; both are re-sampled
        # per episode below (geom_rgba is per-geom in the scalar backend).
        a_color0 = (
            config.cube_a_colors
            if isinstance(config.cube_a_colors, str)
            else config.cube_a_colors[0]
        )
        b_color0 = (
            config.cube_b_colors
            if isinstance(config.cube_b_colors, str)
            else config.cube_b_colors[0]
        )
        self.cube_a_color_name = a_color0
        self.cube_b_color_name = b_color0
        cube_a_obj = CubeObject(
            half_size=config.cube_half_size, mass=config.cube_mass, color=a_color0
        )
        cube_b_obj = CubeObject(
            half_size=config.cube_half_size, mass=config.cube_mass, color=b_color0
        )

        ground_name = (
            config.ground_colors
            if isinstance(config.ground_colors, str)
            else config.ground_colors[0]
        )
        xml_string = build_object_scene_xml(
            [cube_a_obj, cube_b_obj],
            ["cube_a", "cube_b"],
            COLOR_MAP[ground_name],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
            model_name="stack_cube_scene",
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        slots: list[ObjectSlot] = extract_object_slots(
            self.model, ["cube_a", "cube_b"], [cube_a_obj, cube_b_obj]
        )
        self._slot_a, self._slot_b = slots
        # Grasp detection always targets cube A; cube B is never picked up.
        self._obj_geom_id = self._slot_a.geom_id

        self._prev_task_potential: float = 0.0
        self._prev_reach_progress: float = 0.0
        self._prev_grasp_progress: float = 0.0
        self.task_description = config.task_description

        self._finish_model_setup()

    def _get_cube_a_pose(self) -> np.ndarray:
        addr = self._slot_a.qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_cube_b_pose(self) -> np.ndarray:
        addr = self._slot_b.qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _is_cube_a_static(self) -> bool:
        """Return True if cube A's speeds are below the static thresholds.

        ManiSkill's ``is_cubeA_static`` check on the selected cube A slot.
        """
        vel = self.data.qvel[self._slot_a.dof_addr : self._slot_a.dof_addr + 6]
        return bool(
            cube_static_ok(
                float(np.linalg.norm(vel[:3])),
                float(np.linalg.norm(vel[3:])),
                lin_threshold=self.config.cube_static_lin_threshold,
                ang_threshold=self.config.cube_static_ang_threshold,
            )
        )

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus.observations import (
            ObjectOffset as _ObjectOffset,
        )
        from so101_nexus.observations import (
            ObjectPose as _ObjectPose,
        )
        from so101_nexus.observations import (
            TargetOffset as _TargetOffset,
        )
        from so101_nexus.observations import (
            TargetPosition as _TargetPosition,
        )

        if isinstance(component, _ObjectPose):
            return self._get_cube_a_pose()
        if isinstance(component, _ObjectOffset):
            return self._get_cube_a_pose()[:3] - self._get_tcp_pose()[:3]
        if isinstance(component, _TargetPosition):
            return self._get_cube_b_pose()[:3]
        if isinstance(component, _TargetOffset):
            return self._get_cube_b_pose()[:3] - self._get_cube_a_pose()[:3]
        return super()._get_component_data(component)

    def _stack_state(self, a_pos: np.ndarray, b_pos: np.ndarray) -> tuple[float, bool]:
        """Return ``(cube_a_to_goal_dist, is_stacked)`` for the given cube poses.

        ``goal`` is the point ``2 * cube_half_size`` directly above cube B (its
        current position, not a per-episode baseline, so bumping cube B moves
        the goal), matching ManiSkill's ``StackCubeEnv.compute_dense_reward``.
        """
        offset = a_pos - b_pos
        goal = b_pos.copy()
        goal[2] = b_pos[2] + 2.0 * self.cube_half_size
        cube_a_to_goal_dist = float(np.linalg.norm(a_pos - goal))
        is_stacked = cube_stack_offset_ok(
            float(offset[0]),
            float(offset[1]),
            float(offset[2]),
            cube_half_size=self.cube_half_size,
            margin=self.config.stack_alignment_margin,
        )
        return cube_a_to_goal_dist, is_stacked

    def _task_potential(
        self, a_pos: np.ndarray, b_pos: np.ndarray, is_grasped: float, is_stacked: bool
    ) -> float:
        """``Phi_stack(s)``: staged transport-then-settle progress toward completion.

        Reuses ``rewards.place_task_potential`` with ``height_gap=0.0``: cube
        A's goal is an elevated 3D point (``2 * cube_half_size`` above cube B,
        above the cube's own rest height), so the mandatory lift shrinks the
        3D transport distance and the potential rises through the lift on its
        own. The Chebyshev ``max`` that keeps PickAndPlace's ground-level goal
        from penalizing its lift would collapse to ``max(dist, 0) == dist``
        here, so the plain 3D norm is the correct transport measure for this
        task. See
        ``so101_nexus.mujoco.pick_and_place.PickAndPlaceEnv._task_potential``
        for the sibling formula this is adapted from.
        """
        cube_a_to_goal_dist, _ = self._stack_state(a_pos, b_pos)
        arm_speed = float(np.linalg.norm(self.data.qvel[self._arm_qvel_addrs]))
        return place_task_potential(
            cube_a_to_goal_dist,
            0.0,
            arm_speed,
            is_grasped,
            is_stacked,
            scale=self.config.reward.tanh_shaping_scale,
            velocity_scale=self.config.reward.velocity_shaping_scale,
        )

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        a_pos = self._get_cube_a_pose()[:3]
        b_pos = self._get_cube_b_pose()[:3]
        is_grasped = self._is_grasping()

        cube_a_to_goal_dist, is_stacked = self._stack_state(a_pos, b_pos)
        is_robot_static = self._is_robot_static()
        is_cube_a_static = self._is_cube_a_static()
        # Releasing cube A is mandatory, and both the arm and cube A must be
        # static: a stacked-but-still-grasped hold or a cube still settling
        # through the tolerance band does not count as success (ManiSkill
        # StackCubeEnv.evaluate).
        success = is_stacked and is_robot_static and is_cube_a_static and is_grasped < 0.5

        info = {
            "cube_a_to_goal_dist": cube_a_to_goal_dist,
            "is_stacked": is_stacked,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "is_cube_a_static": is_cube_a_static,
            "success": success,
            "tcp_to_obj_dist": float(np.linalg.norm(a_pos - tcp_pos)),
            "task_potential": self._task_potential(a_pos, b_pos, is_grasped, is_stacked),
        }
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        scale = self.config.reward.tanh_shaping_scale
        # reaching/grasping are potential-shaped deltas, not raw state values
        # (dwelling at "reached and grasped, never stacked" must pay ~0/step,
        # matching PickAndPlaceEnv._compute_reward), and both potentials are
        # held up by is_stacked so the mandatory release on cube B and the
        # post-stack retreat pay no negative delta.
        reach_now = place_reach_potential(info["tcp_to_obj_dist"], info["is_stacked"], scale=scale)
        grasp_now = place_grasp_potential(info["is_grasped"], info["is_stacked"])
        reach_delta = potential_shaping(reach_now, self._prev_reach_progress)
        grasp_delta = potential_shaping(grasp_now, self._prev_grasp_progress)
        self._prev_reach_progress = reach_now
        self._prev_grasp_progress = grasp_now
        # task_progress is a potential-based delta (Ng, Harada & Russell, ICML
        # 1999; see _task_potential), not the raw potential -- dwelling at any
        # fixed state pays ~0 per step instead of the potential's full value.
        task_potential = info["task_potential"]
        task_progress = potential_shaping(task_potential, self._prev_task_potential)
        self._prev_task_potential = task_potential
        components = self.config.reward.compute_components(
            reach_progress=reach_delta,
            is_grasped=grasp_delta,
            task_progress=task_progress,
            is_complete=info["success"],
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
        info["reward_components"] = components
        return sum(components.values())

    def _refresh_reset_reference_state(self) -> None:
        """Refresh the stack, reach, and grasp baselines from the post-settle pose."""
        a_pos = self._get_cube_a_pose()[:3]
        b_pos = self._get_cube_b_pose()[:3]
        is_grasped = self._is_grasping()
        _, is_stacked = self._stack_state(a_pos, b_pos)
        self._prev_task_potential = self._task_potential(a_pos, b_pos, is_grasped, is_stacked)
        scale = self.config.reward.tanh_shaping_scale
        tcp_to_obj_dist = float(np.linalg.norm(a_pos - self._get_tcp_pose()[:3]))
        self._prev_reach_progress = place_reach_potential(tcp_to_obj_dist, is_stacked, scale=scale)
        self._prev_grasp_progress = place_grasp_potential(is_grasped, is_stacked)

    def _task_reset(self) -> None:
        rng = self.np_random
        min_r = self.config.spawn_min_radius
        max_r = self.config.spawn_max_radius
        angle_half = float(np.radians(self.config.spawn_angle_half_range_deg))

        self.cube_a_color_name = sample_color_name(self.config.cube_a_colors, rng)
        self.cube_b_color_name = sample_color_name(self.config.cube_b_colors, rng)
        self.model.geom_rgba[self._slot_a.geom_id] = COLOR_MAP[self.cube_a_color_name]
        self.model.geom_rgba[self._slot_b.geom_id] = COLOR_MAP[self.cube_b_color_name]

        radii = [self._slot_a.bounding_radius, self._slot_b.bounding_radius]
        positions = sample_separated_positions(
            rng,
            2,
            min_r,
            max_r,
            angle_half,
            self.config.min_cube_separation,
            radii,
            center=self.config.spawn_center,
        )
        place_freejoint_slot(self.model, self.data, self._slot_a, rng, positions[0])
        place_freejoint_slot(self.model, self.data, self._slot_b, rng, positions[1])

        self.task_description = describe_stack_target(
            CubeObject(color=cast("ColorName", self.cube_a_color_name)),
            CubeObject(color=cast("ColorName", self.cube_b_color_name)),
        )
