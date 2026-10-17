"""GPU-batched stack-cube environment for SO-101 on MuJoCo Warp.

Warp's ``geom_rgba`` is a model-global array (not per-world), so per-episode
colour variety is realized the same way ``WarpPickLiftVectorEnv`` varies object
identity: one freejoint cube slot is compiled per configured colour in
``cube_a_colors`` / ``cube_b_colors``, each world selects one slot per role at
reset, and inactive slots are parked in an off-world band. This matches the
MuJoCo backend's per-episode resampling distribution (uniform over each
configured list, sampled independently per role) -- the divergence documented
here previously (compile-time-fixed colours) is removed.

``StackCubeConfig.max_distractors > 0`` additionally compiles one slot per entry in
``StackCubeConfig.distractors``; each world activates a sampled number of them
per reset and the remainder stay parked in the hidden band.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch

from so101_nexus.config import ControlMode, StackCubeConfig, describe_stack_target
from so101_nexus.constants import COLOR_MAP, ColorName
from so101_nexus.object_slots import (
    compile_object_scene,
)
from so101_nexus.objects import CubeObject, SceneObject
from so101_nexus.observations import (
    GazeDirection,
    GazeState,
    ObjectOffset,
    ObjectPose,
    ObjectVelocity,
    TargetOffset,
    TargetPosition,
)
from so101_nexus.rewards import (
    cube_stack_offset_ok,
    object_static_ok,
    place_grasp_potential,
    place_reach_potential,
    place_task_potential,
)
from so101_nexus.scene import WARP_SCENE_OPTION_XML
from so101_nexus.warp.base_env import SO101NexusWarpVectorEnv
from so101_nexus.warp.object_slots import (
    _contact_budget,
    _ObjectSlotRuntime,
    sample_separated_polar,
)


def _as_list(colors: ColorName | list[ColorName]) -> list[ColorName]:
    return [colors] if isinstance(colors, str) else list(colors)


class WarpStackCubeVectorEnv(SO101NexusWarpVectorEnv, _ObjectSlotRuntime):
    """Batched stack-cube: pick up cube A and stack it on top of cube B.

    Success requires cube A to rest directly on cube B (within
    ``config.stack_alignment_margin``), the arm to be static, cube A itself to
    be static (``config.cube_static_lin_threshold`` /
    ``config.cube_static_ang_threshold``), and cube A released
    (``is_grasped < 0.5``) -- a strict superset of ManiSkill's
    ``StackCubeEnv.evaluate`` predicate, matching ``MuJoCoStackCube-v1``.

    Default obs (43,): joint_positions(6) + joint_velocities(6) +
    end_effector_pose(7) + grasp_state(1) + gaze_state(1) + object_pose(7) +
    object_velocity(6) + object_offset(3) + target_position(3) +
    target_offset(3), matching ``MuJoCoStackCube-v1``.

    ``cube_a_color_names`` / ``cube_b_color_names`` hold the selected colour per
    world and are resampled at every reset; ``task_descriptions`` tracks them.
    """

    config: StackCubeConfig
    default_config_cls: ClassVar[type[StackCubeConfig]] = StackCubeConfig

    def __init__(
        self,
        num_envs: int,
        config: StackCubeConfig | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        device: str = "cuda",
        max_episode_steps: int = 1024,
        seed: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if config is None:
            config = StackCubeConfig()

        self._a_colors = _as_list(config.cube_a_colors)
        self._b_colors = _as_list(config.cube_b_colors)
        self._a_pool = len(self._a_colors)
        self._b_pool = len(self._b_colors)
        distractor_pool = list(config.distractors) if config.max_distractors else []
        self._d_pool = len(distractor_pool)
        self._d_offset = self._a_pool + self._b_pool
        scene_objects: list[SceneObject] = [
            CubeObject(half_size=config.cube_half_size, mass=config.cube_mass, color=c)
            for c in self._a_colors + self._b_colors
        ]
        scene_objects += distractor_pool
        slot_names = (
            [f"cube_a_{i}" for i in range(self._a_pool)]
            + [f"cube_b_{i}" for i in range(self._b_pool)]
            + [f"distractor_slot_{i}" for i in range(self._d_pool)]
        )

        ground_name = (
            config.ground_colors
            if isinstance(config.ground_colors, str)
            else config.ground_colors[0]
        )
        mjm, slots = compile_object_scene(
            scene_objects,
            slot_names,
            COLOR_MAP[ground_name],
            option_xml=WARP_SCENE_OPTION_XML,
            model_name="stack_cube_scene",
            overhead_camera_xml=self._world_camera_xml(config, render_mode),
        )
        default_nconmax, default_njmax = _contact_budget(
            len(slots), sum(len(slot.geom_ids) for slot in slots)
        )

        super().__init__(
            num_envs=num_envs,
            config=config,
            mjm=mjm,
            control_mode=control_mode,
            device=device,
            max_episode_steps=max_episode_steps,
            seed=seed,
            nconmax=default_nconmax if nconmax is None else nconmax,
            njmax=default_njmax if njmax is None else njmax,
            render_mode=render_mode,
        )
        self._mjm = mjm
        self.cube_half_size = config.cube_half_size

        self._init_slot_tensors(slots, mjm.ngeom, config.spawn_max_radius, config.spawn_center)

        # Per-world slot selection (set at reset; defaults track the first
        # configured pair until then). Grasp detection always targets the
        # selected cube A slot; cube B is never picked up.
        self._a_qadr = self._slot_qadr[0].expand(num_envs).clone()
        self._a_dadr = self._slot_dadr[0].expand(num_envs).clone()
        self._b_qadr = self._slot_qadr[self._a_pool].expand(num_envs).clone()
        self._b_dadr = self._slot_dadr[self._a_pool].expand(num_envs).clone()
        self._obj_geom_mask = self._slot_geom_masks[0].expand(num_envs, -1).clone()
        self.cube_a_color_names = [self._a_colors[0]] * num_envs
        self.cube_b_color_names = [self._b_colors[0]] * num_envs
        self._world_rows = torch.arange(num_envs, device=self.device)

        self._prev_task_potential = torch.zeros(num_envs, device=self.device)
        self._prev_reach_progress = torch.zeros(num_envs, device=self.device)
        self._prev_grasp_progress = torch.zeros(num_envs, device=self.device)
        self.task_descriptions = [
            self._describe_pair(self._a_colors[0], self._b_colors[0])
        ] * num_envs

    @staticmethod
    def _describe_pair(a_color: ColorName, b_color: ColorName) -> str:
        return describe_stack_target(CubeObject(color=a_color), CubeObject(color=b_color))

    def _generic_task_description(self) -> str:
        return "Pick up cube A and stack it on top of cube B."

    def _supported_obs_components(self) -> set[type]:
        return {
            ObjectPose,
            ObjectVelocity,
            ObjectOffset,
            TargetPosition,
            TargetOffset,
            GazeDirection,
            GazeState,
        }

    def _gather(self, buf: torch.Tensor, base_cols: torch.Tensor, width: int) -> torch.Tensor:
        cols = base_cols[:, None] + torch.arange(width, device=self.device)
        return buf[self._world_rows[:, None], cols]

    def _cube_a_pos(self) -> torch.Tensor:
        return self._gather(self.qpos, self._a_qadr, 3)

    def _cube_a_pose7(self) -> torch.Tensor:
        return self._gather(self.qpos, self._a_qadr, 7)

    def _cube_b_pos(self) -> torch.Tensor:
        return self._gather(self.qpos, self._b_qadr, 3)

    def _cube_a_vel(self) -> torch.Tensor:
        """Return ``(N, 6)`` cube A free-joint velocity ``[lin(3), ang(3)]`` per world."""
        return self._gather(self.qvel, self._a_dadr, 6)

    def _gaze_target_pos(self) -> torch.Tensor:
        return self._cube_a_pos()

    def _is_cube_a_static(self) -> torch.Tensor:
        """Return ``(N,)`` bool: cube A's speeds below the static thresholds.

        ManiSkill's ``is_cubeA_static`` check on the per-world selected slot.
        """
        vel = self._cube_a_vel()
        return object_static_ok(
            torch.linalg.norm(vel[:, :3], dim=1),
            torch.linalg.norm(vel[:, 3:], dim=1),
            lin_threshold=self.config.cube_static_lin_threshold,
            ang_threshold=self.config.cube_static_ang_threshold,
        )

    def _stack_state(
        self, a_pos: torch.Tensor, b_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cube_a_to_goal_dist, is_stacked)`` batched over worlds."""
        offset = a_pos - b_pos
        goal = b_pos.clone()
        goal[:, 2] = b_pos[:, 2] + 2.0 * self.cube_half_size
        cube_a_to_goal_dist = torch.linalg.norm(a_pos - goal, dim=1)
        is_stacked = cube_stack_offset_ok(
            offset[:, 0],
            offset[:, 1],
            offset[:, 2],
            cube_half_size=self.cube_half_size,
            margin=self.config.stack_alignment_margin,
        )
        return cube_a_to_goal_dist, is_stacked

    def _task_potential(
        self,
        a_pos: torch.Tensor,
        b_pos: torch.Tensor,
        is_grasped: torch.Tensor,
        is_stacked: torch.Tensor,
    ) -> torch.Tensor:
        """``Phi_stack(s)``: staged transport-then-settle progress toward completion.

        Batched physics-query wrapper around ``rewards.place_task_potential``,
        the same formula (and same ``height_gap=0.0`` reasoning) as the MuJoCo
        backend's ``StackCubeEnv._task_potential``.
        """
        cube_a_to_goal_dist, _ = self._stack_state(a_pos, b_pos)
        arm_speed = torch.linalg.norm(self.qvel.index_select(1, self._arm_dof_adr), dim=1)
        return place_task_potential(
            cube_a_to_goal_dist,
            0.0,
            arm_speed,
            is_grasped,
            is_stacked,
            scale=self.config.reward.tanh_shaping_scale,
            velocity_scale=self.config.reward.velocity_shaping_scale,
        )

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        """Refresh the stack, reach, and grasp baselines from the post-settle pose."""
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        a_pos = self._cube_a_pos()
        b_pos = self._cube_b_pos()
        is_grasped = self._is_grasping()
        _, is_stacked = self._stack_state(a_pos, b_pos)
        potential = self._task_potential(a_pos, b_pos, is_grasped, is_stacked)
        self._prev_task_potential[idx] = potential[idx]
        scale = self.config.reward.tanh_shaping_scale
        tcp_to_obj = torch.linalg.norm(a_pos - self._tcp_pos(), dim=1)
        self._prev_reach_progress[idx] = place_reach_potential(tcp_to_obj, is_stacked, scale=scale)[
            idx
        ]
        self._prev_grasp_progress[idx] = place_grasp_potential(is_grasped, is_stacked)[idx]

    def _task_reset(self, mask: torch.Tensor) -> None:
        for idx, count in self._distractor_reset_groups(
            mask, self.config.min_distractors, self.config.max_distractors
        ):
            self._reset_slots(idx, count)

    def _reset_slots(self, idx: torch.Tensor, n_distractors: int) -> None:
        cfg = self.config
        # One slot per configured colour entry; sample one slot per role per
        # world, matching the MuJoCo backend's per-episode colour resampling.
        a_sel = (self._rng.rand("task", idx) * self._a_pool).to(torch.long)
        b_sel_local = (self._rng.rand("task", idx) * self._b_pool).to(torch.long)
        b_sel = self._a_pool + b_sel_local

        self._hide_all_slots(idx)

        angle = float(np.radians(cfg.spawn_angle_half_range_deg))
        # Columns: 0 = cube A, 1 = cube B, 2.. = distractors.
        sel = torch.stack([a_sel, b_sel], dim=1)
        if n_distractors:
            d_rank = self._rng.rand("task", idx, self._d_pool).argsort(dim=1)
            sel = torch.cat([sel, self._d_offset + d_rank[:, :n_distractors]], dim=1)
        radii = self._slot_bradius[sel]  # (n, n_active)
        positions = sample_separated_polar(
            self._rng,
            idx,
            radii,
            cfg.min_cube_separation,
            cfg.spawn_min_radius,
            cfg.spawn_max_radius,
            angle,
            cfg.spawn_center,
        )  # (n, n_active, 2)

        self._place_active_slots(idx, sel, positions)

        self._a_qadr[idx] = self._slot_qadr[a_sel]
        self._a_dadr[idx] = self._slot_dadr[a_sel]
        self._b_qadr[idx] = self._slot_qadr[b_sel]
        self._b_dadr[idx] = self._slot_dadr[b_sel]
        obj_mask = self._obj_geom_mask
        assert obj_mask is not None  # set to a tensor in __init__
        obj_mask[idx] = self._slot_geom_masks[a_sel]
        # One device read per tensor instead of three per world: indexing a CUDA
        # tensor element by element synchronizes on every element.
        for w, a_i, b_i in zip(idx.tolist(), a_sel.tolist(), b_sel_local.tolist(), strict=True):
            a_color = self._a_colors[a_i]
            b_color = self._b_colors[b_i]
            self.cube_a_color_names[w] = a_color
            self.cube_b_color_names[w] = b_color
            self.task_descriptions[w] = self._describe_pair(a_color, b_color)

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, ObjectPose):
            return self._cube_a_pose7()
        if isinstance(component, ObjectVelocity):
            return self._cube_a_vel()
        if isinstance(component, ObjectOffset):
            return self._cube_a_pos() - self._tcp_pos()
        if isinstance(component, TargetPosition):
            return self._cube_b_pos()
        if isinstance(component, TargetOffset):
            return self._cube_b_pos() - self._cube_a_pos()
        return super()._get_component_data(component)

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        a_pos = self._cube_a_pos()
        b_pos = self._cube_b_pos()
        tcp_pos = self._tcp_pos()
        tcp_to_obj = torch.linalg.norm(a_pos - tcp_pos, dim=1)
        cube_a_to_goal_dist, is_stacked = self._stack_state(a_pos, b_pos)
        is_grasped = self._is_grasping()
        is_robot_static = self._is_robot_static()
        is_cube_a_static = self._is_cube_a_static()
        # Releasing cube A is mandatory, and both the arm and cube A must be
        # static: a stacked-but-still-grasped hold or a cube still settling
        # through the tolerance band does not count as success (ManiSkill
        # StackCubeEnv.evaluate). Mirrors StackCubeEnv._get_info (MuJoCo).
        success = is_stacked & is_robot_static & is_cube_a_static & (is_grasped < 0.5)
        scale = self.config.reward.tanh_shaping_scale
        # reaching/grasping are potential-shaped deltas, not raw state values --
        # mirrors StackCubeEnv._compute_reward (MuJoCo backend). Baselines
        # seeded post-settle by _refresh_reset_reference_state.
        reach_now = place_reach_potential(tcp_to_obj, is_stacked, scale=scale)
        grasp_now = place_grasp_potential(is_grasped, is_stacked)
        task_potential = self._task_potential(a_pos, b_pos, is_grasped, is_stacked)
        reach_delta, grasp_delta, task_progress = self._update_potentials(
            reach_now, grasp_now, task_potential
        )
        reward = self.config.reward.compute(
            reach_progress=reach_delta,
            is_grasped=grasp_delta,
            task_progress=task_progress,
            is_complete=success,
            action_delta_norm=action_delta_norm,
            energy_norm=energy_norm,
        )
        info = {
            "cube_a_to_goal_dist": cube_a_to_goal_dist,
            "is_stacked": is_stacked,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "is_cube_a_static": is_cube_a_static,
            "success": success,
            "tcp_to_obj_dist": tcp_to_obj,
        }
        return reward.to(torch.float32), success, info
