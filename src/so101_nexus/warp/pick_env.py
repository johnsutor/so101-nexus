"""GPU-batched pick-lift environment for SO-101 on MuJoCo Warp.

Heterogeneous object pools are supported through compiled object slots (one
freejoint body per pool object, shared with the MuJoCo backend via
``so101_nexus.object_slots``). Per world, the chosen target slot is placed in the
spawn arc, distractor slots are placed too, and every other slot is parked at a
far-off resting position. Object identity varies per world through slot selection
and per-world task descriptions; per-world ``geom_rgba`` colour randomization of a
*single* object is still unsupported (model colour is global), but distinct
coloured cube slots give per-world colour variation through selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from so101_nexus.config import ControlMode, PickConfig, describe_pick_target
from so101_nexus.constants import sample_color
from so101_nexus.object_slots import (
    compile_object_scene,
)
from so101_nexus.observations import (
    GazeDirection,
    GazeState,
    ObjectOffset,
    ObjectPose,
    ObjectVelocity,
)
from so101_nexus.rewards import lift_progress, reach_progress
from so101_nexus.scene import WARP_SCENE_OPTION_XML
from so101_nexus.warp.base_env import SO101NexusWarpVectorEnv
from so101_nexus.warp.object_slots import (
    _contact_budget,
    _ObjectSlotRuntime,
    sample_separated_polar,
)

if TYPE_CHECKING:
    from so101_nexus.objects import SceneObject


class WarpPickLiftVectorEnv(SO101NexusWarpVectorEnv, _ObjectSlotRuntime):
    """Batched pick-lift: grasp the per-world target object and lift it.

    Default obs (31,): joint_positions(6) + joint_velocities(6) +
    end_effector_pose(7) + grasp_state(1) + gaze_state(1) + object_pose(7) +
    object_offset(3), matching ``MuJoCoPickLift-v1``.
    """

    config: PickConfig
    default_config_cls: ClassVar[type[PickConfig]] = PickConfig

    def __init__(
        self,
        num_envs: int,
        config: PickConfig | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        device: str = "cuda",
        max_episode_steps: int = 1024,
        seed: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if config is None:
            config = PickConfig()
        self._build_slot_model(
            scene_objects=list(config.objects),
            n_active=1 + config.n_distractors,
            config=config,
            num_envs=num_envs,
            control_mode=control_mode,
            device=device,
            max_episode_steps=max_episode_steps,
            seed=seed,
            nconmax=nconmax,
            njmax=njmax,
            model_name="pick_scene",
            render_mode=render_mode,
        )

    def _build_slot_model(
        self,
        *,
        scene_objects: list[SceneObject],
        n_active: int,
        config,
        num_envs: int,
        control_mode: ControlMode,
        device: str,
        max_episode_steps: int,
        seed: int | None,
        nconmax: int | None,
        njmax: int | None,
        model_name: str,
        extra_bodies: str = "",
        render_mode: str | None = None,
        n_target_pool: int | None = None,
        slot_names: list[str] | None = None,
    ) -> None:
        """Compile the shared object-slot model and build per-slot tensors.

        Shared by the pick/touch and pick-and-place backends; the latter passes
        ``extra_bodies`` for the mocap goal disc, ``n_active=1``, and a
        ``n_target_pool`` that excludes its trailing distractor slots.

        Parameters
        ----------
        n_target_pool : int, optional
            Number of leading slots a target may be drawn from, which is also
            what ``reset(options={"target_index": k})`` is validated against.
            ``None`` (default) allows every compiled slot.
        slot_names : list[str], optional
            Body-name stems for the compiled slots. ``None`` (default) names
            them ``pick_slot_{i}``.
        """
        self._n_total_slots = len(scene_objects)
        # ``_n_pool`` is the carried/distractor boundary: ``_select_active_slots``
        # draws from ``[0, _n_pool)`` and ``_parse_target_index`` range-checks a
        # pin against it, so a value past the compiled slots would make a
        # distractor pinnable and index past the per-slot tensors.
        if n_target_pool is not None and not 0 < n_target_pool <= self._n_total_slots:
            raise ValueError(
                f"n_target_pool must be in [1, {self._n_total_slots}], got {n_target_pool}"
            )
        self._n_pool = self._n_total_slots if n_target_pool is None else n_target_pool
        self._n_active = n_active
        if slot_names is None:
            slot_names = [f"pick_slot_{i}" for i in range(self._n_total_slots)]
        elif len(slot_names) != self._n_total_slots:
            raise ValueError(
                f"slot_names must have one entry per compiled slot "
                f"({self._n_total_slots}), got {len(slot_names)}"
            )

        mjm, slots = compile_object_scene(
            scene_objects,
            slot_names,
            sample_color(config.ground_colors),
            option_xml=WARP_SCENE_OPTION_XML,
            model_name=model_name,
            extra_bodies=extra_bodies,
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

        self._slot_objs = scene_objects
        self._init_slot_tensors(slots, mjm.ngeom, config.spawn_max_radius, config.spawn_center)

        # Per-world target tracking (set at reset). ``_obj_geom_mask`` drives grasp
        # detection in the base; ``_target_qadr`` indexes the selected slot pose.
        self._obj_geom_mask = torch.zeros(
            (num_envs, mjm.ngeom), dtype=torch.bool, device=self.device
        )
        self._target_qadr = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._target_dadr = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._target_slot = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._target_slot_host = [0] * num_envs
        self._initial_obj_z = torch.zeros(num_envs, device=self.device)
        self._prev_task_potential = torch.zeros(num_envs, device=self.device)
        self._prev_reach_progress = torch.zeros(num_envs, device=self.device)
        self._prev_grasp_progress = torch.zeros(num_envs, device=self.device)
        self._world_rows = torch.arange(num_envs, device=self.device)
        self.task_descriptions = [self._describe_target(scene_objects[0])] * num_envs

    def _describe_target(self, obj: SceneObject) -> str:
        """Per-world task description for the chosen target (overridable per task)."""
        return describe_pick_target(obj)

    def _generic_task_description(self) -> str:
        return "Pick up the selected object."

    def _supported_obs_components(self) -> set[type]:
        return {ObjectPose, ObjectVelocity, ObjectOffset, GazeDirection, GazeState}

    def _gather(self, base_cols: torch.Tensor, width: int) -> torch.Tensor:
        cols = base_cols[:, None] + torch.arange(width, device=self.device)
        return self.qpos[self._world_rows[:, None], cols]

    def _target_pose7(self) -> torch.Tensor:
        return self._gather(self._target_qadr, 7)

    def _target_pos(self) -> torch.Tensor:
        return self._gather(self._target_qadr, 3)

    def _target_vel(self) -> torch.Tensor:
        """Return ``(N, 6)`` target free-joint velocity ``[lin(3), ang(3)]`` per world."""
        cols = self._target_dadr[:, None] + torch.arange(6, device=self.device)
        return self.qvel[self._world_rows[:, None], cols]

    def _gaze_target_pos(self) -> torch.Tensor:
        return self._target_pos()

    def _target_bounding_radius(self) -> torch.Tensor:
        return self._slot_bradius[self._target_slot]

    def _select_active_slots(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(n, n_active)`` distinct target-pool indices and the ``(n,)`` target.

        The active set is one seeded draw, unchanged by
        ``reset(options={"target_index": k})``. Worlds whose pin is already in
        the draw only relabel their target, so the placements stay byte-identical
        and the pair of episodes differs solely in which object the task names -
        the counterfactual a language-conditioned policy needs to prove it reads
        the instruction. A pin outside the draw displaces column 0 instead, which
        does move objects.
        """
        if self._n_active == 1:
            sel = (self._rng.rand("task", idx, 1) * self._n_pool).to(torch.long)
        else:
            perm = self._rng.rand("task", idx, self._n_pool)
            sel = perm.argsort(dim=1)[:, : self._n_active]
        override = self._target_index_override
        if override is None:
            return sel, sel[:, 0]
        forced = override[idx]
        present = (sel == forced[:, None]).any(dim=1)
        sel = sel.clone()
        sel[:, 0] = torch.where(present, sel[:, 0], forced)
        return sel, forced

    def _set_target_tracking(self, idx: torch.Tensor, target: torch.Tensor) -> None:
        """Record the per-world target slot and refresh its task description."""
        self._target_slot[idx] = target
        obj_mask = self._obj_geom_mask
        assert obj_mask is not None  # set to a tensor in _build_slot_model
        obj_mask[idx] = self._slot_geom_masks[target]
        self._target_qadr[idx] = self._slot_qadr[target]
        self._target_dadr[idx] = self._slot_dadr[target]
        self._initial_obj_z[idx] = self._slot_spawn_z[target]
        # One device read per tensor instead of one per world: indexing a CUDA
        # tensor element by element synchronizes on every element.
        for world, slot in zip(idx.tolist(), target.tolist(), strict=True):
            self._target_slot_host[world] = slot
            self.task_descriptions[world] = self._describe_target(self._slot_objs[slot])

    def _task_reset(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        sel, target = self._select_active_slots(idx)  # (n, n_active), (n,)
        self._hide_all_slots(idx)
        radii = self._slot_bradius[sel]  # (n, n_active)
        positions = sample_separated_polar(
            self._rng,
            idx,
            radii,
            self.config.min_object_separation,
            self.config.spawn_min_radius,
            self.config.spawn_max_radius,
            float(np.radians(self.config.spawn_angle_half_range_deg)),
            self.config.spawn_center,
        )
        self._place_active_slots(idx, sel, positions)
        self._set_target_tracking(idx, target)

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        self._initial_obj_z[idx] = self._target_pos()[idx, 2]
        # lift_progress(0, ...) == 0 regardless of grasped (tanh(0) == 0), so the
        # potential baseline is always 0 here: the object sits at its own baseline.
        self._prev_task_potential[idx] = 0.0
        scale = self.config.reward.tanh_shaping_scale
        tcp_to_obj = torch.linalg.norm(self._target_pos() - self._tcp_pos(), dim=1)
        self._prev_reach_progress[idx] = reach_progress(tcp_to_obj, scale=scale)[idx]
        self._prev_grasp_progress[idx] = self._is_grasping()[idx]

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, ObjectPose):
            return self._target_pose7()
        if isinstance(component, ObjectVelocity):
            return self._target_vel()
        if isinstance(component, ObjectOffset):
            return self._target_pos() - self._tcp_pos()
        return super()._get_component_data(component)

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        obj_pos = self._target_pos()
        tcp_pos = self._tcp_pos()
        tcp_to_obj = torch.linalg.norm(obj_pos - tcp_pos, dim=1)
        is_grasped = self._is_grasping()
        lift_height = obj_pos[:, 2] - self._initial_obj_z
        grasped = is_grasped > 0.5
        success = (lift_height > self.config.lift_threshold) & grasped
        scale = self.config.reward.tanh_shaping_scale
        # reaching/grasping are potential-shaped deltas, not raw state values --
        # like task_progress below, both are a strict subset of `success`'s
        # completion surface (must reach and grasp before lifting), so a raw
        # (dwelling) value lets a policy park at "reached and grasped, never
        # lifted" and collect up to their combined budget every step forever.
        reach_now = reach_progress(tcp_to_obj, scale=scale)
        lift_potential = lift_progress(lift_height, scale=scale, grasped=grasped)
        reach_delta, grasp_delta, task_progress = self._update_potentials(
            reach_now, is_grasped, lift_potential
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
            "is_grasped": is_grasped,
            "is_robot_static": self._is_robot_static(),
            "lift_height": lift_height,
            "tcp_to_obj_dist": tcp_to_obj,
            "success": success,
            # No target_object counterpart to the MuJoCo backend's: info values
            # here are per-world batched tensors, so the object identity travels
            # as a string tuple in info["task_description"] instead.
            "target_index": self._target_slot.clone(),
        }
        return reward.to(torch.float32), success, info
