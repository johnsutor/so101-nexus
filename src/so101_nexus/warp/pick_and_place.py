"""GPU-batched pick-and-place environment for SO-101 on MuJoCo Warp.

The carried object is chosen per episode from the shared compiled object-slot pool
(see ``so101_nexus.warp.pick_env``); the goal is a non-colliding mocap disc whose
per-world position lives in ``data.mocap_pos``. The disc colour is fixed at
model-build time to the first configured target colour (``geom_rgba`` is global), a
documented divergence from the MuJoCo backend, which randomizes it per episode.

``PickAndPlaceConfig.n_distractors > 0`` additionally compiles one slot per entry
in ``PickAndPlaceConfig.distractors`` after the carried pool; each world activates
``n_distractors`` of them per reset and the remainder stay parked in the hidden
band. Target selection stays inside the carried pool.
"""

from __future__ import annotations

from typing import ClassVar

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from so101_nexus.config import (
    ControlMode,
    PickAndPlaceConfig,
    PickAndPlaceV2Config,
    describe_place_target,
)
from so101_nexus.constants import COLOR_MAP
from so101_nexus.object_slots import extract_object_slots
from so101_nexus.objects import CubeObject
from so101_nexus.observations import TargetOffset, TargetPosition
from so101_nexus.placement import (
    SlotGeometry,
    compiled_slot_geometry,
    resolved_placement_contract,
    support_state,
    visual_reference,
)
from so101_nexus.rewards import (
    object_static_ok,
    place_grasp_potential,
    place_reach_potential,
    place_task_potential,
    potential_shaping,
)
from so101_nexus.warp.object_slots import sample_polar, sample_separated_polar
from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

_TARGET_Z = 0.001
# Object placed (not lifted) when within this vertical slack of its rest height.
_PLACE_Z_SLACK = 0.01


def _target_disc_body(target_disc_radius: float, rgba: list[float]) -> str:
    r, g, b, a = rgba
    return (
        f'    <body name="target" pos="0.2 0 {_TARGET_Z}" mocap="true">\n'
        f'      <geom name="target_disc" type="cylinder" size="{target_disc_radius} 0.001"\n'
        f'            rgba="{r} {g} {b} {a}" contype="0" conaffinity="0"/>\n'
        f"    </body>\n"
    )


class WarpPickAndPlaceVectorEnv(WarpPickLiftVectorEnv):
    """Batched pick-and-place: carry the per-world object onto the goal disc.

    Default obs (43,): joint_positions(6) + joint_velocities(6) +
    end_effector_pose(7) + grasp_state(1) + gaze_state(1) + target_position(3) +
    object_pose(7) + object_velocity(6) + object_offset(3) + target_offset(3),
    matching ``MuJoCoPickAndPlace-v1``.
    """

    config: PickAndPlaceConfig
    default_config_cls: ClassVar[type[PickAndPlaceConfig]] = PickAndPlaceConfig

    def __init__(
        self,
        num_envs: int,
        config: PickAndPlaceConfig | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        device: str = "cuda",
        max_episode_steps: int = 1024,
        seed: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if config is None:
            config = self.default_config_cls()
        is_v2 = issubclass(self.default_config_cls, PickAndPlaceV2Config)
        if not isinstance(config, PickAndPlaceConfig) or (
            isinstance(config, PickAndPlaceV2Config) != is_v2
        ):
            raise TypeError(f"{type(self).__name__} requires {self.default_config_cls.__name__}.")
        scene_objects = config.object_pool()
        n_carried = len(scene_objects)
        # Distractor slots are only compiled when requested, so the default
        # single-object scene (and its contact budget) stays unchanged.
        self._n_distractors = config.n_distractors
        distractor_pool = list(config.distractors) if self._n_distractors else []
        self._d_pool = len(distractor_pool)
        self._d_offset = n_carried
        scene_objects = scene_objects + distractor_pool
        self.target_color_name = (
            config.target_colors
            if isinstance(config.target_colors, str)
            else config.target_colors[0]
        )
        disc_xml = _target_disc_body(config.target_disc_radius, COLOR_MAP[self.target_color_name])
        self._build_slot_model(
            scene_objects=scene_objects,
            n_active=1,
            config=config,
            num_envs=num_envs,
            control_mode=control_mode,
            device=device,
            max_episode_steps=max_episode_steps,
            seed=seed,
            nconmax=nconmax,
            njmax=njmax,
            model_name="pick_and_place_scene",
            extra_bodies=disc_xml,
            render_mode=render_mode,
            n_target_pool=n_carried,
            slot_names=[f"pick_slot_{i}" for i in range(n_carried)]
            + [f"distractor_slot_{i}" for i in range(self._d_pool)],
        )
        target_bid = mujoco.mj_name2id(self._mjm, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap_id = int(self._mjm.body_mocapid[target_bid])
        self._mocap_pos = wp.to_torch(self.data.mocap_pos)  # (N, nmocap, 3)
        first = scene_objects[0]
        self.cube_color_name = first.color if isinstance(first, CubeObject) else ""
        self._prev_task_potential = torch.zeros(num_envs, device=self.device)

    def _describe_target(self, obj) -> str:
        return describe_place_target(obj, self.target_color_name)

    def _generic_task_description(self) -> str:
        return f"Pick up the object and place it on the {self.target_color_name} circle."

    def _supported_obs_components(self) -> set[type]:
        return {*super()._supported_obs_components(), TargetPosition, TargetOffset}

    def _target_disc_pos(self) -> torch.Tensor:
        return self._mocap_pos[:, self._target_mocap_id, :]

    def _placement_reference_pos(self) -> torch.Tensor:
        """Return the reference point for the placement reward."""
        return self._target_pos()

    def _placement_success(
        self,
        is_obj_placed: torch.Tensor,
        is_obj_static: torch.Tensor,
        is_grasped: torch.Tensor,
    ) -> torch.Tensor:
        return is_obj_placed & is_obj_static & (is_grasped < 0.5)

    def _obj_placement_state(
        self, obj_pos: torch.Tensor, target_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(obj_to_target_xy_dist, is_obj_placed)`` batched over worlds."""
        obj_to_target = torch.linalg.norm(obj_pos[:, :2] - target_pos[:, :2], dim=1)
        is_obj_placed = (obj_to_target <= self.config.goal_thresh) & (
            obj_pos[:, 2] < self._initial_obj_z + _PLACE_Z_SLACK
        )
        return obj_to_target, is_obj_placed

    def _is_obj_static(self) -> torch.Tensor:
        """Return ``(N,)`` bool: the carried object's speeds below the static thresholds."""
        vel = self._target_vel()  # (N, 6)
        return object_static_ok(
            torch.linalg.norm(vel[:, :3], dim=1),
            torch.linalg.norm(vel[:, 3:], dim=1),
            lin_threshold=self.config.object_static_lin_threshold,
            ang_threshold=self.config.object_static_ang_threshold,
        )

    def _task_potential(
        self,
        obj_pos: torch.Tensor,
        target_pos: torch.Tensor,
        is_grasped: torch.Tensor,
        is_obj_placed: torch.Tensor,
    ) -> torch.Tensor:
        """``Phi_place(s)``: staged transport-then-settle progress toward completion.

        Batched physics-query wrapper around ``rewards.place_task_potential``,
        the same formula as the MuJoCo backend's ``PickAndPlaceEnv._task_potential``.
        """
        obj_to_target, _ = self._obj_placement_state(obj_pos, target_pos)
        height_gap = (obj_pos[:, 2] - self._initial_obj_z).clamp(min=0.0)
        arm_speed = torch.linalg.norm(self.qvel.index_select(1, self._arm_dof_adr), dim=1)
        return place_task_potential(
            obj_to_target,
            height_gap,
            arm_speed,
            is_grasped,
            is_obj_placed,
            scale=self.config.reward.tanh_shaping_scale,
            velocity_scale=self.config.reward.velocity_shaping_scale,
        )

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        """Refresh the placement baseline and facet potentials from the post-settle pose."""
        super()._refresh_reset_reference_state(mask)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        obj_pos = self._placement_reference_pos()
        self._initial_obj_z[idx] = obj_pos[idx, 2]
        target_pos = self._target_disc_pos()
        is_grasped = self._is_grasping()
        _, is_obj_placed = self._obj_placement_state(obj_pos, target_pos)
        potential = self._task_potential(obj_pos, target_pos, is_grasped, is_obj_placed)
        self._prev_task_potential[idx] = potential[idx]
        # Re-seed reach/grasp over the parent's raw pick-lift potentials with
        # the is_obj_placed-held place forms, matching _compute_reward_terminated.
        scale = self.config.reward.tanh_shaping_scale
        tcp_to_obj = torch.linalg.norm(obj_pos - self._tcp_pos(), dim=1)
        self._prev_reach_progress[idx] = place_reach_potential(
            tcp_to_obj, is_obj_placed, scale=scale
        )[idx]
        self._prev_grasp_progress[idx] = place_grasp_potential(is_grasped, is_obj_placed)[idx]

    def _task_reset(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        sel, target = self._select_active_slots(idx)  # (n, 1), (n,)
        dev = self.device
        if self._n_distractors:
            # Distinct distractor slots per world, as in WarpStackCubeVectorEnv.
            d_rank = self._rng.rand("task", idx, self._d_pool).argsort(dim=1)
            sel = torch.cat([sel, self._d_offset + d_rank[:, : self._n_distractors]], dim=1)
        self._hide_all_slots(idx)

        cfg = self.config
        angle = float(np.radians(cfg.spawn_angle_half_range_deg))
        disc_xy = sample_polar(
            self._rng,
            idx,
            cfg.spawn_min_radius,
            cfg.spawn_max_radius,
            angle,
            cfg.spawn_center,
        )
        radii = self._slot_bradius[sel]  # (n, n_placed)
        obj_xy = sample_separated_polar(
            self._rng,
            idx,
            radii,
            cfg.min_object_separation,
            cfg.spawn_min_radius,
            cfg.spawn_max_radius,
            angle,
            cfg.spawn_center,
        )  # (n, n_placed, 2)
        # Bounding-radius-aware object/disc separation (disc fixed, objects
        # resampled), applied to the carried object and every distractor so no
        # object spawns on the goal. Resampling would undo the object/object
        # clearance the separated sampler established, so with distractors
        # present that check gates the loop too.
        disc_sep = cfg.min_object_target_separation + radii  # (n, n_placed)
        # Carried object plus its distractors; distinct from ``self._n_active``,
        # which stays 1 for this env (one carried slot drawn from the pool).
        n_placed = sel.shape[1]
        pair_sep = cfg.min_object_separation + radii[:, :, None] + radii[:, None, :]
        off_diag = ~torch.eye(n_placed, dtype=torch.bool, device=dev)
        for _ in range(100):
            bad = torch.linalg.norm(obj_xy - disc_xy[:, None, :], dim=2) < disc_sep
            if n_placed > 1:
                pair_dist = torch.linalg.norm(obj_xy[:, :, None, :] - obj_xy[:, None, :, :], dim=3)
                bad |= ((pair_dist < pair_sep) & off_diag).any(dim=2)
            if not bool(bad.any()):
                break
            for column in range(n_placed):
                column_bad = bad[:, column]
                if bool(column_bad.any()):
                    obj_xy[column_bad, column] = sample_polar(
                        self._rng,
                        idx[column_bad],
                        cfg.spawn_min_radius,
                        cfg.spawn_max_radius,
                        angle,
                        cfg.spawn_center,
                    )

        self._mocap_pos[idx, self._target_mocap_id, 0] = disc_xy[:, 0]
        self._mocap_pos[idx, self._target_mocap_id, 1] = disc_xy[:, 1]
        self._mocap_pos[idx, self._target_mocap_id, 2] = _TARGET_Z
        self._place_active_slots(idx, sel, obj_xy)
        self._set_target_tracking(idx, target)

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, TargetPosition):
            return self._target_disc_pos()
        if isinstance(component, TargetOffset):
            return self._target_disc_pos() - self._target_pos()
        return super()._get_component_data(component)

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        obj_pos = self._placement_reference_pos()
        target_pos = self._target_disc_pos()
        tcp_to_obj = torch.linalg.norm(obj_pos - self._tcp_pos(), dim=1)
        obj_to_target, is_obj_placed = self._obj_placement_state(obj_pos, target_pos)
        is_grasped = self._is_grasping()
        is_robot_static = self._is_robot_static()
        is_obj_static = self._is_obj_static()
        # Completion is measured on the OBJECT, not the arm: the goal is a disc on
        # the table, so the intended terminal behaviour is release-and-retreat and
        # an arm-velocity gate would score the retreat itself as failure. Requiring
        # the object to be settled and released is strictly stronger (it rejects
        # flung placements the arm-static check never inspects, and the still-held
        # placement the arm-static check accepts) and keeps the predicate
        # perceivable from vision. ``is_robot_static`` stays in ``info`` as a
        # diagnostic. Mirrors PickAndPlaceEnv._get_info (MuJoCo).
        success = self._placement_success(is_obj_placed, is_obj_static, is_grasped)
        scale = self.config.reward.tanh_shaping_scale
        # Dwelling while grasped must pay ~0/step. Placement holds both potentials
        # so releasing on the goal and retreating do not incur negative deltas.
        # Baselines seeded post-settle by _refresh_reset_reference_state.
        reach_now = place_reach_potential(tcp_to_obj, is_obj_placed, scale=scale)
        grasp_now = place_grasp_potential(is_grasped, is_obj_placed)
        reach_delta = potential_shaping(reach_now, self._prev_reach_progress)
        grasp_delta = potential_shaping(grasp_now, self._prev_grasp_progress)
        self._prev_reach_progress = reach_now
        self._prev_grasp_progress = grasp_now
        # task_progress is a potential-based delta (Ng, Harada & Russell, ICML
        # 1999; see _task_potential), not the raw potential -- dwelling at any
        # fixed state pays ~0 per step instead of the potential's full value.
        task_potential = self._task_potential(obj_pos, target_pos, is_grasped, is_obj_placed)
        task_progress = potential_shaping(task_potential, self._prev_task_potential)
        self._prev_task_potential = task_potential
        reward = self.config.reward.compute(
            reach_progress=reach_delta,
            is_grasped=grasp_delta,
            task_progress=task_progress,
            is_complete=success,
            action_delta_norm=action_delta_norm,
            energy_norm=energy_norm,
        )
        info = {
            "obj_to_target_dist": obj_to_target,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "is_obj_static": is_obj_static,
            "lift_height": obj_pos[:, 2] - self._initial_obj_z,
            "success": success,
            "tcp_to_obj_dist": tcp_to_obj,
            # target_object is MuJoCo-only; see WarpPickLiftVectorEnv's info dict.
            "target_index": self._target_slot.clone(),
        }
        return reward.to(torch.float32), success, info


class WarpPickAndPlaceV2VectorEnv(WarpPickAndPlaceVectorEnv):
    """Center the visual footprint on the disc with sustained table support.

    CUDA captures physics, support checks, and dwell updates together while
    preserving every substep check. CPU and capture failures use the same loop.
    """

    config: PickAndPlaceV2Config
    default_config_cls: ClassVar[type[PickAndPlaceV2Config]] = PickAndPlaceV2Config

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        slots = extract_object_slots(
            self._mjm,
            [f"pick_slot_{i}" for i in range(self._n_pool)],
            self._slot_objs[: self._n_pool],
        )
        self._placement_geometry = [
            SlotGeometry(
                torch.as_tensor(geometry.triangles, dtype=torch.float32, device=self.device),
                None
                if geometry.symmetry_center is None
                else torch.as_tensor(
                    geometry.symmetry_center, dtype=torch.float32, device=self.device
                ),
            )
            for geometry in compiled_slot_geometry(self._mjm, slots)
        ]
        body_ids = [
            int(self._mjm.body_rootid[self._mjm.geom_bodyid[slot.geom_ids[0]]]) for slot in slots
        ]
        self._placement_body_ids = torch.tensor(body_ids, device=self.device)
        weights = [
            float(self._mjm.body_subtreemass[body]) * float(np.linalg.norm(self._mjm.opt.gravity))
            for body in body_ids
        ]
        self._placement_weights = torch.tensor(weights, device=self.device)
        self._placement_weights_host = weights
        self._placement_timesteps = wp.to_torch(self.model.opt.timestep)
        base_id = mujoco.mj_name2id(self._mjm, mujoco.mjtObj.mjOBJ_BODY, "base")
        self._placement_robot_geom_mask = torch.as_tensor(
            self._mjm.body_rootid[self._mjm.geom_bodyid] == base_id, device=self.device
        )
        self._placement_floor_geom = mujoco.mj_name2id(self._mjm, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._placement_xpos = wp.to_torch(self.data.xpos)
        self._placement_xmat = wp.to_torch(self.data.xmat)
        self._placement_xipos = wp.to_torch(self.data.xipos)
        self._placement_subtree_com = wp.to_torch(self.data.subtree_com)
        self._placement_cvel = wp.to_torch(self.data.cvel)
        self._placement_dwell = torch.zeros(self.num_envs, device=self.device, dtype=torch.float64)
        self._placement_contact_rows = torch.arange(self.data.naconmax, device=self.device)
        self._ensure_contact_force_buffers()
        self._placement_evaluation: dict | None = None
        self._capture_step_graph()

    def _capture_step_graph(self) -> None:
        # Base construction precedes the support buffers; capture once they exist.
        self._captured_support_parameters = self._support_parameters()
        self._step_graph = (
            self._capture_torch_graph(self._step_with_support, "Placement")
            if hasattr(self, "_placement_dwell")
            else None
        )

    def _support_parameters(self) -> tuple[float, ...]:
        return (
            self.config.support_min_weight_fraction,
            self.config.support_force_tolerance,
            self.config.support_relative_force_tolerance,
            self.config.object_static_lin_threshold,
            self.config.object_static_ang_threshold,
        )

    def _describe_target(self, obj) -> str:
        return self.config.describe_target(obj, self.target_color_name)

    def _generic_task_description(self) -> str:
        return (
            f"Pick up the object, center it on the {self.target_color_name} circle, "
            "and release it onto the table."
        )

    def _visual_reference_pos(self) -> torch.Tensor:
        """Return footprint union XY and lowest visual Z without host geometry copies."""
        reference = torch.empty((self.num_envs, 3), device=self.device)
        for slot, geometry in enumerate(self._placement_geometry):
            worlds = (self._target_slot == slot).nonzero(as_tuple=True)[0]
            body = self._placement_body_ids[slot]
            # Bound transformed triangles independently of the total world count.
            chunk_size = max(1, min(32, 262144 // max(1, geometry.triangles.shape[0])))
            for start in range(0, worlds.numel(), chunk_size):
                rows = worlds[start : start + chunk_size]
                rotation = self._placement_xmat[rows, body]
                position = self._placement_xpos[rows, body]
                reference[rows] = visual_reference(
                    geometry, rotation, position, scanlines=self.config.footprint_scanlines
                )
        return reference

    def _placement_reference_pos(self) -> torch.Tensor:
        if self._placement_evaluation is not None:
            return self._placement_evaluation["reference"]
        return self._visual_reference_pos()

    def _placement_contact_forces(self) -> torch.Tensor:
        """Read solved contact forces without a device-to-host contact count."""
        return self._contact_forces()[0]

    def _support_info(self) -> dict[str, torch.Tensor]:
        assert self._obj_geom_mask is not None
        force = self._placement_contact_forces()
        geoms = self._contact_geom_view.long()
        raw_worlds = self._contact_world_view.long()
        worlds = raw_worlds.clamp(0, self.num_envs - 1)
        g1 = geoms[:, 0].clamp(0, self._mjm.ngeom - 1)
        g2 = geoms[:, 1].clamp(0, self._mjm.ngeom - 1)
        valid = (
            (self._placement_contact_rows < self._nacon_view[0])
            & (raw_worlds >= 0)
            & (raw_worlds < self.num_envs)
            & (geoms >= 0).all(dim=1)
            & (geoms < self._mjm.ngeom).all(dim=1)
        )
        target1 = self._obj_geom_mask[worlds, g1]
        target2 = self._obj_geom_mask[worlds, g2]
        valid &= target1 ^ target2
        other = torch.where(target1, g2, g1)
        table_contact = valid & (other == self._placement_floor_geom)
        robot_contact = valid & self._placement_robot_geom_mask[other]
        other_contact = valid & ~table_contact & ~robot_contact
        world_force = (self._contact_frame_view * force[:, :3, None]).sum(dim=1)
        sign = target2.to(force.dtype) - target1.to(force.dtype)
        magnitude = torch.linalg.norm(force[:, :3], dim=1)
        table_force = torch.zeros(self.num_envs, device=self.device)
        robot_force = torch.zeros_like(table_force)
        other_force = torch.zeros_like(table_force)
        table_force.scatter_add_(
            0, worlds, torch.where(table_contact, world_force[:, 2] * sign, 0.0)
        )
        robot_force.scatter_add_(0, worlds, torch.where(robot_contact, magnitude, 0.0))
        other_force.scatter_add_(0, worlds, torch.where(other_contact, magnitude, 0.0))
        bodies = self._placement_body_ids[self._target_slot]
        cvel = self._placement_cvel[self._world_rows, bodies]
        # cvel uses the root subtree COM as its spatial reference. Translate it
        # to the target body's inertial COM, not the free-joint frame origin.
        offset = (
            self._placement_xipos[self._world_rows, bodies]
            - self._placement_subtree_com[self._world_rows, bodies]
        )
        com_velocity = cvel[:, 3:] + torch.linalg.cross(cvel[:, :3], offset)
        linear_speed = torch.linalg.norm(com_velocity, dim=1)
        angular_speed = torch.linalg.norm(cvel[:, :3], dim=1)
        table_ok, robot_ok, other_ok, static, qualified = support_state(
            table_force,
            robot_force,
            other_force,
            self._placement_weights[self._target_slot],
            linear_speed,
            angular_speed,
            min_weight_fraction=self.config.support_min_weight_fraction,
            force_tolerance=self.config.support_force_tolerance,
            relative_force_tolerance=self.config.support_relative_force_tolerance,
            lin_threshold=self.config.object_static_lin_threshold,
            ang_threshold=self.config.object_static_ang_threshold,
        )
        return {
            "supported_by_table": table_ok,
            "supported_by_robot": robot_ok,
            "supported_by_other": other_ok,
            "table_support_force": table_force,
            "robot_support_force": robot_force,
            "other_support_force": other_force,
            "object_linear_speed": linear_speed,
            "object_angular_speed": angular_speed,
            "is_obj_static": static,
            "qualified": qualified,
        }

    def _placement_info(self) -> dict:
        support = self._support_info()
        reference = self._visual_reference_pos()
        distance = torch.linalg.norm(reference[:, :2] - self._target_disc_pos()[:, :2], dim=1)
        geometric_ok = distance <= self.config.target_disc_radius
        placed = geometric_ok & support.pop("qualified")
        return {
            **support,
            "placement_contract": resolved_placement_contract(
                self.config,
                self._slot_objs[: self._n_pool],
                self._placement_weights_host,
                self._placement_timesteps.cpu().tolist(),
                self._target_slot_host.copy(),
            ),
            "placement_centroid_xy": reference[:, :2],
            "geometric_placement_ok": geometric_ok,
            # Snapshot timers before same-step autoreset clears completed worlds.
            "placement_dwell": self._placement_dwell.clone(),
            "is_obj_placed": placed,
            "success": placed & (self._placement_dwell >= self.config.placement_dwell_time),
            "obj_to_target_dist": distance,
            "reference": reference,
        }

    def _obj_placement_state(
        self, obj_pos: torch.Tensor, target_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._placement_evaluation is not None:
            return (
                self._placement_evaluation["obj_to_target_dist"],
                self._placement_evaluation["is_obj_placed"],
            )
        distance = torch.linalg.norm(obj_pos[:, :2] - target_pos[:, :2], dim=1)
        return distance, (distance <= self.config.target_disc_radius) & self._support_info()[
            "qualified"
        ]

    def _is_obj_static(self) -> torch.Tensor:
        if self._placement_evaluation is not None:
            return self._placement_evaluation["is_obj_static"]
        return self._support_info()["is_obj_static"]

    def _placement_success(
        self,
        is_obj_placed: torch.Tensor,
        is_obj_static: torch.Tensor,
        is_grasped: torch.Tensor,
    ) -> torch.Tensor:
        return is_obj_placed & (self._placement_dwell >= self.config.placement_dwell_time)

    def _advance_physics(self) -> None:
        # Captured scalar thresholds must never override later config changes.
        if (
            self._step_graph is not None
            and self._support_parameters() == self._captured_support_parameters
        ):
            self._step_graph.replay()
        else:
            self._step_with_support()

    def _step_with_support(self) -> None:
        for _ in range(self._N_SUBSTEPS):
            mjw.step(self.model, self.data)
            qualified = self._support_info()["qualified"]
            self._placement_dwell.copy_(
                torch.where(qualified, self._placement_dwell + self._placement_timesteps, 0.0)
            )
        # Match geometry and observations to the final integrated state.
        mjw.forward(self.model, self.data)

    def _task_reset(self, mask: torch.Tensor) -> None:
        super()._task_reset(mask)
        self._placement_dwell[mask] = 0.0

    def reset(
        self,
        *,
        seed: int | list[int] | tuple[int, ...] | None = None,
        options: dict | None = None,
    ):
        """Reset all worlds and return their initial placement diagnostics."""
        obs, info = super().reset(seed=seed, options=options)
        placement = self._placement_info()
        placement.pop("reference")
        info.update(placement)
        return obs, info

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        placement = self._placement_info()
        self._placement_evaluation = placement
        try:
            reward, success, info = super()._compute_reward_terminated(
                energy_norm, action_delta_norm
            )
        finally:
            self._placement_evaluation = None
        placement.pop("reference")
        info.update(placement)
        return reward, success, info
