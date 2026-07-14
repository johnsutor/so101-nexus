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

import tempfile

import mujoco
import numpy as np
import torch

from so101_nexus import (
    ensure_ycb_assets,
    get_so101_mujoco_model_dir,
    get_so101_mujoco_model_path,
)
from so101_nexus.config import ControlMode, PickConfig, describe_pick_target
from so101_nexus.constants import sample_color
from so101_nexus.object_slots import build_object_scene_xml, extract_object_slots
from so101_nexus.objects import SceneObject, YCBObject
from so101_nexus.observations import ObjectOffset, ObjectPose
from so101_nexus.rewards import lift_progress, potential_shaping, reach_progress
from so101_nexus.scene import WARP_SCENE_OPTION_XML
from so101_nexus.warp.base_env import SO101NexusWarpVectorEnv
from so101_nexus.warp.object_slots import (
    quat_mul_wxyz,
    random_yaw_quat_batch,
    sample_separated_polar,
)

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

# Contact budget per world. The single-cube scene needs a generous floor for
# active grasping; pools add resting contacts for hidden/distractor slots, so the
# budget scales with the pool size. naconmax = nconmax * num_envs.
_PICK_NCONMAX_BASE = 192
_PICK_NCONMAX_PER_SLOT = 16

# Clearance (m) between the off-world parking band for inactive slots and the
# reachable spawn annulus. Parking positions are derived from the configured
# spawn bounds and object radii so hidden slots never overlap active samples.
_HIDE_CLEARANCE = 0.1


def _contact_budget(n_pool: int) -> tuple[int, int]:
    nconmax = _PICK_NCONMAX_BASE + _PICK_NCONMAX_PER_SLOT * n_pool
    return nconmax, nconmax * 2


class WarpPickLiftVectorEnv(SO101NexusWarpVectorEnv):
    """Batched pick-lift: grasp the per-world target object and lift it.

    Default obs (24,): joint_positions(6) + end_effector_pose(7) + grasp_state(1) +
    object_pose(7) + object_offset(3), matching ``MuJoCoPickLift-v1``.
    """

    config: PickConfig

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
    ) -> None:
        """Compile the shared object-slot model and build per-slot tensors.

        Shared by the pick/touch and pick-and-place backends; the latter passes
        ``extra_bodies`` for the mocap goal disc and ``n_active=1``.
        """
        for obj in scene_objects:
            if isinstance(obj, YCBObject):
                ensure_ycb_assets(obj.model_id)
        self._n_pool = len(scene_objects)
        self._n_active = n_active
        slot_names = [f"pick_slot_{i}" for i in range(self._n_pool)]

        xml_string = build_object_scene_xml(
            scene_objects,
            slot_names,
            sample_color(config.ground_colors),
            option_xml=WARP_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
            model_name=model_name,
            extra_bodies=extra_bodies,
            overhead_camera_xml=self._overhead_camera_xml(config),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            mjm = mujoco.MjModel.from_xml_path(f.name)
        slots = extract_object_slots(mjm, slot_names, scene_objects)
        default_nconmax, default_njmax = _contact_budget(self._n_pool)
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
        self._slot_qadr = torch.tensor([s.qpos_addr for s in slots], device=self.device)
        self._slot_dadr = torch.tensor([s.dof_addr for s in slots], device=self.device)
        self._slot_geom = torch.tensor(
            [s.geom_id for s in slots], dtype=torch.long, device=self.device
        )
        self._slot_spawn_z = torch.tensor(
            [s.spawn_z for s in slots], dtype=torch.float32, device=self.device
        )
        self._slot_bradius = torch.tensor(
            [s.bounding_radius for s in slots], dtype=torch.float32, device=self.device
        )
        self._slot_rest_quat = torch.tensor(
            np.stack([s.rest_quat for s in slots]), dtype=torch.float32, device=self.device
        )
        # Park inactive slots in a band beyond the reachable spawn annulus, spaced
        # by object diameter so neither active samples nor adjacent hidden bodies
        # overlap (Warp contact bits are global, so hidden slots stay collidable).
        cx, cy = config.spawn_center
        max_br = float(self._slot_bradius.max())
        step = 2.0 * max_br + _HIDE_CLEARANCE
        base = config.spawn_max_radius + 2.0 * max_br + _HIDE_CLEARANCE
        hide_x = cx - base - step * torch.arange(self._n_pool, device=self.device)
        self._hide_xy = torch.stack(
            [hide_x, torch.full((self._n_pool,), cy, device=self.device)], dim=1
        )

        # Per-world target tracking (set at reset). ``_obj_geom`` drives grasp
        # detection in the base; ``_target_qadr`` indexes the selected slot pose.
        self._obj_geom = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._target_qadr = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._target_slot = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._initial_obj_z = torch.zeros(num_envs, device=self.device)
        self._prev_task_potential = torch.zeros(num_envs, device=self.device)
        self._world_rows = torch.arange(num_envs, device=self.device)
        self.task_descriptions = [self._describe_target(scene_objects[0])] * num_envs

    def _describe_target(self, obj: SceneObject) -> str:
        """Per-world task description for the chosen target (overridable per task)."""
        return describe_pick_target(obj)

    def _generic_task_description(self) -> str:
        return "Pick up the selected object."

    def _supported_obs_components(self) -> set[type]:
        return {ObjectPose, ObjectOffset}

    def _gather(self, base_cols: torch.Tensor, width: int) -> torch.Tensor:
        cols = base_cols[:, None] + torch.arange(width, device=self.device)
        return self.qpos[self._world_rows[:, None], cols]

    def _target_pose7(self) -> torch.Tensor:
        return self._gather(self._target_qadr, 7)

    def _target_pos(self) -> torch.Tensor:
        return self._gather(self._target_qadr, 3)

    def _target_bounding_radius(self) -> torch.Tensor:
        return self._slot_bradius[self._target_slot]

    def _select_active_slots(self, n: int) -> torch.Tensor:
        """Return ``(n, n_active)`` distinct pool indices (col 0 = target)."""
        if self._n_active == 1:
            return torch.randint(
                0, self._n_pool, (n, 1), generator=self._generator, device=self.device
            )
        perm = torch.rand(n, self._n_pool, generator=self._generator, device=self.device)
        return perm.argsort(dim=1)[:, : self._n_active]

    def _hide_all_slots(self, idx: torch.Tensor) -> None:
        """Park every slot at its hidden resting pose for the reset worlds.

        Slot qpos columns are shared across worlds, so hiding uses fixed slices;
        active slots are overwritten afterward by ``_place_active_slots``.
        """
        for j in range(self._n_pool):
            qa = int(self._slot_qadr[j])
            self.qpos[idx, qa] = self._hide_xy[j, 0]
            self.qpos[idx, qa + 1] = self._hide_xy[j, 1]
            self.qpos[idx, qa + 2] = self._slot_spawn_z[j]
            self.qpos[idx, qa + 3 : qa + 7] = self._slot_rest_quat[j]
            da = int(self._slot_dadr[j])
            self.qvel[idx, da : da + 6] = 0.0

    def _place_active_slots(
        self, idx: torch.Tensor, sel: torch.Tensor, positions: torch.Tensor
    ) -> None:
        """Place each rank's selected slot at ``positions[:, k]`` with random yaw."""
        rows = idx[:, None]
        n = int(idx.numel())
        for k in range(sel.shape[1]):
            sel_k = sel[:, k]  # (n,) pool idx per reset world
            base = self._slot_qadr[sel_k]
            yaw = random_yaw_quat_batch(self._generator, self.device, n)
            quat = quat_mul_wxyz(yaw, self._slot_rest_quat[sel_k])
            self.qpos[rows, base[:, None] + torch.arange(3, device=self.device)] = torch.cat(
                [positions[:, k], self._slot_spawn_z[sel_k][:, None]], dim=1
            )
            self.qpos[rows, base[:, None] + 3 + torch.arange(4, device=self.device)] = quat

    def _set_target_tracking(self, idx: torch.Tensor, sel: torch.Tensor) -> None:
        """Record the rank-0 slot as the per-world target and update descriptions."""
        target = sel[:, 0]
        self._target_slot[idx] = target
        obj_geom = self._obj_geom
        assert obj_geom is not None  # set to a tensor in _build_slot_model
        obj_geom[idx] = self._slot_geom[target]
        self._target_qadr[idx] = self._slot_qadr[target]
        self._initial_obj_z[idx] = self._slot_spawn_z[target]
        for i in range(int(idx.numel())):
            self.task_descriptions[int(idx[i])] = self._describe_target(
                self._slot_objs[int(sel[i, 0])]
            )

    def _task_reset(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        sel = self._select_active_slots(n)  # (n, n_active)
        self._hide_all_slots(idx)
        radii = self._slot_bradius[sel]  # (n, n_active)
        positions = sample_separated_polar(
            self._generator,
            self.device,
            radii,
            self.config.min_object_separation,
            self.config.spawn_min_radius,
            self.config.spawn_max_radius,
            float(np.radians(self.config.spawn_angle_half_range_deg)),
            self.config.spawn_center,
        )
        self._place_active_slots(idx, sel, positions)
        self._set_target_tracking(idx, sel)

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        self._initial_obj_z[idx] = self._target_pos()[idx, 2]
        # lift_progress(0, ...) == 0 regardless of grasped (tanh(0) == 0), so the
        # potential baseline is always 0 here: the object sits at its own baseline.
        self._prev_task_potential[idx] = 0.0

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, ObjectPose):
            return self._target_pose7()
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
        # task_progress is a potential-based delta (Ng, Harada & Russell, ICML
        # 1999; see rewards.potential_shaping), not the raw lift potential --
        # dwelling at a fixed lift height pays ~0 per step instead of the
        # potential's full value every step.
        lift_potential = lift_progress(lift_height, scale=scale, grasped=grasped)
        task_progress = potential_shaping(lift_potential, self._prev_task_potential)
        self._prev_task_potential = lift_potential
        reward = self.config.reward.compute(
            reach_progress=reach_progress(tcp_to_obj, scale=scale),
            is_grasped=is_grasped,
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
        }
        return reward.to(torch.float32), success, info
