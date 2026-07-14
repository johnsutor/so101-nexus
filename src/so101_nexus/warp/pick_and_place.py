"""GPU-batched pick-and-place environment for SO-101 on MuJoCo Warp.

The carried object is chosen per episode from the shared compiled object-slot pool
(see ``so101_nexus.warp.pick_env``); the goal is a non-colliding mocap disc whose
per-world position lives in ``data.mocap_pos``. The disc colour is fixed at
model-build time to the first configured target colour (``geom_rgba`` is global), a
documented divergence from the MuJoCo backend, which randomizes it per episode.
"""

from __future__ import annotations

import mujoco
import numpy as np
import torch
import warp as wp

from so101_nexus.config import ControlMode, PickAndPlaceConfig, describe_place_target
from so101_nexus.constants import COLOR_MAP
from so101_nexus.objects import CubeObject
from so101_nexus.observations import ObjectOffset, ObjectPose, TargetOffset, TargetPosition
from so101_nexus.rewards import potential_shaping, reach_progress
from so101_nexus.warp.object_slots import sample_polar
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

    Default obs (30,): joint_positions(6) + end_effector_pose(7) + grasp_state(1)
    + target_position(3) + object_pose(7) + object_offset(3) + target_offset(3),
    matching ``MuJoCoPickAndPlace-v1``.
    """

    config: PickAndPlaceConfig

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
            config = PickAndPlaceConfig()
        scene_objects = config.object_pool()
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
        return {ObjectPose, ObjectOffset, TargetPosition, TargetOffset}

    def _target_disc_pos(self) -> torch.Tensor:
        return self._mocap_pos[:, self._target_mocap_id, :]

    def _obj_placement_state(
        self, obj_pos: torch.Tensor, target_pos: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(obj_to_target_xy_dist, is_obj_placed)`` batched over worlds."""
        obj_to_target = torch.linalg.norm(obj_pos[:, :2] - target_pos[:, :2], dim=1)
        is_obj_placed = (obj_to_target <= self.config.goal_thresh) & (
            obj_pos[:, 2] < self._initial_obj_z + _PLACE_Z_SLACK
        )
        return obj_to_target, is_obj_placed

    def _task_potential(
        self,
        obj_pos: torch.Tensor,
        target_pos: torch.Tensor,
        is_grasped: torch.Tensor,
        is_obj_placed: torch.Tensor,
    ) -> torch.Tensor:
        """``Phi_place(s)``: joint xy/height/stillness progress toward completion.

        Batched analogue of the MuJoCo backend's ``PickAndPlaceEnv._task_potential``
        -- zero unless the object has been grasped or is already placed, otherwise
        the product of three ``[0, 1]`` progress factors (goal xy proximity, height
        back near rest, arm stillness). Fed through ``rewards.potential_shaping`` as
        a step-to-step delta rather than used directly as ``task_progress``. See
        docs/superpowers/plans/2026-07-12-potential-based-task-progress-shaping.md.
        """
        scale = self.config.reward.tanh_shaping_scale
        vscale = self.config.reward.velocity_shaping_scale
        obj_to_target, _ = self._obj_placement_state(obj_pos, target_pos)
        height_gap = (obj_pos[:, 2] - self._initial_obj_z).clamp(min=0.0)
        arm_speed = torch.linalg.norm(self.qvel.index_select(1, self._arm_dof_adr), dim=1)
        gate = ((is_grasped > 0.5) | is_obj_placed).to(torch.float32)
        return (
            reach_progress(obj_to_target, scale=scale)
            * reach_progress(height_gap, scale=scale)
            * reach_progress(arm_speed, scale=vscale)
            * gate
        )

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        """Refresh the placement baseline and task potential from the post-settle pose."""
        super()._refresh_reset_reference_state(mask)
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        obj_pos = self._target_pos()
        target_pos = self._target_disc_pos()
        is_grasped = self._is_grasping()
        _, is_obj_placed = self._obj_placement_state(obj_pos, target_pos)
        potential = self._task_potential(obj_pos, target_pos, is_grasped, is_obj_placed)
        self._prev_task_potential[idx] = potential[idx]

    def _task_reset(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        sel = self._select_active_slots(n)  # (n, 1)
        self._hide_all_slots(idx)

        cfg = self.config
        angle = float(np.radians(cfg.spawn_angle_half_range_deg))
        disc_xy = sample_polar(
            self._generator,
            self.device,
            n,
            cfg.spawn_min_radius,
            cfg.spawn_max_radius,
            angle,
            cfg.spawn_center,
        )
        obj_xy = sample_polar(
            self._generator,
            self.device,
            n,
            cfg.spawn_min_radius,
            cfg.spawn_max_radius,
            angle,
            cfg.spawn_center,
        )
        # Bounding-radius-aware object/disc separation (disc fixed, object resampled).
        sep = cfg.min_object_target_separation + self._slot_bradius[sel[:, 0]]
        for _ in range(100):
            bad = torch.linalg.norm(obj_xy - disc_xy, dim=1) < sep
            k = int(bad.sum())
            if k == 0:
                break
            obj_xy[bad] = sample_polar(
                self._generator,
                self.device,
                k,
                cfg.spawn_min_radius,
                cfg.spawn_max_radius,
                angle,
                cfg.spawn_center,
            )

        self._mocap_pos[idx, self._target_mocap_id, 0] = disc_xy[:, 0]
        self._mocap_pos[idx, self._target_mocap_id, 1] = disc_xy[:, 1]
        self._mocap_pos[idx, self._target_mocap_id, 2] = _TARGET_Z
        self._place_active_slots(idx, sel, obj_xy[:, None, :])
        self._set_target_tracking(idx, sel)

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, TargetPosition):
            return self._target_disc_pos()
        if isinstance(component, TargetOffset):
            return self._target_disc_pos() - self._target_pos()
        return super()._get_component_data(component)

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        obj_pos = self._target_pos()
        target_pos = self._target_disc_pos()
        tcp_to_obj = torch.linalg.norm(obj_pos - self._tcp_pos(), dim=1)
        obj_to_target, is_obj_placed = self._obj_placement_state(obj_pos, target_pos)
        is_grasped = self._is_grasping()
        is_robot_static = self._is_robot_static()
        success = is_obj_placed & is_robot_static
        scale = self.config.reward.tanh_shaping_scale
        # task_progress is a potential-based delta (Ng, Harada & Russell, ICML
        # 1999; see _task_potential), not the raw potential -- dwelling at any
        # fixed state pays ~0 per step instead of the potential's full value.
        task_potential = self._task_potential(obj_pos, target_pos, is_grasped, is_obj_placed)
        task_progress = potential_shaping(task_potential, self._prev_task_potential)
        self._prev_task_potential = task_potential
        reward = self.config.reward.compute(
            reach_progress=reach_progress(tcp_to_obj, scale=scale),
            is_grasped=is_grasped,
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
            "lift_height": obj_pos[:, 2] - self._initial_obj_z,
            "success": success,
            "tcp_to_obj_dist": tcp_to_obj,
        }
        return reward.to(torch.float32), success, info
