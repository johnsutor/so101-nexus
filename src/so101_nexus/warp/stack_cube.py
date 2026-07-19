"""GPU-batched stack-cube environment for SO-101 on MuJoCo Warp.

Warp's ``geom_rgba`` is a model-global array (not per-world), so per-episode
colour variety is realized the same way ``WarpPickLiftVectorEnv`` varies object
identity: one freejoint cube slot is compiled per configured colour in
``cube_a_colors`` / ``cube_b_colors``, each world selects one slot per role at
reset, and inactive slots are parked in an off-world band. This matches the
MuJoCo backend's per-episode resampling distribution (uniform over each
configured list, sampled independently per role) -- the divergence documented
here previously (compile-time-fixed colours) is removed.
"""

from __future__ import annotations

import tempfile

import mujoco
import numpy as np
import torch

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, StackCubeConfig, describe_stack_target
from so101_nexus.constants import COLOR_MAP, ColorName
from so101_nexus.object_slots import build_object_scene_xml, extract_object_slots
from so101_nexus.objects import CubeObject
from so101_nexus.observations import ObjectOffset, ObjectPose, TargetOffset, TargetPosition
from so101_nexus.rewards import (
    cube_stack_offset_ok,
    cube_static_ok,
    place_grasp_potential,
    place_reach_potential,
    place_task_potential,
    potential_shaping,
)
from so101_nexus.scene import WARP_SCENE_OPTION_XML
from so101_nexus.warp.base_env import SO101NexusWarpVectorEnv
from so101_nexus.warp.object_slots import (
    hidden_slot_band_xy,
    random_yaw_quat_batch,
    sample_separated_polar,
)

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

# Contact budget per world, mirroring so101_nexus.warp.pick_env._contact_budget:
# a generous floor for active grasping plus a per-slot allowance for the parked
# slots' resting contacts. naconmax = nconmax * num_envs.
_STACK_NCONMAX_BASE = 192
_STACK_NCONMAX_PER_SLOT = 16


def _contact_budget(n_slots: int) -> tuple[int, int]:
    nconmax = _STACK_NCONMAX_BASE + _STACK_NCONMAX_PER_SLOT * n_slots
    return nconmax, nconmax * 2


def _as_list(colors: ColorName | list[ColorName]) -> list[ColorName]:
    return [colors] if isinstance(colors, str) else list(colors)


class WarpStackCubeVectorEnv(SO101NexusWarpVectorEnv):
    """Batched stack-cube: pick up cube A and stack it on top of cube B.

    Success requires cube A to rest directly on cube B (within
    ``config.stack_alignment_margin``), the arm to be static, cube A itself to
    be static (``config.cube_static_lin_threshold`` /
    ``config.cube_static_ang_threshold``), and cube A released
    (``is_grasped < 0.5``) -- a strict superset of ManiSkill's
    ``StackCubeEnv.evaluate`` predicate, matching ``MuJoCoStackCube-v1``.

    Default obs (30,): joint_positions(6) + end_effector_pose(7) + grasp_state(1)
    + object_pose(7) + object_offset(3) + target_position(3) + target_offset(3),
    matching ``MuJoCoStackCube-v1``.

    ``cube_a_color_names`` / ``cube_b_color_names`` hold the selected colour per
    world and are resampled at every reset; ``task_descriptions`` tracks them.
    """

    config: StackCubeConfig

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
        scene_objects = [
            CubeObject(half_size=config.cube_half_size, mass=config.cube_mass, color=c)
            for c in self._a_colors + self._b_colors
        ]
        slot_names = [f"cube_a_{i}" for i in range(self._a_pool)] + [
            f"cube_b_{i}" for i in range(self._b_pool)
        ]

        ground_name = (
            config.ground_colors
            if isinstance(config.ground_colors, str)
            else config.ground_colors[0]
        )
        xml_string = build_object_scene_xml(
            scene_objects,
            slot_names,
            COLOR_MAP[ground_name],
            option_xml=WARP_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
            model_name="stack_cube_scene",
            overhead_camera_xml=self._overhead_camera_xml(config),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            mjm = mujoco.MjModel.from_xml_path(f.name)
        slots = extract_object_slots(mjm, slot_names, scene_objects)
        default_nconmax, default_njmax = _contact_budget(len(slots))

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

        n_slots = len(slots)
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
        self._hide_xy = hidden_slot_band_xy(
            self.device,
            n_slots,
            float(self._slot_bradius.max()),
            config.spawn_max_radius,
            config.spawn_center,
        )
        # Cube slots rest at an identity quaternion (no mesh rest pose).
        self._identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        # Per-world slot selection (set at reset; defaults track the first
        # configured pair until then). Grasp detection always targets the
        # selected cube A slot; cube B is never picked up.
        self._a_qadr = self._slot_qadr[0].expand(num_envs).clone()
        self._a_dadr = self._slot_dadr[0].expand(num_envs).clone()
        self._b_qadr = self._slot_qadr[self._a_pool].expand(num_envs).clone()
        self._b_dadr = self._slot_dadr[self._a_pool].expand(num_envs).clone()
        self._obj_geom = self._slot_geom[0].expand(num_envs).clone()
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
        return {ObjectPose, ObjectOffset, TargetPosition, TargetOffset}

    def _gather(self, buf: torch.Tensor, base_cols: torch.Tensor, width: int) -> torch.Tensor:
        cols = base_cols[:, None] + torch.arange(width, device=self.device)
        return buf[self._world_rows[:, None], cols]

    def _cube_a_pos(self) -> torch.Tensor:
        return self._gather(self.qpos, self._a_qadr, 3)

    def _cube_a_pose7(self) -> torch.Tensor:
        return self._gather(self.qpos, self._a_qadr, 7)

    def _cube_b_pos(self) -> torch.Tensor:
        return self._gather(self.qpos, self._b_qadr, 3)

    def _is_cube_a_static(self) -> torch.Tensor:
        """Return ``(N,)`` bool: cube A's speeds below the static thresholds.

        ManiSkill's ``is_cubeA_static`` check on the per-world selected slot.
        """
        vel = self._gather(self.qvel, self._a_dadr, 6)  # (N, 6)
        return cube_static_ok(
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
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        cfg = self.config
        gen, dev = self._generator, self.device
        # One slot per configured colour entry; sample one slot per role per
        # world, matching the MuJoCo backend's per-episode colour resampling.
        a_sel = torch.randint(0, self._a_pool, (n,), generator=gen, device=dev)
        b_sel_local = torch.randint(0, self._b_pool, (n,), generator=gen, device=dev)
        b_sel = self._a_pool + b_sel_local

        # Park every slot off-world with zeroed velocity (Warp contact bits are
        # model-global, so inactive cubes cannot have collisions disabled);
        # the two selected slots per world are overwritten below.
        for j in range(len(self._slot_qadr)):
            qa = int(self._slot_qadr[j])
            self.qpos[idx, qa] = self._hide_xy[j, 0]
            self.qpos[idx, qa + 1] = self._hide_xy[j, 1]
            self.qpos[idx, qa + 2] = self._slot_spawn_z[j]
            self.qpos[idx, qa + 3 : qa + 7] = self._identity_quat
            da = int(self._slot_dadr[j])
            self.qvel[idx, da : da + 6] = 0.0

        angle = float(np.radians(cfg.spawn_angle_half_range_deg))
        sel = torch.stack([a_sel, b_sel], dim=1)  # (n, 2)
        radii = self._slot_bradius[sel]  # (n, 2)
        positions = sample_separated_polar(
            gen,
            dev,
            radii,
            cfg.min_cube_separation,
            cfg.spawn_min_radius,
            cfg.spawn_max_radius,
            angle,
            cfg.spawn_center,
        )  # (n, 2, 2): column 0 = cube A, column 1 = cube B

        rows = idx[:, None]
        for k in range(2):
            sel_k = sel[:, k]
            base = self._slot_qadr[sel_k]  # (n,) qpos address per reset world
            yaw = random_yaw_quat_batch(gen, dev, n)
            # Cubes rest at an identity quaternion, so a pure yaw sample is the
            # final orientation (no rest_quat multiply needed, unlike mesh slots).
            self.qpos[rows, base[:, None] + torch.arange(3, device=dev)] = torch.cat(
                [positions[:, k], self._slot_spawn_z[sel_k][:, None]], dim=1
            )
            self.qpos[rows, base[:, None] + 3 + torch.arange(4, device=dev)] = yaw
            dadr = self._slot_dadr[sel_k]
            self.qvel[rows, dadr[:, None] + torch.arange(6, device=dev)] = 0.0

        self._a_qadr[idx] = self._slot_qadr[a_sel]
        self._a_dadr[idx] = self._slot_dadr[a_sel]
        self._b_qadr[idx] = self._slot_qadr[b_sel]
        self._b_dadr[idx] = self._slot_dadr[b_sel]
        obj_geom = self._obj_geom
        assert obj_geom is not None  # set to a tensor in __init__
        obj_geom[idx] = self._slot_geom[a_sel]
        for i in range(n):
            w = int(idx[i])
            a_color = self._a_colors[int(a_sel[i])]
            b_color = self._b_colors[int(b_sel_local[i])]
            self.cube_a_color_names[w] = a_color
            self.cube_b_color_names[w] = b_color
            self.task_descriptions[w] = self._describe_pair(a_color, b_color)

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, ObjectPose):
            return self._cube_a_pose7()
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
        reach_delta = potential_shaping(reach_now, self._prev_reach_progress)
        grasp_delta = potential_shaping(grasp_now, self._prev_grasp_progress)
        self._prev_reach_progress = reach_now
        self._prev_grasp_progress = grasp_now
        task_potential = self._task_potential(a_pos, b_pos, is_grasped, is_stacked)
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
            "cube_a_to_goal_dist": cube_a_to_goal_dist,
            "is_stacked": is_stacked,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "is_cube_a_static": is_cube_a_static,
            "success": success,
            "tcp_to_obj_dist": tcp_to_obj,
        }
        return reward.to(torch.float32), success, info
