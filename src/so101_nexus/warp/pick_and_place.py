"""GPU-batched pick-and-place environment for SO-101 on MuJoCo Warp.

The cube is a freejoint body; the goal disc is a non-colliding mocap body so its
pose can vary per world (``data.mocap_pos`` is per-world, unlike ``model``
fields). Cube and target colors are fixed at model-build time to the first
configured color, since ``geom_rgba`` is shared across all worlds; this is a
documented divergence from the MuJoCo backend, which randomizes color per
episode. ``task_description`` reflects the fixed colors.
"""

from __future__ import annotations

import tempfile

import mujoco
import torch
import warp as wp

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, PickAndPlaceConfig
from so101_nexus.constants import COLOR_MAP, sample_color
from so101_nexus.observations import (
    ObjectOffset,
    ObjectPose,
    TargetOffset,
    TargetPosition,
)
from so101_nexus.rewards import reach_progress
from so101_nexus.scene import WARP_SCENE_OPTION_XML
from so101_nexus.warp.base_env import SO101NexusWarpVectorEnv

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

# Contact-rich scene (cube + gripper + floor); see WarpPickLiftVectorEnv.
_PNP_NCONMAX = 192
_PNP_NJMAX = 384

_TARGET_Z = 0.001


def _resolve_color(colors) -> str:
    return colors if isinstance(colors, str) else colors[0]


def _build_scene_xml(
    ground_rgba: list[float],
    cube_half_size: float,
    cube_mass: float,
    cube_rgba: list[float],
    target_rgba: list[float],
    target_disc_radius: float,
) -> str:
    """Build the pick-and-place MJCF: robot + floor + freejoint cube + mocap disc."""
    gr, gg, gb, ga = ground_rgba
    cr, cg, cb, ca = cube_rgba
    tr, tg, tb, ta = target_rgba
    hs = cube_half_size
    return f"""\
<mujoco model="pick_and_place_scene">
  <compiler angle="radian"/>

  <include file="{_SO101_XML}"/>
  {WARP_SCENE_OPTION_XML}

  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>
    <body name="cube" pos="0.15 0 {hs}">
      <freejoint name="cube_joint"/>
      <geom name="cube_geom" type="box" size="{hs} {hs} {hs}"
            rgba="{cr} {cg} {cb} {ca}" mass="{cube_mass}"
            contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
    <body name="target" pos="0.2 0 {_TARGET_Z}" mocap="true">
      <geom name="target_disc" type="cylinder" size="{target_disc_radius} 0.001"
            rgba="{tr} {tg} {tb} {ta}" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class WarpPickAndPlaceVectorEnv(SO101NexusWarpVectorEnv):
    """Batched pick-and-place: move the cube onto the goal disc and settle.

    Default obs (24,): end_effector_pose(7) + grasp_state(1) + target_position(3)
    + object_pose(7) + object_offset(3) + target_offset(3), matching
    ``MuJoCoPickAndPlace-v1``.
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
    ) -> None:
        if config is None:
            config = PickAndPlaceConfig()
        self.cube_color_name = _resolve_color(config.cube_colors)
        self.target_color_name = _resolve_color(config.target_colors)
        xml_string = _build_scene_xml(
            sample_color(config.ground_colors),
            config.cube_half_size,
            config.cube_mass,
            COLOR_MAP[self.cube_color_name],
            COLOR_MAP[self.target_color_name],
            config.target_disc_radius,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            mjm = mujoco.MjModel.from_xml_path(f.name)
        super().__init__(
            num_envs=num_envs,
            config=config,
            mjm=mjm,
            control_mode=control_mode,
            device=device,
            max_episode_steps=max_episode_steps,
            seed=seed,
            nconmax=_PNP_NCONMAX if nconmax is None else nconmax,
            njmax=_PNP_NJMAX if njmax is None else njmax,
        )
        cube_jid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        cube_qadr = int(mjm.jnt_qposadr[cube_jid])
        cube_dadr = int(mjm.jnt_dofadr[cube_jid])
        self._cube_qadr = cube_qadr
        self._cube_pos_cols = torch.arange(cube_qadr, cube_qadr + 3, device=self.device)
        self._cube_quat_cols = torch.arange(cube_qadr + 3, cube_qadr + 7, device=self.device)
        self._cube_dof_cols = torch.arange(cube_dadr, cube_dadr + 6, device=self.device)
        cube_geom = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self._obj_geom = torch.full((num_envs,), cube_geom, dtype=torch.long, device=self.device)
        target_bid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_BODY, "target")
        self._target_mocap_id = int(mjm.body_mocapid[target_bid])
        self._mocap_pos = wp.to_torch(self.data.mocap_pos)  # (N, nmocap, 3)
        self._initial_obj_z = torch.zeros(num_envs, device=self.device)
        self.task_description = config.task_description

    def _supported_obs_components(self) -> set[type]:
        return {ObjectPose, ObjectOffset, TargetPosition, TargetOffset}

    def _cube_pose7(self) -> torch.Tensor:
        return self.qpos[:, self._cube_qadr : self._cube_qadr + 7]

    def _cube_pos(self) -> torch.Tensor:
        return self.qpos[:, self._cube_qadr : self._cube_qadr + 3]

    def _target_pos(self) -> torch.Tensor:
        return self._mocap_pos[:, self._target_mocap_id, :]

    def _sample_polar(self, n: int) -> torch.Tensor:
        cfg = self.config
        r = (
            torch.rand(n, generator=self._generator, device=self.device)
            * (cfg.spawn_max_radius - cfg.spawn_min_radius)
            + cfg.spawn_min_radius
        )
        half_ang = torch.deg2rad(torch.tensor(cfg.spawn_angle_half_range_deg, device=self.device))
        theta = (
            torch.rand(n, generator=self._generator, device=self.device) * 2.0 - 1.0
        ) * half_ang
        cx, cy = cfg.spawn_center
        xy = torch.empty((n, 2), device=self.device)
        xy[:, 0] = cx + r * torch.cos(theta)
        xy[:, 1] = cy + r * torch.sin(theta)
        return xy

    def _task_reset(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        target_xy = self._sample_polar(n)
        cube_xy = self._sample_polar(n)
        # Resample cube positions that violate the minimum separation, matching
        # the MuJoCo rejection loop (target fixed, cube resampled).
        sep = self.config.min_cube_target_separation
        for _ in range(100):
            bad = torch.linalg.norm(cube_xy - target_xy, dim=1) < sep
            k = int(bad.sum())
            if k == 0:
                break
            cube_xy[bad] = self._sample_polar(k)

        self._mocap_pos[idx, self._target_mocap_id, 0] = target_xy[:, 0]
        self._mocap_pos[idx, self._target_mocap_id, 1] = target_xy[:, 1]
        self._mocap_pos[idx, self._target_mocap_id, 2] = _TARGET_Z

        yaw = torch.rand(n, generator=self._generator, device=self.device) * 2.0 * torch.pi
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 0] = torch.cos(yaw / 2.0)
        quat[:, 3] = torch.sin(yaw / 2.0)
        rows = idx[:, None]
        pos = torch.empty((n, 3), device=self.device)
        pos[:, :2] = cube_xy
        pos[:, 2] = self.config.cube_half_size
        self.qpos[rows, self._cube_pos_cols] = pos
        self.qpos[rows, self._cube_quat_cols] = quat
        self.qvel[rows, self._cube_dof_cols] = 0.0
        self._initial_obj_z[idx] = self.config.cube_half_size

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        self._initial_obj_z[idx] = self._cube_pos()[idx, 2]

    def _get_component_data(self, component: object) -> torch.Tensor:
        handlers = {
            ObjectPose: self._cube_pose7,
            ObjectOffset: lambda: self._cube_pos() - self._tcp_pos(),
            TargetPosition: self._target_pos,
            TargetOffset: lambda: self._target_pos() - self._cube_pos(),
        }
        for comp_type, handler in handlers.items():
            if isinstance(component, comp_type):
                return handler()
        return super()._get_component_data(component)

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        cube_pos = self._cube_pos()
        target_pos = self._target_pos()
        tcp_to_obj = torch.linalg.norm(cube_pos - self._tcp_pos(), dim=1)
        obj_to_target = torch.linalg.norm(cube_pos[:, :2] - target_pos[:, :2], dim=1)
        is_grasped = self._is_grasping()
        grasped = is_grasped > 0.5
        is_obj_placed = (obj_to_target <= self.config.goal_thresh) & (
            cube_pos[:, 2] < self.config.cube_half_size + 0.01
        )
        is_robot_static = self._is_robot_static()
        success = is_obj_placed & is_robot_static
        scale = self.config.reward.tanh_shaping_scale
        # Placement progress counts only while grasped (mirrors MuJoCo PickAndPlace).
        placement = reach_progress(obj_to_target, scale=scale) * grasped.to(torch.float32)
        reward = self.config.reward.compute(
            reach_progress=reach_progress(tcp_to_obj, scale=scale),
            is_grasped=is_grasped,
            task_progress=placement,
            is_complete=success,
            action_delta_norm=action_delta_norm,
            energy_norm=energy_norm,
        )
        info = {
            "obj_to_target_dist": obj_to_target,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "lift_height": cube_pos[:, 2] - self._initial_obj_z,
            "success": success,
            "tcp_to_obj_dist": tcp_to_obj,
        }
        return reward.to(torch.float32), success, info
