"""GPU-batched pick-lift environment for SO-101 on MuJoCo Warp.

Supports a single ``CubeObject`` target. The batched Warp backend shares one
model across all worlds, so per-world target-geom selection, distractor hiding
(``model`` contact bits are global), and per-episode color randomization
(``geom_rgba`` is global) are out of scope; multi-object pools and YCB/Mesh
objects raise ``NotImplementedError``. The default ``PickConfig`` (one cube, no
distractors) is fully supported, matching ``MuJoCoPickLift-v1``.
"""

from __future__ import annotations

import tempfile

import mujoco
import torch

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, PickConfig, describe_pick_target
from so101_nexus.constants import COLOR_MAP, sample_color
from so101_nexus.objects import CubeObject
from so101_nexus.observations import ObjectOffset, ObjectPose
from so101_nexus.rewards import lift_progress, reach_progress
from so101_nexus.scene import WARP_SCENE_OPTION_XML
from so101_nexus.warp.base_env import SO101NexusWarpVectorEnv

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

# Contact-rich scene (cube + gripper + floor); auto-sizing overflows under active
# control, so size generously. naconmax = nconmax * num_envs is sized from these.
_PICK_NCONMAX = 192
_PICK_NJMAX = 384


def _build_pick_scene_xml(ground_rgba: list[float], cube: CubeObject) -> str:
    """Build the pick-scene MJCF: robot + floor + one freejoint cube."""
    gr, gg, gb, ga = ground_rgba
    hs = cube.half_size
    r, g, b, a = COLOR_MAP[cube.color]
    return f"""\
<mujoco model="pick_scene">
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
            rgba="{r} {g} {b} {a}" mass="{cube.mass}"
            contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


class WarpPickLiftVectorEnv(SO101NexusWarpVectorEnv):
    """Batched pick-lift: grasp the cube and lift it above ``lift_threshold``.

    Default obs (18,): end_effector_pose(7) + grasp_state(1) + object_pose(7) +
    object_offset(3), matching ``MuJoCoPickLift-v1``.
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
    ) -> None:
        if config is None:
            config = PickConfig()
        if len(config.objects) != 1 or not isinstance(config.objects[0], CubeObject):
            raise NotImplementedError(
                "WarpPickLift supports a single CubeObject target; multi-object pools "
                "and YCB/Mesh objects are not supported on the batched Warp backend"
            )
        if config.n_distractors != 0:
            raise NotImplementedError(
                "WarpPickLift does not support distractors on the batched Warp backend"
            )
        self._cube: CubeObject = config.objects[0]
        ground_rgba = sample_color(config.ground_colors)
        xml_string = _build_pick_scene_xml(ground_rgba, self._cube)
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
            nconmax=_PICK_NCONMAX if nconmax is None else nconmax,
            njmax=_PICK_NJMAX if njmax is None else njmax,
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
        self._initial_obj_z = torch.zeros(num_envs, device=self.device)
        self.task_description = describe_pick_target(self._cube)

    def _supported_obs_components(self) -> set[type]:
        return {ObjectPose, ObjectOffset}

    def _cube_pose7(self) -> torch.Tensor:
        return self.qpos[:, self._cube_qadr : self._cube_qadr + 7]

    def _cube_pos(self) -> torch.Tensor:
        return self.qpos[:, self._cube_qadr : self._cube_qadr + 3]

    def _task_reset(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
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
        pos = torch.empty((n, 3), device=self.device)
        pos[:, 0] = cx + r * torch.cos(theta)
        pos[:, 1] = cy + r * torch.sin(theta)
        pos[:, 2] = self._cube.half_size
        yaw = torch.rand(n, generator=self._generator, device=self.device) * 2.0 * torch.pi
        quat = torch.zeros((n, 4), device=self.device)
        quat[:, 0] = torch.cos(yaw / 2.0)
        quat[:, 3] = torch.sin(yaw / 2.0)
        rows = idx[:, None]
        self.qpos[rows, self._cube_pos_cols] = pos
        self.qpos[rows, self._cube_quat_cols] = quat
        self.qvel[rows, self._cube_dof_cols] = 0.0
        self._initial_obj_z[idx] = self._cube.half_size

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        self._initial_obj_z[idx] = self._cube_pos()[idx, 2]

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, ObjectPose):
            return self._cube_pose7()
        if isinstance(component, ObjectOffset):
            return self._cube_pos() - self._tcp_pos()
        return super()._get_component_data(component)

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        cube_pos = self._cube_pos()
        tcp_pos = self._tcp_pos()
        tcp_to_obj = torch.linalg.norm(cube_pos - tcp_pos, dim=1)
        is_grasped = self._is_grasping()
        lift_height = cube_pos[:, 2] - self._initial_obj_z
        grasped = is_grasped > 0.5
        success = (lift_height > self.config.lift_threshold) & grasped
        # Shared reward source (so101_nexus.rewards + RewardConfig.compute).
        scale = self.config.reward.tanh_shaping_scale
        reward = self.config.reward.compute(
            reach_progress=reach_progress(tcp_to_obj, scale=scale),
            is_grasped=is_grasped,
            task_progress=lift_progress(lift_height, scale=scale, grasped=grasped),
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
