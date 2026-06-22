"""GPU-batched look-at environment for SO-101 on MuJoCo Warp."""

from __future__ import annotations

import tempfile

import mujoco
import numpy as np
import torch
import warp as wp

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, LookAtConfig
from so101_nexus.constants import sample_color
from so101_nexus.objects import CubeObject
from so101_nexus.observations import GazeDirection
from so101_nexus.rewards import orientation_progress, simple_reward
from so101_nexus.scene import WARP_SCENE_OPTION_XML, build_robot_floor_scene_xml
from so101_nexus.warp.base_env import SO101NexusWarpVectorEnv

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

# Contact-free scene (robot + floor); mujoco_warp auto-sizing overflows under
# active control, so size generously.
_LOOK_AT_NCONMAX = 128
_LOOK_AT_NJMAX = 256


class WarpLookAtVectorEnv(SO101NexusWarpVectorEnv):
    """Batched look-at primitive: orient every world's TCP toward a target object.

    The target is a position sampled in the spawn square and stored as a tensor
    (the MuJoCo backend's kinematic cube is purely visual, and the Warp backend
    does not render). Default obs (6,): joint_positions, matching
    ``MuJoCoLookAt-v1``. Add ``GazeDirection`` to make the target observable.
    """

    config: LookAtConfig

    def __init__(
        self,
        num_envs: int,
        config: LookAtConfig | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        device: str = "cuda",
        max_episode_steps: int = 256,
        seed: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
    ) -> None:
        if config is None:
            config = LookAtConfig()
        ground_rgba = sample_color(config.ground_colors)
        xml_string = build_robot_floor_scene_xml(
            ground_rgba,
            option_xml=WARP_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            mjm = mujoco.MjModel.from_xml_path(f.name)
        self._wrist_cam_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")
        super().__init__(
            num_envs=num_envs,
            config=config,
            mjm=mjm,
            control_mode=control_mode,
            device=device,
            max_episode_steps=max_episode_steps,
            seed=seed,
            nconmax=_LOOK_AT_NCONMAX if nconmax is None else nconmax,
            njmax=_LOOK_AT_NJMAX if njmax is None else njmax,
        )
        self._targets = torch.zeros((num_envs, 3), device=self.device)
        target = config.objects[0]
        assert isinstance(target, CubeObject)
        self._spawn_z = float(target.half_size)
        cx, cy = config.spawn_center
        self._spawn_center = torch.as_tensor([cx, cy], device=self.device)
        # Zero-copy view of camera rotation matrices: (num_envs, ncam, 3, 3).
        self._cam_xmat = wp.to_torch(self.data.cam_xmat)
        # In-frame boundary: half the wrist-camera vertical FOV. Warp does not
        # render or randomize the camera, so the static model fovy is the live
        # value; config.fov_deg overrides it when pinned.
        fovy = (
            config.fov_deg
            if config.fov_deg is not None
            else float(mjm.cam_fovy[self._wrist_cam_id])
        )
        self._success_half_fov_rad = float(np.radians(fovy) / 2.0)
        self.task_descriptions = [config.task_description] * num_envs

    def _supported_obs_components(self) -> set[type]:
        return {GazeDirection}

    def _task_reset(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        half = self.config.spawn_half_size
        xy = (
            self._spawn_center
            + (torch.rand((n, 2), generator=self._generator, device=self.device) * 2.0 - 1.0) * half
        )
        self._targets[idx, :2] = xy
        self._targets[idx, 2] = self._spawn_z

    def _gaze_dir(self) -> torch.Tensor:
        """Return the unit vector from each TCP toward its target (GazeDirection)."""
        to_target = self._targets - self._tcp_pos()
        norm = torch.linalg.norm(to_target, dim=1, keepdim=True)
        return to_target / norm.clamp(min=1e-8)

    def _gaze_axis(self) -> torch.Tensor:
        """Return the wrist-camera optical axis in world frame, per world."""
        # MuJoCo cameras look along local -z; the optical axis is the negative
        # third column of the camera rotation matrix.
        return -self._cam_xmat[:, self._wrist_cam_id, :, 2]

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, GazeDirection):
            return self._gaze_dir()
        return super()._get_component_data(component)

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        cos_sim = (self._gaze_axis() * self._gaze_dir()).sum(dim=1)
        orientation_error = torch.arccos(cos_sim.clamp(-1.0, 1.0))
        success = orientation_error <= self._success_half_fov_rad
        # orientation_progress(cos(error)) == orientation_progress(cos_sim).
        progress = orientation_progress(cos_sim)
        base = simple_reward(
            progress=progress,
            completion_bonus=self.config.reward.completion_bonus,
            success=success,
        )
        reward = self.config.reward.apply_penalties(
            base, action_delta_norm=action_delta_norm, energy_norm=energy_norm
        )
        info = {"orientation_error": orientation_error, "success": success}
        return reward.to(torch.float32), success, info
