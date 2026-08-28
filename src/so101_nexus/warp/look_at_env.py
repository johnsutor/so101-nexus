"""GPU-batched look-at environment for SO-101 on MuJoCo Warp."""

from __future__ import annotations

import tempfile
from typing import ClassVar

import mujoco
import numpy as np
import torch
import warp as wp

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, LookAtConfig
from so101_nexus.constants import sample_color
from so101_nexus.gaze import gaze_angle_rad, object_in_view
from so101_nexus.object_slots import primitive_visual_xml_geom, pyramid_xml_asset
from so101_nexus.objects import PrimitiveObject, PyramidObject
from so101_nexus.observations import CameraObservation, GazeDirection, GazeState
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

    The target is a position sampled in the spawn square and stored as a tensor.
    When a camera observation is configured, a visual-only marker geom is added to
    the scene and tracked to the target so the rendered image shows it (present in
    the MuJoCo backend's image too, though the two backends' shading of it differs;
    see ``so101_nexus.warp.base_env``). Default obs (23,): joint_positions(6) +
    joint_velocities(6) + end_effector_pose(7) + gaze_direction(3) +
    gaze_state(1), matching ``MuJoCoLookAt-v1``.
    """

    config: LookAtConfig
    default_config_cls: ClassVar[type[LookAtConfig]] = LookAtConfig

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
        render_mode: str | None = None,
    ) -> None:
        if config is None:
            config = LookAtConfig()
        ground_rgba = sample_color(config.ground_colors)
        target = config.objects[0]
        assert isinstance(target, PrimitiveObject)
        marker_assets = pyramid_xml_asset(0, target) if isinstance(target, PyramidObject) else ""
        marker_xml = ""
        if any(isinstance(c, CameraObservation) for c in (config.observations or [])):
            marker_xml = primitive_visual_xml_geom("look_target", target)
        xml_string = build_robot_floor_scene_xml(
            ground_rgba,
            option_xml=WARP_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
            overhead_camera_xml=SO101NexusWarpVectorEnv._overhead_camera_xml(config),
            extra_assets=marker_assets,
            extra_bodies=marker_xml,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            mjm = mujoco.MjModel.from_xml_path(f.name)
        # In-frame boundary: half the wrist-camera vertical FOV. The base reads
        # the live per-world value, which wrist-camera domain randomization moves;
        # config.fov_deg pins it instead (see _half_fov_rad).
        self._pinned_half_fov_rad = (
            None if config.fov_deg is None else float(np.radians(config.fov_deg) / 2.0)
        )
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
            render_mode=render_mode,
        )
        self._targets = torch.zeros((num_envs, 3), device=self.device)
        self._spawn_z = float(target.half_size)
        cx, cy = config.spawn_center
        self._spawn_center = torch.as_tensor([cx, cy], device=self.device)
        self.task_descriptions = [config.task_description] * num_envs
        if self._has_cameras:
            self._marker_gid = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, "look_target")
            self._geom_xpos = wp.to_torch(self.data.geom_xpos)  # (N, ngeom, 3)

    def _supported_obs_components(self) -> set[type]:
        return {GazeDirection, GazeState}

    def _update_render_markers(self) -> None:
        self._geom_xpos[:, self._marker_gid] = self._targets

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

    def _gaze_target_pos(self) -> torch.Tensor:
        return self._targets

    def _half_fov_rad(self) -> torch.Tensor | float:
        """Half the wrist-camera vertical FOV, or ``config.fov_deg``'s pin."""
        if self._pinned_half_fov_rad is None:
            return super()._half_fov_rad()
        return self._pinned_half_fov_rad

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        cos_sim = self._gaze_cosine()
        orientation_error = gaze_angle_rad(cos_sim)
        success = object_in_view(orientation_error, self._half_fov_rad())
        progress = orientation_progress(cos_sim)
        base = simple_reward(
            progress=progress,
            completion_bonus=self.config.reward.completion_bonus,
            success=success,
        )
        reward = self.config.reward.apply_penalties(
            base, action_delta_norm=action_delta_norm, energy_norm=energy_norm, is_complete=success
        )
        info = {"orientation_error": orientation_error, "success": success}
        return reward.to(torch.float32), success, info
