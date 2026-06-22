"""GPU-batched directional move environment for SO-101 on MuJoCo Warp."""

from __future__ import annotations

import tempfile

import mujoco
import torch

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import DIRECTION_VECTORS, ControlMode, MoveConfig
from so101_nexus.constants import sample_color
from so101_nexus.observations import TargetOffset
from so101_nexus.rewards import reach_progress, simple_reward
from so101_nexus.scene import WARP_SCENE_OPTION_XML, build_robot_floor_scene_xml
from so101_nexus.warp.base_env import SO101NexusWarpVectorEnv

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

# Contact-free scene (robot + floor); mujoco_warp auto-sizing overflows once an
# active policy drives the arm into the floor and joint limits, so size generously.
_MOVE_NCONMAX = 128
_MOVE_NJMAX = 256

# Floor clearance for the move target (cross-backend contract): downward moves
# clamp here, so the initial distance may be < target_distance.
_TARGET_MIN_Z = 0.02


class WarpMoveVectorEnv(SO101NexusWarpVectorEnv):
    """Batched move primitive: translate every world's TCP a fixed distance.

    The per-world target is the post-settle TCP plus the configured direction
    vector scaled by ``target_distance`` (settle-then-place contract), stored as
    a tensor rather than a placed site since the Warp backend does not render.
    Default obs (6,): joint_positions, matching ``MuJoCoMove-v1``. Add
    ``TargetOffset`` in the config to make the target observable.
    """

    config: MoveConfig

    def __init__(
        self,
        num_envs: int,
        config: MoveConfig | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        device: str = "cuda",
        max_episode_steps: int = 256,
        seed: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
    ) -> None:
        if config is None:
            config = MoveConfig()
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
        super().__init__(
            num_envs=num_envs,
            config=config,
            mjm=mjm,
            control_mode=control_mode,
            device=device,
            max_episode_steps=max_episode_steps,
            seed=seed,
            nconmax=_MOVE_NCONMAX if nconmax is None else nconmax,
            njmax=_MOVE_NJMAX if njmax is None else njmax,
        )
        self._targets = torch.zeros((num_envs, 3), device=self.device)
        self._start_pos = torch.zeros((num_envs, 3), device=self.device)
        self._target_displacement = torch.zeros(num_envs, device=self.device)
        self._dir_vec = torch.as_tensor(
            DIRECTION_VECTORS[config.direction], dtype=torch.float32, device=self.device
        )

        self.task_descriptions = [config.task_description] * num_envs

    def _supported_obs_components(self) -> set[type]:
        return {TargetOffset}

    def _refresh_reset_reference_state(self, mask: torch.Tensor) -> None:
        # Place each reset world's target target_distance from the post-forward
        # (settle-independent) TCP along the move direction, z clamped above the
        # floor (cross-backend). Captured pre-settle so reset() and same-step
        # autoreset place identical targets.
        idx = mask.nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            return
        tcp = self._tcp_pos()[idx]
        self._start_pos[idx] = tcp
        target = tcp + self._dir_vec * self.config.target_distance
        target[:, 2] = torch.clamp(target[:, 2], min=_TARGET_MIN_Z)
        self._targets[idx] = target
        # Compare success against the clamped (reachable) displacement along the
        # move direction, not the raw configured distance: a downward move clamped
        # above the floor cannot reach the full target_distance.
        self._target_displacement[idx] = ((target - tcp) * self._dir_vec).sum(dim=1)

    def _task_reset(self, mask: torch.Tensor) -> None:
        # Target depends on the post-forward TCP; placed in _refresh_reset_reference_state.
        pass

    def _get_component_data(self, component: object) -> torch.Tensor:
        if isinstance(component, TargetOffset):
            return self._targets - self._tcp_pos()
        return super()._get_component_data(component)

    def _compute_reward_terminated(
        self, energy_norm: torch.Tensor, action_delta_norm: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        dist = torch.linalg.norm(self._targets - self._tcp_pos(), dim=1)
        # Tensor path of so101_nexus.rewards.reach_progress / simple_reward.
        progress = reach_progress(dist, scale=self.config.reward.tanh_shaping_scale)
        displacement = ((self._tcp_pos() - self._start_pos) * self._dir_vec).sum(dim=1)
        success = displacement >= self._target_displacement - self.config.success_threshold
        base = simple_reward(
            progress=progress,
            completion_bonus=self.config.reward.completion_bonus,
            success=success,
        )
        reward = self.config.reward.apply_penalties(
            base, action_delta_norm=action_delta_norm, energy_norm=energy_norm
        )
        info = {
            "tcp_to_target_dist": dist,
            "move_displacement": displacement,
            "success": success,
        }
        return reward.to(torch.float32), success, info
