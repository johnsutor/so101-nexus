"""GPU-batched reach-to-target environment for SO-101 on MuJoCo Warp."""

from __future__ import annotations

import tempfile

import mujoco
import torch

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, ReachConfig
from so101_nexus.constants import sample_color
from so101_nexus.observations import TargetOffset
from so101_nexus.rewards import reach_progress, simple_reward
from so101_nexus.scene import WARP_SCENE_OPTION_XML, build_reach_scene_xml
from so101_nexus.warp.base_env import SO101NexusWarpVectorEnv

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

# Reach workspace (cross-backend contract): center [0.15, 0, 0.15], per-axis
# offset in [-half, +half], z clamped to >= 0.05.
_WORKSPACE_CENTER = (0.15, 0.0, 0.15)

# Per-world contact/constraint buffer budgets. mujoco_warp's auto-sizing (~10*nv)
# is too small once an active policy drives the arm into the floor and joint
# limits (observed nefc overflow requesting ~78), which silently drops
# constraints and corrupts the batched physics. Size generously; override via the
# constructor for a denser scene.
_REACH_NCONMAX = 128
_REACH_NJMAX = 256


class WarpReachVectorEnv(SO101NexusWarpVectorEnv):
    """Batched reach primitive: move every world's TCP to its sampled 3-D target.

    Default obs (6,): joint_positions, matching ``MuJoCoReach-v1``. Add
    ``TargetOffset`` in the config to make the target observable.
    """

    config: ReachConfig

    task_description = "Move the robot's end-effector to the target position."

    def __init__(
        self,
        num_envs: int,
        config: ReachConfig | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        device: str = "cuda",
        max_episode_steps: int = 512,
        seed: int | None = None,
        nconmax: int | None = None,
        njmax: int | None = None,
    ) -> None:
        if config is None:
            config = ReachConfig()
        ground_rgba = sample_color(config.ground_colors)
        xml_string = build_reach_scene_xml(
            ground_rgba,
            config.target_radius,
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
            nconmax=_REACH_NCONMAX if nconmax is None else nconmax,
            njmax=_REACH_NJMAX if njmax is None else njmax,
        )
        self._targets = torch.zeros((num_envs, 3), device=self.device)
        self._center = torch.as_tensor(_WORKSPACE_CENTER, device=self.device)

    def _task_reset(self, mask: torch.Tensor) -> None:
        idx = mask.nonzero(as_tuple=True)[0]
        n = int(idx.numel())
        if n == 0:
            return
        half = self.config.target_workspace_half_extent
        offset = (
            torch.rand((n, 3), generator=self._generator, device=self.device) * 2.0 - 1.0
        ) * half
        pos = self._center + offset
        pos[:, 2] = torch.clamp(pos[:, 2], min=0.05)
        self._targets[idx] = pos

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
        success = dist < self.config.success_threshold
        base = simple_reward(
            progress=progress,
            completion_bonus=self.config.reward.completion_bonus,
            success=success,
        )
        reward = self.config.reward.apply_penalties(
            base, action_delta_norm=action_delta_norm, energy_norm=energy_norm
        )
        info = {"tcp_to_target_dist": dist, "success": success}
        return reward.to(torch.float32), success, info
