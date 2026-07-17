"""MuJoCo pick-and-place environment.

The carried object is chosen per episode from a compiled object pool (shared with
the unified pick env via ``so101_nexus.object_slots``); the goal is a visible,
non-colliding disc whose colour is randomized per episode. Supported carried
objects: ``CubeObject``, ``YCBObject``, ``MeshObject``.
"""

from __future__ import annotations

import tempfile
from typing import ClassVar

import mujoco
import numpy as np

from so101_nexus import (
    ensure_ycb_assets,
    get_so101_mujoco_model_dir,
    get_so101_mujoco_model_path,
)
from so101_nexus.config import (
    ControlMode,
    PickAndPlaceConfig,
    describe_place_target,
)
from so101_nexus.constants import COLOR_MAP, sample_color_name
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus.mujoco.spawn_utils import hide_freejoint_slot, place_freejoint_slot
from so101_nexus.object_slots import ObjectSlot, build_object_scene_xml, extract_object_slots
from so101_nexus.objects import CubeObject, YCBObject
from so101_nexus.rewards import (
    place_grasp_potential,
    place_reach_potential,
    place_task_potential,
    potential_shaping,
)
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()

# Object placed (not lifted) when within this vertical slack of its rest height.
_PLACE_Z_SLACK = 0.01
_TARGET_Z = 0.001


def _target_disc_body(target_disc_radius: float, target_rgba: list[float]) -> str:
    tr, tg, tb, ta = target_rgba
    return (
        f'    <body name="target" pos="0.15 0 {_TARGET_Z}">\n'
        f'      <geom name="target_disc" type="cylinder" size="{target_disc_radius} 0.001"\n'
        f'            rgba="{tr} {tg} {tb} {ta}" contype="0" conaffinity="0"/>\n'
        f"    </body>\n"
    )


class PickAndPlaceEnv(SO101NexusMuJoCoBaseEnv):
    """Pick-and-place environment: carry a pooled object onto a goal disc."""

    config: PickAndPlaceConfig
    default_config_cls: ClassVar[type[PickAndPlaceConfig]] = PickAndPlaceConfig

    def __init__(
        self,
        config: PickAndPlaceConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        if config is None:
            config = PickAndPlaceConfig()
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        scene_objects = config.object_pool()
        for obj in scene_objects:
            if isinstance(obj, YCBObject):
                ensure_ycb_assets(obj.model_id)
        slot_names = [f"pick_slot_{i}" for i in range(len(scene_objects))]

        self.cube_half_size = config.cube_half_size
        self.target_disc_radius = config.target_disc_radius
        # First configured colours seed the compiled model; the disc colour is
        # re-sampled per episode (geom_rgba is per-geom in the scalar backend).
        first = scene_objects[0]
        self.cube_color_name = first.color if isinstance(first, CubeObject) else ""
        self.target_color_name = (
            config.target_colors
            if isinstance(config.target_colors, str)
            else config.target_colors[0]
        )

        ground_name = (
            config.ground_colors
            if isinstance(config.ground_colors, str)
            else config.ground_colors[0]
        )
        xml_string = build_object_scene_xml(
            scene_objects,
            slot_names,
            COLOR_MAP[ground_name],
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
            model_name="pick_and_place_scene",
            extra_bodies=_target_disc_body(
                config.target_disc_radius, COLOR_MAP[self.target_color_name]
            ),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        self._slots: list[ObjectSlot] = extract_object_slots(self.model, slot_names, scene_objects)
        self._target_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "target_disc"
        )
        self._target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")

        self._target_slot_idx: int = 0
        self._initial_obj_z: float = 0.0
        self._prev_task_potential: float = 0.0
        self._prev_reach_progress: float = 0.0
        self._prev_grasp_progress: float = 0.0
        self._obj_geom_id: int = self._slots[0].geom_id
        self.task_description = config.task_description

        self._finish_model_setup()

    def _get_object_pose(self) -> np.ndarray:
        addr = self._slots[self._target_slot_idx].qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_target_pos(self) -> np.ndarray:
        return self.data.xpos[self._target_body_id].copy()

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus.observations import (
            ObjectOffset as _ObjectOffset,
        )
        from so101_nexus.observations import (
            ObjectPose as _ObjectPose,
        )
        from so101_nexus.observations import (
            TargetOffset as _TargetOffset,
        )
        from so101_nexus.observations import (
            TargetPosition as _TargetPosition,
        )

        if isinstance(component, _ObjectPose):
            return self._get_object_pose()
        if isinstance(component, _ObjectOffset):
            return self._get_object_pose()[:3] - self._get_tcp_pose()[:3]
        if isinstance(component, _TargetPosition):
            return self._get_target_pos()
        if isinstance(component, _TargetOffset):
            return self._get_target_pos() - self._get_object_pose()[:3]
        return super()._get_component_data(component)

    def _obj_placement_state(
        self, obj_pos: np.ndarray, target_pos: np.ndarray
    ) -> tuple[float, bool]:
        """Return ``(obj_to_target_xy_dist, is_obj_placed)`` for the given pose."""
        obj_to_target_dist = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
        is_obj_placed = (
            obj_to_target_dist <= self.config.goal_thresh
            and obj_pos[2] < self._initial_obj_z + _PLACE_Z_SLACK
        )
        return obj_to_target_dist, is_obj_placed

    def _task_potential(
        self, obj_pos: np.ndarray, target_pos: np.ndarray, is_grasped: float, is_obj_placed: bool
    ) -> float:
        """``Phi_place(s)``: staged transport-then-settle progress toward completion.

        Thin physics-query wrapper around ``rewards.place_task_potential``,
        which is monotone non-decreasing along the ideal grasp-lift-carry-
        lower-settle trajectory so ``rewards.potential_shaping`` pays a
        positive delta for forward progress and ~0 for dwelling at any fixed
        state. See docs/superpowers/plans/2026-07-16-monotone-place-potential.md.
        """
        obj_to_target_dist, _ = self._obj_placement_state(obj_pos, target_pos)
        height_gap = max(0.0, float(obj_pos[2]) - self._initial_obj_z)
        arm_speed = float(np.linalg.norm(self.data.qvel[self._arm_qvel_addrs]))
        return place_task_potential(
            obj_to_target_dist,
            height_gap,
            arm_speed,
            is_grasped,
            is_obj_placed,
            scale=self.config.reward.tanh_shaping_scale,
            velocity_scale=self.config.reward.velocity_shaping_scale,
        )

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        obj_pos = self._get_object_pose()[:3]
        target_pos = self._get_target_pos()
        is_grasped = self._is_grasping()

        obj_to_target_dist, is_obj_placed = self._obj_placement_state(obj_pos, target_pos)
        is_robot_static = self._is_robot_static()
        lift_height = float(obj_pos[2] - self._initial_obj_z)
        success = is_obj_placed and is_robot_static

        info = {
            "obj_to_target_dist": obj_to_target_dist,
            "is_obj_placed": is_obj_placed,
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "lift_height": lift_height,
            "success": success,
            "tcp_to_obj_dist": float(np.linalg.norm(obj_pos - tcp_pos)),
            "task_potential": self._task_potential(obj_pos, target_pos, is_grasped, is_obj_placed),
        }
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        scale = self.config.reward.tanh_shaping_scale
        # reaching/grasping are potential-shaped deltas, not raw state values
        # (dwelling at "reached and grasped, never placed" must pay ~0/step, see
        # docs/superpowers/plans/2026-07-16-pick-grasp-potential-shaping.md), and
        # both potentials are held up by is_obj_placed so the mandatory release
        # on the goal and the post-place retreat pay no negative delta (see
        # docs/superpowers/plans/2026-07-16-monotone-place-potential.md).
        reach_now = place_reach_potential(
            info["tcp_to_obj_dist"], info["is_obj_placed"], scale=scale
        )
        grasp_now = place_grasp_potential(info["is_grasped"], info["is_obj_placed"])
        reach_delta = potential_shaping(reach_now, self._prev_reach_progress)
        grasp_delta = potential_shaping(grasp_now, self._prev_grasp_progress)
        self._prev_reach_progress = reach_now
        self._prev_grasp_progress = grasp_now
        # task_progress is a potential-based delta (Ng, Harada & Russell, ICML
        # 1999; see _task_potential), not the raw potential -- dwelling at any
        # fixed state pays ~0 per step instead of the potential's full value.
        task_potential = info["task_potential"]
        task_progress = potential_shaping(task_potential, self._prev_task_potential)
        self._prev_task_potential = task_potential
        components = self.config.reward.compute_components(
            reach_progress=reach_delta,
            is_grasped=grasp_delta,
            task_progress=task_progress,
            is_complete=info["success"],
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
        info["reward_components"] = components
        return sum(components.values())

    def _refresh_reset_reference_state(self) -> None:
        """Refresh the placement, reach, and grasp baselines from the post-settle pose."""
        self._initial_obj_z = float(self._get_object_pose()[2])
        obj_pos = self._get_object_pose()[:3]
        target_pos = self._get_target_pos()
        is_grasped = self._is_grasping()
        _, is_obj_placed = self._obj_placement_state(obj_pos, target_pos)
        self._prev_task_potential = self._task_potential(
            obj_pos, target_pos, is_grasped, is_obj_placed
        )
        scale = self.config.reward.tanh_shaping_scale
        tcp_to_obj_dist = float(np.linalg.norm(obj_pos - self._get_tcp_pose()[:3]))
        self._prev_reach_progress = place_reach_potential(
            tcp_to_obj_dist, is_obj_placed, scale=scale
        )
        self._prev_grasp_progress = place_grasp_potential(is_grasped, is_obj_placed)

    def _task_reset(self) -> None:
        rng = self.np_random
        n_pool = len(self._slots)

        # Restore collisions on every slot; the chosen target collides, the rest
        # are hidden below the floor with their contact bits zeroed.
        for slot in self._slots:
            self.model.geom_contype[slot.geom_id] = 1
            self.model.geom_conaffinity[slot.geom_id] = 1

        target_idx = int(rng.choice(n_pool))
        target_slot = self._slots[target_idx]
        target_obj = target_slot.obj
        self._target_slot_idx = target_idx
        self._obj_geom_id = target_slot.geom_id
        self.cube_color_name = target_obj.color if isinstance(target_obj, CubeObject) else ""

        # The disc colour is sampled per episode (reproducible under reset(seed=...)).
        self.target_color_name = sample_color_name(self.config.target_colors, rng)
        self.model.geom_rgba[self._target_geom_id] = COLOR_MAP[self.target_color_name]
        self.task_description = describe_place_target(target_obj, self.target_color_name)

        min_r = self.config.spawn_min_radius
        max_r = self.config.spawn_max_radius
        angle_half = float(np.radians(self.config.spawn_angle_half_range_deg))
        cx, cy = self.config.spawn_center

        r_t = rng.uniform(min_r, max_r)
        theta_t = rng.uniform(-angle_half, angle_half)
        target_x = cx + r_t * np.cos(theta_t)
        target_y = cy + r_t * np.sin(theta_t)
        self.model.body_pos[self._target_body_id] = [target_x, target_y, _TARGET_Z]

        sep = self.config.min_object_target_separation + target_slot.bounding_radius
        obj_x, obj_y = target_x, target_y
        for _ in range(100):
            r_c = rng.uniform(min_r, max_r)
            theta_c = rng.uniform(-angle_half, angle_half)
            obj_x = cx + r_c * np.cos(theta_c)
            obj_y = cy + r_c * np.sin(theta_c)
            if np.hypot(obj_x - target_x, obj_y - target_y) >= sep:
                break

        place_freejoint_slot(self.model, self.data, target_slot, rng, (obj_x, obj_y))
        for idx, slot in enumerate(self._slots):
            if idx != target_idx:
                hide_freejoint_slot(self.model, self.data, slot)

        self._initial_obj_z = target_slot.spawn_z
