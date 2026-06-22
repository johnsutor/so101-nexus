"""MuJoCo unified pick environment.

Provides ``PickEnv`` (reach-only reward) and ``PickLiftEnv`` (lift-to-success)
backed by a MuJoCo scene built dynamically from a ``PickConfig`` object list.
The shared object-slot machinery (XML builders, ``ObjectSlot`` metadata) lives
in ``so101_nexus.object_slots``.

Supported object types: ``CubeObject``, ``YCBObject``, ``MeshObject``.
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
    PickConfig,
    describe_pick_target,
)
from so101_nexus.constants import sample_color
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus.mujoco.spawn_utils import (
    hide_freejoint_slot,
    place_freejoint_slot,
    sample_separated_positions,
)
from so101_nexus.object_slots import (
    ObjectSlot,
    build_object_scene_xml,
    extract_object_slots,
)
from so101_nexus.objects import SceneObject, YCBObject
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()


class PickEnv(SO101NexusMuJoCoBaseEnv):
    """Unified MuJoCo pick environment with reach-only reward.

    Handles ``CubeObject``, ``YCBObject``, and ``MeshObject`` from
    ``PickConfig.objects``. One object is randomly chosen as the target per
    episode; ``config.n_distractors`` others are placed as distractors.

    Observation (18-dim):
        tcp_pos(3) + tcp_quat(4) + is_grasped(1) + obj_pos(3) + obj_quat(4)
        + tcp_to_obj(3)
    """

    config: PickConfig
    default_config_cls: ClassVar[type[PickConfig]] = PickConfig

    def __init__(
        self,
        config: PickConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = PickConfig()
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        self._n_distractors = config.n_distractors
        # Build the XML for ALL objects in the pool so any object can be
        # selected as the target or a distractor at reset time.
        scene_objects = list(config.objects)
        n_pool = len(scene_objects)
        n_slots = 1 + self._n_distractors

        # Ensure YCB assets are on disk before building XML
        for obj in scene_objects:
            if isinstance(obj, YCBObject):
                ensure_ycb_assets(obj.model_id)

        # Body slot names: one per pool object (not just active slots).
        # Slots beyond n_slots will be hidden off-world at reset time.
        slot_names = [f"pick_slot_{i}" for i in range(n_pool)]

        xml_string = build_object_scene_xml(
            scene_objects,
            slot_names,
            sample_color(config.ground_colors),
            option_xml=MUJOCO_SCENE_OPTION_XML,
            robot_xml_path=str(_SO101_XML),
            model_name="pick_scene",
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        # Per-slot runtime metadata (one entry per pool object).
        self._slots: list[ObjectSlot] = extract_object_slots(self.model, slot_names, scene_objects)

        # n_slots = active slots (target + distractors); the rest are hidden.
        self._n_slots = n_slots
        # Bookkeeping updated at each _task_reset
        self._target_slot_idx: int = 0
        self._task_description: str = ""
        self._initial_obj_z: float = 0.0
        # _obj_geom_id required by base _is_grasping(); will be set at reset
        self._obj_geom_id: int = self._slots[0].geom_id

        self._finish_model_setup()

    @property
    def task_description(self) -> str:
        """Return the current episode task description."""
        return self._task_description

    def _get_target_pose(self) -> np.ndarray:
        slot = self._slots[self._target_slot_idx]
        addr = slot.qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _describe_target(self, target_obj: SceneObject) -> str:
        """Return the task description for the chosen target (overridable per task)."""
        return describe_pick_target(target_obj)

    def _target_bounding_radius(self) -> float:
        """Return the horizontal bounding radius of the current target object."""
        return self._slots[self._target_slot_idx].bounding_radius

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus.observations import ObjectOffset as _ObjectOffset
        from so101_nexus.observations import ObjectPose as _ObjectPose

        if isinstance(component, _ObjectPose):
            return self._get_target_pose()
        if isinstance(component, _ObjectOffset):
            tcp_pos = self._get_tcp_pose()[:3]
            obj_pos = self._get_target_pose()[:3]
            return obj_pos - tcp_pos
        return super()._get_component_data(component)

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        obj_pose = self._get_target_pose()
        obj_pos = obj_pose[:3]
        is_grasped = self._is_grasping()
        lift_height = float(obj_pos[2] - self._initial_obj_z)

        info = {
            "is_grasped": is_grasped,
            "is_robot_static": self._is_robot_static(),
            "lift_height": lift_height,
            "tcp_to_obj_dist": float(np.linalg.norm(obj_pos - tcp_pos)),
        }
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        return self._reach_only_reward(info)

    def _refresh_reset_reference_state(self) -> None:
        """Refresh lift baseline from the post-settle target object pose."""
        self._initial_obj_z = float(self._get_target_pose()[2])

    def _task_reset(self) -> None:
        rng = self.np_random
        min_r = self.config.spawn_min_radius
        max_r = self.config.spawn_max_radius
        angle_half = float(np.radians(self.config.spawn_angle_half_range_deg))

        n_pool = len(self._slots)
        n_slots = self._n_slots  # number of active slots (target + distractors)

        # Restore collisions on every slot at the start of each reset. Slots
        # that remain unchosen below are re-zeroed; slots that become active
        # need contype/conaffinity = 1 so they collide with the floor and gripper.
        for slot in self._slots:
            self.model.geom_contype[slot.geom_id] = 1
            self.model.geom_conaffinity[slot.geom_id] = 1

        # Sample n_slots distinct slot indices from the pool without replacement.
        chosen_indices = list(rng.choice(n_pool, size=n_slots, replace=False))
        target_pool_idx = int(chosen_indices[0])
        target_obj = self._slots[target_pool_idx].obj

        # The first chosen slot becomes the target; the rest are distractors.
        self._target_slot_idx = target_pool_idx
        self._obj_geom_id = self._slots[target_pool_idx].geom_id

        # Gather bounding radii only for active slots (for position sampling).
        active_bounding_radii = [self._slots[int(i)].bounding_radius for i in chosen_indices]

        positions = sample_separated_positions(
            rng,
            n_slots,
            min_r,
            max_r,
            angle_half,
            self.config.min_object_separation,
            active_bounding_radii,
            center=self.config.spawn_center,
        )

        # Place active slots at their sampled positions; park the rest off-world.
        for pos_idx, pool_idx in enumerate(chosen_indices):
            place_freejoint_slot(
                self.model, self.data, self._slots[int(pool_idx)], rng, positions[pos_idx]
            )
        unchosen = set(range(n_pool)) - {int(i) for i in chosen_indices}
        for pool_idx in unchosen:
            hide_freejoint_slot(self.model, self.data, self._slots[pool_idx])

        self._task_description = self._describe_target(target_obj)


class PickLiftEnv(PickEnv):
    """Pick-lift variant: success requires grasping and lifting the target.

    Extends ``PickEnv`` with a lift reward and ``success`` flag in the info
    dict. Uses ``config.lift_threshold`` as the minimum height.
    """

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["success"] = (info["lift_height"] > self.config.lift_threshold) and (
            info["is_grasped"] > 0.5
        )
        return info

    def _compute_reward(self, info: dict) -> float:
        return self._lift_reward(info)
