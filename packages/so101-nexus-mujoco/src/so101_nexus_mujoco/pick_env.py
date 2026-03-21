"""MuJoCo unified pick environment.

Provides ``PickEnv`` (reach-only reward) and ``PickLiftEnv`` (lift-to-success)
backed by a MuJoCo scene built dynamically from a ``PickConfig`` object list.

Supported object types: ``CubeObject``, ``YCBObject``, ``MeshObject``.
"""

from __future__ import annotations

import tempfile
from typing import Literal

import mujoco
import numpy as np

from so101_nexus_core import (
    ensure_ycb_assets,
    get_mujoco_ycb_rest_pose,
    get_so101_simulation_dir,
    get_ycb_collision_mesh,
    get_ycb_visual_mesh,
)
from so101_nexus_core.config import (
    ControlMode,
    PickConfig,
)
from so101_nexus_core.constants import COLOR_MAP, sample_color
from so101_nexus_core.objects import CubeObject, MeshObject, SceneObject, YCBObject
from so101_nexus_core.observations import (
    EndEffectorPose,
    GraspState,
    ObjectOffset,
    ObjectPose,
)
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus_mujoco.spawn_utils import random_yaw_quat, sample_separated_positions

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"

# Default bounding radius used when we cannot compute one from the object.
_DEFAULT_BOUNDING_RADIUS = 0.025


def _cube_bounding_radius(obj: CubeObject) -> float:
    return float(obj.half_size * np.sqrt(2))


def _cube_xml_body(slot_name: str, obj: CubeObject) -> str:
    hs = obj.half_size
    r, g, b, a = COLOR_MAP[obj.color]
    return (
        f'    <body name="{slot_name}" pos="0.15 0 {hs}">\n'
        f'      <freejoint name="{slot_name}_joint"/>\n'
        f'      <geom name="{slot_name}_geom" type="box" size="{hs} {hs} {hs}"\n'
        f'            rgba="{r} {g} {b} {a}" mass="{obj.mass}"\n'
        f'            contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"\n'
        f'            solref="0.01 1" solimp="0.95 0.99 0.001"/>\n'
        f"    </body>\n"
    )


def _mesh_xml_body(slot_name: str, asset_index: int, mass: float) -> str:
    return (
        f'    <body name="{slot_name}" pos="0.15 0 0.01">\n'
        f'      <freejoint name="{slot_name}_joint"/>\n'
        f'      <geom name="{slot_name}_collision" type="mesh" '
        f'mesh="pick_coll_{asset_index}"\n'
        f'            mass="{mass}" contype="1" conaffinity="1"\n'
        f'            condim="4" friction="1 0.05 0.001" solref="0.01 1"\n'
        f'            solimp="0.95 0.99 0.001"/>\n'
        f'      <geom name="{slot_name}_visual" type="mesh" '
        f'mesh="pick_vis_{asset_index}"\n'
        f'            contype="0" conaffinity="0" mass="0"/>\n'
        f"    </body>\n"
    )


def _build_scene_xml(
    objects: list[SceneObject],
    slot_names: list[str],
    ground_color: list[float],
) -> str:
    """Build the full MuJoCo XML string for all objects.

    Parameters
    ----------
    objects:
        Ordered list of objects; index matches slot_names.
    slot_names:
        MuJoCo body names for each object in order.
    ground_color:
        RGBA ground plane colour.
    """
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_color

    asset_entries = ""
    body_entries = ""

    for i, (obj, slot) in enumerate(zip(objects, slot_names)):
        if isinstance(obj, YCBObject):
            collision_path = str(get_ycb_collision_mesh(obj.model_id))
            visual_path = str(get_ycb_visual_mesh(obj.model_id))
            asset_entries += f'    <mesh name="pick_coll_{i}" file="{collision_path}"/>\n'
            asset_entries += f'    <mesh name="pick_vis_{i}" file="{visual_path}"/>\n'
            mass = obj.mass_override if obj.mass_override is not None else 0.01
            body_entries += _mesh_xml_body(slot, i, mass)
        elif isinstance(obj, MeshObject):
            asset_entries += (
                f'    <mesh name="pick_coll_{i}" file="{obj.collision_mesh_path}"'
                f' scale="{obj.scale} {obj.scale} {obj.scale}"/>\n'
            )
            asset_entries += (
                f'    <mesh name="pick_vis_{i}" file="{obj.visual_mesh_path}"'
                f' scale="{obj.scale} {obj.scale} {obj.scale}"/>\n'
            )
            body_entries += _mesh_xml_body(slot, i, obj.mass)
        elif isinstance(obj, CubeObject):
            body_entries += _cube_xml_body(slot, obj)
        else:
            raise TypeError(f"Unsupported object type: {type(obj)}")

    asset_section = f"  <asset>\n{asset_entries}  </asset>\n\n" if asset_entries else ""

    return f"""\
<mujoco model="pick_scene">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" noslip_iterations="3"/>
  <compiler angle="radian"/>

  <include file="{robot_path}"/>

{asset_section}  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>

{body_entries}  </worldbody>
</mujoco>
"""


def _primary_geom_name(slot_name: str, obj: SceneObject) -> str:
    """Return the name of the collision/contact geom for an object slot."""
    if isinstance(obj, CubeObject):
        return f"{slot_name}_geom"
    # YCBObject and MeshObject both use _collision suffix
    return f"{slot_name}_collision"


class _SlotInfo:
    """Runtime data for one object slot in the MuJoCo model."""

    __slots__ = (
        "qpos_addr",
        "geom_id",
        "rest_quat",
        "spawn_z",
        "bounding_radius",
        "obj",
    )

    def __init__(
        self,
        qpos_addr: int,
        geom_id: int,
        rest_quat: np.ndarray,
        spawn_z: float,
        bounding_radius: float,
        obj: SceneObject,
    ) -> None:
        self.qpos_addr = qpos_addr
        self.geom_id = geom_id
        self.rest_quat = rest_quat
        self.spawn_z = spawn_z
        self.bounding_radius = bounding_radius
        self.obj = obj


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

    def __init__(
        self,
        config: PickConfig | None = None,
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = PickConfig()
        if config.observations is None:
            config.observations = [EndEffectorPose(), GraspState(), ObjectPose(), ObjectOffset()]
        self._init_common(
            config=config,
            render_mode=render_mode,
            camera_mode=camera_mode,
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

        xml_string = _build_scene_xml(
            scene_objects,
            slot_names,
            sample_color(config.ground_colors),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        # Build per-slot runtime info (one entry per pool object)
        self._slots: list[_SlotInfo] = []
        for slot, obj in zip(slot_names, scene_objects):
            geom_name = _primary_geom_name(slot, obj)
            geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{slot}_joint")
            qpos_addr = self.model.jnt_qposadr[joint_id]

            if isinstance(obj, (YCBObject, MeshObject)):
                mesh_id = self.model.geom_dataid[geom_id]
                vert_start = self.model.mesh_vertadr[mesh_id]
                vert_count = self.model.mesh_vertnum[mesh_id]
                verts = self.model.mesh_vert[vert_start : vert_start + vert_count]
                rest_quat, spawn_z = get_mujoco_ycb_rest_pose(verts)
                xy_extent = np.ptp(verts[:, :2], axis=0)
                bounding_radius = float(np.linalg.norm(xy_extent) / 2)
            else:
                assert isinstance(obj, CubeObject)
                rest_quat = np.array([1.0, 0.0, 0.0, 0.0])
                spawn_z = obj.half_size
                bounding_radius = _cube_bounding_radius(obj)

            self._slots.append(
                _SlotInfo(
                    qpos_addr=qpos_addr,
                    geom_id=geom_id,
                    rest_quat=rest_quat,
                    spawn_z=spawn_z,
                    bounding_radius=bounding_radius,
                    obj=obj,
                )
            )

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

    def _get_obs(self) -> np.ndarray | dict:
        state = self._compute_obs_components()
        if self.camera_mode == "wrist":
            assert self._wrist_renderer is not None
            assert self._wrist_cam_id is not None
            self._wrist_renderer.update_scene(self.data, camera=self._wrist_cam_id)
            wrist_image = self._wrist_renderer.render()
            if self.config.obs_mode == "visual":
                self._privileged_state = state
                return {"state": self._get_current_qpos(), "wrist_camera": wrist_image}
            return {"state": state, "wrist_camera": wrist_image}
        return state

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus_core.observations import ObjectOffset as _ObjectOffset
        from so101_nexus_core.observations import ObjectPose as _ObjectPose

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

    def _task_reset(self) -> None:
        rng = self.np_random
        min_r = self.config.spawn_min_radius
        max_r = self.config.spawn_max_radius
        angle_half = float(np.radians(self.config.spawn_angle_half_range_deg))

        n_pool = len(self._slots)
        n_slots = self._n_slots  # number of active slots (target + distractors)

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
        )

        # Place active slots at their sampled positions.
        for pos_idx, pool_idx in enumerate(chosen_indices):
            slot = self._slots[int(pool_idx)]
            obj = slot.obj
            x, y = positions[pos_idx]

            if isinstance(obj, CubeObject):
                spawn_z = obj.half_size
                yaw_quat = random_yaw_quat(rng)
                obj_quat = yaw_quat
                rgba = COLOR_MAP[obj.color]
                self.model.geom_rgba[slot.geom_id] = rgba
            elif isinstance(obj, (YCBObject, MeshObject)):
                spawn_z = slot.spawn_z
                yaw_quat = random_yaw_quat(rng)
                obj_quat = np.zeros(4)
                mujoco.mju_mulQuat(obj_quat, yaw_quat, slot.rest_quat)
            else:
                raise TypeError(f"Unsupported object type: {type(obj)}")

            addr = slot.qpos_addr
            self.data.qpos[addr : addr + 3] = [x, y, spawn_z]
            self.data.qpos[addr + 3 : addr + 7] = obj_quat

            if pos_idx == 0:
                self._initial_obj_z = spawn_z

        # Move all inactive slots far off-world so they are invisible and
        # do not participate in collision.
        unchosen = set(range(n_pool)) - {int(i) for i in chosen_indices}
        for pool_idx in unchosen:
            slot = self._slots[pool_idx]
            addr = slot.qpos_addr
            self.data.qpos[addr : addr + 3] = [0.0, 0.0, -10.0]
            self.data.qpos[addr + 3 : addr + 7] = [1.0, 0.0, 0.0, 0.0]

        self._task_description = f"Pick up the {repr(target_obj)}."


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
