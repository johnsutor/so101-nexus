from __future__ import annotations

import tempfile
from typing import Literal

import mujoco
import numpy as np

from so101_nexus_core import (
    ensure_ycb_assets,
    get_so101_simulation_dir,
    get_ycb_collision_mesh,
    get_ycb_visual_mesh,
)
from so101_nexus_core.config import (
    YCB_OBJECTS,
    ControlMode,
    PickYCBMultipleConfig,
    YcbModelId,
    sample_color,
)
from so101_nexus_core.ycb_geometry import get_mujoco_ycb_rest_pose
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus_mujoco.spawn_utils import sample_separated_positions

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"


def _build_ycb_multiple_scene_xml(
    model_ids: list[str],
    collision_paths: list[str],
    visual_paths: list[str],
    ground_color: list[float],
) -> str:
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_color

    asset_entries = ""
    for i, (coll, vis) in enumerate(zip(collision_paths, visual_paths)):
        asset_entries += f'    <mesh name="ycb_collision_{i}" file="{coll}"/>\n'
        asset_entries += f'    <mesh name="ycb_visual_{i}" file="{vis}"/>\n'

    body_entries = ""
    for i in range(len(model_ids)):
        name = "ycb_target" if i == 0 else f"ycb_distractor_{i - 1}"
        body_entries += f"""\
    <body name="{name}" pos="0.15 0 0.01">
      <freejoint name="{name}_joint"/>
      <geom name="{name}_collision" type="mesh" mesh="ycb_collision_{i}"
            mass="0.01" contype="1" conaffinity="1"
            condim="4" friction="1 0.05 0.001" solref="0.01 1" solimp="0.95 0.99 0.001"/>
      <geom name="{name}_visual" type="mesh" mesh="ycb_visual_{i}"
            contype="0" conaffinity="0" mass="0"/>
    </body>
"""

    return f"""\
<mujoco model="pick_ycb_multiple_scene">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" noslip_iterations="3"/>
  <compiler angle="radian"/>

  <include file="{robot_path}"/>

  <asset>
{asset_entries}  </asset>

  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>

{body_entries}\
  </worldbody>
</mujoco>
"""


class PickYCBMultipleEnv(SO101NexusMuJoCoBaseEnv):
    """MuJoCo pick-YCB environment with distractor YCB objects."""

    config: PickYCBMultipleConfig

    def __init__(
        self,
        config: PickYCBMultipleConfig = PickYCBMultipleConfig(),
        model_id: YcbModelId = "058_golf_ball",
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        if model_id not in YCB_OBJECTS:
            raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {model_id!r}")

        self._init_common(
            config=config,
            render_mode=render_mode,
            camera_mode=camera_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        self.model_id = model_id
        self.num_distractors = config.num_distractors
        self.min_object_separation = config.min_object_separation
        self.task_description = (
            f"Pick up the {YCB_OBJECTS[model_id]} from among {config.num_distractors} distractors"
        )

        # Sample distractor model IDs (excluding target)
        other_ids = [mid for mid in YCB_OBJECTS if mid != model_id]
        rng = np.random.default_rng()
        self.distractor_model_ids: list[str] = list(
            rng.choice(other_ids, size=config.num_distractors, replace=True)
        )

        # All model IDs: target first, then distractors
        all_model_ids = [model_id] + self.distractor_model_ids

        # Download assets and collect paths
        collision_paths: list[str] = []
        visual_paths: list[str] = []
        for mid in all_model_ids:
            ensure_ycb_assets(mid)
            collision_paths.append(str(get_ycb_collision_mesh(mid)))
            visual_paths.append(str(get_ycb_visual_mesh(mid)))

        xml_string = _build_ycb_multiple_scene_xml(
            all_model_ids,
            collision_paths,
            visual_paths,
            sample_color(config.ground_colors),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        # Compute rest poses and bounding radii for all objects
        self._all_rest_quats: list[np.ndarray] = []
        self._all_spawn_zs: list[float] = []
        self._all_bounding_radii: list[float] = []

        for i in range(len(all_model_ids)):
            name = "ycb_target" if i == 0 else f"ycb_distractor_{i - 1}"
            coll_geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{name}_collision"
            )
            mesh_id = self.model.geom_dataid[coll_geom_id]
            vert_start = self.model.mesh_vertadr[mesh_id]
            vert_count = self.model.mesh_vertnum[mesh_id]
            verts = self.model.mesh_vert[vert_start : vert_start + vert_count]
            rest_quat, spawn_z = get_mujoco_ycb_rest_pose(verts)
            self._all_rest_quats.append(rest_quat)
            self._all_spawn_zs.append(spawn_z)
            # Bounding radius from XY extents of the mesh
            xy_extent = np.ptp(verts[:, :2], axis=0)
            self._all_bounding_radii.append(float(np.linalg.norm(xy_extent) / 2))

        # Target object IDs
        self._obj_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "ycb_target_collision"
        )
        target_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "ycb_target_joint"
        )
        self._obj_qpos_addr = self.model.jnt_qposadr[target_joint_id]

        # Distractor IDs
        self._distractor_qpos_addrs: list[int] = []
        for i in range(config.num_distractors):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"ycb_distractor_{i}_joint"
            )
            self._distractor_qpos_addrs.append(self.model.jnt_qposadr[joint_id])

        self._finish_model_setup()

    def _get_obj_pose(self) -> np.ndarray:
        addr = self._obj_qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_obs(self) -> np.ndarray | dict:
        tcp_pose = self._get_tcp_pose()
        is_grasped = np.array([self._is_grasping()])
        obj_pose = self._get_obj_pose()
        tcp_to_obj = obj_pose[:3] - tcp_pose[:3]
        state = np.concatenate([tcp_pose, is_grasped, obj_pose, tcp_to_obj])

        if self.camera_mode == "wrist":
            assert self._wrist_renderer is not None
            assert self._wrist_cam_id is not None
            self._wrist_renderer.update_scene(self.data, camera=self._wrist_cam_id)
            return {"state": state, "wrist_camera": self._wrist_renderer.render()}
        return state

    def _get_info(self) -> dict:
        tcp_pos = self._get_tcp_pose()[:3]
        obj_pose = self._get_obj_pose()
        obj_pos = obj_pose[:3]
        is_grasped = self._is_grasping()

        is_robot_static = self._is_robot_static()
        lift_height = float(obj_pos[2] - self._initial_obj_z)

        return {
            "is_grasped": is_grasped,
            "is_robot_static": is_robot_static,
            "lift_height": lift_height,
            "tcp_to_obj_dist": float(np.linalg.norm(obj_pos - tcp_pos)),
        }

    def _compute_reward(self, info: dict) -> float:
        return self._reach_only_reward(info)

    def _task_reset(self) -> None:
        rng = self.np_random
        min_r = self.config.spawn_min_radius
        max_r = self.config.spawn_max_radius
        angle_half = float(np.radians(self.config.spawn_angle_half_range_deg))

        total_objects = 1 + self.num_distractors
        positions = sample_separated_positions(
            rng,
            total_objects,
            min_r,
            max_r,
            angle_half,
            self.min_object_separation,
            self._all_bounding_radii,
        )

        # Place target
        obj_x, obj_y = positions[0]
        obj_z = self._all_spawn_zs[0]
        angle = rng.uniform(0, 2 * np.pi)
        yaw_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        obj_quat = np.zeros(4)
        mujoco.mju_mulQuat(obj_quat, yaw_quat, self._all_rest_quats[0])

        addr = self._obj_qpos_addr
        self.data.qpos[addr : addr + 3] = [obj_x, obj_y, obj_z]
        self.data.qpos[addr + 3 : addr + 7] = obj_quat
        self._initial_obj_z = obj_z

        # Place distractors
        for i in range(self.num_distractors):
            dx, dy = positions[1 + i]
            dz = self._all_spawn_zs[1 + i]
            d_angle = rng.uniform(0, 2 * np.pi)
            d_yaw = np.array([np.cos(d_angle / 2), 0, 0, np.sin(d_angle / 2)])
            d_quat = np.zeros(4)
            mujoco.mju_mulQuat(d_quat, d_yaw, self._all_rest_quats[1 + i])

            d_addr = self._distractor_qpos_addrs[i]
            self.data.qpos[d_addr : d_addr + 3] = [dx, dy, dz]
            self.data.qpos[d_addr + 3 : d_addr + 7] = d_quat


class PickYCBMultipleLiftEnv(PickYCBMultipleEnv):
    """Pick-YCB-multiple variant where success requires lifting the object above threshold."""

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["success"] = (info["lift_height"] > self.config.lift_threshold) and (
            info["is_grasped"] > 0.5
        )
        return info

    def _compute_reward(self, info: dict) -> float:
        return self._lift_reward(info)
