from __future__ import annotations

import tempfile
from typing import Literal

import mujoco
import numpy as np

from so101_nexus_core import get_so101_simulation_dir
from so101_nexus_core.config import (
    CUBE_COLOR_MAP,
    ColorName,
    ControlMode,
    PickCubeMultipleConfig,
    sample_color,
)
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"


def _sample_separated_positions(
    rng: np.random.Generator,
    count: int,
    min_r: float,
    max_r: float,
    angle_half: float,
    min_clearance: float,
    bounding_radii: list[float],
    max_attempts: int = 100,
) -> list[tuple[float, float]]:
    """Sample 2D positions in a polar arc with bounding-radius-aware separation."""
    positions: list[tuple[float, float]] = []
    for idx in range(count):
        for _ in range(max_attempts):
            r = rng.uniform(min_r, max_r)
            theta = rng.uniform(-angle_half, angle_half)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            if all(
                np.sqrt((x - px) ** 2 + (y - py) ** 2)
                >= bounding_radii[idx] + bounding_radii[j] + min_clearance
                for j, (px, py) in enumerate(positions)
            ):
                positions.append((x, y))
                break
        else:
            positions.append((x, y))
    return positions


def _build_scene_xml(
    num_distractors: int,
    cube_half_size: float,
    cube_color: list[float],
    cube_mass: float,
    ground_color: list[float],
) -> str:
    r, g, b, a = cube_color
    hs = cube_half_size
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_color

    cube_bodies = f"""\
    <body name="cube_target" pos="0.15 0 {hs}">
      <freejoint name="cube_target_joint"/>
      <geom name="cube_target_geom" type="box" size="{hs} {hs} {hs}"
            rgba="{r} {g} {b} {a}" mass="{cube_mass}"
            contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
"""

    for i in range(num_distractors):
        cube_bodies += f"""\
    <body name="cube_distractor_{i}" pos="0.15 0 {hs}">
      <freejoint name="cube_distractor_{i}_joint"/>
      <geom name="cube_distractor_{i}_geom" type="box" size="{hs} {hs} {hs}"
            rgba="0.5 0.5 0.5 1.0" mass="{cube_mass}"
            contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"
            solref="0.01 1" solimp="0.95 0.99 0.001"/>
    </body>
"""

    return f"""\
<mujoco model="pick_cube_multiple_scene">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" noslip_iterations="3"/>
  <compiler angle="radian"/>

  <include file="{robot_path}"/>

  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>

{cube_bodies}\
  </worldbody>
</mujoco>
"""


class PickCubeMultipleEnv(SO101NexusMuJoCoBaseEnv):
    """MuJoCo pick-cube environment with distractor cubes."""

    config: PickCubeMultipleConfig

    def __init__(
        self,
        config: PickCubeMultipleConfig = PickCubeMultipleConfig(),
        cube_color: ColorName = "red",
        render_mode: str | None = None,
        camera_mode: Literal["state_only", "wrist"] = "state_only",
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ):
        if cube_color not in CUBE_COLOR_MAP:
            raise ValueError(
                f"cube_color must be one of {list(CUBE_COLOR_MAP)}, got {cube_color!r}"
            )

        self._init_common(
            config=config,
            render_mode=render_mode,
            camera_mode=camera_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        self.cube_color_name = cube_color
        self.cube_half_size = config.cube_half_size
        self.num_distractors = config.num_distractors
        self.min_object_separation = config.min_object_separation
        self.task_description = (
            f"Pick up the small {cube_color} cube from among {config.num_distractors} distractors"
        )

        xml_string = _build_scene_xml(
            config.num_distractors,
            config.cube_half_size,
            CUBE_COLOR_MAP[cube_color],
            config.cube_mass,
            sample_color(config.ground_colors),
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        # Target cube IDs
        self._cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_target")
        self._obj_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_target_geom"
        )
        cube_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_target_joint"
        )
        self._cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        # Distractor IDs
        self._distractor_geom_ids: list[int] = []
        self._distractor_qpos_addrs: list[int] = []
        for i in range(config.num_distractors):
            geom_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, f"cube_distractor_{i}_geom"
            )
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"cube_distractor_{i}_joint"
            )
            self._distractor_geom_ids.append(geom_id)
            self._distractor_qpos_addrs.append(self.model.jnt_qposadr[joint_id])

        self._finish_model_setup()

    def _get_cube_pose(self) -> np.ndarray:
        addr = self._cube_qpos_addr
        return self.data.qpos[addr : addr + 7].copy()

    def _get_obs(self) -> np.ndarray | dict:
        tcp_pose = self._get_tcp_pose()
        is_grasped = np.array([self._is_grasping()])
        obj_pose = self._get_cube_pose()
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
        obj_pose = self._get_cube_pose()
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
        reach_progress = 1.0 - float(np.tanh(5.0 * info["tcp_to_obj_dist"]))
        is_grasped = info["is_grasped"] > 0.5

        return self.config.reward.compute(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=0.0,
            is_complete=info.get("success", False),
            action_delta_norm=info.get("action_delta_norm", 0.0),
        )

    def _task_reset(self) -> None:
        rng = self.np_random
        min_r = self.config.spawn_min_radius
        max_r = self.config.spawn_max_radius
        angle_half = float(np.radians(self.config.spawn_angle_half_range_deg))

        total_objects = 1 + self.num_distractors
        cube_radius = self.cube_half_size * np.sqrt(2)
        bounding_radii = [cube_radius] * total_objects
        positions = _sample_separated_positions(
            rng, total_objects, min_r, max_r, angle_half, self.min_object_separation, bounding_radii
        )

        # Place target cube
        cube_x, cube_y = positions[0]
        cube_z = self.cube_half_size
        angle = rng.uniform(0, 2 * np.pi)
        cube_quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])

        addr = self._cube_qpos_addr
        self.data.qpos[addr : addr + 3] = [cube_x, cube_y, cube_z]
        self.data.qpos[addr + 3 : addr + 7] = cube_quat
        self._initial_obj_z = cube_z

        # Place distractors with random colors
        distractor_colors = [c for c in CUBE_COLOR_MAP if c != self.cube_color_name]
        for i in range(self.num_distractors):
            dx, dy = positions[1 + i]
            dz = self.cube_half_size
            d_angle = rng.uniform(0, 2 * np.pi)
            d_quat = np.array([np.cos(d_angle / 2), 0, 0, np.sin(d_angle / 2)])

            d_addr = self._distractor_qpos_addrs[i]
            self.data.qpos[d_addr : d_addr + 3] = [dx, dy, dz]
            self.data.qpos[d_addr + 3 : d_addr + 7] = d_quat

            # Randomize distractor color
            color_name = distractor_colors[rng.integers(len(distractor_colors))]
            rgba = CUBE_COLOR_MAP[color_name]
            self.model.geom_rgba[self._distractor_geom_ids[i]] = rgba


class PickCubeMultipleLiftEnv(PickCubeMultipleEnv):
    """Pick-cube-multiple variant where success requires lifting the cube above a threshold."""

    def _get_info(self) -> dict:
        info = super()._get_info()
        info["success"] = (info["lift_height"] > self.config.lift_threshold) and (
            info["is_grasped"] > 0.5
        )
        return info

    def _compute_reward(self, info: dict) -> float:
        reach_progress = 1.0 - float(np.tanh(5.0 * info["tcp_to_obj_dist"]))
        is_grasped = info["is_grasped"] > 0.5
        lift_progress = float(np.tanh(5.0 * max(info["lift_height"], 0.0))) if is_grasped else 0.0

        return self.config.reward.compute(
            reach_progress=reach_progress,
            is_grasped=is_grasped,
            task_progress=lift_progress,
            is_complete=info["success"],
            action_delta_norm=info.get("action_delta_norm", 0.0),
        )
