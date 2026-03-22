"""Primitive look-at environment for SO-101."""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from so101_nexus_core import get_so101_simulation_dir
from so101_nexus_core.config import ControlMode, LookAtConfig
from so101_nexus_core.constants import COLOR_MAP, sample_color
from so101_nexus_core.rewards import simple_reward
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv

if TYPE_CHECKING:
    from so101_nexus_core.objects import CubeObject

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"


def _build_look_at_scene_xml(obj: CubeObject, ground_rgba: list[float]) -> str:
    """Build MuJoCo XML string for the look-at scene (robot + floor + target object).

    Only CubeObject is supported for the look-at target; the cube is placed as
    a static visual/collision body so the robot can orient toward it.
    """
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_rgba
    hs = obj.half_size
    cr, cg, cb, ca = COLOR_MAP[obj.color]
    return f"""\
<mujoco model="look_at_scene">
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
    <body name="look_target" pos="0.15 0 {hs}">
      <freejoint name="look_target_joint"/>
      <geom name="look_target_geom" type="box" size="{hs} {hs} {hs}"
            rgba="{cr} {cg} {cb} {ca}" mass="{obj.mass}"
            contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"/>
    </body>
  </worldbody>
</mujoco>
"""


class LookAtEnv(SO101NexusMuJoCoBaseEnv):
    """LookAt primitive: orient the wrist camera toward a sampled target object.

    Default obs (6,): joint_positions.
    Info: orientation_error (radians), success.
    task_description is auto-generated: "Look at the <repr(obj)>."

    DO NOT call _is_grasping() in this env — there is no graspable object
    (the object is present for visual targeting, not grasping).
    """

    config: LookAtConfig

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        config: LookAtConfig | None = None,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        if config is None:
            config = LookAtConfig()
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )

        # Use the first (and typically only) target object from config.
        self._target_obj: CubeObject = config.objects[0]  # type: ignore[assignment]

        ground_rgba = sample_color(config.ground_colors)
        xml_string = _build_look_at_scene_xml(self._target_obj, ground_rgba)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        self._look_target_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "look_target_joint"
        )
        self._look_target_qpos_addr = self.model.jnt_qposadr[self._look_target_joint_id]
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "look_target"
        )

        self._task_description: str = f"Look at the {self._target_obj!r}."

        self._finish_model_setup()

    @property
    def task_description(self) -> str:
        """Return the current episode task description."""
        return self._task_description

    def _task_reset(self) -> None:
        # Place the object randomly in the workspace in front of the robot.
        half = self.config.spawn_half_size
        cx, cy = self.config.spawn_center
        x = cx + self.np_random.uniform(-half, half)
        y = cy + self.np_random.uniform(-half, half)
        obj = self._target_obj
        spawn_z = obj.half_size
        addr = self._look_target_qpos_addr
        self.data.qpos[addr : addr + 3] = [x, y, spawn_z]
        self.data.qpos[addr + 3 : addr + 7] = [1.0, 0.0, 0.0, 0.0]

    def _get_target_pos(self) -> np.ndarray:
        """Return the current world position of the look-at target body."""
        return self.data.xpos[self._target_body_id].copy()

    def _get_tcp_forward(self) -> np.ndarray:
        """Return the TCP z-axis (forward / gaze direction) in world frame."""
        mat = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        # Third column of the rotation matrix = local z-axis in world frame.
        return mat[:, 2].copy()

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus_core.observations import GazeDirection

        if isinstance(component, GazeDirection):
            target_pos = self._get_target_pos()
            tcp_pos = self._get_tcp_pose()[:3]
            gaze = target_pos - tcp_pos
            norm = float(np.linalg.norm(gaze))
            if norm > 1e-8:
                gaze = gaze / norm
            return gaze
        return super()._get_component_data(component)

    def _get_info(self) -> dict:
        tcp_forward = self._get_tcp_forward()
        target_pos = self._get_target_pos()
        tcp_pos = self._get_tcp_pose()[:3]
        to_target = target_pos - tcp_pos
        norm = float(np.linalg.norm(to_target))
        if norm > 1e-8:
            to_target = to_target / norm
        # Angular error in [0, pi]
        cos_sim = float(np.dot(tcp_forward, to_target) / (np.linalg.norm(tcp_forward) + 1e-8))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        orientation_error = float(np.arccos(cos_sim))
        threshold_rad: float = self.config._orientation_success_threshold_rad
        info = {
            "orientation_error": orientation_error,
            "success": orientation_error < threshold_rad,
        }
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        tcp_forward = self._get_tcp_forward()
        target_pos = self._get_target_pos()
        tcp_pos = self._get_tcp_pose()[:3]
        to_target = target_pos - tcp_pos
        orient = self._orientation_toward_reward(tcp_forward, to_target)
        return simple_reward(
            progress=orient,
            completion_bonus=self.config.reward.completion_bonus,
            success=info.get("success", False),
        )
