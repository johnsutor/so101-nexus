"""Primitive look-at environment for SO-101."""

from __future__ import annotations

import math
import tempfile
from typing import TYPE_CHECKING, ClassVar

import mujoco
import numpy as np

from so101_nexus import get_so101_mujoco_model_dir, get_so101_mujoco_model_path
from so101_nexus.config import ControlMode, LookAtConfig
from so101_nexus.constants import COLOR_MAP, sample_color
from so101_nexus.mujoco.base_env import SO101NexusMuJoCoBaseEnv
from so101_nexus.rewards import orientation_progress
from so101_nexus.scene import MUJOCO_SCENE_OPTION_XML, SCENE_LIGHTS_XML, SCENE_VISUAL_XML

if TYPE_CHECKING:
    from so101_nexus.objects import CubeObject

_SO101_DIR = get_so101_mujoco_model_dir()
_SO101_XML = get_so101_mujoco_model_path()


def _build_look_at_scene_xml(obj: CubeObject, ground_rgba: list[float]) -> str:
    """Build MuJoCo XML string for the look-at scene (robot + floor + target object).

    Only CubeObject is supported for the look-at target; the cube is placed as a
    kinematic mocap body so the robot can orient toward it. Per the cross-backend
    contract (body_type="kinematic"), the target is purely visual: collision is
    disabled (contype=0 conaffinity=0) so the arm passes through it, and as a mocap
    body it is unaffected by gravity or contact (its pose is driven via
    data.mocap_pos/mocap_quat, set each reset). This prevents the arm from bumping
    the reference frame, which a dynamic freejoint box previously allowed.
    """
    robot_path = str(_SO101_XML)
    gr, gg, gb, ga = ground_rgba
    hs = obj.half_size
    cr, cg, cb, ca = COLOR_MAP[obj.color]
    return f"""\
<mujoco model="look_at_scene">
  <compiler angle="radian"/>

  <include file="{robot_path}"/>
  {MUJOCO_SCENE_OPTION_XML}

{SCENE_VISUAL_XML}

  <worldbody>
{SCENE_LIGHTS_XML}
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>
    <body name="look_target" pos="0.15 0 {hs}" mocap="true">
      <geom name="look_target_geom" type="box" size="{hs} {hs} {hs}"
            rgba="{cr} {cg} {cb} {ca}"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


class LookAtEnv(SO101NexusMuJoCoBaseEnv):
    """LookAt primitive: orient the wrist camera toward a sampled target object.

    Default obs (16,): joint_positions(6) + end_effector_pose(7) + gaze_direction(3).
    Info: orientation_error (radians), success.
    task_description is auto-generated: "Look at the <repr(obj)>."

    DO NOT call _is_grasping() in this env - there is no graspable object
    (the object is present for visual targeting, not grasping).
    """

    config: LookAtConfig
    default_config_cls: ClassVar[type[LookAtConfig]] = LookAtConfig

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

        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "look_target"
        )
        # The target is a mocap body (kinematic): its pose is driven via the
        # data.mocap_pos / mocap_quat arrays indexed by body_mocapid, not qpos.
        self._look_target_mocap_id = int(self.model.body_mocapid[self._target_body_id])
        self._wrist_cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam")

        self._finish_model_setup()

    @property
    def task_description(self) -> str:
        """Return the current episode task description."""
        return self.config.task_description

    def _task_reset(self) -> None:
        # Place the object randomly in the workspace in front of the robot.
        half = self.config.spawn_half_size
        cx, cy = self.config.spawn_center
        x = cx + self.np_random.uniform(-half, half)
        y = cy + self.np_random.uniform(-half, half)
        obj = self._target_obj
        spawn_z = obj.half_size
        # Drive the kinematic mocap body via the mocap arrays (unaffected by
        # dynamics) instead of a freejoint qpos. The arm cannot bump it.
        mid = self._look_target_mocap_id
        self.data.mocap_pos[mid] = [x, y, spawn_z]
        self.data.mocap_quat[mid] = [1.0, 0.0, 0.0, 0.0]

    def _get_target_pos(self) -> np.ndarray:
        """Return the current world position of the look-at target body."""
        return self.data.xpos[self._target_body_id].copy()

    def _gaze_axis(self) -> np.ndarray:
        """Return the wrist-camera optical axis in world frame (where it points)."""
        # MuJoCo cameras look along their local -z axis; the optical axis is the
        # negative third column of the camera rotation matrix. Using the real
        # camera axis (not the gripperframe proxy) keeps success tied to what the
        # camera actually sees, and tracks any mount/FOV change automatically.
        mat = self.data.cam_xmat[self._wrist_cam_id].reshape(3, 3)
        return -mat[:, 2].copy()

    def _success_half_fov_rad(self) -> float:
        """Half the wrist-camera vertical FOV (radians): the in-frame boundary."""
        if self.config.fov_deg is not None:
            fovy = self.config.fov_deg
        else:
            fovy = float(self.model.cam_fovy[self._wrist_cam_id].item())
        return float(np.radians(fovy) / 2.0)

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus.observations import GazeDirection

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
        gaze = self._gaze_axis()
        target_pos = self._get_target_pos()
        tcp_pos = self._get_tcp_pose()[:3]
        to_target = target_pos - tcp_pos
        norm = float(np.linalg.norm(to_target))
        if norm > 1e-8:
            to_target = to_target / norm
        # Angle between the wrist-camera optical axis and the direction to the
        # object. Success when the object is within the camera's field of view.
        cos_sim = float(np.dot(gaze, to_target) / (np.linalg.norm(gaze) + 1e-8))
        cos_sim = float(np.clip(cos_sim, -1.0, 1.0))
        orientation_error = float(np.arccos(cos_sim))
        info = {
            "orientation_error": orientation_error,
            "success": orientation_error <= self._success_half_fov_rad(),
        }
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        orient = orientation_progress(math.cos(float(info["orientation_error"])))
        components = self.config.reward.compute_simple_components(
            orient,
            info.get("success", False),
            progress_key="task_objective",
            action_delta_norm=info.get("action_delta_norm", 0.0),
            energy_norm=info.get("energy_norm", 0.0),
        )
        info["reward_components"] = components
        return sum(components.values())
