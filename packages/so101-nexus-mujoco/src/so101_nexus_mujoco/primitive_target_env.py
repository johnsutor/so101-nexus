"""MuJoCo backend for the PrimitiveTargetSpec abstraction.

Consumed by the thin per-task shim env classes (ReachEnv, MoveEnv,
LookAtEnv). Handles XML scene construction (template branches on
TargetMarker.is_kinematic), target sampling at reset, observation hooks
for TargetOffset/GazeDirection, and reward via the spec's metric+shaper.
"""

from __future__ import annotations

import tempfile
from typing import TYPE_CHECKING, Any

import mujoco
import numpy as np

from so101_nexus_core import get_so101_simulation_dir
from so101_nexus_core.constants import COLOR_MAP, sample_color
from so101_nexus_core.tasks import NumpyContext, PrimitiveTargetSpec, resolve_task_description
from so101_nexus_mujoco.base_env import SO101NexusMuJoCoBaseEnv

if TYPE_CHECKING:
    from so101_nexus_core.config import EnvironmentConfig
    from so101_nexus_core.config._types import ControlMode

_SO101_DIR = get_so101_simulation_dir()
_SO101_XML = _SO101_DIR / "so101_new_calib.xml"


def _kinematic_marker_xml(marker) -> str:
    """Render the visual-only site marker (Reach, Move)."""
    r, g, b, a = marker.rgba
    return (
        f'    <site name="{marker.name}" type="{marker.shape}" '
        f'size="{marker.size}" rgba="{r} {g} {b} {a}" '
        f'pos="0.15 0 0.1" group="1"/>\n'
    )


def _dynamic_marker_xml(marker, color_rgba: tuple[float, float, float, float]) -> str:
    """Render a dynamic body+freejoint marker (LookAt cube)."""
    hs = marker.size
    cr, cg, cb, ca = color_rgba
    return (
        f'    <body name="{marker.name}" pos="0.15 0 {hs}">\n'
        f'      <freejoint name="{marker.name}_joint"/>\n'
        f'      <geom name="{marker.name}_geom" type="box" size="{hs} {hs} {hs}"\n'
        f'            rgba="{cr} {cg} {cb} {ca}" mass="{marker.mass}"\n'
        f'            contype="1" conaffinity="1" condim="4" friction="1 0.05 0.001"/>\n'
        f"    </body>\n"
    )


def _build_primitive_scene_xml(marker, ground_rgba: list[float], color_rgba=None) -> str:
    """Single shared scene template branching on marker.is_kinematic."""
    if marker.is_kinematic:
        marker_block = _kinematic_marker_xml(marker)
    else:
        assert color_rgba is not None, "dynamic marker requires resolved color_rgba"
        marker_block = _dynamic_marker_xml(marker, color_rgba)
    gr, gg, gb, ga = ground_rgba
    robot_path = str(_SO101_XML)
    return f"""\
<mujoco model="primitive_scene">
  <option timestep="0.002" gravity="0 0 -9.81" cone="elliptic" noslip_iterations="3"/>
  <compiler angle="radian"/>
  <include file="{robot_path}"/>
  <visual><headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/></visual>
  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>
{marker_block}  </worldbody>
</mujoco>
"""


class PrimitiveTargetMuJoCoEnv(SO101NexusMuJoCoBaseEnv):
    """Generic MuJoCo env that consumes a PrimitiveTargetSpec.

    Reach, Move, and LookAt env classes subclass this with only
    ``default_config_cls`` and an ``__init__`` that builds and forwards the spec.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        spec: PrimitiveTargetSpec,
        config: EnvironmentConfig,
        *,
        render_mode: str | None = None,
        control_mode: ControlMode = "pd_joint_pos",
        robot_init_qpos_noise: float = 0.02,
    ) -> None:
        self._spec = spec
        self._init_common(
            config=config,
            render_mode=render_mode,
            control_mode=control_mode,
            robot_init_qpos_noise=robot_init_qpos_noise,
        )
        self._task_description = resolve_task_description(spec, config)

        ground_rgba = sample_color(config.ground_colors)
        color_rgba = None
        if not spec.marker.is_kinematic:
            color_rgba = COLOR_MAP[spec.marker.color_name]
        xml_string = _build_primitive_scene_xml(spec.marker, ground_rgba, color_rgba)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=_SO101_DIR, delete=True) as f:
            f.write(xml_string)
            f.flush()
            self.model = mujoco.MjModel.from_xml_path(f.name)
        self.data = mujoco.MjData(self.model)

        if spec.marker.is_kinematic:
            self._target_site_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, spec.marker.name
            )
            self._target_qpos_addr = None
            self._target_body_id = None
        else:
            self._target_site_id = None
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{spec.marker.name}_joint"
            )
            self._target_qpos_addr = self.model.jnt_qposadr[joint_id]
            self._target_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, spec.marker.name
            )

        self._target_pos: np.ndarray = np.zeros(3)
        self._finish_model_setup()

    @property
    def task_description(self) -> str:  # noqa: D102
        return self._task_description

    def _set_target_world_pos(self, pos: np.ndarray) -> None:
        if self._spec.marker.is_kinematic:
            self.model.site_pos[self._target_site_id] = pos
        else:
            addr = self._target_qpos_addr
            self.data.qpos[addr : addr + 3] = pos
            self.data.qpos[addr + 3 : addr + 7] = [1.0, 0.0, 0.0, 0.0]

    def _current_target_world_pos(self) -> np.ndarray:
        """Read the live target position (LookAt's dynamic body can drift)."""
        if self._spec.marker.is_kinematic:
            return np.asarray(self._target_pos, dtype=np.float64)
        return self.data.xpos[self._target_body_id].copy()

    def _get_tcp_forward(self) -> np.ndarray:
        mat = self.data.site_xmat[self._tcp_site_id].reshape(3, 3)
        return mat[:, 2].copy()

    def _task_reset(self) -> None:
        spec = self._spec
        if spec.requires_tcp_pos_for_sampling:
            mujoco.mj_forward(self.model, self.data)
            tcp_pos = self.data.site_xpos[self._tcp_site_id].copy()
        else:
            tcp_pos = None
        ctx = NumpyContext(rng=self.np_random, config=self.config, tcp_pos=tcp_pos)
        pos = spec.sampler.sample_numpy(ctx)
        self._target_pos = np.asarray(pos, dtype=np.float64)
        self._set_target_world_pos(self._target_pos)

    def _get_component_data(self, component: object) -> np.ndarray:
        from so101_nexus_core.observations import GazeDirection, TargetOffset

        if isinstance(component, TargetOffset):
            return self._current_target_world_pos() - self._get_tcp_pose()[:3]
        if isinstance(component, GazeDirection):
            target = self._current_target_world_pos()
            tcp_pos = self._get_tcp_pose()[:3]
            gaze = target - tcp_pos
            norm = float(np.linalg.norm(gaze))
            return gaze / norm if norm > 1e-8 else gaze
        return super()._get_component_data(component)

    def _get_info(self) -> dict:
        spec = self._spec
        tcp_pos = self._get_tcp_pose()[:3]
        target_pos = self._current_target_world_pos()
        tcp_forward = self._get_tcp_forward() if spec.requires_tcp_forward_for_metric else None
        ctx = NumpyContext(rng=self.np_random, config=self.config, tcp_pos=tcp_pos)
        metric, success = spec.metric.evaluate_numpy(
            target_pos=target_pos, tcp_pos=tcp_pos, tcp_forward=tcp_forward, ctx=ctx
        )
        metric_key = (
            "orientation_error" if spec.requires_tcp_forward_for_metric else "tcp_to_target_dist"
        )
        info: dict[str, Any] = {metric_key: metric, "success": bool(success)}
        if self._privileged_state is not None:
            info["privileged_state"] = self._privileged_state
        return info

    def _compute_reward(self, info: dict) -> float:
        spec = self._spec
        ctx = NumpyContext(rng=self.np_random, config=self.config)
        metric_key = (
            "orientation_error" if spec.requires_tcp_forward_for_metric else "tcp_to_target_dist"
        )
        progress = spec.shaper.shape_numpy(info[metric_key], ctx)
        completion = self.config.reward.completion_bonus
        return (1.0 - completion) * progress + completion * float(info.get("success", False))
