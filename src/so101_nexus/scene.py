"""Shared MuJoCo scene templates for SO101-Nexus backends.

The scene wrapper emits a robot ``<include>`` followed by an ``<option>`` that
overrides the vendored menagerie model's option (MuJoCo: the last top-level
option wins). Two presets exist because MuJoCo Warp does not support the
``implicitfast`` integrator or the ``noslip`` solver: the Warp preset switches
to the ``implicit`` integrator and drops noslip, so contact and friction
dynamics will not be bit-identical to the CPU backend. For the contact-free
primitive scenes the only effect is the integrator on free arm motion.
"""

from __future__ import annotations

MUJOCO_SCENE_OPTION_XML = (
    '<option timestep="0.005" gravity="0 0 -9.81" cone="elliptic" '
    'integrator="implicitfast" impratio="10" iterations="10" '
    'ls_iterations="20" noslip_iterations="3"/>'
)

WARP_SCENE_OPTION_XML = (
    '<option timestep="0.005" gravity="0 0 -9.81" cone="elliptic" '
    'integrator="implicit" impratio="10" iterations="10" ls_iterations="20"/>'
)

# Shared ``<visual>`` block for every backend scene. ``shadowsize`` and
# ``offsamples`` raise the shadow-map resolution and offscreen multisample count
# above the MuJoCo defaults (4096 / 4), and ``shadowclip`` tightens the
# directional-light shadow frustum (default 1.0) to concentrate shadow texels on
# the workspace. Together they remove the blocky shadow-edge and silhouette
# aliasing seen in rendered camera observations (overhead corners, distant floor
# in the wrist view). Values match the vendored menagerie ``scene.xml``.
SCENE_VISUAL_XML = """\
  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map shadowclip="0.5"/>
  </visual>"""

# Two directional lights: an angled key light that casts the scene's single
# shadow plus an overhead fill light with ``castshadow="false"``. Both lights
# casting shadows previously produced a doubled robot shadow.
SCENE_LIGHTS_XML = """\
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"
           castshadow="false"/>"""


def build_robot_floor_scene_xml(
    ground_rgba: list[float],
    *,
    option_xml: str,
    robot_xml_path: str,
    overhead_camera_xml: str = "",
    extra_bodies: str = "",
) -> str:
    """Build a robot + floor MJCF with no in-scene target marker.

    Shared by the Warp move and look-at envs, whose per-world targets live in
    torch tensors rather than a placed site or body. The floor keeps arm/floor
    contact consistent with the reach scene.

    Parameters
    ----------
    ground_rgba : list of float
        Floor colour ``[r, g, b, a]``.
    option_xml : str
        The ``<option>`` element controlling physics (a preset above).
    robot_xml_path : str
        Path to the vendored menagerie SO101 model to ``<include>``.
    overhead_camera_xml : str
        Optional ``<camera>`` element injected into the worldbody, used when an
        ``OverheadCamera`` observation renders on the Warp backend. Empty by
        default (no overhead camera).
    extra_bodies : str
        Optional extra ``<worldbody>`` XML (for example a visual-only target
        marker geom rendered by the Warp camera path). Empty by default.
    """
    gr, gg, gb, ga = ground_rgba
    return f"""\
<mujoco model="robot_floor_scene">
  <compiler angle="radian"/>

  <include file="{robot_xml_path}"/>
  {option_xml}

{SCENE_VISUAL_XML}

  <worldbody>
{SCENE_LIGHTS_XML}
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>
{extra_bodies}{overhead_camera_xml}  </worldbody>
</mujoco>
"""
