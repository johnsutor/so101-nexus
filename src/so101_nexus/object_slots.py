"""Shared object-slot abstraction for SO101-Nexus manipulation backends.

Both the scalar MuJoCo backend and the batched MuJoCo Warp backend build a pool
of freejoint object bodies ("slots") into one compiled ``MjModel`` and select
which slot is the active target (and which are distractors) per episode. This
module holds the backend-neutral pieces of that machinery:

- MJCF fragment builders for ``CubeObject``, ``YCBObject``, and ``MeshObject``.
- The full scene builder ``build_object_scene_xml`` parameterized by the
  ``<option>`` preset and robot model path, so MuJoCo and Warp emit identical
  bodies and assets while differing only in the integrator/solver preset.
- Runtime metadata extraction from a compiled ``MjModel`` (``ObjectSlot``).

Only the ``mujoco`` third-party library is imported here (for metadata
extraction); the ``so101_nexus.mujoco`` *package* is not, so the Warp backend can
import this module without triggering MuJoCo env registration. No torch import.
"""

from __future__ import annotations

import mujoco
import numpy as np

from so101_nexus.constants import COLOR_MAP
from so101_nexus.objects import CubeObject, MeshObject, SceneObject, YCBObject
from so101_nexus.ycb_assets import (
    get_ycb_collision_mesh,
    get_ycb_texture_file,
    get_ycb_visual_mesh,
)
from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose

# Default bounding radius used when an object exposes no geometry to measure.
DEFAULT_BOUNDING_RADIUS = 0.025

# Default per-object mass (kg) for YCB objects without a mass override.
_DEFAULT_YCB_MASS = 0.01


def cube_bounding_radius(obj: CubeObject) -> float:
    """Return the horizontal bounding radius of a cube (half-diagonal)."""
    return float(obj.half_size * np.sqrt(2))


def primary_geom_name(slot_name: str, obj: SceneObject) -> str:
    """Return the name of the collision/contact geom for an object slot."""
    if isinstance(obj, CubeObject):
        return f"{slot_name}_geom"
    # YCBObject and MeshObject both use the _collision suffix.
    return f"{slot_name}_collision"


def cube_xml_body(slot_name: str, obj: CubeObject) -> str:
    """Return the MJCF ``<body>`` fragment for one freejoint cube slot."""
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


def mesh_xml_body(
    slot_name: str,
    asset_index: int,
    mass: float,
    material_name: str | None = None,
) -> str:
    """Return the MJCF ``<body>`` fragment for one freejoint mesh slot.

    Mesh slots carry a hidden collision geom (group 3) and a non-colliding visual
    geom (group 2); the latter is mostly for MuJoCo rendering parity (Warp
    training reads state tensors, not rendered images).
    """
    material_attr = f' material="{material_name}"' if material_name else ""
    return (
        f'    <body name="{slot_name}" pos="0.15 0 0.01">\n'
        f'      <freejoint name="{slot_name}_joint"/>\n'
        f'      <geom name="{slot_name}_collision" type="mesh" '
        f'mesh="pick_coll_{asset_index}"\n'
        f'            mass="{mass}" contype="1" conaffinity="1"\n'
        f'            group="3" condim="4" friction="1 0.05 0.001" solref="0.01 1"\n'
        f'            solimp="0.95 0.99 0.001"/>\n'
        f'      <geom name="{slot_name}_visual" type="mesh" '
        f'mesh="pick_vis_{asset_index}"\n'
        f'            group="2" contype="0" conaffinity="0" mass="0"{material_attr}/>\n'
        f"    </body>\n"
    )


def build_object_scene_xml(
    objects: list[SceneObject],
    slot_names: list[str],
    ground_color: list[float],
    *,
    option_xml: str,
    robot_xml_path: str,
    model_name: str = "object_scene",
    extra_bodies: str = "",
) -> str:
    """Build a robot + floor MJCF with one freejoint body per pool object.

    Parameters
    ----------
    objects:
        Ordered pool of scene objects; index matches ``slot_names``.
    slot_names:
        MuJoCo body name for each object, in order.
    ground_color:
        RGBA floor colour.
    option_xml:
        The physics ``<option>`` preset (MuJoCo or Warp; see ``so101_nexus.scene``).
    robot_xml_path:
        Path to the vendored menagerie SO101 model to ``<include>``.
    model_name:
        MJCF model name (cosmetic).
    extra_bodies:
        Additional ``<worldbody>`` XML appended after the object slots (for
        example a pick-and-place goal disc).
    """
    gr, gg, gb, ga = ground_color
    asset_entries = ""
    body_entries = ""

    for i, (obj, slot) in enumerate(zip(objects, slot_names, strict=True)):
        if isinstance(obj, YCBObject):
            collision_path = get_ycb_collision_mesh(obj.model_id).as_posix()
            visual_path = get_ycb_visual_mesh(obj.model_id).as_posix()
            asset_entries += f'    <mesh name="pick_coll_{i}" file="{collision_path}"/>\n'
            asset_entries += f'    <mesh name="pick_vis_{i}" file="{visual_path}"/>\n'
            material_name = None
            texture_path = get_ycb_texture_file(obj.model_id)
            if texture_path.exists():
                texture_name = f"pick_tex_{i}"
                material_name = f"pick_mat_{i}"
                asset_entries += (
                    f'    <texture name="{texture_name}" type="2d" '
                    f'file="{texture_path.as_posix()}"/>\n'
                )
                asset_entries += (
                    f'    <material name="{material_name}" texture="{texture_name}" '
                    'texuniform="false"/>\n'
                )
            mass = obj.mass_override if obj.mass_override is not None else _DEFAULT_YCB_MASS
            body_entries += mesh_xml_body(slot, i, mass, material_name=material_name)
        elif isinstance(obj, MeshObject):
            asset_entries += (
                f'    <mesh name="pick_coll_{i}" file="{obj.collision_mesh_path}"'
                f' scale="{obj.scale} {obj.scale} {obj.scale}"/>\n'
            )
            asset_entries += (
                f'    <mesh name="pick_vis_{i}" file="{obj.visual_mesh_path}"'
                f' scale="{obj.scale} {obj.scale} {obj.scale}"/>\n'
            )
            body_entries += mesh_xml_body(slot, i, obj.mass)
        elif isinstance(obj, CubeObject):
            body_entries += cube_xml_body(slot, obj)
        else:
            raise TypeError(f"Unsupported object type: {type(obj)}")

    asset_section = f"  <asset>\n{asset_entries}  </asset>\n\n" if asset_entries else ""

    return f"""\
<mujoco model="{model_name}">
  <compiler angle="radian"/>

  <include file="{robot_xml_path}"/>
  {option_xml}

{asset_section}  <visual>
    <headlight diffuse="0.0 0.0 0.0" ambient="0.3 0.3 0.3" specular="0 0 0"/>
  </visual>

  <worldbody>
    <light pos="1 1 3.5" dir="-0.27 -0.27 -0.92" directional="true" diffuse="0.5 0.5 0.5"/>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true" diffuse="0.5 0.5 0.5"/>
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>

{body_entries}{extra_bodies}  </worldbody>
</mujoco>
"""


class ObjectSlot:
    """Runtime metadata for one freejoint object slot in a compiled model.

    Backend-neutral: ``qpos_addr``/``dof_addr`` index the shared ``MjModel``
    layout and apply to a scalar ``MjData`` (MuJoCo) and a batched
    ``mjw.Data`` column (Warp) alike. ``rest_quat`` is a NumPy ``wxyz`` vector;
    backends convert to tensors as needed.
    """

    __slots__ = (
        "bounding_radius",
        "dof_addr",
        "geom_id",
        "obj",
        "qpos_addr",
        "rest_quat",
        "spawn_z",
    )

    def __init__(
        self,
        qpos_addr: int,
        dof_addr: int,
        geom_id: int,
        rest_quat: np.ndarray,
        spawn_z: float,
        bounding_radius: float,
        obj: SceneObject,
    ) -> None:
        self.qpos_addr = qpos_addr
        self.dof_addr = dof_addr
        self.geom_id = geom_id
        self.rest_quat = rest_quat
        self.spawn_z = spawn_z
        self.bounding_radius = bounding_radius
        self.obj = obj


def extract_object_slots(
    mjm: mujoco.MjModel,
    slot_names: list[str],
    objects: list[SceneObject],
) -> list[ObjectSlot]:
    """Read per-slot runtime metadata from a compiled ``MjModel``.

    For cubes the rest pose is identity and the spawn height is the half-size;
    for mesh-backed objects (YCB, custom) the stable rest orientation and floor
    clearance come from the compiled mesh vertices.
    """
    slots: list[ObjectSlot] = []
    for slot_name, obj in zip(slot_names, objects, strict=True):
        geom_name = primary_geom_name(slot_name, obj)
        geom_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        joint_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, f"{slot_name}_joint")
        qpos_addr = int(mjm.jnt_qposadr[joint_id])
        dof_addr = int(mjm.jnt_dofadr[joint_id])

        if isinstance(obj, (YCBObject, MeshObject)):
            mesh_id = mjm.geom_dataid[geom_id]
            vert_start = mjm.mesh_vertadr[mesh_id]
            vert_count = mjm.mesh_vertnum[mesh_id]
            verts = mjm.mesh_vert[vert_start : vert_start + vert_count]
            rest_quat, spawn_z = get_mujoco_ycb_rest_pose(verts)
            # Footprint is measured in the resting orientation: the stable rest
            # pose can rotate a thin X/Y axis up, changing the horizontal extent
            # that the spawn separation samplers rely on.
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, rest_quat)
            rotated_xy = (verts @ rot.reshape(3, 3).T)[:, :2]
            xy_extent = np.ptp(rotated_xy, axis=0)
            bounding_radius = float(np.linalg.norm(xy_extent) / 2)
        elif isinstance(obj, CubeObject):
            rest_quat = np.array([1.0, 0.0, 0.0, 0.0])
            spawn_z = obj.half_size
            bounding_radius = cube_bounding_radius(obj)
        else:
            raise TypeError(f"Unsupported object type: {type(obj)}")

        slots.append(
            ObjectSlot(
                qpos_addr=qpos_addr,
                dof_addr=dof_addr,
                geom_id=geom_id,
                rest_quat=rest_quat,
                spawn_z=float(spawn_z),
                bounding_radius=bounding_radius,
                obj=obj,
            )
        )
    return slots


def object_bounding_radius(obj: SceneObject, compiled_verts: np.ndarray | None = None) -> float:
    """Return an object's horizontal bounding radius.

    Cubes are computed analytically; mesh-backed objects require the compiled
    mesh vertices (pass ``compiled_verts``), falling back to
    ``DEFAULT_BOUNDING_RADIUS`` when unavailable.
    """
    if isinstance(obj, CubeObject):
        return cube_bounding_radius(obj)
    if compiled_verts is not None:
        xy_extent = np.ptp(compiled_verts[:, :2], axis=0)
        return float(np.linalg.norm(xy_extent) / 2)
    return DEFAULT_BOUNDING_RADIUS
