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

from typing import TYPE_CHECKING

import mujoco
import numpy as np

from so101_nexus.constants import COLOR_MAP
from so101_nexus.objects import CubeObject, MeshObject, SceneObject, YCBObject
from so101_nexus.scene import SCENE_LIGHTS_XML, SCENE_VISUAL_XML
from so101_nexus.ycb_assets import (
    get_ycb_collision_parts,
    get_ycb_texture_file,
    get_ycb_visual_mesh,
)
from so101_nexus.ycb_geometry import get_mujoco_ycb_rest_pose

if TYPE_CHECKING:
    from collections.abc import Sequence

# Default bounding radius used when an object exposes no geometry to measure.
DEFAULT_BOUNDING_RADIUS = 0.025

# Default per-object mass (kg) for YCB objects without a mass override.
_DEFAULT_YCB_MASS = 0.01


def cube_bounding_radius(obj: CubeObject) -> float:
    """Return the horizontal bounding radius of a cube (half-diagonal)."""
    return float(obj.half_size * np.sqrt(2))


def collision_geom_name(slot_name: str, obj: SceneObject, part: int = 0) -> str:
    """Return the name of a slot's collision/contact geom.

    Mesh-backed slots (YCB, custom) carry one geom per convex part of their
    collision decomposition, numbered from zero; cubes carry a single box geom.
    """
    if isinstance(obj, CubeObject):
        return f"{slot_name}_geom"
    return _mesh_collision_geom_name(slot_name, part)


def _mesh_collision_geom_name(slot_name: str, part: int) -> str:
    return f"{slot_name}_collision_{part}"


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
    mass_fractions: Sequence[float] = (1.0,),
) -> str:
    """Return the MJCF ``<body>`` fragment for one freejoint mesh slot.

    Mesh slots carry hidden collision geoms (group 3) and a non-colliding visual
    geom (group 2); the latter is mostly for MuJoCo rendering parity (Warp
    training reads state tensors, not rendered images).

    The collision hull is a convex decomposition, so there is one collision geom
    per part. Each part declares ``mass * mass_fractions[k]`` rather than the
    full mass: MuJoCo sums geom masses into the body, so declaring ``mass`` on
    every part would multiply the object's weight by the part count.
    """
    material_attr = f' material="{material_name}"' if material_name else ""
    collision_geoms = "".join(
        f'      <geom name="{_mesh_collision_geom_name(slot_name, part)}" type="mesh" '
        f'mesh="pick_coll_{asset_index}_{part}"\n'
        f'            mass="{float(mass) * float(fraction)!r}" contype="1" conaffinity="1"\n'
        f'            group="3" condim="4" friction="1 0.05 0.001" solref="0.01 1"\n'
        f'            solimp="0.95 0.99 0.001"/>\n'
        for part, fraction in enumerate(mass_fractions)
    )
    return (
        f'    <body name="{slot_name}" pos="0.15 0 0.01">\n'
        f'      <freejoint name="{slot_name}_joint"/>\n'
        f"{collision_geoms}"
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
    overhead_camera_xml: str = "",
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
    overhead_camera_xml:
        Optional ``<camera>`` element injected into the worldbody, used when an
        ``OverheadCamera`` observation renders on the Warp backend.
    """
    gr, gg, gb, ga = ground_color
    asset_entries = ""
    body_entries = ""

    for i, (obj, slot) in enumerate(zip(objects, slot_names, strict=True)):
        if isinstance(obj, YCBObject):
            parts = get_ycb_collision_parts(obj.model_id)
            for k, part in enumerate(parts):
                asset_entries += (
                    f'    <mesh name="pick_coll_{i}_{k}" file="{part.path.as_posix()}"/>\n'
                )
            visual_path = get_ycb_visual_mesh(obj.model_id).as_posix()
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
            body_entries += mesh_xml_body(
                slot,
                i,
                mass,
                material_name=material_name,
                mass_fractions=[part.mass_fraction for part in parts],
            )
        elif isinstance(obj, MeshObject):
            asset_entries += (
                f'    <mesh name="pick_coll_{i}_0" file="{obj.collision_mesh_path}"'
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

{asset_section}{SCENE_VISUAL_XML}

  <worldbody>
{SCENE_LIGHTS_XML}
    <geom name="floor" type="plane" size="0 0 0.01" rgba="{gr} {gg} {gb} {ga}"
          pos="0 0 0" contype="1" conaffinity="1"/>

{body_entries}{extra_bodies}{overhead_camera_xml}  </worldbody>
</mujoco>
"""


class ObjectSlot:
    """Runtime metadata for one freejoint object slot in a compiled model.

    Backend-neutral: ``qpos_addr``/``dof_addr`` index the shared ``MjModel``
    layout and apply to a scalar ``MjData`` (MuJoCo) and a batched
    ``mjw.Data`` column (Warp) alike. ``rest_quat`` is a NumPy ``wxyz`` vector;
    backends convert to tensors as needed. ``geom_ids`` holds every collision
    geom of the slot (one per convex part of a decomposed mesh, one for a cube),
    which is what contact scans must aggregate over.
    """

    __slots__ = (
        "bounding_radius",
        "dof_addr",
        "geom_ids",
        "obj",
        "qpos_addr",
        "rest_quat",
        "spawn_z",
    )

    def __init__(
        self,
        qpos_addr: int,
        dof_addr: int,
        geom_ids: tuple[int, ...],
        rest_quat: np.ndarray,
        spawn_z: float,
        bounding_radius: float,
        obj: SceneObject,
    ) -> None:
        self.qpos_addr = qpos_addr
        self.dof_addr = dof_addr
        self.geom_ids = geom_ids
        self.rest_quat = rest_quat
        self.spawn_z = spawn_z
        self.bounding_radius = bounding_radius
        self.obj = obj

    @property
    def geom_id(self) -> int:
        """First collision geom of the slot; the whole object is ``geom_ids``."""
        return self.geom_ids[0]


def _slot_collision_geom_ids(
    mjm: mujoco.MjModel, slot_name: str, obj: SceneObject
) -> tuple[int, ...]:
    """Return every collision geom id of a slot body, in XML order."""
    geom_ids: list[int] = []
    if isinstance(obj, CubeObject):
        geom_id = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, f"{slot_name}_geom"))
        geom_ids = [geom_id] if geom_id >= 0 else []
    else:
        while True:
            name = _mesh_collision_geom_name(slot_name, len(geom_ids))
            geom_id = int(mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_GEOM, name))
            if geom_id < 0:
                break
            geom_ids.append(geom_id)
    if not geom_ids:
        # mj_name2id's -1 sentinel would silently address the model's last geom.
        raise ValueError(f"slot {slot_name!r} has no collision geom in the compiled model")
    return tuple(geom_ids)


def _body_frame_collision_verts(mjm: mujoco.MjModel, geom_ids: tuple[int, ...]) -> np.ndarray:
    """Return the union of a slot's compiled collision vertices, in body frame.

    The compiler recenters every mesh on its own center of mass and folds the
    offset into the geom frame, so raw ``mesh_vert`` blocks of a multi-part
    decomposition do not share an origin. Rest pose and footprint must see the
    whole object, not one part.
    """
    chunks = []
    rot = np.zeros(9)
    for geom_id in geom_ids:
        mesh_id = int(mjm.geom_dataid[geom_id])
        vert_start = int(mjm.mesh_vertadr[mesh_id])
        vert_count = int(mjm.mesh_vertnum[mesh_id])
        verts = mjm.mesh_vert[vert_start : vert_start + vert_count]
        mujoco.mju_quat2Mat(rot, mjm.geom_quat[geom_id])
        chunks.append(verts @ rot.reshape(3, 3).T + mjm.geom_pos[geom_id])
    return np.concatenate(chunks)


def extract_object_slots(
    mjm: mujoco.MjModel,
    slot_names: list[str],
    objects: list[SceneObject],
) -> list[ObjectSlot]:
    """Read per-slot runtime metadata from a compiled ``MjModel``.

    For cubes the rest pose is identity and the spawn height is the half-size;
    for mesh-backed objects (YCB, custom) the stable rest orientation and floor
    clearance come from the union of the compiled collision-part vertices.
    """
    slots: list[ObjectSlot] = []
    for slot_name, obj in zip(slot_names, objects, strict=True):
        geom_ids = _slot_collision_geom_ids(mjm, slot_name, obj)
        joint_id = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, f"{slot_name}_joint")
        qpos_addr = int(mjm.jnt_qposadr[joint_id])
        dof_addr = int(mjm.jnt_dofadr[joint_id])

        if isinstance(obj, (YCBObject, MeshObject)):
            verts = _body_frame_collision_verts(mjm, geom_ids)
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
                geom_ids=geom_ids,
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
