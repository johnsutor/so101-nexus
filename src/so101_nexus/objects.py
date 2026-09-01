"""Pure-data scene object classes.

Backend builders in so101_nexus.mujoco consume these classes to instantiate
simulator objects. No simulator imports here.

The ``__repr__`` of each class emits a natural-language description that
environments use to auto-generate task description strings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from so101_nexus.constants import COLOR_MAP, GSO_OBJECTS, YCB_OBJECTS, ColorName


def _resolve_half_size(
    half_size: float | None,
    *,
    length_mm: float | None,
    length_name: str,
) -> float:
    """Return a half-size in metres from one millimeter size input."""
    if half_size is not None and length_mm is not None:
        raise ValueError(f"Specify either half_size or {length_name}, not both")
    if length_mm is None:
        return 0.0125 if half_size is None else half_size
    if length_mm <= 0:
        raise ValueError(f"{length_name} must be positive, got {length_mm}")
    return length_mm / 2000.0


class SceneObject(ABC):
    """Abstract base class for all scene objects.

    Every concrete object type must implement ``__repr__`` to return a
    natural-language description (e.g. "red cube", "gelatin box"). This
    description is used by environments to auto-generate task strings and
    is the canonical string identity of the object for logging and display.

    Subclasses are also expected to validate their construction arguments
    and raise ``ValueError`` on invalid inputs.
    """

    @abstractmethod
    def __repr__(self) -> str:
        """Return a natural-language description of this object."""


class PrimitiveObject(SceneObject):
    """Shared base class for solid-color geometric primitives.

    Parameters
    ----------
    half_size : float
        Half the side length of the cube that contains the primitive, in metres.
    mass : float
        Object mass in kg.
    color : ColorName
        Named color from COLOR_MAP (e.g. "red", "blue").
    """

    shape_name: ClassVar[str]

    def __init__(
        self,
        half_size: float = 0.0125,
        mass: float = 0.01,
        color: ColorName = "red",
    ) -> None:
        if type(self) is PrimitiveObject:
            raise TypeError("PrimitiveObject is an abstract base class")
        if half_size <= 0:
            raise ValueError(f"half_size must be positive, got {half_size}")
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        if color not in COLOR_MAP:
            raise ValueError(f"color must be one of {list(COLOR_MAP)}, got {color!r}")
        self.half_size = half_size
        self.mass = mass
        self.color = color

    def __repr__(self) -> str:  # noqa: D105
        return f"{self.color} {self.shape_name}"


class CubeObject(PrimitiveObject):
    """Axis-aligned box for use in simulation scenes."""

    shape_name = "cube"

    def __init__(
        self,
        half_size: float | None = None,
        mass: float = 0.01,
        color: ColorName = "red",
        *,
        side_length_mm: float | None = None,
    ) -> None:
        super().__init__(
            half_size=_resolve_half_size(
                half_size, length_mm=side_length_mm, length_name="side_length_mm"
            ),
            mass=mass,
            color=color,
        )


class CylinderObject(PrimitiveObject):
    """Cylinder that fills the same cube as a :class:`CubeObject`."""

    shape_name = "cylinder"

    def __init__(
        self,
        half_size: float | None = None,
        mass: float = 0.01,
        color: ColorName = "red",
        *,
        diameter_mm: float | None = None,
    ) -> None:
        super().__init__(
            half_size=_resolve_half_size(
                half_size, length_mm=diameter_mm, length_name="diameter_mm"
            ),
            mass=mass,
            color=color,
        )


class SphereObject(PrimitiveObject):
    """Sphere that fills the same cube as a :class:`CubeObject`."""

    shape_name = "sphere"

    def __init__(
        self,
        half_size: float | None = None,
        mass: float = 0.01,
        color: ColorName = "red",
        *,
        diameter_mm: float | None = None,
    ) -> None:
        super().__init__(
            half_size=_resolve_half_size(
                half_size, length_mm=diameter_mm, length_name="diameter_mm"
            ),
            mass=mass,
            color=color,
        )


class PyramidObject(PrimitiveObject):
    """Square pyramid that fills the same cube as a :class:`CubeObject`."""

    shape_name = "pyramid"

    def __init__(
        self,
        half_size: float | None = None,
        mass: float = 0.01,
        color: ColorName = "red",
        *,
        side_length_mm: float | None = None,
    ) -> None:
        super().__init__(
            half_size=_resolve_half_size(
                half_size, length_mm=side_length_mm, length_name="side_length_mm"
            ),
            mass=mass,
            color=color,
        )


class ScannedMeshObject(SceneObject):
    """Shared base for dataset-scanned mesh objects (``YCBObject``, ``GSOObject``).

    Both datasets feed the same convex-hull collision pipeline (see
    ``so101_nexus.mesh_assets``): the visual mesh is the original scan, the
    collision geometry is a measured convex decomposition of it, and a
    surface-error gate keeps a single hull when that hull already matches the
    scan. Backend builders dispatch on this base type rather than repeating
    per-source ``isinstance`` checks.

    Parameters
    ----------
    model_id : str
        Dataset identifier. Must be a key in the subclass's object map.
    mass_override : float, optional
        Mass in kg to override the default mesh mass.
    """

    model_id: str
    mass_override: float | None

    def __init__(self, model_id: str, mass_override: float | None = None) -> None:
        if mass_override is not None and mass_override <= 0:
            raise ValueError(f"mass_override must be positive, got {mass_override}")
        self.model_id = model_id
        self.mass_override = mass_override


class YCBObject(ScannedMeshObject):
    """YCB dataset object identified by ``model_id``.

    The visual mesh is the original YCB scan. The collision geometry is a
    measured convex decomposition of it (see ``ensure_ycb_assets``), so physics
    sees concavities such as a fork handle or a spatula neck. A surface-error
    gate keeps a single hull when that hull already matches the scan.

    The decomposition needs the optional ``decomp`` extra
    (``pip install so101-nexus[decomp]``). Without it, collision geometry falls
    back to one convex hull:

    - The hull can bury the feature that makes an object graspable.
    - A hull wider than the open gripper makes a grasp impossible.

    Installing the extra after asset preparation rebuilds the cache on the next
    ``ensure_ycb_assets`` call. The manifest records the scan hash, generator
    configuration, and part properties. ``get_ycb_collision_meshes(model_id)``
    returns the OBJ parts that physics uses.

    Parameters
    ----------
    model_id : str
        YCB dataset identifier. Must be a key in YCB_OBJECTS.
    mass_override : float, optional
        Mass in kg to override the default mesh mass.
    """

    def __init__(
        self,
        model_id: str,
        mass_override: float | None = None,
    ) -> None:
        if model_id not in YCB_OBJECTS:
            raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {model_id!r}")
        super().__init__(model_id, mass_override)

    def __repr__(self) -> str:  # noqa: D105
        return YCB_OBJECTS[self.model_id]


class GSOObject(ScannedMeshObject):
    """Google Scanned Objects (GSO) dataset object identified by ``model_id``.

    Shares the ``YCBObject`` convex-hull collision pipeline (see
    ``so101_nexus.mesh_assets``); the visual mesh is the mirrored GSO scan
    (real-world scale, no rescale needed) and the collision geometry is a
    measured convex decomposition of it, gated the same way as YCB's.

    GSO ships no benchmark masses (unlike YCB, which has measured masses from
    its physical objects), so the default mass for each ``model_id`` is a
    hand-estimated volume-from-hull x assumed-density figure (see
    ``GSO_MASSES`` in ``so101_nexus.constants``); pass ``mass_override`` to
    replace it with a measured value.

    Parameters
    ----------
    model_id : str
        GSO dataset identifier. Must be a key in GSO_OBJECTS.
    mass_override : float, optional
        Mass in kg to override the default (hand-estimated) mesh mass.
    """

    def __init__(
        self,
        model_id: str,
        mass_override: float | None = None,
    ) -> None:
        if model_id not in GSO_OBJECTS:
            raise ValueError(f"model_id must be one of {list(GSO_OBJECTS)}, got {model_id!r}")
        super().__init__(model_id, mass_override)

    def __repr__(self) -> str:  # noqa: D105
        return GSO_OBJECTS[self.model_id]


class MeshObject(SceneObject):
    """Arbitrary mesh object (collision + visual) for .obj/.stl support.

    Parameters
    ----------
    collision_mesh_path : str
        Absolute path to the collision mesh file.
    visual_mesh_path : str
        Absolute path to the visual mesh file.
    mass : float
        Object mass in kg.
    name : str
        Human-readable name used in task descriptions and ``__repr__``.
    scale : float
        Uniform scale factor applied to the mesh.
    """

    def __init__(
        self,
        collision_mesh_path: str,
        visual_mesh_path: str,
        mass: float,
        name: str,
        scale: float = 1.0,
    ) -> None:
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.collision_mesh_path = collision_mesh_path
        self.visual_mesh_path = visual_mesh_path
        self.mass = mass
        self.name = name
        self.scale = scale

    def __repr__(self) -> str:  # noqa: D105
        return self.name
