"""Pure-data scene object classes.

Backend builders in so101_nexus_mujoco and so101_nexus_maniskill consume
these classes to instantiate simulator objects. No simulator imports here.

The ``__repr__`` of each class emits a natural-language description that
environments use to auto-generate task description strings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from so101_nexus_core.config import COLOR_MAP, YCB_OBJECTS


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


class CubeObject(SceneObject):
    """Axis-aligned box for use in simulation scenes.

    Args:
        half_size: Half-extent of each side in metres.
        mass: Object mass in kg.
        color: Named color key from COLOR_MAP (e.g. "red", "blue").
    """

    def __init__(
        self,
        half_size: float = 0.0125,
        mass: float = 0.01,
        color: str = "red",
    ) -> None:
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
        return f"{self.color} cube"


class YCBObject(SceneObject):
    """YCB dataset object identified by model_id (e.g. '003_cracker_box').

    Args:
        model_id: YCB dataset identifier. Must be a key in YCB_OBJECTS.
        mass_override: Optional mass in kg to override the default mesh mass.
    """

    def __init__(
        self,
        model_id: str,
        mass_override: float | None = None,
    ) -> None:
        if model_id not in YCB_OBJECTS:
            raise ValueError(f"model_id must be one of {list(YCB_OBJECTS)}, got {model_id!r}")
        if mass_override is not None and mass_override <= 0:
            raise ValueError(f"mass_override must be positive, got {mass_override}")
        self.model_id = model_id
        self.mass_override = mass_override

    def __repr__(self) -> str:  # noqa: D105
        return YCB_OBJECTS[self.model_id]


class MeshObject(SceneObject):
    """Arbitrary mesh object (collision + visual) for .obj/.stl support.

    Args:
        collision_mesh_path: Absolute path to the collision mesh file.
        visual_mesh_path: Absolute path to the visual mesh file.
        mass: Object mass in kg.
        name: Human-readable name used in task descriptions and ``__repr__``.
        scale: Uniform scale factor applied to the mesh.
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
