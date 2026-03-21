# Composable Observations + ManiSkill Primitive Environments

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a composable observation system to SO101-Nexus and implement Reach, LookAt, and Move environments on the ManiSkill backend.

**Architecture:** Observation components are lightweight descriptor classes (like `SceneObject`) defined in `so101_nexus_core`. Each backend's base env iterates over the component list to build the observation vector. Primitive env configs (`ReachConfig`, `LookAtConfig`, `MoveConfig`) and shared constants (`MoveDirection`, `DIRECTION_VECTORS`) move to core. ManiSkill gets three new env files with SO100/SO101 registration variants.

**Tech Stack:** Python 3.10+, Gymnasium, MuJoCo, ManiSkill, NumPy, PyTorch, SAPIEN

---

## File Map

### New Files

| File | Responsibility |
|------|---------------|
| `packages/so101-nexus-core/src/so101_nexus_core/observations.py` | `Observation` ABC + concrete component classes |
| `packages/so101-nexus-core/tests/test_observations.py` | Unit tests for observation component classes |
| `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/reach_env.py` | ManiSkill Reach env + SO100/SO101 variants |
| `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/look_at_env.py` | ManiSkill LookAt env + SO100/SO101 variants |
| `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/move_env.py` | ManiSkill Move env + SO100/SO101 variants |
| `packages/so101-nexus-maniskill/tests/test_reach_env.py` | Tests for ManiSkill Reach envs |
| `packages/so101-nexus-maniskill/tests/test_look_at_env.py` | Tests for ManiSkill LookAt envs |
| `packages/so101-nexus-maniskill/tests/test_move_env.py` | Tests for ManiSkill Move envs |
| `docs/content/docs/environments/maniskill-reach-so100.mdx` | Docs page |
| `docs/content/docs/environments/maniskill-reach-so101.mdx` | Docs page |
| `docs/content/docs/environments/maniskill-look-at-so100.mdx` | Docs page |
| `docs/content/docs/environments/maniskill-look-at-so101.mdx` | Docs page |
| `docs/content/docs/environments/maniskill-move-so100.mdx` | Docs page |
| `docs/content/docs/environments/maniskill-move-so101.mdx` | Docs page |

### Modified Files

| File | Changes |
|------|---------|
| `packages/so101-nexus-core/src/so101_nexus_core/config.py` | Add `MoveDirection`, `DIRECTION_VECTORS`, `ReachConfig`, `LookAtConfig`, `MoveConfig`; add `observations` param to `EnvironmentConfig` |
| `packages/so101-nexus-core/src/so101_nexus_core/__init__.py` | Export new observation classes, configs, and constants |
| `packages/so101-nexus-core/src/so101_nexus_core/env_ids.py` | Add 6 new ManiSkill env IDs |
| `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py` | Add `_compute_obs_components()` method; update `_finish_model_setup()` to compute obs size from components |
| `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/reach_env.py` | Import `ReachConfig` from core; use component-based observations; default `[JointPositions()]` |
| `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/look_at_env.py` | Import `LookAtConfig` from core; use component-based observations; default `[JointPositions()]` |
| `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/move_env.py` | Import `MoveConfig`, `MoveDirection`, `DIRECTION_VECTORS` from core; use component-based observations; default `[JointPositions()]` |
| `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_env.py` | Use component-based observations; default preserves current 18-dim |
| `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_and_place.py` | Use component-based observations; default preserves current shape |
| `packages/so101-nexus-mujoco/tests/test_reach_env.py` | Update expected obs shape from `(10,)` to `(6,)` |
| `packages/so101-nexus-mujoco/tests/test_look_at_env.py` | Update expected obs shape from `(10,)` to `(6,)` |
| `packages/so101-nexus-mujoco/tests/test_move_env.py` | Update expected obs shape from `(10,)` to `(6,)` |
| `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/__init__.py` | Import new env modules |
| `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/base_env.py` | Add `_compute_obs_components()` for observation building |
| `docs/content/docs/environments/index.mdx` | Add 6 new rows to table |
| `docs/content/docs/getting-started/environment-ids.mdx` | Add 6 new env IDs |
| `docs/content/docs/environments/mujoco-reach.mdx` | Update observation section |
| `docs/content/docs/environments/mujoco-look-at.mdx` | Update observation section |
| `docs/content/docs/environments/mujoco-move.mdx` | Update observation section |

---

## Task 1: Observation Component Classes in Core

**Files:**
- Create: `packages/so101-nexus-core/src/so101_nexus_core/observations.py`
- Create: `packages/so101-nexus-core/tests/test_observations.py`
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/__init__.py`

- [ ] **Step 1: Write failing tests for observation components**

```python
"""Tests for composable observation components."""

import pytest

from so101_nexus_core.observations import (
    EndEffectorPose,
    GazeDirection,
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    Observation,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
)


class TestObservationBase:
    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            Observation()

    def test_subclass_requires_name_and_size(self):
        class Bad(Observation):
            pass
        with pytest.raises(TypeError):
            Bad()


class TestStateComponents:
    @pytest.mark.parametrize(
        "cls,expected_name,expected_size",
        [
            (JointPositions, "joint_positions", 6),
            (EndEffectorPose, "end_effector_pose", 7),
            (TargetOffset, "target_offset", 3),
            (GazeDirection, "gaze_direction", 3),
            (GraspState, "grasp_state", 1),
            (ObjectPose, "object_pose", 7),
            (ObjectOffset, "object_offset", 3),
            (TargetPosition, "target_position", 3),
        ],
    )
    def test_name_and_size(self, cls, expected_name, expected_size):
        comp = cls()
        assert comp.name == expected_name
        assert comp.size == expected_size

    def test_repr_includes_name(self):
        comp = JointPositions()
        assert "JointPositions" in repr(comp)


class TestCameraComponents:
    def test_wrist_camera_defaults(self):
        cam = WristCamera()
        assert cam.name == "wrist_camera"
        assert cam.width == 224
        assert cam.height == 224

    def test_wrist_camera_custom_resolution(self):
        cam = WristCamera(width=128, height=128)
        assert cam.width == 128
        assert cam.height == 128

    def test_overhead_camera_defaults(self):
        cam = OverheadCamera()
        assert cam.name == "overhead_camera"
        assert cam.width == 224
        assert cam.height == 224

    def test_camera_invalid_resolution(self):
        with pytest.raises(ValueError):
            WristCamera(width=0, height=224)
        with pytest.raises(ValueError):
            OverheadCamera(width=224, height=-1)

    def test_camera_is_observation(self):
        assert isinstance(WristCamera(), Observation)
        assert isinstance(OverheadCamera(), Observation)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package so101-nexus-core pytest packages/so101-nexus-core/tests/test_observations.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement observation component classes**

Create `packages/so101-nexus-core/src/so101_nexus_core/observations.py`:

```python
"""Composable observation components for SO101-Nexus environments.

Observation components are lightweight descriptor classes that tell
environments which data to include in the observation vector. They
follow the same pattern as ``SceneObject`` subclasses: pure-data
config objects consumed by backend-specific environment code.

State components produce fixed-size slices of the flat observation
vector. Camera components add image tensors to a dict-style
observation space.

Typical usage::

    from so101_nexus_core.observations import JointPositions, TargetOffset

    config = ReachConfig(observations=[JointPositions(), TargetOffset()])
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Observation(ABC):
    """Abstract base for observation components.

    Every concrete component must define ``name`` (unique string key)
    and ``size`` (number of scalar dimensions for state components,
    or ``0`` for camera components whose shape depends on resolution).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this observation component."""

    @property
    @abstractmethod
    def size(self) -> int:
        """Number of scalar dimensions (0 for camera components)."""

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# State components — fixed-size slices of the flat observation vector
# ---------------------------------------------------------------------------


class JointPositions(Observation):
    """Current angle of each robot joint in radians (6-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "joint_positions"

    @property
    def size(self) -> int:  # noqa: D102
        return 6


class EndEffectorPose(Observation):
    """Position (x, y, z) and orientation (qw, qx, qy, qz) of the gripper tip in world coordinates (7-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "end_effector_pose"

    @property
    def size(self) -> int:  # noqa: D102
        return 7


class TargetOffset(Observation):
    """3D vector pointing from the gripper tip to the goal position (3-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "target_offset"

    @property
    def size(self) -> int:  # noqa: D102
        return 3


class GazeDirection(Observation):
    """Unit vector pointing from the gripper tip toward the target object, normalized to length 1 (3-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "gaze_direction"

    @property
    def size(self) -> int:  # noqa: D102
        return 3


class GraspState(Observation):
    """Whether the robot is currently holding an object: 1.0 = yes, 0.0 = no (1-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "grasp_state"

    @property
    def size(self) -> int:  # noqa: D102
        return 1


class ObjectPose(Observation):
    """Position (x, y, z) and orientation (qw, qx, qy, qz) of the target object in world coordinates (7-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "object_pose"

    @property
    def size(self) -> int:  # noqa: D102
        return 7


class ObjectOffset(Observation):
    """3D vector pointing from the gripper tip to the target object (3-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "object_offset"

    @property
    def size(self) -> int:  # noqa: D102
        return 3


class TargetPosition(Observation):
    """Absolute position (x, y, z) of the goal location in world coordinates (3-dim)."""

    @property
    def name(self) -> str:  # noqa: D102
        return "target_position"

    @property
    def size(self) -> int:  # noqa: D102
        return 3


# ---------------------------------------------------------------------------
# Camera components — add image tensors to dict-style observation spaces
# ---------------------------------------------------------------------------


class _CameraObservation(Observation):
    """Base for camera observation components.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
    """

    _name: str  # set by subclasses

    def __init__(self, width: int = 224, height: int = 224) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"Camera dimensions must be > 0, got {width}x{height}")
        self.width = width
        self.height = height

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def size(self) -> int:  # noqa: D102
        return 0

    def __repr__(self) -> str:  # noqa: D105
        return f"{type(self).__name__}(width={self.width}, height={self.height})"


class WristCamera(_CameraObservation):
    """RGB image from the camera mounted on the robot's wrist."""

    _name = "wrist_camera"


class OverheadCamera(_CameraObservation):
    """RGB image from the stationary camera above the workspace."""

    _name = "overhead_camera"
```

- [ ] **Step 4: Export from core `__init__.py`**

Add to `packages/so101-nexus-core/src/so101_nexus_core/__init__.py` after the existing object imports (after line 64):

```python
from so101_nexus_core.observations import (  # noqa: F401
    EndEffectorPose,
    GazeDirection,
    GraspState,
    JointPositions,
    ObjectOffset,
    ObjectPose,
    Observation,
    OverheadCamera,
    TargetOffset,
    TargetPosition,
    WristCamera,
)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --package so101-nexus-core pytest packages/so101-nexus-core/tests/test_observations.py -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add packages/so101-nexus-core/src/so101_nexus_core/observations.py \
      packages/so101-nexus-core/tests/test_observations.py \
      packages/so101-nexus-core/src/so101_nexus_core/__init__.py
git commit -m "feat: add composable observation component classes to core"
```

---

## Task 2: Move Shared Configs and Constants to Core

**Files:**
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/config.py`
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/__init__.py`

- [ ] **Step 1: Add `MoveDirection`, `DIRECTION_VECTORS` to `config.py`**

Add after the `SO101_JOINT_NAMES` tuple (after line 45):

```python
MoveDirection = Literal["up", "down", "left", "right", "forward", "backward"]

DIRECTION_VECTORS: dict[str, tuple[float, float, float]] = {
    "up": (0.0, 0.0, 1.0),
    "down": (0.0, 0.0, -1.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
}
```

- [ ] **Step 2: Add `observations` parameter to `EnvironmentConfig.__init__`**

In `EnvironmentConfig.__init__` (line 349), add `observations` parameter and store it. Add after `robot_init_qpos_noise` parameter:

```python
    def __init__(
        self,
        camera: CameraConfig | None = None,
        reward: RewardConfig | None = None,
        robot: RobotConfig | None = None,
        ground_colors: ColorConfig = "gray",
        max_episode_steps: int = 1024,
        goal_thresh: float = 0.025,
        spawn_half_size: float = 0.05,
        spawn_center: tuple[float, float] = (0.15, 0.0),
        spawn_min_radius: float = 0.20,
        spawn_max_radius: float = 0.40,
        spawn_angle_half_range_deg: float = 90.0,
        camera_mode: CameraMode = "fixed",
        obs_mode: ObsMode = "state",
        robot_colors: ColorConfig = "yellow",
        robot_init_qpos_noise: float = 0.02,
        observations: list[Observation] | None = None,
    ) -> None:
```

Add the import at the top of `config.py` (inside `TYPE_CHECKING` block):

```python
if TYPE_CHECKING:
    from so101_nexus_core.objects import SceneObject
    from so101_nexus_core.observations import Observation
```

Store in `__init__` body:

```python
        self.observations = observations
```

- [ ] **Step 3: Add `ReachConfig` to `config.py`**

Add after `PickAndPlaceConfig` (after line 527, before `SQRT_HALF`):

```python
class ReachConfig(EnvironmentConfig):
    """Config for the reach-to-target primitive task.

    Args:
        target_radius: Visual radius of the target site sphere (metres).
        target_workspace_half_extent: Half-width of the cubic workspace to
            sample target positions from (metres).
        success_threshold: TCP-to-target distance (m) that counts as success.
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        target_radius: float = 0.02,
        target_workspace_half_extent: float = 0.15,
        success_threshold: float = 0.02,
        **kwargs,
    ) -> None:
        kwargs.setdefault("max_episode_steps", 512)
        super().__init__(**kwargs)
        self.target_radius = target_radius
        self.target_workspace_half_extent = target_workspace_half_extent
        self.success_threshold = success_threshold
```

- [ ] **Step 4: Add `LookAtConfig` to `config.py`**

Add after `ReachConfig`:

```python
class LookAtConfig(EnvironmentConfig):
    """Config for the look-at primitive task.

    Args:
        objects: Object(s) to sample as the look-at target. Accepts a single
            SceneObject, a list, or None (defaults to [CubeObject()]).
            Only CubeObject targets are currently supported.
        orientation_success_threshold_deg: Max angular error in degrees for success.
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        objects: list[SceneObject] | SceneObject | None = None,
        orientation_success_threshold_deg: float = 5.73,
        **kwargs,
    ) -> None:
        kwargs.setdefault("max_episode_steps", 256)
        super().__init__(**kwargs)
        from so101_nexus_core.objects import CubeObject, SceneObject as _SceneObject  # noqa: PLC0415

        if objects is None:
            self.objects: list[SceneObject] = [CubeObject()]
        elif isinstance(objects, _SceneObject):
            self.objects = [objects]
        else:
            self.objects = list(objects)
        self.orientation_success_threshold_deg = orientation_success_threshold_deg
        for obj in self.objects:
            if not isinstance(obj, CubeObject):
                raise TypeError(
                    f"LookAtConfig only supports CubeObject targets, got {type(obj).__name__}"
                )

    @property
    def _orientation_success_threshold_rad(self) -> float:
        """Orientation success threshold converted to radians (internal use only)."""
        return float(np.radians(self.orientation_success_threshold_deg))
```

- [ ] **Step 5: Add `MoveConfig` to `config.py`**

Add after `LookAtConfig`:

```python
class MoveConfig(EnvironmentConfig):
    """Config for the directional move primitive task.

    Args:
        direction: Cardinal direction to move the TCP.
        target_distance: Distance in metres to travel from the initial TCP position.
        success_threshold: Max residual distance (m) to count as success.
        **kwargs: Forwarded to EnvironmentConfig.
    """

    def __init__(
        self,
        direction: MoveDirection = "up",
        target_distance: float = 0.10,
        success_threshold: float = 0.01,
        **kwargs,
    ) -> None:
        kwargs.setdefault("max_episode_steps", 256)
        if direction not in DIRECTION_VECTORS:
            raise ValueError(
                f"direction must be one of {list(DIRECTION_VECTORS)}, got {direction!r}"
            )
        super().__init__(**kwargs)
        self.direction = direction
        self.target_distance = target_distance
        self.success_threshold = success_threshold
```

- [ ] **Step 6: Export new types from core `__init__.py`**

Add exports for `MoveDirection`, `DIRECTION_VECTORS`, `ReachConfig`, `LookAtConfig`, `MoveConfig` following the existing grouped-import pattern.

- [ ] **Step 7: Run existing core tests to verify no regressions**

Run: `uv run --package so101-nexus-core pytest packages/so101-nexus-core/tests/ -v`
Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add packages/so101-nexus-core/src/so101_nexus_core/config.py \
      packages/so101-nexus-core/src/so101_nexus_core/__init__.py
git commit -m "feat: add ReachConfig, LookAtConfig, MoveConfig and observations param to core"
```

---

## Task 3: MuJoCo Base Env — Component-Based Observation Building

**Files:**
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py`

- [ ] **Step 1: Add `_compute_obs_components()` method to `SO101NexusMuJoCoBaseEnv`**

Add after `_get_current_qpos()` (after line 301). This method iterates over `self.config.observations` and builds the flat state vector. Each component type dispatches to the relevant data source. Subclasses override `_get_component_data(component)` for task-specific components (e.g., `TargetOffset`, `GazeDirection`).

```python
    def _compute_obs_components(self) -> np.ndarray:
        """Build the flat state vector from the observation component list."""
        from so101_nexus_core.observations import (
            EndEffectorPose,
            GazeDirection,
            GraspState,
            JointPositions,
            ObjectOffset,
            ObjectPose,
            TargetOffset,
            TargetPosition,
        )

        parts: list[np.ndarray] = []
        for comp in self.config.observations:
            if isinstance(comp, JointPositions):
                parts.append(self._get_current_qpos())
            elif isinstance(comp, EndEffectorPose):
                parts.append(self._get_tcp_pose())
            elif isinstance(comp, GraspState):
                parts.append(np.array([self._is_grasping()]))
            elif isinstance(comp, (TargetOffset, GazeDirection, ObjectPose, ObjectOffset, TargetPosition)):
                parts.append(self._get_component_data(comp))
            else:
                raise ValueError(f"Unsupported observation component: {comp!r}")
        return np.concatenate(parts)

    def _get_component_data(self, component: object) -> np.ndarray:
        """Return data for a task-specific observation component.

        Subclasses override this for components like TargetOffset or GazeDirection
        that depend on task state (target position, object position, etc.).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support observation component {component!r}"
        )
```

- [ ] **Step 2: Update `_state_obs_size()` to compute from components**

Replace the existing `_state_obs_size` method (line 237-243) to compute size from components when they are configured:

```python
    def _state_obs_size(self) -> int:
        """Return the dimensionality of the flat state observation vector."""
        if self.config.observations is not None:
            return sum(c.size for c in self.config.observations if c.size > 0)
        # Legacy default for pick envs that haven't migrated yet
        return 18
```

- [ ] **Step 3: Run existing MuJoCo tests (expect them to still pass — pick envs unchanged)**

Run: `uv run --package so101-nexus-mujoco pytest packages/so101-nexus-mujoco/tests/test_pick_env.py -v`
Expected: All PASS (pick envs don't set `observations`, so legacy default 18 is used)

- [ ] **Step 4: Commit**

```bash
git add packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py
git commit -m "feat: add component-based observation building to MuJoCo base env"
```

---

## Task 4: Refactor MuJoCo Reach, LookAt, Move to Use Observation Components

**Files:**
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/reach_env.py`
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/look_at_env.py`
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/move_env.py`
- Modify: `packages/so101-nexus-mujoco/tests/test_reach_env.py`
- Modify: `packages/so101-nexus-mujoco/tests/test_look_at_env.py`
- Modify: `packages/so101-nexus-mujoco/tests/test_move_env.py`

- [ ] **Step 1: Update MuJoCo tests to expect 6-dim observations**

In each test file, change `assert obs.shape == (10,)` to `assert obs.shape == (6,)`.

Also add a test verifying custom observation components work:

```python
    def test_custom_observations(self):
        from so101_nexus_core.observations import JointPositions, EndEffectorPose
        # For reach_env:
        from so101_nexus_core.config import ReachConfig
        config = ReachConfig(observations=[JointPositions(), EndEffectorPose()])
        env = gym.make("MuJoCoReach-v1", config=config)
        obs, _ = env.reset()
        assert obs.shape == (13,)  # 6 + 7
        env.close()
```

(Similar for LookAt with `LookAtConfig` and Move with `MoveConfig`.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package so101-nexus-mujoco pytest packages/so101-nexus-mujoco/tests/test_reach_env.py packages/so101-nexus-mujoco/tests/test_look_at_env.py packages/so101-nexus-mujoco/tests/test_move_env.py -v`
Expected: FAIL (still 10-dim, custom observations not wired)

- [ ] **Step 3: Refactor `reach_env.py`**

Key changes:
1. Remove `ReachConfig` class definition — import from `so101_nexus_core.config`
2. Set default observations to `[JointPositions()]` when `config.observations is None`
3. Replace hardcoded `_get_obs()` with `_compute_obs_components()`
4. Implement `_get_component_data()` for `TargetOffset` (returns `self._target_pos - tcp_pos`)
5. Keep `_state_obs_size()` override removed (base class computes from components)

```python
from so101_nexus_core.config import ReachConfig, sample_color
from so101_nexus_core.observations import JointPositions

# In __init__, after config is created:
if config.observations is None:
    config.observations = [JointPositions()]

# Replace _get_obs:
def _get_obs(self) -> np.ndarray | dict:
    state = self._compute_obs_components()
    if self.camera_mode == "wrist":
        # ... wrist camera handling unchanged ...
        if self.config.obs_mode == "visual":
            self._privileged_state = state
            return {"state": self._get_current_qpos(), "wrist_camera": wrist_image}
        return {"state": state, "wrist_camera": wrist_image}
    return state

# Add _get_component_data:
def _get_component_data(self, component: object) -> np.ndarray:
    from so101_nexus_core.observations import TargetOffset
    if isinstance(component, TargetOffset):
        return self._target_pos - self._get_tcp_pose()[:3]
    return super()._get_component_data(component)
```

Remove `_state_obs_size()` override (base class handles it).

- [ ] **Step 4: Refactor `look_at_env.py`**

Same pattern. Key differences:
1. Remove `LookAtConfig` — import from `so101_nexus_core.config`
2. Default observations: `[JointPositions()]`
3. Implement `_get_component_data()` for `GazeDirection` (returns normalized direction from TCP to target)

```python
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
```

- [ ] **Step 5: Refactor `move_env.py`**

Same pattern. Key differences:
1. Remove `MoveConfig`, `MoveDirection`, `_DIRECTION_VEC` — import from `so101_nexus_core.config`
2. Default observations: `[JointPositions()]`
3. Implement `_get_component_data()` for `TargetOffset`

```python
from so101_nexus_core.config import DIRECTION_VECTORS, MoveConfig, sample_color
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --package so101-nexus-mujoco pytest packages/so101-nexus-mujoco/tests/test_reach_env.py packages/so101-nexus-mujoco/tests/test_look_at_env.py packages/so101-nexus-mujoco/tests/test_move_env.py -v`
Expected: All PASS

- [ ] **Step 7: Run full MuJoCo test suite to check for regressions**

Run: `make test-mujoco`
Expected: All PASS (pick envs still use legacy 18-dim default)

- [ ] **Step 8: Commit**

```bash
git add packages/so101-nexus-mujoco/src/so101_nexus_mujoco/reach_env.py \
      packages/so101-nexus-mujoco/src/so101_nexus_mujoco/look_at_env.py \
      packages/so101-nexus-mujoco/src/so101_nexus_mujoco/move_env.py \
      packages/so101-nexus-mujoco/tests/test_reach_env.py \
      packages/so101-nexus-mujoco/tests/test_look_at_env.py \
      packages/so101-nexus-mujoco/tests/test_move_env.py
git commit -m "refactor: MuJoCo Reach/LookAt/Move use composable observations, default to JointPositions"
```

---

## Task 5: Refactor MuJoCo Pick and PickAndPlace to Use Observation Components

**Files:**
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_env.py`
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_and_place.py`

- [ ] **Step 1: Refactor `pick_env.py`**

Set default observations to preserve current 18-dim behavior:

```python
from so101_nexus_core.observations import (
    EndEffectorPose,
    GraspState,
    ObjectPose,
    ObjectOffset,
)

# In __init__:
if config.observations is None:
    config.observations = [EndEffectorPose(), GraspState(), ObjectPose(), ObjectOffset()]
```

Replace `_get_obs()` body with `_compute_obs_components()` call. Implement `_get_component_data()` for `ObjectPose` and `ObjectOffset` using existing `_get_target_pose()` and TCP pose.

- [ ] **Step 2: Refactor `pick_and_place.py`**

Preserving current 24-dim: `[EndEffectorPose(), GraspState(), TargetPosition(), ObjectPose(), ObjectOffset(), TargetOffset()]` = 7+1+3+7+3+3 = 24. Here `TargetPosition` is the absolute position of the target disc, `ObjectOffset` is the vector from TCP to the cube, and `TargetOffset` is the vector from the cube to the target disc (`obj_to_target_pos`). Implement `_get_component_data()` accordingly.

- [ ] **Step 3: Run full MuJoCo tests**

Run: `make test-mujoco`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_env.py \
      packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_and_place.py
git commit -m "refactor: MuJoCo Pick/PickAndPlace use composable observations"
```

---

## Task 6: ManiSkill Base Env — Component-Based Observation Support

**Files:**
- Modify: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/base_env.py`

- [ ] **Step 1: Add observation component support to ManiSkill base env**

Add a `_build_obs_extra_from_components()` method that iterates over `self.config.observations` and builds the `_get_obs_extra` dict. In ManiSkill, `agent_qpos` and `agent_qvel` are automatically included by the framework — so `JointPositions` is a no-op in `_get_obs_extra` (it's always present).

```python
    def _build_obs_extra_from_components(self, info: dict) -> dict[str, torch.Tensor]:
        """Build obs_extra dict from observation components.

        ManiSkill automatically includes agent qpos/qvel. This method adds
        task-specific components from config.observations.
        """
        from so101_nexus_core.observations import (
            EndEffectorPose,
            GraspState,
            JointPositions,
        )

        obs: dict[str, torch.Tensor] = {}
        if self.config.observations is None:
            return obs
        for comp in self.config.observations:
            if isinstance(comp, JointPositions):
                continue  # ManiSkill includes qpos automatically
            elif isinstance(comp, EndEffectorPose):
                obs["tcp_pose"] = self.agent.tcp_pose.raw_pose
            elif isinstance(comp, GraspState):
                obs["is_grasped"] = info.get("is_grasped", torch.zeros(self.num_envs, device=self.device))
            else:
                self._add_component_obs(obs, comp, info)
        return obs

    def _add_component_obs(
        self, obs: dict[str, torch.Tensor], component: object, info: dict
    ) -> None:
        """Add a task-specific component to obs dict. Subclasses override."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support observation component {component!r}"
        )
```

- [ ] **Step 2: Commit**

```bash
git add packages/so101-nexus-maniskill/src/so101_nexus_maniskill/base_env.py
git commit -m "feat: add component-based observation support to ManiSkill base env"
```

---

## Task 7: ManiSkill Reach Environment

**Files:**
- Create: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/reach_env.py`
- Create: `packages/so101-nexus-maniskill/tests/test_reach_env.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for ManiSkill Reach environments."""

import gymnasium as gym
import pytest
import torch

import so101_nexus_maniskill  # noqa: F401
from so101_nexus_core.config import ReachConfig
from so101_nexus_core.observations import JointPositions, TargetOffset

BASE_KWARGS = dict(obs_mode="state", num_envs=1, render_mode=None)

REACH_ENV_IDS = [
    ("ManiSkillReachSO100-v1", "so100"),
    ("ManiSkillReachSO101-v1", "so101"),
]


@pytest.fixture(scope="module")
def reach_so100_env():
    env = gym.make("ManiSkillReachSO100-v1", **BASE_KWARGS)
    yield env
    env.close()


@pytest.fixture(scope="module")
def reach_so101_env():
    env = gym.make("ManiSkillReachSO101-v1", **BASE_KWARGS)
    yield env
    env.close()


def _get_env(request, env_id):
    mapping = {
        "ManiSkillReachSO100-v1": "reach_so100_env",
        "ManiSkillReachSO101-v1": "reach_so101_env",
    }
    return request.getfixturevalue(mapping[env_id])


class TestEnvCreation:
    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_env_creates(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert isinstance(env, gym.Env)

    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_env_reset_and_step(self, request, env_id, robot):
        env = _get_env(request, env_id)
        obs, info = env.reset()
        assert isinstance(obs, torch.Tensor)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert reward is not None

    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_action_space_shape(self, request, env_id, robot):
        env = _get_env(request, env_id)
        assert env.action_space.shape == (6,)


class TestEpisodeLogic:
    EVALUATE_KEYS = {"tcp_to_target_dist", "success"}

    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_evaluate_keys(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        info = env.unwrapped.evaluate()
        assert self.EVALUATE_KEYS <= set(info.keys())

    @pytest.mark.parametrize("env_id,robot", REACH_ENV_IDS)
    def test_reward_range(self, request, env_id, robot):
        env = _get_env(request, env_id)
        env.reset()
        _, reward, _, _, _ = env.step(env.action_space.sample())
        assert (reward >= 0).all()
        assert (reward <= 1).all()


class TestTaskDescription:
    def test_task_description_nonempty(self, reach_so100_env):
        assert reach_so100_env.unwrapped.task_description
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package so101-nexus-maniskill pytest packages/so101-nexus-maniskill/tests/test_reach_env.py -v --prerelease=allow`
Expected: FAIL (env not registered)

- [ ] **Step 3: Implement `reach_env.py`**

Create `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/reach_env.py`:

Follow the pattern from `pick_env.py`. Key structure:

```python
"""ManiSkill reach-to-target environment."""

from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose

from so101_nexus_core.config import ReachConfig, sample_color
from so101_nexus_core.observations import JointPositions
from so101_nexus_core.robot_presets import build_maniskill_robot_configs
from so101_nexus_maniskill.base_env import SO101NexusManiSkillBaseEnv

_DEFAULT_CONFIG = ReachConfig()


class ReachEnv(SO101NexusManiSkillBaseEnv):
    """Reach primitive: move TCP to a randomly sampled 3-D target position.

    The target is a visual-only sphere. No graspable objects.
    """

    config: ReachConfig
    task_description = "Move the robot's end-effector to the target position."

    def __init__(
        self,
        *args,
        config: ReachConfig = ReachConfig(),
        robot_uids: str = "so100",
        num_envs: int = 1,
        reconfiguration_freq: int | None = None,
        **kwargs,
    ):
        if config.observations is None:
            config.observations = [JointPositions()]

        robot_cfgs = build_maniskill_robot_configs(config=config)
        self._setup_base(config=config, robot_uids=robot_uids, robot_cfgs=robot_cfgs)

        if reconfiguration_freq is None:
            reconfiguration_freq = 1 if config.camera_mode in ("wrist", "both") else 0

        self._target_pos: torch.Tensor | None = None

        super().__init__(
            *args,
            robot_uids=robot_uids,
            reconfiguration_freq=reconfiguration_freq,
            num_envs=num_envs,
            **kwargs,
        )

    def _load_scene(self, options: dict) -> None:
        self._build_ground()
        # Build target site as a kinematic actor (visual only)
        targets: list[Actor] = []
        for i in range(self.num_envs):
            builder = self.scene.create_actor_builder()
            builder.add_sphere_visual(
                radius=self.config.target_radius,
                material=sapien.render.RenderMaterial(
                    base_color=[1.0, 0.5, 0.0, 0.7],
                ),
            )
            builder.initial_pose = sapien.Pose(p=[0.15, 0, 0.15])
            builder.set_scene_idxs([i])
            target = builder.build_kinematic(name=f"reach_target-{i}")
            targets.append(target)
            self.remove_from_state_dict_registry(target)
        self.target_site = Actor.merge(targets, name="reach_target")
        self.add_to_state_dict_registry(self.target_site)
        self._apply_robot_color_if_needed()

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict) -> None:
        with torch.device(self.device):
            b = len(env_idx)
            self._reset_robot(env_idx)

            half = self.config.target_workspace_half_extent
            center = torch.tensor([0.15, 0.0, 0.15], device=self.device)
            offset = (torch.rand(b, 3, device=self.device) * 2 - 1) * half
            pos = center.unsqueeze(0) + offset
            pos[:, 2] = pos[:, 2].clamp(min=0.05)
            q = torch.tensor([[1, 0, 0, 0]], device=self.device, dtype=torch.float32).expand(b, -1)
            self.target_site.set_pose(Pose.create_from_pq(p=pos, q=q))
            self._target_pos = pos

    def evaluate(self) -> dict[str, torch.Tensor]:
        tcp_to_target = self.target_site.pose.p - self.agent.tcp_pose.p
        dist = torch.linalg.norm(tcp_to_target, axis=1)
        return {
            "tcp_to_target_dist": dist,
            "success": dist < self.config.success_threshold,
        }

    def _get_obs_extra(self, info: dict) -> dict[str, torch.Tensor]:
        return self._build_obs_extra_from_components(info)

    def _add_component_obs(
        self, obs: dict[str, torch.Tensor], component: object, info: dict
    ) -> None:
        from so101_nexus_core.observations import TargetOffset
        if isinstance(component, TargetOffset):
            obs["target_offset"] = self.target_site.pose.p - self.agent.tcp_pose.p
        else:
            super()._add_component_obs(obs, component, info)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict) -> torch.Tensor:
        reach = self._reach_progress(info["tcp_to_target_dist"])
        bonus = self.config.reward.completion_bonus
        return (1.0 - bonus) * reach + bonus * info["success"]
```

- [ ] **Step 4: Register SO100 and SO101 variants**

At the bottom of `reach_env.py`, use the same `_register_robot_variant` pattern from `pick_env.py`:

```python
def _register_robot_variant(*, class_name, env_id, base_cls, robot_uid):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("robot_uids", robot_uid)
        base_cls.__init__(self, *args, **kwargs)
    cls = type(class_name, (base_cls,), {"__init__": __init__})
    cls = register_env(env_id, max_episode_steps=_DEFAULT_CONFIG.max_episode_steps)(cls)
    globals()[class_name] = cls
    return cls

ReachSO100Env = _register_robot_variant(
    class_name="ReachSO100Env", env_id="ManiSkillReachSO100-v1",
    base_cls=ReachEnv, robot_uid="so100",
)
ReachSO101Env = _register_robot_variant(
    class_name="ReachSO101Env", env_id="ManiSkillReachSO101-v1",
    base_cls=ReachEnv, robot_uid="so101",
)
```

- [ ] **Step 5: Add import to `__init__.py`**

Add `reach_env` to imports in `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/__init__.py`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run --package so101-nexus-maniskill pytest packages/so101-nexus-maniskill/tests/test_reach_env.py -v --prerelease=allow`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add packages/so101-nexus-maniskill/src/so101_nexus_maniskill/reach_env.py \
      packages/so101-nexus-maniskill/src/so101_nexus_maniskill/__init__.py \
      packages/so101-nexus-maniskill/tests/test_reach_env.py
git commit -m "feat: add ManiSkill Reach environment (SO100 + SO101)"
```

---

## Task 8: ManiSkill LookAt Environment

**Files:**
- Create: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/look_at_env.py`
- Create: `packages/so101-nexus-maniskill/tests/test_look_at_env.py`

- [ ] **Step 1: Write failing tests**

Same structure as Reach tests but with:
- Env IDs: `ManiSkillLookAtSO100-v1`, `ManiSkillLookAtSO101-v1`
- Evaluate keys: `{"orientation_error", "success"}`
- Task description check includes target object

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --package so101-nexus-maniskill pytest packages/so101-nexus-maniskill/tests/test_look_at_env.py -v --prerelease=allow`

- [ ] **Step 3: Implement `look_at_env.py`**

Follow Reach pattern. Key differences:
- Uses `LookAtConfig` — spawns a `CubeObject` as visual target (kinematic actor, not dynamic)
- `_initialize_episode` places the cube randomly in spawn region
- `evaluate()` computes `orientation_error` using TCP forward vector (z-axis of TCP link rotation) vs direction to target; `success` is `orientation_error < threshold_rad`
- Reward uses cosine similarity: `(cos_sim + 1) / 2`
- `_get_component_data` / `_add_component_obs` handles `GazeDirection`
- `task_description` property auto-generated: `"Look at the {repr(target_obj)}."`

```python
def evaluate(self) -> dict[str, torch.Tensor]:
    tcp_pose = self.agent.tcp_pose
    # TCP forward = z-axis of rotation matrix (third column)
    rot_mat = tcp_pose.to_transformation_matrix()[..., :3, :3]
    tcp_forward = rot_mat[..., :, 2]  # (num_envs, 3)
    to_target = self.target_obj_actor.pose.p - tcp_pose.p
    to_target_norm = to_target / (torch.linalg.norm(to_target, dim=1, keepdim=True) + 1e-8)
    cos_sim = (tcp_forward * to_target_norm).sum(dim=1).clamp(-1, 1)
    orientation_error = torch.arccos(cos_sim)
    return {
        "orientation_error": orientation_error,
        "success": orientation_error < self.config._orientation_success_threshold_rad,
    }
```

- [ ] **Step 4: Register SO100/SO101 variants with `max_episode_steps=256`**

- [ ] **Step 5: Add import to `__init__.py`**

- [ ] **Step 6: Run tests**

Run: `uv run --package so101-nexus-maniskill pytest packages/so101-nexus-maniskill/tests/test_look_at_env.py -v --prerelease=allow`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add packages/so101-nexus-maniskill/src/so101_nexus_maniskill/look_at_env.py \
      packages/so101-nexus-maniskill/src/so101_nexus_maniskill/__init__.py \
      packages/so101-nexus-maniskill/tests/test_look_at_env.py
git commit -m "feat: add ManiSkill LookAt environment (SO100 + SO101)"
```

---

## Task 9: ManiSkill Move Environment

**Files:**
- Create: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/move_env.py`
- Create: `packages/so101-nexus-maniskill/tests/test_move_env.py`

- [ ] **Step 1: Write failing tests**

Same structure as Reach tests but with:
- Env IDs: `ManiSkillMoveSO100-v1`, `ManiSkillMoveSO101-v1`
- Evaluate keys: `{"tcp_to_target_dist", "success"}`
- Task description format: `"Move the end-effector up by 0.10 m."`
- `max_episode_steps=256`

- [ ] **Step 2: Run tests to verify they fail**

- [ ] **Step 3: Implement `move_env.py`**

Key differences from Reach:
- Uses `MoveConfig` with `direction` and `target_distance`
- `_initialize_episode` computes target from initial TCP position + direction_vector * distance
- Target is a visual-only sphere (kinematic actor)
- After `_reset_robot`, need to step physics once to get accurate TCP position, then compute target
- `task_description` property: `f"Move the end-effector {direction} by {distance:.2f} m."`
- Same tanh-shaped reach reward as Reach

```python
from so101_nexus_core.config import DIRECTION_VECTORS, MoveConfig

# In _initialize_episode:
self._reset_robot(env_idx)
self.scene._gpu_apply_all()
self.scene.px.gpu_update_articulation_kinematics()
self.scene._gpu_fetch_all()

tcp_pos = self.agent.tcp_pose.p.clone()
dir_vec = torch.tensor(
    DIRECTION_VECTORS[self.config.direction], device=self.device, dtype=torch.float32
)
target = tcp_pos + dir_vec * self.config.target_distance
target[:, 2] = target[:, 2].clamp(min=0.02)
```

- [ ] **Step 4: Register SO100/SO101 variants with `max_episode_steps=256`**

- [ ] **Step 5: Add import to `__init__.py`**

- [ ] **Step 6: Run tests**

Run: `uv run --package so101-nexus-maniskill pytest packages/so101-nexus-maniskill/tests/test_move_env.py -v --prerelease=allow`
Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add packages/so101-nexus-maniskill/src/so101_nexus_maniskill/move_env.py \
      packages/so101-nexus-maniskill/src/so101_nexus_maniskill/__init__.py \
      packages/so101-nexus-maniskill/tests/test_move_env.py
git commit -m "feat: add ManiSkill Move environment (SO100 + SO101)"
```

---

## Task 10: Update Environment Registry and Run Full Test Suite

**Files:**
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/env_ids.py`

- [ ] **Step 1: Add new env IDs to `all_registered_env_ids()`**

```python
def all_registered_env_ids() -> list[str]:
    return [
        "MuJoCoPickLift-v1",
        "MuJoCoPickAndPlace-v1",
        "MuJoCoReach-v1",
        "MuJoCoLookAt-v1",
        "MuJoCoMove-v1",
        "ManiSkillPickLiftSO100-v1",
        "ManiSkillPickLiftSO101-v1",
        "ManiSkillReachSO100-v1",
        "ManiSkillReachSO101-v1",
        "ManiSkillLookAtSO100-v1",
        "ManiSkillLookAtSO101-v1",
        "ManiSkillMoveSO100-v1",
        "ManiSkillMoveSO101-v1",
    ]
```

- [ ] **Step 2: Run full test suites**

Run: `make test`
Expected: All PASS

- [ ] **Step 3: Run lint and typecheck**

Run: `make format && make lint && make typecheck`
Expected: Clean

- [ ] **Step 4: Commit**

```bash
git add packages/so101-nexus-core/src/so101_nexus_core/env_ids.py
git commit -m "feat: register ManiSkill Reach/LookAt/Move env IDs"
```

---

## Task 11: Documentation Updates

**Files:**
- Create: 6 new MDX files in `docs/content/docs/environments/`
- Modify: `docs/content/docs/environments/index.mdx`
- Modify: `docs/content/docs/getting-started/environment-ids.mdx`
- Modify: `docs/content/docs/environments/mujoco-reach.mdx`
- Modify: `docs/content/docs/environments/mujoco-look-at.mdx`
- Modify: `docs/content/docs/environments/mujoco-move.mdx`

- [ ] **Step 1: Create ManiSkill Reach SO100 docs page**

Create `docs/content/docs/environments/maniskill-reach-so100.mdx` following the template from `maniskill-pick-lift-so100.mdx`:

```mdx
---
title: ManiSkillReachSO100-v1
description: Move the TCP to a target position with the SO-100 arm on the ManiSkill backend.
---

# ManiSkillReachSO100-v1

Move the robot's end-effector (TCP) to a randomly sampled target position using the SO-100 robot arm in ManiSkill.

## Overview

| Property | Value |
|----------|-------|
| Environment ID | `ManiSkillReachSO100-v1` |
| Backend | ManiSkill |
| Robot | SO-100 |
| Max episode steps | 512 |
| Config class | `ReachConfig` |

## Observation Space

Default observation includes only joint positions (configurable via `observations` parameter):

| Component | Class | Dimensions | Description |
|-----------|-------|------------|-------------|
| Joint positions | `JointPositions` | 6 | Current angle of each robot joint in radians |

## Objects

No graspable objects. The target is a visual-only sphere rendered in the scene.

## Success Condition

The episode succeeds when the distance from the TCP to the target is less than `success_threshold` (default 0.02 m).

## Vectorized Simulation

ManiSkill supports running multiple environment instances in parallel via the `num_envs` argument.

## Example

\`\`\`python
import gymnasium as gym
import so101_nexus_maniskill

from so101_nexus_core.config import ReachConfig
from so101_nexus_core.observations import JointPositions, TargetOffset

# Default (joint positions only)
env = gym.make("ManiSkillReachSO100-v1", obs_mode="state", num_envs=1)

# With custom observations
config = ReachConfig(observations=[JointPositions(), TargetOffset()])
env = gym.make("ManiSkillReachSO100-v1", config=config, obs_mode="state", num_envs=1)
\`\`\`
```

- [ ] **Step 2: Create remaining 5 ManiSkill env doc pages**

Create analogous pages for:
- `maniskill-reach-so101.mdx`
- `maniskill-look-at-so100.mdx`
- `maniskill-look-at-so101.mdx`
- `maniskill-move-so100.mdx`
- `maniskill-move-so101.mdx`

Each follows the same template with appropriate env ID, robot name, config class, evaluate keys, and success condition.

- [ ] **Step 3: Update `environments/index.mdx`**

Add 6 new rows to the table:

```
| [`ManiSkillReachSO100-v1`](/docs/environments/maniskill-reach-so100) | ManiSkill | Move TCP to target (SO-100) | 512 |
| [`ManiSkillReachSO101-v1`](/docs/environments/maniskill-reach-so101) | ManiSkill | Move TCP to target (SO-101) | 512 |
| [`ManiSkillLookAtSO100-v1`](/docs/environments/maniskill-look-at-so100) | ManiSkill | Orient TCP toward object (SO-100) | 256 |
| [`ManiSkillLookAtSO101-v1`](/docs/environments/maniskill-look-at-so101) | ManiSkill | Orient TCP toward object (SO-101) | 256 |
| [`ManiSkillMoveSO100-v1`](/docs/environments/maniskill-move-so100) | ManiSkill | Move TCP in a direction (SO-100) | 256 |
| [`ManiSkillMoveSO101-v1`](/docs/environments/maniskill-move-so101) | ManiSkill | Move TCP in a direction (SO-101) | 256 |
```

- [ ] **Step 4: Update `getting-started/environment-ids.mdx`**

Add the 6 new IDs to the core registry table.

- [ ] **Step 5: Update MuJoCo primitive env docs**

Update `mujoco-reach.mdx`, `mujoco-look-at.mdx`, `mujoco-move.mdx` to reflect the new default observation (6-dim joint positions instead of 10-dim) and document the `observations` config parameter.

- [ ] **Step 6: Commit**

```bash
git add docs/
git commit -m "docs: add ManiSkill Reach/LookAt/Move pages, update observation docs"
```
