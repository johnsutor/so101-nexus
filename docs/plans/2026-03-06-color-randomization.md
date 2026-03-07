# Color Randomization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add configurable robot color (default yellow) and color randomization support across all color config fields.

**Architecture:** All color configs become plural (e.g. `cube_colors`), accept `str | list[str]`, and resolve to RGBA at reset via `sample_color()`. A unified `COLOR_MAP` replaces `CUBE_COLOR_MAP`/`TARGET_COLOR_MAP`. Validation uses `ValueError` consistently; PickAndPlace overlap uses `warnings.warn`.

**Tech Stack:** Python dataclasses, numpy, warnings module

---

### Task 1: Update core config types and COLOR_MAP

**Files:**
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/config.py`

**Step 1: Update ColorName, add ColorConfig, add gray, rename color map, update configs, add sample_color, update validation**

In `config.py`:

1. Add `"gray"` to `ColorName` literal (line 15):
```python
ColorName = Literal["red", "orange", "yellow", "green", "blue", "purple", "black", "white", "gray"]
```

2. Remove `CubeColorName` and `TargetColorName` aliases (lines 16-17). Add `ColorConfig` union type:
```python
ColorConfig = ColorName | list[ColorName]
```

3. Add `import warnings` at the top.

4. Update `EnvironmentConfig` (lines 148, 154):
```python
ground_colors: ColorConfig = "gray"
robot_colors: ColorConfig = "yellow"
```

5. Add validation in `EnvironmentConfig.__post_init__` for `ground_colors` and `robot_colors`:
```python
_validate_color_config(self.ground_colors, "ground_colors")
_validate_color_config(self.robot_colors, "robot_colors")
```

6. Update `PickCubeConfig` (line 170):
```python
cube_colors: ColorConfig = "red"
```
Update its `__post_init__` to validate `cube_colors` instead of `cube_color`.

7. Update `PickAndPlaceConfig` (lines 190-191):
```python
cube_colors: ColorConfig = "red"
target_colors: ColorConfig = "blue"
```
Update its `__post_init__`: validate both, replace the equality check with overlap warning:
```python
_validate_color_config(self.cube_colors, "cube_colors")
_validate_color_config(self.target_colors, "target_colors")
cube_set = {self.cube_colors} if isinstance(self.cube_colors, str) else set(self.cube_colors)
target_set = {self.target_colors} if isinstance(self.target_colors, str) else set(self.target_colors)
overlap = cube_set & target_set
if overlap:
    warnings.warn(
        f"cube_colors and target_colors overlap on {overlap}; "
        "the cube and target may be the same color in some episodes",
        stacklevel=2,
    )
```

8. Rename `CUBE_COLOR_MAP` to `COLOR_MAP` (line 229+). Add `"gray"`. Remove `TARGET_COLOR_MAP` alias:
```python
COLOR_MAP: dict[str, list[float]] = {
    "red": [1.0, 0.0, 0.0, 1.0],
    "orange": [1.0, 0.5, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0, 1.0],
    "green": [0.0, 1.0, 0.0, 1.0],
    "blue": [0.0, 0.0, 1.0, 1.0],
    "purple": [0.5, 0.0, 0.5, 1.0],
    "black": [0.0, 0.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0, 1.0],
    "gray": [0.5, 0.5, 0.5, 1.0],
}
```

9. Add helper functions before the config classes:
```python
def _validate_color_config(colors: ColorConfig, field_name: str) -> None:
    names = [colors] if isinstance(colors, str) else colors
    for name in names:
        if name not in COLOR_MAP:
            raise ValueError(
                f"{field_name} must be one of {list(COLOR_MAP)}, got {name!r}"
            )


def sample_color(colors: ColorConfig, rng: np.random.Generator | None = None) -> list[float]:
    if isinstance(colors, str):
        return COLOR_MAP[colors]
    if rng is None:
        rng = np.random.default_rng()
    chosen = rng.choice(colors)
    return COLOR_MAP[chosen]
```

**Step 2: Verify config tests fail (expected since field names changed)**

Run: `cd /home/johnsutor/Desktop/so101-nexus && python -m pytest packages/so101-nexus-core/tests/ -x -q 2>&1 | head -30`

---

### Task 2: Update core `__init__.py` exports

**Files:**
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/__init__.py`

**Step 1: Update exports**

- Replace `CUBE_COLOR_MAP` import with `COLOR_MAP`
- Remove `TARGET_COLOR_MAP` import
- Remove `CubeColorName` and `TargetColorName` imports
- Add `ColorConfig` and `sample_color` imports:

```python
from so101_nexus_core.config import (
    COLOR_MAP as COLOR_MAP,
)
from so101_nexus_core.config import (
    ColorConfig as ColorConfig,
)
from so101_nexus_core.config import (
    sample_color as sample_color,
)
```

---

### Task 3: Update core tests

**Files:**
- Modify: `packages/so101-nexus-core/tests/test_config.py`
- Modify: `packages/so101-nexus-core/tests/test_config_new_fields.py`

**Step 1: Update test_config.py**

- Remove `TestColorAliases` class (tests for removed `CubeColorName`/`TargetColorName`)
- Update imports: `COLOR_MAP` instead of `CUBE_COLOR_MAP`/`TARGET_COLOR_MAP`, remove `CubeColorName`/`TargetColorName`, add `ColorConfig`, `sample_color`
- Update `TestColorMaps`:
  - Remove `test_target_color_map_aliases_cube_map`
  - Update `test_rgba_entries_are_valid` to use `COLOR_MAP`
  - Add test for gray in `COLOR_MAP`
- Add `TestSampleColor`:
```python
class TestSampleColor:
    def test_single_color_returns_rgba(self):
        assert sample_color("red") == COLOR_MAP["red"]

    def test_list_returns_valid_rgba(self):
        rng = np.random.default_rng(42)
        result = sample_color(["red", "blue"], rng)
        assert result in [COLOR_MAP["red"], COLOR_MAP["blue"]]

    def test_single_element_list(self):
        assert sample_color(["green"]) == COLOR_MAP["green"]
```

**Step 2: Update test_config_new_fields.py**

- `test_environment_config_has_robot_color`: change to check `cfg.robot_colors == "yellow"`
- `test_pick_cube_has_cube_color`: change to check `cfg.cube_colors == "red"`
- `test_pick_cube_custom_color`: change to `PickCubeConfig(cube_colors="green")` and assert `cfg.cube_colors == "green"`
- `test_pick_and_place_has_cube_color`: change to `cfg.cube_colors == "red"`
- `test_pick_and_place_has_target_color`: change to `cfg.target_colors == "blue"`
- `test_robot_color_custom`: change to `EnvironmentConfig(robot_colors="red")` and assert `cfg.robot_colors == "red"`
- `test_invalid_cube_color_pick_cube`: change match to `"cube_colors"`, use `cube_colors="neon"`
- `test_invalid_cube_color_pick_and_place`: change match to `"cube_colors"`, use `cube_colors="neon"`
- `test_invalid_target_color`: change match to `"target_colors"`, use `target_colors="neon"`
- `test_same_cube_and_target_color`: change to `pytest.warns` instead of `pytest.raises`:
```python
def test_same_cube_and_target_color_warns(self):
    with pytest.warns(UserWarning, match="overlap"):
        PickAndPlaceConfig(cube_colors="red", target_colors="red")
```
- Add tests for list colors:
```python
def test_pick_cube_list_colors(self):
    cfg = PickCubeConfig(cube_colors=["red", "blue"])
    assert cfg.cube_colors == ["red", "blue"]

def test_invalid_color_in_list(self):
    with pytest.raises(ValueError, match="cube_colors"):
        PickCubeConfig(cube_colors=["red", "neon"])

def test_ground_colors_default(self):
    cfg = EnvironmentConfig()
    assert cfg.ground_colors == "gray"

def test_robot_colors_default(self):
    cfg = EnvironmentConfig()
    assert cfg.robot_colors == "yellow"

def test_robot_colors_list(self):
    cfg = EnvironmentConfig(robot_colors=["red", "blue"])
    assert cfg.robot_colors == ["red", "blue"]
```

**Step 3: Run tests to verify they pass**

Run: `cd /home/johnsutor/Desktop/so101-nexus && python -m pytest packages/so101-nexus-core/tests/ -x -v 2>&1 | tail -30`

**Step 4: Commit**

```bash
git add packages/so101-nexus-core/
git commit -m "feat: add color randomization support to core config

- Add gray to color palette, rename CUBE_COLOR_MAP to COLOR_MAP
- All color fields now plural (cube_colors, target_colors, etc.)
- Support str | list[str] for uniform random color sampling
- Add sample_color() utility function
- Robot color defaults to yellow, ground to gray
- PickAndPlace overlap is now a warning instead of error"
```

---

### Task 4: Update ManiSkill base_env

**Files:**
- Modify: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/base_env.py`

**Step 1: Update ground and robot color handling**

1. Add import of `sample_color` and `COLOR_MAP`:
```python
from so101_nexus_core.config import CameraMode, EnvironmentConfig, sample_color, COLOR_MAP
```

2. In `_setup_base` (line 39): remove `self.robot_color = config.robot_color`

3. In `_build_ground` (line 183): change `mat.base_color = list(self.config.ground_color)` to:
```python
mat.base_color = sample_color(self.config.ground_colors)
```

4. In `_apply_robot_color_if_needed` (lines 196-209): replace the None check with color sampling:
```python
def _apply_robot_color_if_needed(self) -> None:
    color = sample_color(self.config.robot_colors)
    for link in self.agent.robot.links:
        for obj in link._objs:
            render_body: RenderBodyComponent = obj.entity.find_component_by_type(
                RenderBodyComponent
            )
            if render_body is None:
                continue
            for render_shape in render_body.render_shapes:
                for part in render_shape.parts:
                    part.material.set_base_color(color)
```

---

### Task 5: Update ManiSkill pick_cube.py

**Files:**
- Modify: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_cube.py`

**Step 1: Update color handling**

1. Update import: `COLOR_MAP, sample_color` instead of `CUBE_COLOR_MAP`

2. In `__init__` (lines 36-39): use config's `cube_colors`:
```python
self.cube_colors = config.cube_colors
self.cube_half_size = config.cube_half_size
cube_color_name = config.cube_colors if isinstance(config.cube_colors, str) else config.cube_colors[0]
self.task_description = f"Pick up the small {cube_color_name} cube"
```

3. In `_load_scene` (line 68): sample color at scene load:
```python
cube = actors.build_cube(
    self.scene,
    half_size=self.cube_half_size,
    color=sample_color(self.cube_colors),
    ...
)
```

---

### Task 6: Update ManiSkill pick_and_place.py

**Files:**
- Modify: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_and_place.py`

**Step 1: Update color handling**

1. Update import: `COLOR_MAP, sample_color` instead of `CUBE_COLOR_MAP, TARGET_COLOR_MAP`

2. In `__init__` (lines 38-47):
```python
self.cube_colors = config.cube_colors
self.target_colors = config.target_colors
self.cube_half_size = config.cube_half_size
self.target_disc_radius = config.target_disc_radius
cube_name = config.cube_colors if isinstance(config.cube_colors, str) else config.cube_colors[0]
target_name = config.target_colors if isinstance(config.target_colors, str) else config.target_colors[0]
self.task_description = (
    f"Pick up the small {cube_name} cube"
    f" and place it on the {target_name} circle"
)
```

3. In `_load_scene` (line 76): `color=sample_color(self.cube_colors)`
4. In `_load_scene` (line 93): `material=sapien.render.RenderMaterial(base_color=sample_color(self.target_colors))`

---

### Task 7: Update MuJoCo pick_cube.py

**Files:**
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_cube.py`

**Step 1: Update color handling**

1. Update imports: `COLOR_MAP, sample_color` instead of `CUBE_COLOR_MAP`. Remove `CubeColorName`.

2. `_build_scene_xml` signature: `ground_color` param stays as `list[float]` (it receives resolved RGBA).

3. In `__init__`: remove `cube_color` parameter. Use config fields instead:
```python
def __init__(
    self,
    config: PickCubeConfig = PickCubeConfig(),
    render_mode: str | None = None,
    camera_mode: Literal["state_only", "wrist"] = "state_only",
    control_mode: ControlMode = "pd_joint_pos",
    robot_init_qpos_noise: float = 0.02,
):
    if not (0.01 <= config.cube_half_size <= 0.05):
        raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {config.cube_half_size}")

    self._init_common(...)

    cube_color_rgba = sample_color(config.cube_colors)
    cube_color_name = config.cube_colors if isinstance(config.cube_colors, str) else config.cube_colors[0]
    self.cube_color_name = cube_color_name
    self.cube_half_size = config.cube_half_size
    self.task_description = f"Pick up the small {cube_color_name} cube"

    xml_string = _build_scene_xml(
        config.cube_half_size,
        cube_color_rgba,
        config.cube_mass,
        sample_color(config.ground_colors),
        config.goal_thresh,
    )
    ...
```

Note: `_build_scene_xml` already takes `list[float]` for colors and `tuple` for ground_color. Update `ground_color` param type to `list[float]` to match.

---

### Task 8: Update MuJoCo pick_and_place.py

**Files:**
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_and_place.py`

**Step 1: Update color handling**

1. Update imports: `COLOR_MAP, sample_color` instead of `CUBE_COLOR_MAP, TARGET_COLOR_MAP`. Remove `CubeColorName, TargetColorName`.

2. Remove `cube_color` and `target_color` parameters from `__init__`. Use config fields:
```python
def __init__(
    self,
    config: PickAndPlaceConfig = PickAndPlaceConfig(),
    render_mode: str | None = None,
    camera_mode: Literal["state_only", "wrist"] = "state_only",
    control_mode: ControlMode = "pd_joint_pos",
    robot_init_qpos_noise: float = 0.02,
):
    if not (0.01 <= config.cube_half_size <= 0.05):
        raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {config.cube_half_size}")

    self._init_common(...)

    cube_color_rgba = sample_color(config.cube_colors)
    target_color_rgba = sample_color(config.target_colors)
    cube_name = config.cube_colors if isinstance(config.cube_colors, str) else config.cube_colors[0]
    target_name = config.target_colors if isinstance(config.target_colors, str) else config.target_colors[0]
    self.cube_color_name = cube_name
    self.target_color_name = target_name
    ...

    xml_string = _build_scene_xml(
        config.cube_half_size,
        cube_color_rgba,
        target_color_rgba,
        config.target_disc_radius,
        config.cube_mass,
        sample_color(config.ground_colors),
    )
```

3. Remove the duplicate validation that was in the MuJoCo `__init__` (color map checks, same-color check) — this is now handled by the config dataclass.

---

### Task 9: Update MuJoCo pick_ycb.py

**Files:**
- Modify: `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_ycb.py`

**Step 1: Update ground color handling**

1. Add import of `sample_color`.
2. In `__init__` (line 117): change `config.ground_color` to `sample_color(config.ground_colors)`.
3. Update `_build_ycb_scene_xml` signature: `ground_color: list[float]` instead of `tuple[float, float, float, float]`.

---

### Task 10: Update ManiSkill tests

**Files:**
- Modify: `packages/so101-nexus-maniskill/tests/test_pick_cube.py`
- Modify: `packages/so101-nexus-maniskill/tests/test_pick_and_place.py`

**Step 1: Update test_pick_cube.py**

- `test_invalid_cube_color`: match on `"cube_colors"`, use `cube_colors="neon"`
- `test_task_description_includes_color`: use `cube_colors="green"`

**Step 2: Update test_pick_and_place.py**

- `test_invalid_cube_color`: match on `"cube_colors"`, use `cube_colors="neon"`
- `test_invalid_target_color`: match on `"target_colors"`, use `target_colors="neon"`
- `test_same_cube_and_target_color_raises`: change to `pytest.warns(UserWarning, match="overlap")` with `cube_colors="red", target_colors="red"`
- `test_task_description_starts_with_capital`: use `cube_colors=` and `target_colors=`
- `test_includes_cube_color` / `test_includes_target_color`: use `cube_colors=` and `target_colors=`

---

### Task 11: Update MuJoCo tests

**Files:**
- Modify: `packages/so101-nexus-mujoco/tests/test_pick_cube.py`
- Modify: `packages/so101-nexus-mujoco/tests/test_pick_and_place.py`

**Step 1: Update test_pick_cube.py**

- Remove `cube_color` constructor args from `PickCubeEnv(cube_color=...)` calls. Use config instead: `PickCubeEnv(config=PickCubeConfig(cube_colors="red"))` or just `PickCubeEnv()` (defaults).
- `test_invalid_cube_color`: change to `PickCubeEnv(config=PickCubeConfig(cube_colors="neon"))` with match `"cube_colors"`
- `test_task_description_exists`: use `config=PickCubeConfig(cube_colors="red")`
- `test_task_description_starts_with_capital`: use `config=PickCubeConfig(cube_colors="blue")`

**Step 2: Update test_pick_and_place.py**

- Remove `cube_color`/`target_color` constructor args. Use config instead.
- `test_invalid_cube_color`: change to `PickAndPlaceEnv(config=PickAndPlaceConfig(cube_colors="neon"))` with match `"cube_colors"`
- `test_invalid_target_color`: change to `PickAndPlaceEnv(config=PickAndPlaceConfig(target_colors="neon"))` with match `"target_colors"`
- `test_same_cube_and_target_color_raises`: change to `pytest.warns` with `PickAndPlaceEnv(config=PickAndPlaceConfig(cube_colors="red", target_colors="red"))`
- `test_task_description_exists`: use `config=PickAndPlaceConfig(cube_colors="red", target_colors="blue")`

---

### Task 12: Update visual tests and utility scripts

**Files:**
- Modify: `tests/visual/test_maniskill_visual.py`
- Modify: `tests/visual/test_mujoco_visual.py`
- Modify: `utils/visualize_env.py`
- Modify: `utils/visualize_env_maniskill.py`

**Step 1: Update test_maniskill_visual.py**

In `_env_kwargs`: change `cube_color="red"` to `config=PickCubeConfig(camera=_CAM, cube_colors="red")` for pick cube envs, and similarly for pick and place.

Actually, since configs now have the color field, just pass it via config:
```python
def _env_kwargs(env_id: str) -> dict:
    if env_id in _PICK_CUBE_ENVS:
        return dict(config=PickCubeConfig(camera=_CAM, cube_colors="red"))
    if env_id in _PICK_AND_PLACE_ENVS:
        return dict(config=PickAndPlaceConfig(camera=_CAM, cube_colors="red"))
    return dict(config=PickYCBConfig(camera=_CAM))
```

**Step 2: Update test_mujoco_visual.py**

Same pattern — pass colors via config, remove standalone `cube_color` kwarg.

**Step 3: Update utils/visualize_env.py**

Line 84: remove `cube_color="red"` kwarg (default config handles it).

**Step 4: Update utils/visualize_env_maniskill.py**

Line 52: remove `cube_color="red"` kwarg.

---

### Task 13: Run full test suite and commit

**Step 1: Run core tests**

Run: `cd /home/johnsutor/Desktop/so101-nexus && python -m pytest packages/so101-nexus-core/tests/ -x -v 2>&1 | tail -30`

**Step 2: Run MuJoCo tests**

Run: `cd /home/johnsutor/Desktop/so101-nexus && python -m pytest packages/so101-nexus-mujoco/tests/ -x -v 2>&1 | tail -30`

**Step 3: Run ManiSkill tests (if available)**

Run: `cd /home/johnsutor/Desktop/so101-nexus && python -m pytest packages/so101-nexus-maniskill/tests/ -x -v 2>&1 | tail -30`

**Step 4: Commit all changes**

```bash
git add -A
git commit -m "feat: implement color randomization across all environments

- Update ManiSkill and MuJoCo backends to use sample_color()
- Remove standalone cube_color/target_color constructor params from MuJoCo envs
- Update all tests for new plural field names and warning-based overlap
- Update visual tests and utility scripts"
```
