# Absorb Loose Constructor Params Into Config Dataclasses

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move `cube_color`, `robot_color`, `camera_mode`, `robot_init_qpos_noise`, `target_color`, and `model_id` from environment constructor kwargs into the existing config dataclasses, so that config is the single source of truth — HuggingFace style.

**Architecture:** Add fields to `EnvironmentConfig` (base-level params) and task-specific subclasses (`PickAndPlaceConfig`, `PickYCBConfig`). Update `_setup_base` and each env `__init__` to read from config. Update tests to pass params via config objects. Color validation moves into config `__post_init__`.

**Tech Stack:** Python dataclasses (frozen), ManiSkill, gymnasium, pytest

---

### Task 1: Add base-level fields to `EnvironmentConfig`

**Files:**
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/config.py:118-133`

**Step 1: Write the failing test**

Create a test that verifies the new fields exist on EnvironmentConfig with correct defaults.

File: `packages/so101-nexus-core/tests/test_config_new_fields.py`

```python
from so101_nexus_core.config import (
    CameraMode as CameraModeType,
    EnvironmentConfig,
    PickAndPlaceConfig,
    PickCubeConfig,
    PickYCBConfig,
)


class TestNewConfigFields:
    def test_environment_config_has_camera_mode(self):
        cfg = EnvironmentConfig()
        assert cfg.camera_mode == "fixed"

    def test_environment_config_has_robot_color(self):
        cfg = EnvironmentConfig()
        assert cfg.robot_color is None

    def test_environment_config_has_robot_init_qpos_noise(self):
        cfg = EnvironmentConfig()
        assert cfg.robot_init_qpos_noise == 0.02

    def test_pick_cube_has_cube_color(self):
        cfg = PickCubeConfig()
        assert cfg.cube_color == "red"

    def test_pick_cube_custom_color(self):
        cfg = PickCubeConfig(cube_color="green")
        assert cfg.cube_color == "green"

    def test_pick_and_place_has_cube_color(self):
        cfg = PickAndPlaceConfig()
        assert cfg.cube_color == "red"

    def test_pick_and_place_has_target_color(self):
        cfg = PickAndPlaceConfig()
        assert cfg.target_color == "blue"

    def test_pick_ycb_has_model_id(self):
        cfg = PickYCBConfig()
        assert cfg.model_id == "058_golf_ball"

    def test_pick_ycb_custom_model(self):
        cfg = PickYCBConfig(model_id="011_banana")
        assert cfg.model_id == "011_banana"

    def test_camera_mode_custom(self):
        cfg = EnvironmentConfig(camera_mode="wrist")
        assert cfg.camera_mode == "wrist"

    def test_robot_color_custom(self):
        cfg = EnvironmentConfig(robot_color=(1.0, 0.0, 0.0, 1.0))
        assert cfg.robot_color == (1.0, 0.0, 0.0, 1.0)
```

**Step 2: Run test to verify it fails**

Run: `cd packages/so101-nexus-core && python -m pytest tests/test_config_new_fields.py -v`
Expected: FAIL — fields don't exist yet

**Step 3: Implement the config changes**

In `packages/so101-nexus-core/src/so101_nexus_core/config.py`:

1. Add `CameraMode` type alias (already exists as a string literal in base_env, move it here):
   ```python
   CameraMode = Literal["fixed", "wrist", "both"]
   ```

2. Add fields to `EnvironmentConfig` (after existing fields):
   ```python
   camera_mode: CameraMode = "fixed"
   robot_color: tuple[float, float, float, float] | None = None
   robot_init_qpos_noise: float = 0.02
   ```

3. Add `cube_color` to `PickCubeConfig`:
   ```python
   cube_color: CubeColorName = "red"
   ```

4. Add `cube_color` and `target_color` to `PickAndPlaceConfig`:
   ```python
   cube_color: CubeColorName = "red"
   target_color: TargetColorName = "blue"
   ```

5. Add `model_id` to `PickYCBConfig`:
   ```python
   model_id: YcbModelId = "058_golf_ball"
   ```

**Step 4: Run test to verify it passes**

Run: `cd packages/so101-nexus-core && python -m pytest tests/test_config_new_fields.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/so101-nexus-core/src/so101_nexus_core/config.py packages/so101-nexus-core/tests/test_config_new_fields.py
git commit -m "feat: add cube_color, camera_mode, robot_color, model_id to config dataclasses"
```

---

### Task 2: Add validation `__post_init__` to task configs

**Files:**
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/config.py`
- Modify: `packages/so101-nexus-core/tests/test_config_new_fields.py`

**Step 1: Write failing tests for validation**

Append to `packages/so101-nexus-core/tests/test_config_new_fields.py`:

```python
import pytest


class TestConfigValidation:
    def test_invalid_cube_color_pick_cube(self):
        with pytest.raises(ValueError, match="cube_color"):
            PickCubeConfig(cube_color="neon")

    def test_invalid_cube_color_pick_and_place(self):
        with pytest.raises(ValueError, match="cube_color"):
            PickAndPlaceConfig(cube_color="neon")

    def test_invalid_target_color(self):
        with pytest.raises(ValueError, match="target_color"):
            PickAndPlaceConfig(target_color="neon")

    def test_same_cube_and_target_color(self):
        with pytest.raises(ValueError, match="must differ"):
            PickAndPlaceConfig(cube_color="red", target_color="red")

    def test_invalid_model_id(self):
        with pytest.raises(ValueError, match="model_id"):
            PickYCBConfig(model_id="invalid_object")

    def test_invalid_camera_mode(self):
        with pytest.raises(ValueError, match="camera_mode"):
            EnvironmentConfig(camera_mode="overhead")

    def test_invalid_cube_half_size_too_small(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickCubeConfig(cube_half_size=0.001)

    def test_invalid_cube_half_size_too_large(self):
        with pytest.raises(ValueError, match="cube_half_size"):
            PickCubeConfig(cube_half_size=0.1)
```

**Step 2: Run test to verify it fails**

Run: `cd packages/so101-nexus-core && python -m pytest tests/test_config_new_fields.py::TestConfigValidation -v`
Expected: FAIL — no validation yet

**Step 3: Add `__post_init__` validation**

Since the dataclasses are `frozen=True`, use `object.__setattr__` pattern is not needed — just validate and raise. Add `__post_init__` to each config:

`EnvironmentConfig`:
```python
def __post_init__(self):
    if self.camera_mode not in ("fixed", "wrist", "both"):
        raise ValueError(f"camera_mode must be fixed|wrist|both, got {self.camera_mode!r}")
    if self.camera.width <= 0 or self.camera.height <= 0:
        raise ValueError(
            f"camera dimensions must be > 0, got {self.camera.width}x{self.camera.height}"
        )
```

`PickCubeConfig`:
```python
def __post_init__(self):
    super().__post_init__()
    if self.cube_color not in CUBE_COLOR_MAP:
        raise ValueError(
            f"cube_color must be one of {list(CUBE_COLOR_MAP)}, got {self.cube_color!r}"
        )
    if not (0.01 <= self.cube_half_size <= 0.05):
        raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {self.cube_half_size}")
```

`PickAndPlaceConfig`:
```python
def __post_init__(self):
    super().__post_init__()
    if self.cube_color not in CUBE_COLOR_MAP:
        raise ValueError(
            f"cube_color must be one of {list(CUBE_COLOR_MAP)}, got {self.cube_color!r}"
        )
    if self.target_color not in TARGET_COLOR_MAP:
        raise ValueError(
            f"target_color must be one of {list(TARGET_COLOR_MAP)}, got {self.target_color!r}"
        )
    if self.cube_color == self.target_color:
        raise ValueError(
            f"cube_color and target_color must differ, both are {self.cube_color!r}"
        )
    if not (0.01 <= self.cube_half_size <= 0.05):
        raise ValueError(f"cube_half_size must be in [0.01, 0.05], got {self.cube_half_size}")
```

`PickYCBConfig`:
```python
def __post_init__(self):
    super().__post_init__()
    if self.model_id not in YCB_OBJECTS:
        raise ValueError(
            f"model_id must be one of {list(YCB_OBJECTS)}, got {self.model_id!r}"
        )
```

**Step 4: Run test to verify it passes**

Run: `cd packages/so101-nexus-core && python -m pytest tests/test_config_new_fields.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add packages/so101-nexus-core/src/so101_nexus_core/config.py packages/so101-nexus-core/tests/test_config_new_fields.py
git commit -m "feat: add __post_init__ validation to config dataclasses"
```

---

### Task 3: Update `base_env._setup_base` to read from config

**Files:**
- Modify: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/base_env.py:30-56`

**Step 1: Simplify `_setup_base` signature**

Remove `robot_color`, `camera_mode`, `robot_init_qpos_noise` params from `_setup_base` — read them from `config` instead. The new signature:

```python
def _setup_base(
    self,
    *,
    config: EnvironmentConfig,
    robot_uids: str,
    robot_cfgs: dict[str, dict],
) -> None:
```

Body reads from config:
```python
    if robot_uids not in robot_cfgs:
        raise ValueError(f"robot_uids must be one of {list(robot_cfgs)}, got {robot_uids!r}")

    self.config = config
    self.robot_color = config.robot_color
    self.camera_mode: CameraMode = config.camera_mode
    self.robot_init_qpos_noise = config.robot_init_qpos_noise
    self.camera_width = config.camera.width
    self.camera_height = config.camera.height
    self._robot_cfg = robot_cfgs[robot_uids]
    self._initial_obj_z: torch.Tensor | None = None
```

Remove the `CameraMode` type alias from `base_env.py` — import it from `so101_nexus_core.config` instead.

Also remove the camera_mode and camera dimension validation from `_setup_base` since it now lives in config `__post_init__`.

**Step 2: Run existing tests (they will fail because callers still pass old args)**

Run: `cd packages/so101-nexus-maniskill && python -m pytest tests/ -x -v 2>&1 | head -30`
Expected: TypeError from old call sites — this is expected, fixed in Task 4.

**Step 3: Commit**

```bash
git add packages/so101-nexus-maniskill/src/so101_nexus_maniskill/base_env.py
git commit -m "refactor: _setup_base reads camera_mode, robot_color, qpos_noise from config"
```

---

### Task 4: Update `pick_cube.py` to use config-only constructor

**Files:**
- Modify: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_cube.py`

**Step 1: Rewrite `PickCubeEnv.__init__`**

Remove `cube_color`, `robot_color`, `camera_mode`, `robot_init_qpos_noise` from `__init__` params. Read them from `config`:

```python
def __init__(
    self,
    *args,
    config: PickCubeConfig = PickCubeConfig(),
    robot_uids: str = "so100",
    num_envs: int = 1,
    reconfiguration_freq: int | None = None,
    **kwargs,
):
    self.cube_color_name = config.cube_color
    self.cube_color_rgba = CUBE_COLOR_MAP[config.cube_color]
    self.cube_half_size = config.cube_half_size
    self.task_description = f"Pick up the small {config.cube_color} cube"

    robot_cfgs = build_maniskill_robot_configs(config=config)

    self._setup_base(
        config=config,
        robot_uids=robot_uids,
        robot_cfgs=robot_cfgs,
    )

    if reconfiguration_freq is None:
        reconfiguration_freq = 1 if config.camera_mode in ("wrist", "both") else 0

    super().__init__(
        *args,
        robot_uids=robot_uids,
        reconfiguration_freq=reconfiguration_freq,
        num_envs=num_envs,
        **kwargs,
    )
```

Remove the validation for `cube_color` and `cube_half_size` from `__init__` — it's now in config `__post_init__`.

Also remove the `CubeColorName` import (no longer needed in env file).

Update `_register_robot_variant` — no changes needed since it just passes kwargs through.

**Step 2: Run pick_cube tests (will fail due to tests passing old kwargs)**

Run: `cd packages/so101-nexus-maniskill && python -m pytest tests/test_pick_cube.py -x 2>&1 | head -20`
Expected: Failures from tests passing `cube_color=` directly to `gym.make`

**Step 3: Commit**

```bash
git add packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_cube.py
git commit -m "refactor: PickCubeEnv reads cube_color from config"
```

---

### Task 5: Update `pick_and_place.py` to use config-only constructor

**Files:**
- Modify: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_and_place.py`

**Step 1: Rewrite `PickAndPlaceEnv.__init__`**

Same pattern as Task 4. Remove `cube_color`, `target_color`, `robot_color`, `camera_mode`, `robot_init_qpos_noise`:

```python
def __init__(
    self,
    *args,
    config: PickAndPlaceConfig = PickAndPlaceConfig(),
    robot_uids: str = "so100",
    num_envs: int = 1,
    reconfiguration_freq: int | None = None,
    **kwargs,
):
    self.cube_color_name = config.cube_color
    self.cube_color_rgba = CUBE_COLOR_MAP[config.cube_color]
    self.target_color_name = config.target_color
    self.target_color_rgba = TARGET_COLOR_MAP[config.target_color]
    self.cube_half_size = config.cube_half_size
    self.target_disc_radius = config.target_disc_radius
    self.task_description = (
        f"Pick up the small {config.cube_color} cube and place it on the {config.target_color} circle"
    )

    robot_cfgs = build_maniskill_robot_configs(config=config)

    self._setup_base(
        config=config,
        robot_uids=robot_uids,
        robot_cfgs=robot_cfgs,
    )

    if reconfiguration_freq is None:
        reconfiguration_freq = 1 if config.camera_mode in ("wrist", "both") else 0

    super().__init__(
        *args,
        robot_uids=robot_uids,
        reconfiguration_freq=reconfiguration_freq,
        num_envs=num_envs,
        **kwargs,
    )
```

Remove validation and old color imports (`CubeColorName`, `TargetColorName`).

**Step 2: Commit**

```bash
git add packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_and_place.py
git commit -m "refactor: PickAndPlaceEnv reads cube_color, target_color from config"
```

---

### Task 6: Update `pick_ycb.py` to use config-only constructor

**Files:**
- Modify: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_ycb.py`

**Step 1: Rewrite `PickYCBEnv.__init__`**

Remove `model_id`, `robot_color`, `camera_mode`, `robot_init_qpos_noise`:

```python
def __init__(
    self,
    *args,
    config: PickYCBConfig = PickYCBConfig(),
    robot_uids: str = "so100",
    num_envs: int = 1,
    reconfiguration_freq: int | None = None,
    **kwargs,
):
    self.model_id = config.model_id
    self._obj_spawn_z = 0.0
    self.task_description = f"Pick up the {YCB_OBJECTS[config.model_id]}"

    robot_cfgs = build_maniskill_robot_configs(config=config)

    self._setup_base(
        config=config,
        robot_uids=robot_uids,
        robot_cfgs=robot_cfgs,
    )

    if reconfiguration_freq is None:
        reconfiguration_freq = 1 if config.camera_mode in ("wrist", "both") else 0

    super().__init__(
        *args,
        robot_uids=robot_uids,
        reconfiguration_freq=reconfiguration_freq,
        num_envs=num_envs,
        **kwargs,
    )
```

Update the YCB variant registration loop — `model_id` now goes through config:

```python
for _model_id, _env_name in YCB_ENV_NAME_MAP.items():
    for _task, _base_cls in [("Goal", PickYCBEnv), ("Lift", PickYCBLiftEnv)]:
        for _robot in ["SO100", "SO101"]:
            _env_id = f"ManiSkillPick{_env_name}{_task}{_robot}-v1"
            _robot_uid = _robot.lower()

            def _make_init(_mid=_model_id, _ruid=_robot_uid, _base=_base_cls):
                def __init__(self, *args, **kwargs):
                    kwargs.setdefault("robot_uids", _ruid)
                    kwargs.setdefault("config", PickYCBConfig(model_id=_mid))
                    _base.__init__(self, *args, **kwargs)
                return __init__

            _cls = type(
                f"Pick{_env_name}{_task}{_robot}Env",
                (_base_cls,),
                {"__init__": _make_init()},
            )
            _cls = register_env(_env_id, max_episode_steps=_DEFAULT_CONFIG.max_episode_steps)(_cls)
            globals()[f"Pick{_env_name}{_task}{_robot}Env"] = _cls
```

Remove `YcbModelId` import (no longer needed in env file).

**Step 2: Commit**

```bash
git add packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_ycb.py
git commit -m "refactor: PickYCBEnv reads model_id from config"
```

---

### Task 7: Update tests — `test_pick_cube.py`

**Files:**
- Modify: `packages/so101-nexus-maniskill/tests/test_pick_cube.py`

**Step 1: Update test calls**

Replace all `cube_color=` and `camera_mode=` kwargs in `gym.make()` with config objects:

- `test_invalid_cube_color`: Now test that constructing `PickCubeConfig(cube_color="neon")` raises `ValueError`
- `test_invalid_cube_half_size`: Now test that constructing `PickCubeConfig(cube_half_size=0.001)` raises `ValueError`
- `test_task_description_includes_color`: Pass `config=PickCubeConfig(cube_color="green")` instead of `cube_color="green"`
- Camera mode fixtures: Pass `config=PickCubeConfig(camera_mode="wrist")` etc.

**Step 2: Run tests**

Run: `cd packages/so101-nexus-maniskill && python -m pytest tests/test_pick_cube.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add packages/so101-nexus-maniskill/tests/test_pick_cube.py
git commit -m "test: update pick_cube tests to use config objects"
```

---

### Task 8: Update tests — `test_pick_and_place.py`

**Files:**
- Modify: `packages/so101-nexus-maniskill/tests/test_pick_and_place.py`

**Step 1: Update test calls**

- `test_invalid_cube_color`: `PickAndPlaceConfig(cube_color="neon")` raises
- `test_invalid_target_color`: `PickAndPlaceConfig(target_color="neon")` raises
- `test_same_cube_and_target_color_raises`: `PickAndPlaceConfig(cube_color="red", target_color="red")` raises
- Task description tests: Pass config objects
- Camera mode fixtures: Pass config objects

**Step 2: Run tests**

Run: `cd packages/so101-nexus-maniskill && python -m pytest tests/test_pick_and_place.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add packages/so101-nexus-maniskill/tests/test_pick_and_place.py
git commit -m "test: update pick_and_place tests to use config objects"
```

---

### Task 9: Update tests — `test_pick_ycb.py`

**Files:**
- Modify: `packages/so101-nexus-maniskill/tests/test_pick_ycb.py`

**Step 1: Update test calls**

- `test_invalid_model_id`: `PickYCBConfig(model_id="invalid_object")` raises
- Camera mode fixtures: Pass config objects

**Step 2: Run tests**

Run: `cd packages/so101-nexus-maniskill && python -m pytest tests/test_pick_ycb.py -v`
Expected: ALL PASS

**Step 3: Commit**

```bash
git add packages/so101-nexus-maniskill/tests/test_pick_ycb.py
git commit -m "test: update pick_ycb tests to use config objects"
```

---

### Task 10: Export `CameraMode` from core and clean up base_env

**Files:**
- Modify: `packages/so101-nexus-core/src/so101_nexus_core/__init__.py` (if CameraMode needs exporting)
- Modify: `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/base_env.py`

**Step 1: Remove `CameraMode` definition from base_env.py**

Replace the local `CameraMode = Literal["fixed", "wrist", "both"]` with an import from core config. Update the import line in `base_env.py`:

```python
from so101_nexus_core.config import CameraMode, EnvironmentConfig
```

**Step 2: Update any files that import CameraMode from base_env**

All three env files import `CameraMode` from `base_env`. Update them to import from `so101_nexus_core.config` instead, or keep the re-export in base_env for convenience. Recommend keeping the re-export for backward compat.

**Step 3: Run full test suite**

Run: `cd packages/so101-nexus-maniskill && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 4: Commit**

```bash
git add -u
git commit -m "refactor: move CameraMode to core config, clean up imports"
```

---

### Task 11: Final verification — full test suite

**Step 1: Run all core tests**

Run: `cd packages/so101-nexus-core && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 2: Run all maniskill tests**

Run: `cd packages/so101-nexus-maniskill && python -m pytest tests/ -v`
Expected: ALL PASS

**Step 3: Run any linting/type checks if configured**

Run: `cd /home/johnsutor/Desktop/so101-nexus && python -m ruff check packages/`
Expected: Clean or only pre-existing warnings
