# Color Randomization Design

## Summary

Add configurable robot color (default yellow) and color randomization support across all color config fields. Any color config accepts a single color name or a list of color names; when a list is provided, one color is sampled uniformly at each episode reset.

## Changes

### Type System (`config.py`)

- Add `"gray"` to `ColorName` literal and color map (`[0.5, 0.5, 0.5, 1.0]`).
- Rename `CUBE_COLOR_MAP` to `COLOR_MAP`. Remove `TARGET_COLOR_MAP` alias.
- Remove `CubeColorName` and `TargetColorName` aliases.
- Define `ColorConfig = ColorName | list[ColorName]`.

### Config Fields

All color fields become plural and use `ColorConfig`:

| Old Field | New Field | Default |
|-----------|-----------|---------|
| `ground_color: tuple[float, float, float, float]` | `ground_colors: ColorConfig` | `"gray"` |
| `robot_color: tuple[float, float, float, float] \| None` | `robot_colors: ColorConfig` | `"yellow"` |
| `cube_color: CubeColorName` | `cube_colors: ColorConfig` | `"red"` |
| `target_color: TargetColorName` | `target_colors: ColorConfig` | `"blue"` |

### Utility Function

`sample_color(colors: ColorConfig, rng: np.random.Generator | None = None) -> list[float]` in `config.py`:
- Single string: return `COLOR_MAP[color]`
- List: sample one uniformly via `rng.choice()`, return its RGBA

### Validation

- Invalid color names raise `ValueError` in `__post_init__` (consistent with existing pattern).
- Each element validated against `COLOR_MAP`.
- PickAndPlace: `warnings.warn()` if cube and target color sets overlap (not an error).

### Affected Files

- `packages/so101-nexus-core/src/so101_nexus_core/config.py` - types, color map, configs, validation, `sample_color()`
- `packages/so101-nexus-core/src/so101_nexus_core/__init__.py` - updated exports
- `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/base_env.py` - ground/robot color sampling at reset
- `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_cube.py` - cube color sampling at reset
- `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_and_place.py` - cube/target color sampling at reset
- `packages/so101-nexus-maniskill/src/so101_nexus_maniskill/pick_ycb.py` - ground color sampling at reset
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/base_env.py` - ground/robot color sampling at reset
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_cube.py` - cube color sampling at reset
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_and_place.py` - cube/target color sampling at reset
- `packages/so101-nexus-mujoco/src/so101_nexus_mujoco/pick_ycb.py` - ground color sampling at reset
- `packages/so101-nexus-core/tests/test_config.py` - updated tests
- `packages/so101-nexus-core/tests/test_config_new_fields.py` - updated tests
