# Review: feat/warp-heterogeneous-objects

## Findings

### P1 - `PickAndPlaceConfig` breaks existing positional callers

- `src/so101_nexus/config.py:704-710`
- The constructor inserts `objects` before the existing `cube_colors` and `target_colors` parameters. Existing public usage such as `PickAndPlaceConfig("blue", "green")` now stores `"blue"` as `objects` and `"green"` as `target_colors`, leaving `cube_colors` at the default `"red"`.
- `object_pool()` then normalizes that string with `list(objects)` at `src/so101_nexus/config.py:782-792`, producing `['b', 'l', 'u', 'e']`. Environment construction later fails with `TypeError: Unsupported object type: <class 'str'>`.
- Suggested fix: keep the old positional order and make `objects` keyword-only, or explicitly detect legacy positional color values and translate/reject them with a clear error. Also validate that every normalized object is a `SceneObject`.

### P2 - Mesh object spacing uses an unrotated footprint

- `src/so101_nexus/object_slots.py:247-254`
- `extract_object_slots()` computes `rest_quat` and `spawn_z` for `YCBObject` and `MeshObject`, but computes `bounding_radius` from the original `verts[:, :2]`. For meshes whose stable rest pose rotates the thin X or Y axis upward, the reset samplers use the wrong horizontal footprint.
- That radius feeds MuJoCo and Warp active-object spacing and pick-and-place object/target spacing. A rotated mesh can be placed closer than `min_object_separation` or `min_object_target_separation` actually allows.
- Suggested fix: compute the XY extent after applying `rest_quat`, or use a conservative yaw-invariant footprint for the resting orientation.

### P2 - Warp same-step autoreset reports new task descriptions for old transitions

- `src/so101_nexus/warp/base_env.py:468-490`
- `step()` computes reward, termination, and task info before same-step autoreset, but appends `info["task_description"]` after `_write_reset_state(done)` can select new target slots. Terminal transitions in heterogeneous pools can therefore report reward/success for one object while labeling the transition with the next episode's object.
- Reproduced with `WarpPickLiftVectorEnv(max_episode_steps=1, objects=[red, green, blue, yellow])`: `before` descriptions differed from `step_info["task_description"]` for 7 of 8 truncated worlds, and `step_info` matched the post-reset descriptions.
- Suggested fix: snapshot `tuple(self.task_descriptions)` before autoreset and return that for the transition, or return separate `final_task_description` and new `task_description` fields.

### P2 - Warp hidden object slots can collide with valid custom spawn regions

- `src/so101_nexus/warp/pick_env.py:168-170` and `src/so101_nexus/warp/pick_env.py:220-224`
- Inactive Warp slots remain collidable and are parked at fixed XY coordinates near `(-2, -2)`. The public `spawn_center`, `spawn_min_radius`, and `spawn_max_radius` can be configured so that valid active-object samples overlap those parked bodies. Large custom meshes can also overlap at the fixed `_HIDE_SPACING`.
- Suggested fix: derive parking positions from configured spawn bounds and object radii, or validate that the fixed parking band is outside the reachable spawn annulus with enough clearance for every hidden slot.

### P3 - Deprecated `min_cube_target_separation` is no longer writable

- `src/so101_nexus/config.py:794-797`
- The compatibility alias is now a read-only property. Existing plain-config-object code can still pass `min_cube_target_separation` to the constructor, but `cfg.min_cube_target_separation = 0.06` now raises `AttributeError` instead of updating the separation used by both backends.
- Suggested fix: add a setter that forwards to `min_object_target_separation` for the deprecation period.

### P3 - `PickAndPlaceConfig.describe()` was removed without a compatibility shim

- `src/so101_nexus/config.py:805-811`
- The previous public `PickAndPlaceConfig.describe(cube_name, target_name)` helper is gone. The new module-level template is not exported from `so101_nexus`, so existing callers now get `AttributeError`.
- Suggested fix: keep `describe()` as a staticmethod that delegates to the new object-based template, or deprecate it explicitly before removal.

## Validation performed

- `uv run pytest tests/core/test_config.py tests/core/test_object_slots.py tests/mujoco/test_envs.py -q` - 302 passed.
- `uv run pytest tests/warp/test_warp_envs.py tests/warp/test_warp_heterogeneous.py tests/warp/test_warp_reset_contract.py tests/warp/test_warp_touch.py -q` - 51 passed.
- Manual checks in Python confirmed the positional `PickAndPlaceConfig("blue", "green")` regression, the read-only alias, the removed `describe()` method, and the Warp autoreset task-description mismatch.
