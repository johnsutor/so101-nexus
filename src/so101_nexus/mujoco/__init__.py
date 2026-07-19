"""MuJoCo backend for SO101-Nexus: registers all MuJoCo Gymnasium environments."""

from __future__ import annotations

import gymnasium

# This package exposes its environments through Gymnasium registration and lazy
# entry points (see ``gymnasium.register`` below) rather than top-level
# re-exports, so the public name surface here is empty.
__all__: list[str] = []

gymnasium.register(
    id="MuJoCoPickLift-v1",
    entry_point="so101_nexus.mujoco.pick_env:PickLiftEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoPickAndPlace-v1",
    entry_point="so101_nexus.mujoco.pick_and_place:PickAndPlaceEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoStackCube-v1",
    entry_point="so101_nexus.mujoco.stack_cube:StackCubeEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoTouch-v1",
    entry_point="so101_nexus.mujoco.touch_env:TouchEnv",
    max_episode_steps=512,
)

gymnasium.register(
    id="MuJoCoLookAt-v1",
    entry_point="so101_nexus.mujoco.look_at_env:LookAtEnv",
    max_episode_steps=256,
)

gymnasium.register(
    id="MuJoCoMove-v1",
    entry_point="so101_nexus.mujoco.move_env:MoveEnv",
    max_episode_steps=256,
)
