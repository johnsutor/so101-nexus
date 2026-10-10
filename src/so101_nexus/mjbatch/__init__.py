"""Optional CPU batch backend with lazy Gymnasium vector registration."""

import gymnasium

gymnasium.register(
    id="MJBatchTouch-v1",
    vector_entry_point="so101_nexus.mjbatch.vector_env:MJBatchVectorEnv",
    max_episode_steps=512,
    kwargs={"task_entry_point": "so101_nexus.mujoco.touch_env:TouchEnv"},
)

gymnasium.register(
    id="MJBatchLookAt-v1",
    vector_entry_point="so101_nexus.mjbatch.vector_env:MJBatchVectorEnv",
    max_episode_steps=256,
    kwargs={"task_entry_point": "so101_nexus.mujoco.look_at_env:LookAtEnv"},
)

gymnasium.register(
    id="MJBatchMove-v1",
    vector_entry_point="so101_nexus.mjbatch.vector_env:MJBatchVectorEnv",
    max_episode_steps=256,
    kwargs={"task_entry_point": "so101_nexus.mujoco.move_env:MoveEnv"},
)

gymnasium.register(
    id="MJBatchPickLift-v1",
    vector_entry_point="so101_nexus.mjbatch.vector_env:MJBatchVectorEnv",
    max_episode_steps=1024,
    kwargs={"task_entry_point": "so101_nexus.mujoco.pick_env:PickLiftEnv"},
)

gymnasium.register(
    id="MJBatchPickReturn-v1",
    vector_entry_point="so101_nexus.mjbatch.vector_env:MJBatchVectorEnv",
    max_episode_steps=1024,
    kwargs={"task_entry_point": "so101_nexus.mujoco.pick_return:PickReturnEnv"},
)

gymnasium.register(
    id="MJBatchPickAndPlace-v1",
    vector_entry_point="so101_nexus.mjbatch.vector_env:MJBatchVectorEnv",
    max_episode_steps=1024,
    kwargs={"task_entry_point": "so101_nexus.mujoco.pick_and_place:PickAndPlaceEnv"},
)

gymnasium.register(
    id="MJBatchPickAndPlace-v2",
    vector_entry_point="so101_nexus.mjbatch.vector_env:MJBatchVectorEnv",
    max_episode_steps=1024,
    kwargs={"task_entry_point": "so101_nexus.mujoco.pick_and_place:PickAndPlaceV2Env"},
)

gymnasium.register(
    id="MJBatchStackCube-v1",
    vector_entry_point="so101_nexus.mjbatch.vector_env:MJBatchVectorEnv",
    max_episode_steps=1024,
    kwargs={"task_entry_point": "so101_nexus.mujoco.stack_cube:StackCubeEnv"},
)
