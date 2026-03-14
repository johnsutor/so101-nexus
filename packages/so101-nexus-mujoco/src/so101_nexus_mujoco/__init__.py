import gymnasium

gymnasium.register(
    id="MuJoCoPickLift-v1",
    entry_point="so101_nexus_mujoco.pick_env:PickLiftEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoPickAndPlace-v1",
    entry_point="so101_nexus_mujoco.pick_and_place:PickAndPlaceEnv",
    max_episode_steps=1024,
)
