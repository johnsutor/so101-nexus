import gymnasium

from so101_nexus_core.config import YCB_ENV_NAME_MAP

gymnasium.register(
    id="MuJoCoPickCubeLift-v1",
    entry_point="so101_nexus_mujoco.pick_cube:PickCubeLiftEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoPickAndPlace-v1",
    entry_point="so101_nexus_mujoco.pick_and_place:PickAndPlaceEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoPickCubeMultipleLift-v1",
    entry_point="so101_nexus_mujoco.pick_cube_multiple:PickCubeMultipleLiftEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoPickYCBMultipleLift-v1",
    entry_point="so101_nexus_mujoco.pick_ycb_multiple:PickYCBMultipleLiftEnv",
    max_episode_steps=1024,
)

gymnasium.register(
    id="MuJoCoPickYCBLift-v1",
    entry_point="so101_nexus_mujoco.pick_ycb:PickYCBLiftEnv",
    max_episode_steps=1024,
)

for _model_id, _env_name in YCB_ENV_NAME_MAP.items():
    gymnasium.register(
        id=f"MuJoCoPick{_env_name}Lift-v1",
        entry_point="so101_nexus_mujoco.pick_ycb:PickYCBLiftEnv",
        max_episode_steps=1024,
        kwargs={"model_id": _model_id},
    )
    gymnasium.register(
        id=f"MuJoCoPick{_env_name}LiftSO101-v1",
        entry_point="so101_nexus_mujoco.pick_ycb:PickYCBLiftEnv",
        max_episode_steps=1024,
        kwargs={"model_id": _model_id},
    )
    gymnasium.register(
        id=f"MuJoCoPick{_env_name}MultipleLift-v1",
        entry_point="so101_nexus_mujoco.pick_ycb_multiple:PickYCBMultipleLiftEnv",
        max_episode_steps=1024,
        kwargs={"model_id": _model_id},
    )
