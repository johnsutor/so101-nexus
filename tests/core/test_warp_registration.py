"""Warp env registration + discovery, proven without the warp extra."""

# Import-light: registration uses string vector_entry_points, so this never
# imports torch or mujoco_warp.
_EXPECTED = {
    "WarpTouch-v1": 512,
    "WarpLookAt-v1": 256,
    "WarpMove-v1": 256,
    "WarpPickLift-v1": 1024,
    "WarpPickAndPlace-v1": 1024,
}


def test_warp_envs_registered_and_discoverable():
    import gymnasium as gym

    import so101_nexus.warp  # noqa: F401
    from so101_nexus.env_ids import env_ids_for_backend

    warp_ids = set(env_ids_for_backend("warp"))
    for env_id, max_steps in _EXPECTED.items():
        assert env_id in gym.envs.registry
        assert gym.envs.registry[env_id].max_episode_steps == max_steps
        assert env_id in warp_ids
    assert not (warp_ids & set(env_ids_for_backend("mujoco")))
