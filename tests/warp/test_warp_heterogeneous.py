"""Heterogeneous-object and per-world task-description tests for the Warp backend."""

import pytest

pytestmark = pytest.mark.warp


def test_touch_threshold_vector_matches_selected_bounding_radius():
    import torch

    from so101_nexus.config import TouchConfig
    from so101_nexus.objects import CubeObject, YCBObject
    from so101_nexus.warp.touch_env import WarpTouchVectorEnv

    pool = [CubeObject(half_size=0.02, color="red"), YCBObject("011_banana")]
    env = WarpTouchVectorEnv(num_envs=16, config=TouchConfig(objects=pool), device="cpu", seed=0)
    env.reset(seed=0)
    radius = env._target_bounding_radius()
    assert torch.allclose(radius, env._slot_bradius[env._target_slot])
    # The two objects have different bounding radii, so the per-world threshold varies.
    assert radius.unique().numel() > 1


def test_pnp_ycb_object_target_offset_and_finite_reward():
    import torch

    from so101_nexus.config import PickAndPlaceConfig
    from so101_nexus.objects import YCBObject
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceVectorEnv

    config = PickAndPlaceConfig(objects=[YCBObject("011_banana")])
    env = WarpPickAndPlaceVectorEnv(num_envs=4, config=config, device="cpu", seed=0)
    obs, _ = env.reset(seed=0)
    assert torch.isfinite(obs).all()
    # TargetOffset (last 3 of the 24-dim obs) equals disc - object.
    expected = env._target_disc_pos() - env._target_pos()
    assert torch.allclose(obs[:, -3:], expected, atol=1e-5)
    _, reward, _, _, info = env.step(torch.zeros((4, 6)))
    assert torch.isfinite(reward).all()
    assert "obj_to_target_dist" in info


def test_autoreset_preserves_per_world_target_metadata():
    import torch

    from so101_nexus.config import PickConfig
    from so101_nexus.objects import CubeObject
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    pool = [CubeObject(color=c) for c in ("red", "green", "blue", "yellow")]
    env = WarpPickLiftVectorEnv(
        num_envs=8, config=PickConfig(objects=pool, n_distractors=1), device="cpu", seed=0
    )
    env.reset(seed=0)
    slot_before = env._target_slot.clone()
    # Truncate only worlds 0 and 1 on the next step.
    env._elapsed[:] = 0
    env._elapsed[[0, 1]] = env.max_episode_steps
    _, _, _, truncated, _ = env.step(torch.zeros((8, 6)))
    assert truncated[:2].all()
    assert not truncated[2:].any()
    # Non-reset worlds keep their target; reset worlds keep a valid geom mapping.
    assert torch.equal(env._target_slot[2:], slot_before[2:])
    assert torch.equal(env._obj_geom, env._slot_geom[env._target_slot])


def test_per_world_task_descriptions_and_reducer():
    import torch

    from so101_nexus.config import PickConfig
    from so101_nexus.objects import CubeObject, YCBObject
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    pool = [CubeObject(color="red"), YCBObject("011_banana")]
    env = WarpPickLiftVectorEnv(num_envs=8, config=PickConfig(objects=pool), device="cpu", seed=0)
    _, info = env.reset(seed=0)
    assert len(env.task_descriptions) == 8
    assert "task_description" in info
    assert len(info["task_description"]) == 8
    # Heterogeneous worlds -> the scalar reducer returns the generic family string.
    if len(set(env.task_descriptions)) > 1:
        assert env.task_description == "Pick up the selected object."
    # step() also surfaces per-world descriptions.
    _, _, _, _, step_info = env.step(torch.zeros((8, 6)))
    assert len(step_info["task_description"]) == 8


def test_uniform_pool_task_description_is_exact():
    from so101_nexus.config import PickConfig
    from so101_nexus.objects import CubeObject
    from so101_nexus.warp.pick_env import WarpPickLiftVectorEnv

    env = WarpPickLiftVectorEnv(
        num_envs=4, config=PickConfig(objects=CubeObject(color="red")), device="cpu", seed=0
    )
    env.reset(seed=0)
    assert env.task_description == "Pick up the red cube."
    assert all(d == "Pick up the red cube." for d in env.task_descriptions)
