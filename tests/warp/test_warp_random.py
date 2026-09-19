"""Per-world random stream contracts for Warp environments."""

import numpy as np
import pytest
import torch

from so101_nexus.warp._random import WorldEpisodeRNG

pytestmark = pytest.mark.warp


def test_seeded_stream_replays_and_scalar_seed_derives_worlds():
    worlds = torch.arange(3)
    rng = WorldEpisodeRNG(3, torch.device("cpu"), 17)
    first = rng.rand("task", worlds, 8)

    rng.seed(17)
    replay = rng.rand("task", worlds, 8)

    torch.testing.assert_close(first, replay, rtol=0, atol=0)
    assert not torch.equal(first[0], first[1])


def test_explicit_seed_is_portable_across_batch_positions():
    batched = WorldEpisodeRNG(3, torch.device("cpu"), [8, 12, 21])
    relocated = WorldEpisodeRNG(2, torch.device("cpu"), [12, 12])

    expected = batched.rand("task", torch.tensor([1]), 16)
    actual = relocated.rand("task", torch.tensor([0, 1]), 16)

    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[0], rtol=0, atol=0)


def test_numpy_integral_seeds_are_accepted():
    numpy_seeded = WorldEpisodeRNG(2, torch.device("cpu"), [np.int64(7), np.int32(9)])
    integer_seeded = WorldEpisodeRNG(2, torch.device("cpu"), [7, 9])
    worlds = torch.arange(2)

    torch.testing.assert_close(
        numpy_seeded.rand("task", worlds, 4),
        integer_seeded.rand("task", worlds, 4),
        rtol=0,
        atol=0,
    )


def test_partial_episode_and_other_streams_do_not_perturb_world():
    world = torch.tensor([1])
    expected = WorldEpisodeRNG(3, torch.device("cpu"), [11, 12, 13])
    actual = WorldEpisodeRNG(3, torch.device("cpu"), [11, 12, 13])

    actual.rand("task", torch.tensor([0, 2]), 5)
    actual.rand("wrist", world, 7)
    actual.begin(torch.tensor([0, 2]), advance=True)

    torch.testing.assert_close(
        actual.rand("task", world, 6), expected.rand("task", world, 6), rtol=0, atol=0
    )


def test_stream_and_episode_domains_do_not_overlap_after_long_sequence():
    world = torch.tensor([0])
    rng = WorldEpisodeRNG(1, torch.device("cpu"), 4)
    task = rng.rand("task", world, 8_193)
    wrist = rng.rand("wrist", world, 2)

    assert not torch.equal(task[0, 8_191:8_193], wrist[0])
    rng.begin(world, advance=True)
    next_episode = rng.rand("task", world, 2)
    assert not torch.equal(task[0, :2], next_episode[0])

    rng.seed(4)
    rng.offsets[rng._STREAMS["task"] - 1, 0] = 2**32
    high_offset = rng.rand("task", world, 2)
    assert not torch.equal(task[0, :2], high_offset[0])


def test_uniform_draws_have_sane_distribution():
    rng = WorldEpisodeRNG(1, torch.device("cpu"), 123)
    values = rng.rand("task", torch.tensor([0]), 20_000)

    assert 0.48 < float(values.mean()) < 0.52
    assert float(values.min()) < 0.001
    assert float(values.max()) > 0.999


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cpu_and_cuda_generate_identical_sequences():
    cpu = WorldEpisodeRNG(2, torch.device("cpu"), 31)
    cuda = WorldEpisodeRNG(2, torch.device("cuda"), 31)
    worlds_cpu = torch.arange(2)
    worlds_cuda = worlds_cpu.cuda()

    torch.testing.assert_close(
        cpu.rand("task", worlds_cpu, 32),
        cuda.rand("task", worlds_cuda, 32).cpu(),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("seed", [[1], [1, 2, 3], [-1, 2], [1, 2**32], [1, True]])
def test_seed_sequence_validation(seed):
    with pytest.raises(ValueError, match=r"seed sequence length|seeds must"):
        WorldEpisodeRNG(2, torch.device("cpu"), seed)


@pytest.mark.parametrize("seed", [1.5, "4", object()])
def test_scalar_seed_validation_is_clear(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        WorldEpisodeRNG(2, torch.device("cpu"), seed)


def test_partial_task_reset_does_not_perturb_other_world_rejection_sampling():
    from so101_nexus.config import PickAndPlaceConfig
    from so101_nexus.warp.pick_and_place import WarpPickAndPlaceVectorEnv

    config = PickAndPlaceConfig(n_distractors=2, min_object_separation=0.04)
    perturbed = WarpPickAndPlaceVectorEnv(2, config=config, device="cpu", seed=0)
    reference = WarpPickAndPlaceVectorEnv(2, config=config, device="cpu", seed=0)
    mask0 = torch.tensor([True, False])
    mask1 = torch.tensor([False, True])
    for env in (perturbed, reference):
        env.reset(seed=[31, 47])

    perturbed._rng.begin(torch.tensor([0]), advance=True)
    perturbed._write_reset_state(mask0)
    for env in (perturbed, reference):
        env._rng.begin(torch.tensor([1]), advance=True)
        env._write_reset_state(mask1)

    torch.testing.assert_close(perturbed.qpos[1], reference.qpos[1], rtol=0, atol=0)


def test_camera_configuration_does_not_change_task_samples():
    from so101_nexus.config import LookAtConfig
    from so101_nexus.observations import GazeDirection, WristCamera
    from so101_nexus.warp.look_at_env import WarpLookAtVectorEnv

    plain = WarpLookAtVectorEnv(2, config=LookAtConfig(), device="cpu", seed=0)
    camera = WarpLookAtVectorEnv(
        2,
        config=LookAtConfig(observations=[GazeDirection(), WristCamera(width=16, height=12)]),
        device="cpu",
        seed=0,
    )

    plain.reset(seed=[19, 23])
    camera.reset(seed=[19, 23])

    torch.testing.assert_close(plain._targets, camera._targets, rtol=0, atol=0)


def test_constructor_seed_first_reset_matches_explicit_seed_then_advances():
    from so101_nexus.config import LookAtConfig
    from so101_nexus.warp.look_at_env import WarpLookAtVectorEnv

    implicit = WarpLookAtVectorEnv(2, config=LookAtConfig(), device="cpu", seed=7)
    explicit = WarpLookAtVectorEnv(2, config=LookAtConfig(), device="cpu", seed=99)

    implicit.reset()
    explicit.reset(seed=7)
    torch.testing.assert_close(implicit._targets, explicit._targets, rtol=0, atol=0)
    torch.testing.assert_close(implicit._rng.episodes, torch.zeros(2, dtype=torch.int64))

    implicit.reset()
    torch.testing.assert_close(implicit._rng.episodes, torch.ones(2, dtype=torch.int64))
    assert not torch.equal(implicit._targets, explicit._targets)
