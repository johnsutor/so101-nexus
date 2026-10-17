"""Episode distractor counts vary reproducibly across both physics backends."""

import numpy as np
import pytest

from so101_nexus import CubeObject, PickAndPlaceConfig, PickConfig, StackCubeConfig


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
@pytest.mark.parametrize("bounds", [(0, 3), (1, 2), (2, 2), (0, 0)])
@pytest.mark.parametrize("task", ["PickLift", "PickAndPlace", "StackCube"])
def test_distractor_count_sampling(env_factory, backend, task, bounds):
    lower, upper = bounds
    kwargs = {"min_distractors": lower, "max_distractors": upper, "reset_settle_frames": 0}
    if task == "PickLift":
        config = PickConfig(
            objects=[CubeObject(color=c) for c in ("red", "green", "blue", "yellow")], **kwargs
        )
        targets = 1
    elif task == "PickAndPlace":
        config = PickAndPlaceConfig(**kwargs)
        targets = 1
    else:
        config = StackCubeConfig(**kwargs)
        targets = 2
    env = env_factory(backend=backend, task=task, config=config)
    inner = env.unwrapped

    def active_counts():
        if backend == "warp":
            xy = inner.qpos[:, inner._slot_qadr[:, None] + inner._qpos_offsets[:2]]
            distances = ((xy - xy.new_tensor(config.spawn_center)) ** 2).sum(dim=2).sqrt()
            return (distances <= config.spawn_max_radius + 0.01).sum(dim=1).cpu().numpy() - targets
        slots = inner._slots if task == "PickLift" else inner._distractor_slots
        active = sum(inner.data.qpos[slot.qpos_addr + 2] > -1 for slot in slots)
        return np.array([active - targets if task == "PickLift" else active])

    counts = set()
    for seed in range(40):
        env.reset(seed=seed, options={} if task == "StackCube" else {"target_index": 0})
        count = active_counts()
        assert np.all((count >= lower) & (count <= upper))
        counts.update(count.tolist())
        snapshot = inner.qpos.clone() if backend == "warp" else inner.data.qpos.copy()
        env.reset(seed=seed, options={} if task == "StackCube" else {"target_index": 0})
        actual = inner.qpos.cpu().numpy() if backend == "warp" else inner.data.qpos
        expected = snapshot.cpu().numpy() if backend == "warp" else snapshot
        np.testing.assert_array_equal(actual, expected)
    assert counts == set(range(lower, upper + 1))
