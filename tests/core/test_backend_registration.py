"""Exact registration contracts for all backends, without loading simulators."""

import importlib
import subprocess
import sys
from functools import partial

import gymnasium as gym
import pytest

_TASKS = {
    "Touch-v1": ("touch_env:TouchEnv", 512),
    "LookAt-v1": ("look_at_env:LookAtEnv", 256),
    "Move-v1": ("move_env:MoveEnv", 256),
    "PickLift-v1": ("pick_env:PickLiftEnv", 1024),
    "PickReturn-v1": ("pick_return:PickReturnEnv", 1024),
    "PickAndPlace-v1": ("pick_and_place:PickAndPlaceEnv", 1024),
    "PickAndPlace-v2": ("pick_and_place:PickAndPlaceV2Env", 1024),
    "StackCube-v1": ("stack_cube:StackCubeEnv", 1024),
}


@pytest.mark.parametrize("backend,prefix", [("mujoco", "MuJoCo"), ("warp", "Warp")])
def test_backend_registration_contract(backend, prefix):
    importlib.import_module(f"so101_nexus.{backend}")
    specs = {key: value for key, value in gym.envs.registry.items() if key.startswith(prefix)}
    tasks = list(_TASKS)
    if backend == "mujoco":
        tasks = tasks[3:] + tasks[:3]
    assert list(specs) == [prefix + task for task in tasks]
    for task, (entry, limit) in _TASKS.items():
        spec = specs[prefix + task]
        assert spec.max_episode_steps == limit
        if backend == "mujoco":
            assert spec.entry_point == f"so101_nexus.mujoco.{entry}"
            assert isinstance(spec.vector_entry_point, partial)
            assert spec.vector_entry_point.keywords == {
                "task_entry_point": f"so101_nexus.mujoco.{entry}"
            }
            assert spec.kwargs == {}
        elif backend == "warp":
            module, cls = entry.split(":")
            assert spec.vector_entry_point == f"so101_nexus.warp.{module}:Warp{cls[:-3]}VectorEnv"
            assert spec.entry_point is None
            assert spec.kwargs == {}


@pytest.mark.parametrize("backend", ["mujoco", "warp"])
def test_registration_import_is_lazy(backend):
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"""
import sys
import gymnasium as gym
import so101_nexus
assert not any(key.startswith(('MuJoCo', 'Warp', 'MJBatch')) for key in gym.envs.registry)
import so101_nexus.{backend}
assert not {{'mujoco', 'torch', 'mujoco_warp', 'mjbatch'}} & sys.modules.keys()
assert not any(
    name.endswith(('.base_env', '.vector_env'))
    for name in sys.modules if name.startswith('so101_nexus.')
)
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
