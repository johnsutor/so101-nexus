<div align="center">

<img src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/assets/so101.png" width="250" alt="SO-101 Arm">

<h3 align="center">
    <p>SO101-Nexus: Simulation environments for the SO-101 robot arm across ManiSkill, Genesis, and MuJoCo</p>
</h3>

<p align="center">
    <a href="https://github.com/johnsutor/so101-nexus/blob/main/LICENSE.md"><img alt="License" src="https://img.shields.io/github/license/johnsutor/so101-nexus.svg?color=blue"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue"></a>
    <a href="https://github.com/haosulab/ManiSkill"><img alt="ManiSkill" src="https://img.shields.io/badge/backend-ManiSkill-orange"></a>
    <a href="https://genesis-world.readthedocs.io/"><img alt="Genesis" src="https://img.shields.io/badge/backend-Genesis-purple"></a>
    <a href="https://mujoco.org/"><img alt="MuJoCo" src="https://img.shields.io/badge/backend-MuJoCo-green"></a>
    <a href="https://github.com/johnsutor/so101-nexus/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/johnsutor/so101-nexus.svg"></a>
</p>

</div>

## Overview

SO101-Nexus is a simulation library providing [Gymnasium](https://gymnasium.farama.org/)-compatible environments for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot arm. It is designed to work across multiple simulation backends — [ManiSkill](https://github.com/haosulab/ManiSkill), [Genesis](https://genesis-world.readthedocs.io/), and [MuJoCo](https://mujoco.org/) — so that researchers can train and evaluate policies without being locked into a single physics engine. SO101-Nexus integrates seamlessly with the [LeRobot](https://github.com/huggingface/lerobot) ecosystem.

## Installation

Install only the backend you need. Each backend is a separate package that automatically pulls in the shared core:

```bash
pip install so101-nexus-mujoco      # MuJoCo backend
pip install so101-nexus-maniskill   # ManiSkill backend
```

### From source (development)

```bash
git clone --recurse-submodules https://github.com/johnsutor/so101-nexus.git
cd so101-nexus

# Install a single backend for development
uv sync --package so101-nexus-mujoco

# Or for ManiSkill
uv sync --package so101-nexus-maniskill --prerelease=allow
```

> [!NOTE]
> The `--recurse-submodules` flag is required to fetch the SO-ARM100 robot assets.

## Quick Start

SO101-Nexus registers environments with Gymnasium. Any registered environment can be instantiated with `gym.make`.

### ManiSkill — Pick and Place

`PickCubeGoalSO101-v1` requires the robot to place a cube at a target goal position while remaining stationary.

```python
import gymnasium as gym
import so101_nexus_maniskill  # noqa: F401

env = gym.make(
    "PickCubeGoalSO101-v1",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="rgb_array",
)

obs, info = env.reset()
for _ in range(256):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### ManiSkill — Pick and Lift

`PickCubeLiftSO101-v1` requires the robot to grasp a cube and lift it above a height threshold.

```python
import gymnasium as gym
import so101_nexus_maniskill

env = gym.make(
    "PickCubeLiftSO101-v1",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    render_mode="rgb_array",
)

obs, info = env.reset()
for _ in range(256):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### MuJoCo — Pick and Place

```python
import gymnasium as gym
import so101_nexus_mujoco  # noqa: F401

env = gym.make("MuJoCoPickCubeGoal-v1", render_mode="rgb_array")

obs, info = env.reset()
for _ in range(256):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Parallel Environments

ManiSkill and Genesis backends support batched simulation for large-scale data collection and training:

```python
import gymnasium as gym
import so101_nexus_maniskill

env = gym.make(
    "PickCubeLiftSO101-v1",
    obs_mode="state",
    control_mode="pd_joint_delta_pos",
    num_envs=512,
)

obs, info = env.reset()
```

### Recording Episodes

Use the `RecordEpisode` wrapper to save trajectories and videos:

```python
import gymnasium as gym
from mani_skill.utils.wrappers.record import RecordEpisode
import so101_nexus_maniskill

env = gym.make("PickCubeLiftSO101-v1", obs_mode="state", render_mode="rgb_array")
env = RecordEpisode(env, output_dir="videos", save_trajectory=True, video_fps=30)

obs, info = env.reset()
for _ in range(100):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Environments

### ManiSkill

| Environment | Robot | Task | Description |
|---|---|---|---|
| `PickCubeGoal-v1` | Configurable | Place | Place cube at goal position (robot must be static) |
| `PickCubeGoalSO100-v1` | SO-100 | Place | SO-100 specific goal placement |
| `PickCubeGoalSO101-v1` | SO-101 | Place | SO-101 specific goal placement |
| `PickCubeLift-v1` | Configurable | Lift | Grasp and lift cube above 0.05m |
| `PickCubeLiftSO100-v1` | SO-100 | Lift | SO-100 specific lift task |
| `PickCubeLiftSO101-v1` | SO-101 | Lift | SO-101 specific lift task |

### MuJoCo

| Environment | Robot | Task | Description |
|---|---|---|---|
| `MuJoCoPickCubeGoal-v1` | SO-101 | Place | Place cube at goal position |
| `MuJoCoPickCubeLift-v1` | SO-101 | Lift | Grasp and lift cube above 0.05m |

All environments have a maximum episode length of **256 steps**.

## Roadmap

- [x] ManiSkill environments for SO-100 and SO-101
- [x] MuJoCo environments for SO-101
- [ ] Genesis environments for SO-100 and SO-101
- [ ] Add more randomization options to environments (such as robot color, more objects, environment, etc)
- [ ] Add a variety of starting poses for the SO-100 and SO-101
- [X] Add consistent controls for the actions (end-effector, joint positions, etc) across simulation environments.
- [ ] Add consistency in the appearance of environments 
- [ ] Add documentation, with demo videos of each environment
- [ ] Additional manipulation tasks beyond pick-and-place/lift
- [ ] Add environments to the [Lerobot Hub](https://huggingface.co/docs/lerobot/en/envhub) 


## Development

To contribute or customize SO101-Nexus, clone and install with development dependencies:

```bash
git clone --recurse-submodules https://github.com/johnsutor/so101-nexus.git
cd so101-nexus
uv sync --package so101-nexus-mujoco
```

Run the test suite:

```bash
make test-mujoco
make test-maniskill
make test
```

Format and lint:

```bash
make format
make lint
```

## License

This repository's source code is available under the [Apache-2.0 License](LICENSE.md).
