<div align="center">

<img src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/assets/so101.png" width="250" alt="SO-101 Arm">

<h3 align="center">
    <p>SO101-Nexus: Simulation environments for the SO-101 robot arm across ManiSkill, Genesis, and MuJoCo</p>
</h3>

<p align="center">
    <a href="https://github.com/johnsutor/so101-nexus/blob/main/LICENSE.md"><img alt="License" src="https://img.shields.io/github/license/johnsutor/so101-nexus.svg?color=blue"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue"></a>
    <a href="https://so101-nexus.com/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-so101--nexus.com-blue"></a>
    <a href="https://github.com/johnsutor/so101-nexus/actions"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/johnsutor/so101-nexus/ci.yml?label=tests"></a>
    <a href="https://github.com/johnsutor/so101-nexus/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/johnsutor/so101-nexus.svg"></a>
</p>

> **Beta** — APIs may change between releases. Feedback and bug reports are welcome.

</div>

## Overview

SO101-Nexus is a simulation library providing [Gymnasium](https://gymnasium.farama.org/)-compatible environments for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot arm. It supports multiple simulation backends ([ManiSkill](https://github.com/haosulab/ManiSkill), [Genesis](https://genesis-world.readthedocs.io/), and [MuJoCo](https://mujoco.org/)) so that researchers can train and evaluate policies without being locked into a single physics engine. SO101-Nexus integrates seamlessly with the [LeRobot](https://github.com/huggingface/lerobot) ecosystem.

For full documentation, visit [so101-nexus.com/docs](https://so101-nexus.com/docs).

## Why

Robust robot policies should generalize across simulators before being deployed in the real world. Sim-to-sim transfer helps identify policies that rely on simulator-specific artifacts.

At the same time, there are very few standardized simulation environments available for the SO-100 and SO-101 robot arms. SO101-Nexus addresses this by providing Gymnasium-compatible environments across multiple physics backends, enabling consistent experimentation and benchmarking.

In addition, SO101-Nexus provides a foundation for training text-conditioned embodied policies via curriculum learning, with environments that expose primitives such as object localization and grasping.

## Installation

Install only the backend you need:

```bash
pip install so101-nexus-mujoco      # MuJoCo backend
pip install so101-nexus-maniskill   # ManiSkill backend
```

### From source (development)

```bash
git clone https://github.com/johnsutor/so101-nexus.git
cd so101-nexus

# Install a single backend for development (swap package name to switch backends)
uv sync --package so101-nexus-mujoco
uv sync --package so101-nexus-maniskill --prerelease=allow
```

## Try Teleop with uvx

You can launch the Gradio teleop recorder against either backend without a permanent install. `uvx` resolves the package, the `teleop` extra, and runs the CLI in an ephemeral environment:

```bash
# MuJoCo
uvx --from "so101-nexus-mujoco[teleop]" so101-nexus-mujoco teleop \
    --leader-port /dev/ttyACM0

# ManiSkill
uvx --from "so101-nexus-maniskill[teleop]" --prerelease=allow \
    so101-nexus-maniskill teleop --leader-port /dev/ttyACM0
```

Pass `--leader-port` to point at your serial device. See the [teleop guide](https://so101-nexus.com/docs/guides/teleop-dataset-recording) for hardware setup and the full session walkthrough.

## Quick Start

```python
import gymnasium as gym
import so101_nexus_mujoco  # noqa: F401  (or so101_nexus_maniskill)

env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")

obs, info = env.reset()
for _ in range(256):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

See the [docs](https://so101-nexus.com/docs) for the full list of environments, parallel environment usage, and training examples.

## Roadmap

- [x] ManiSkill environments for SO-100 and SO-101 (all five tasks)
- [x] MuJoCo environments for SO-101 (all five tasks)
- [x] Primitive tasks: Reach, LookAt, Move, PickLift, PickAndPlace
- [ ] Genesis environments for SO-100 and SO-101
- [ ] TD-MPC baselines and exemplars for every environment
- [ ] Integration with the [LeRobot Hub](https://huggingface.co/docs/lerobot/en/envhub)

## Development

```bash
git clone https://github.com/johnsutor/so101-nexus.git
cd so101-nexus
uv sync --package so101-nexus-mujoco

make test       # run all tests
make format     # format code
make lint       # lint code
```

## License

This repository's source code is available under the [Apache-2.0 License](LICENSE.md).
