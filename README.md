<div align="center">

<img src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/assets/so101.png" width="250" alt="SO-101 Arm">

<h3 align="center">
    <p>SO101-Nexus: Gymnasium-compatible MuJoCo simulation environments for the SO-101 robot arm</p>
</h3>

<p align="center">
    <a href="https://github.com/johnsutor/so101-nexus/blob/main/LICENSE.md"><img alt="License" src="https://img.shields.io/github/license/johnsutor/so101-nexus.svg?color=blue"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue"></a>
    <a href="https://so101-nexus.com/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-so101--nexus.com-blue"></a>
    <a href="https://github.com/johnsutor/so101-nexus/actions"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/johnsutor/so101-nexus/ci.yml?label=tests"></a>
    <a href="https://github.com/johnsutor/so101-nexus/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/johnsutor/so101-nexus.svg"></a>
</p>

> **Beta**: APIs may change between releases. Feedback and bug reports are welcome.

</div>

## Overview

SO101-Nexus is a simulation library providing [Gymnasium](https://gymnasium.farama.org/)-compatible [MuJoCo](https://mujoco.org/) environments for the [SO-101](https://github.com/TheRobotStudio/SO-ARM100) robot arm. It is built for imitation-learning and teleoperation dataset collection and evaluation, and integrates seamlessly with the [LeRobot](https://github.com/huggingface/lerobot) ecosystem.

For full documentation, visit [so101-nexus.com/docs](https://so101-nexus.com/docs).

## Why

There are very few standardized simulation environments for the SO-101 robot arm. SO101-Nexus fills that gap with Gymnasium-compatible MuJoCo environments that drop directly into the LeRobot stack for teleoperation, dataset recording, and real-policy evaluation (for example, MolmoAct).

The environments also expose primitives such as object localization and grasping, providing a foundation for training text-conditioned embodied policies via curriculum learning.

A MuJoCo Warp backend is planned to add GPU-parallel throughput for large-scale training.

## Installation

```bash
pip install so101-nexus
```

### From source (development)

```bash
git clone https://github.com/johnsutor/so101-nexus.git
cd so101-nexus
uv sync
```

## Try Teleop with uvx

You can launch the Gradio teleop recorder without a permanent install. `uvx` resolves the package, the `teleop` extra, and runs the CLI in an ephemeral environment:

```bash
uvx --from "so101-nexus[teleop]" so101-nexus teleop \
    --leader-port /dev/ttyACM0
```

Pass `--leader-port` to point at your serial device. Gradio recordings use
LeRobot SO follower units by default: body joints in degrees, gripper in
`RANGE_0_100`, follower readback in `observation.state`, and camera videos at
`observation.images.wrist` / `observation.images.overhead`. See the [teleop guide](https://so101-nexus.com/docs/guides/teleop-dataset-recording) for hardware setup and the full session walkthrough.

## LeRobot CLI Recording

For LeRobot-compatible sim-real datasets, use upstream `lerobot-record` with the simulator follower adapter:

```bash
lerobot-record \
  --robot.discover_packages_path=so101_nexus.lerobot_adapter \
  --robot.type=sim_so_follower \
  --robot.env_id=MuJoCoReach-v1 \
  --robot.id=my_robot \
  --robot.calibration_dir=~/.cache/huggingface/lerobot/calibration/robots/so_follower \
  --robot.use_degrees=true \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=my_leader \
  --dataset.repo_id=user/my_sim_reach \
  --dataset.num_episodes=10 \
  --dataset.single_task="reach the target"
```

This path requires the `teleop` extra and records the simulated follower state through LeRobot's standard robot/dataset APIs.
Keep `--robot.use_degrees=true` when targeting SO100/101 checkpoints such as `allenai/MolmoAct2-SO100_101`; percent mode is for policies trained or fine-tuned with percent-mode body joints.

## Quick Start

```python
import gymnasium as gym
import so101_nexus.mujoco  # noqa: F401

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

- [x] MuJoCo environments for the SO-101 arm (all five tasks)
- [x] MuJoCo SO-101 tasks: Reach, LookAt, Move, PickLift, PickAndPlace
- [ ] MuJoCo Warp backend for GPU-parallel throughput
- [ ] TD-MPC baselines and exemplars for every environment
- [ ] Integration with the [LeRobot Hub](https://huggingface.co/docs/lerobot/en/envhub)

## Development

```bash
git clone https://github.com/johnsutor/so101-nexus.git
cd so101-nexus
uv sync

make test       # run all tests
make format     # format code
make lint       # lint code
```

## License

This repository's source code is available under the [Apache-2.0 License](LICENSE.md).
