<div align="center">

<img src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/assets/so101.png" width="250" alt="SO-101 Arm">

<h3 align="center">
    <p>SO101-Nexus: SO-101 robot learning, from demos to policies</p>
</h3>

<p align="center">
    <a href="https://github.com/johnsutor/so101-nexus/blob/main/LICENSE.md"><img alt="License" src="https://img.shields.io/github/license/johnsutor/so101-nexus.svg?color=blue"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue"></a>
    <a href="https://so101-nexus.com/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-so101--nexus.com-blue"></a>
    <a href="https://github.com/johnsutor/so101-nexus/actions"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/johnsutor/so101-nexus/ci.yml?label=tests"></a>
    <a href="https://github.com/johnsutor/so101-nexus/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/johnsutor/so101-nexus.svg"></a>
    <a href="https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/bc_ppo_warp_colab.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
</p>

> **Beta**: APIs may change between releases. Feedback and bug reports are welcome.

</div>

SO101-Nexus is an end-to-end Python library for taking an SO-101 robot from demonstrations to a trained policy. It combines physical leader-arm teleoperation, LeRobot-compatible dataset recording, Gymnasium/MuJoCo manipulation environments, and training/evaluation hooks in one installable package.

For full documentation, visit [so101-nexus.com/docs](https://so101-nexus.com/docs).

## Demo Rollouts

<div align="center">
  <video controls muted playsinline width="720" aria-label="MuJoCo PickLift teleoperation rollout">
    <source src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/docs/public/videos/pick-lift-overhead.mp4" type="video/mp4">
    Open the <a href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fjohnsutor%2FMuJoCoPickLift%2Fepisode_0">PickLift episode viewer</a> instead.
  </video>
  <p>
    <a href="https://huggingface.co/datasets/johnsutor/MuJoCoPickLift">PickLift dataset</a>
    ·
    <a href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fjohnsutor%2FMuJoCoPickLift%2Fepisode_0">PickLift viewer</a>
  </p>
</div>

Recorded MuJoCo teleoperation datasets are available on Hugging Face:

| Task | Dataset | Episode viewer |
| --- | --- | --- |
| PickLift | [johnsutor/MuJoCoPickLift](https://huggingface.co/datasets/johnsutor/MuJoCoPickLift) | [episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fjohnsutor%2FMuJoCoPickLift%2Fepisode_0) |
| PickAndPlace | [johnsutor/MuJoCoPickAndPlace](https://huggingface.co/datasets/johnsutor/MuJoCoPickAndPlace) | [episode 0](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fjohnsutor%2FMuJoCoPickAndPlace%2Fepisode_0) |

## Why

There are useful SO-101 tools, but few packages connect teleoperation, LeRobot datasets, environments, and training loops in one workflow, all using simulations. SO101-Nexus is built around the record -> clone -> reinforce path: collect demonstrations, replay and evaluate in matching SO-101 environments, bootstrap with imitation learning, then fine-tune with RL.

MuJoCo is the default backend. An optional MuJoCo Warp backend (`so101-nexus[warp]`) adds GPU-parallel, batched environments for large-scale RL.

## What You Get

- **Teleoperation recorder**: drive a simulated follower with a physical SO-100 or SO-101 leader arm.
- **LeRobot dataset output**: save demonstrations with SO follower state/action units and wrist/overhead camera fields.
- **Gymnasium environments**: run SO-101 MuJoCo tasks for touch, look-at, move, pick-lift, and pick-and-place.
- **Configurable curricula**: swap objects, add distractors, randomize colors, tune rewards, and choose observation components.
- **Training and evaluation hooks**: start with the PPO baseline, LeRobot processors, and policy adapters for real-policy evaluation.
- **GPU-parallel Warp backend** (optional, experimental): batched `Warp*-v1` vector environments for large-scale RL, installed with `so101-nexus[warp]`.

## Installation

```bash
pip install so101-nexus
```

### From source

```bash
git clone https://github.com/johnsutor/so101-nexus.git
cd so101-nexus
uv sync
```

## Start with the Workflow

### Record demonstrations

```bash
uvx --from "so101-nexus[teleop]" so101-nexus teleop \
    --leader-port /dev/ttyACM0
```

See the [teleoperation docs](https://so101-nexus.com/docs/teleoperation/overview) for hardware setup, camera fields, environment customization, and Hub upload.

### Run an environment

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

See the [environment reference](https://so101-nexus.com/docs/environments) for all task IDs.

### Run the GPU-parallel Warp backend

> **Experimental**: The Warp backend's API and physics may change between minor releases while the MuJoCo backend is stable. See [Stability and versioning](https://so101-nexus.com/docs/getting-started/stability).

Install the optional extra and create a batched vector environment:

```bash
pip install "so101-nexus[warp]"
```

```python
import gymnasium as gym
import so101_nexus.warp  # noqa: F401

envs = gym.make_vec("WarpTouch-v1", num_envs=4096, device="cuda")
obs, info = envs.reset(seed=0)
obs, reward, terminated, truncated, info = envs.step(envs.action_space.sample())
envs.close()
```

### Train a policy

The default workflow is demo-seeded: behavior cloning from teleoperation demonstrations, then PPO fine-tuning on the GPU-parallel Warp backend. Train end to end in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/bc_ppo_warp_colab.ipynb)

Or run it locally with [`examples/bc_ppo_warp.py`](examples/bc_ppo_warp.py).

Prefer a from-scratch baseline instead? SO101-Nexus also ships a CleanRL-style PPO baseline for Gymnasium environments (no demonstration seeding). See [Training with PPO](https://so101-nexus.com/docs/training/ppo) for the command-line workflow and tuning notes, or train a strong policy end to end in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/ppo_warp_colab.ipynb)

## Roadmap

- [x] MuJoCo environments for the SO-101 arm
- [x] SO-101 tasks: Touch, LookAt, Move, PickLift, PickAndPlace
- [x] Physical leader-arm teleop recorder for LeRobot datasets
- [x] MuJoCo Warp backend for GPU-parallel throughput
- [x] Stronger training baselines and exemplars for every environment
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
