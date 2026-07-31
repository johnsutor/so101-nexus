<div align="center">

<img src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/assets/so101.png" width="250" alt="SO-101 Arm">

<h3 align="center">
    <p>SO101-Nexus: full-stack robot learning for the SO-101 arm</p>
</h3>

<p align="center">
    <a href="https://github.com/johnsutor/so101-nexus/blob/main/LICENSE.md"><img alt="License" src="https://img.shields.io/github/license/johnsutor/so101-nexus.svg?color=blue"></a>
    <a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue"></a>
    <a href="https://so101-nexus.com/docs"><img alt="Docs" src="https://img.shields.io/badge/docs-so101--nexus.com-blue"></a>
    <a href="https://github.com/johnsutor/so101-nexus/actions"><img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/johnsutor/so101-nexus/ci.yml?label=tests"></a>
    <a href="https://github.com/johnsutor/so101-nexus/releases"><img alt="GitHub release" src="https://img.shields.io/github/release/johnsutor/so101-nexus.svg"></a>
    <a href="https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/bc_ppo_warp_colab.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg"></a>
    <a href="https://discord.gg/37kKRXDh8"><img alt="Discord" src="https://img.shields.io/badge/Discord-Join_Us-5865F2?style=flat&logo=discord&logoColor=white"></a>
</p>

> **Beta**: APIs may change between releases. Feedback and bug reports are welcome.

</div>

Full-stack robot learning for the SO-101 arm: teleoperation, imitation learning, and RL in
MuJoCo. One installable library that takes a robot from demonstrations to a trained policy,
built on [LeRobot](https://github.com/huggingface/lerobot) and Gymnasium.

```bash
pip install so101-nexus
```

Full documentation: **[so101-nexus.com/docs](https://so101-nexus.com/docs)**.

<div align="center">
  <video controls muted playsinline width="720" aria-label="MuJoCo PickAndPlace teleoperation rollout">
    <source src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/docs/public/videos/pick-it-up.mp4" type="video/mp4">
    Open the <a href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fjohnsutor%2FMuJoCoPickAndPlace-v1%2Fepisode_0">PickAndPlace episode viewer</a> instead.
  </video>
</div>

## The workflow

Record, then clone, then reinforce. Each stage hands one artifact to the next.

**1. Record.** Drive a simulated follower with a physical SO-100 or SO-101 leader arm and
save LeRobot v3 datasets.

```bash
uvx --from "so101-nexus[teleop]" so101-nexus teleop --leader-port /dev/ttyACM0
```

**2. Clone.** Bootstrap a policy from those demonstrations with behavior cloning.

**3. Reinforce.** Fine-tune with PPO on the GPU-parallel Warp backend, anchored to the demos.

Stages 2 and 3 are one command. No leader arm? It defaults to a published dataset, so this
runs end to end on its own:

```bash
uv run --extra warp --extra train python examples/bc_ppo_warp.py
```

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/bc_ppo_warp_colab.ipynb)

See the [workflow walkthrough](https://so101-nexus.com/docs/workflow/overview) for the full path.

## Run an environment

```python
import gymnasium as gym
import so101_nexus.mujoco  # registers the MuJoCo env ids

env = gym.make("MuJoCoPickLift-v1", render_mode="rgb_array")
obs, info = env.reset()

for _ in range(256):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

Six SO-101 manipulation tasks ship on MuJoCo: PickLift, PickAndPlace, StackCube, Touch,
LookAt, and Move. The optional MuJoCo Warp backend (`so101-nexus[warp]`) registers the same
six as GPU-parallel batched vector environments for large-scale RL. See the
[environment reference](https://so101-nexus.com/docs/environments).

The same environments are published as a
[LeRobot EnvHub](https://huggingface.co/docs/lerobot/en/envhub) package, so a LeRobot user
can load them without importing this library:

```python
from lerobot.envs.factory import make_env

envs = make_env(
    "johnsutor/so101-nexus-envs:envs/MuJoCoPickLift-v1.py",
    n_envs=4,
    trust_remote_code=True,
)
```

## Why

Plenty of SO-101 tooling exists, but little of it connects teleoperation, LeRobot datasets,
simulated environments, and training loops into one workflow. SO101-Nexus is that connection:
collect demonstrations, replay and evaluate them in matching SO-101 environments, bootstrap
with imitation learning, then fine-tune with RL.

- **Teleoperation recorder** with a Gradio UI, writing LeRobot v3 datasets with SO follower
  state and action units plus wrist and overhead camera fields.
- **Gymnasium environments** with configurable objects, distractors, colors, spawn regions,
  rewards, and observation components.
- **Training baselines** for behavior cloning and PPO, plus LeRobot processors and policy
  adapters for evaluating real policies.
- **Optional GPU-parallel Warp backend** (experimental, NVIDIA and CUDA only) for batched RL,
  and an optional ROCm extra for training on AMD hardware.

Recorded MuJoCo teleoperation datasets are published on Hugging Face:
[MuJoCoPickLift-v1](https://huggingface.co/datasets/johnsutor/MuJoCoPickLift-v1)
([viewer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fjohnsutor%2FMuJoCoPickLift-v1%2Fepisode_0)),
[MuJoCoPickAndPlace-v1](https://huggingface.co/datasets/johnsutor/MuJoCoPickAndPlace-v1)
([viewer](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fjohnsutor%2FMuJoCoPickAndPlace-v1%2Fepisode_0)).

## Roadmap

- [x] MuJoCo environments for the SO-101 arm
- [x] SO-101 tasks: Touch, LookAt, Move, PickLift, PickAndPlace, StackCube
- [x] Physical leader-arm teleop recorder for LeRobot datasets
- [x] MuJoCo Warp backend for GPU-parallel throughput
- [x] Stronger training baselines and exemplars for every environment
- [x] Integration with the [LeRobot Hub](https://huggingface.co/docs/lerobot/en/envhub)

## Development

```bash
git clone https://github.com/johnsutor/so101-nexus.git
cd so101-nexus
uv sync

make test       # run all tests
make format     # format code
make lint       # lint code
```

See [CONTRIBUTING.md](CONTRIBUTING.md), and
[Stability and versioning](https://so101-nexus.com/docs/api/stability) for the public-API and
release policy.

## License

This repository's source code is available under the [Apache-2.0 License](LICENSE.md).
