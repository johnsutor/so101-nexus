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

Record demonstrations, train a policy with behavior cloning, then fine-tune it with reinforcement learning.
SO101-Nexus connects SO-100 and SO-101 leader arms, [LeRobot](https://github.com/huggingface/lerobot)
datasets, and Gymnasium environments in one Python library.

## Start here

| I want to... | Go to |
| --- | --- |
| Install the library | [Installation](https://so101-nexus.com/docs/getting-started/installation) |
| Run a simulated robot | [Quickstart](https://so101-nexus.com/docs/getting-started/quickstart) |
| Record demonstrations with a leader arm | [Teleoperation](https://so101-nexus.com/docs/workflow/teleoperation) |
| Train with published demonstrations | [Training](https://so101-nexus.com/docs/workflow/training) |
| Choose a task | [Environments](https://so101-nexus.com/docs/environments) |
| Use LeRobot EnvHub | [LeRobot compatibility](https://so101-nexus.com/docs/concepts/lerobot#loading-the-environments-from-the-hub-envhub) |

No leader arm? Start with the published demonstrations in the
[BC + PPO Colab notebook](https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/bc_ppo_warp_colab.ipynb).
Select a GPU runtime and run all cells.

<div align="center">
  <video controls muted playsinline width="720" aria-label="MuJoCo PickAndPlace teleoperation rollout">
    <source src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/docs/public/videos/pick-it-up.mp4" type="video/mp4">
    Open the <a href="https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fjohnsutor%2FMuJoCoPickAndPlace-v1%2Fepisode_0">PickAndPlace episode viewer</a> instead.
  </video>
</div>

## How it works

1. **Record.** Control a simulated follower with a physical leader arm and save a LeRobot dataset.
2. **Clone.** Train a policy to reproduce the demonstrations.
3. **Reinforce.** Fine-tune the policy with PPO on the GPU-parallel MuJoCo Warp backend.

The [workflow guide](https://so101-nexus.com/docs/workflow/overview) connects these stages.
The [examples index](examples/README.md) lists the training scripts and notebooks.

MuJoCo provides the default simulation backend. The optional Warp backend supports batched GPU training.
See [Backends](https://so101-nexus.com/docs/concepts/backends) for rendering, hardware requirements, and physics differences.

## Development

Follow the [source installation guide](https://so101-nexus.com/docs/getting-started/installation#install-from-source), then run:

```bash
make format lint typecheck
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution instructions and
[Stability and versioning](https://so101-nexus.com/docs/api/stability) for the release policy.

## License

This repository's source code is available under the [Apache-2.0 License](LICENSE.md).
