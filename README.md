<div align="center">

<img src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/assets/so101.png" width="250" alt="SO-101 Arm">

<h3>SO101-Nexus: from demonstrations to a trained policy</h3>

[Docs](https://so101-nexus.com/docs) · [Releases](https://github.com/johnsutor/so101-nexus/releases) · [Discord](https://discord.gg/37kKRXDh8)

</div>

**Record. Clone. Reinforce.** Control a simulated SO-101 with an SO-100 or SO-101 leader arm,
record [LeRobot](https://github.com/huggingface/lerobot) demonstrations, train with behavior cloning,
then fine-tune with reinforcement learning. One Python library connects the workflow.

## Get started

Follow [Installation](https://so101-nexus.com/docs/getting-started/installation), then
run your first simulated robot with the [Quickstart](https://so101-nexus.com/docs/getting-started/quickstart).
No hardware is required.

To train without a local setup, open the
[BC + PPO Colab notebook](https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/bc_ppo_warp_colab.ipynb).
Select a GPU runtime and run all cells with the published demonstrations.

## From demonstrations to policy

1. [Record](https://so101-nexus.com/docs/workflow/teleoperation) demonstrations with a leader arm.
2. [Clone](https://so101-nexus.com/docs/workflow/training) the demonstrations with behavior cloning.
3. [Reinforce](https://so101-nexus.com/docs/workflow/training#recommended-commands) the policy with GPU-parallel PPO.

[Environments](https://so101-nexus.com/docs/environments) lists the tasks.
[Examples](examples/README.md) lists the scripts and notebooks.
MuJoCo provides Gymnasium simulation and CPU batching; the optional Warp backend provides GPU batching.
See [Backends](https://so101-nexus.com/docs/concepts/backends) for requirements and physics differences.

This library is in beta. See [Stability and versioning](https://so101-nexus.com/docs/api/stability) before you depend on an API.

## Contribute

Follow the [source installation guide](https://so101-nexus.com/docs/getting-started/installation#install-from-source), then run:

```bash
make format lint typecheck
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Source code uses the [Apache-2.0 License](LICENSE.md).
