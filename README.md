<div align="center">

<img src="https://raw.githubusercontent.com/johnsutor/so101-nexus/main/assets/so101.png" width="250" alt="SO-101 Arm">

<h3>SO101-Nexus: from demonstrations to a trained policy</h3>

[Docs](https://so101-nexus.com/docs) · [Releases](https://github.com/johnsutor/so101-nexus/releases) · [Discord](https://discord.gg/37kKRXDh8)

</div>

**Record. Clone. Reinforce.** Control a simulated SO-101 with an SO-100 or SO-101 leader arm,
record [LeRobot](https://github.com/huggingface/lerobot) demonstrations, train with behavior cloning,
then fine-tune with reinforcement learning. One Python library connects the workflow.

## Get started

With Python 3.12 or newer, install the library:

```bash
pip install so101-nexus
```

Run a simulated robot. No hardware is required:

```python
import gymnasium as gym
import so101_nexus.mujoco  # registers the MuJoCo environments

env = gym.make("MuJoCoPickLift-v1", render_mode="human")
obs, info = env.reset(seed=0)
env.action_space.seed(0)

for _ in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

See [Installation](https://so101-nexus.com/docs/getting-started/installation) for optional extras and source setup.
The [Quickstart](https://so101-nexus.com/docs/getting-started/quickstart) explains the example and shows how to change objects.

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
