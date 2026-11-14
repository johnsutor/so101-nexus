# Examples

Choose a script, then follow the linked guide for commands and configuration.
Run scripts from the repository root after [installation from source](https://so101-nexus.com/docs/getting-started/installation#install-from-source).

| Goal | Script | Guide |
| --- | --- | --- |
| Train from demonstrations with BC and PPO | [`bc_ppo_warp.py`](bc_ppo_warp.py) | [Training](https://so101-nexus.com/docs/workflow/training) |
| Train from scratch with PPO | [`ppo_warp.py`](ppo_warp.py) | [Task recipes](https://so101-nexus.com/docs/workflow/training#recommended-commands) |
| Evaluate a saved checkpoint | [`eval_warp.py`](eval_warp.py) | [Train and evaluate](https://so101-nexus.com/docs/workflow/training#quick-start) |
| Run the CPU MuJoCo PPO reference | [`ppo.py`](ppo.py) | [Backend differences](https://so101-nexus.com/docs/concepts/backends) |
| List environment IDs | [`list_envs.py`](list_envs.py) | [Environment reference](https://so101-nexus.com/docs/environments) |

## Train in Colab

Select a GPU runtime and run all cells:

- [BC + PPO notebook](https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/bc_ppo_warp_colab.ipynb): train with published demonstrations.
- [PPO notebook](https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/ppo_warp_colab.ipynb): train from scratch.

## Record demonstrations

Follow [Teleoperation](https://so101-nexus.com/docs/workflow/teleoperation) for the recorder CLI, arm calibration, and dataset details.
For a physical follower, use [Robot identification](https://so101-nexus.com/docs/workflow/teleoperation#robot-identification).
