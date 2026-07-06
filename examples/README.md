# Examples

## Teleoperation (Dataset Recording)

Record [LeRobot v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) datasets by teleoperating an SO100 or SO101 leader arm to control a simulated follower in any SO101-Nexus environment. A Gradio web UI handles session configuration, live recording, and episode review.

### Prerequisites

Before using teleop, you need to set up and calibrate your physical SO101 (or SO100) leader arm:

- **SO-101 setup guide**: https://huggingface.co/docs/lerobot/en/so101
- **SO-100 setup guide**: https://huggingface.co/docs/lerobot/en/so100

Follow your robot's guide to assemble, flash firmware, and run calibration. This ensures your leader arm's joint readings are accurate.

**Find your leader arm's serial port** using the LeRobot port finder:

If you're on Linux, you may need to make the serial devices writable before running the port finder or launching teleop:

```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

```bash
lerobot-find-port
```

This will list connected serial devices. Note the port (e.g. `/dev/ttyACM0` or `/dev/tty.usbmodemXXX`) for the `--leader-port` argument.

### Launch the recorder

Teleoperation lives behind the backend CLI, not a standalone script:

```bash
uv run so101-nexus teleop --leader-port /dev/ttyACM0
```

For deeper coverage of session configuration, dataset layout, and troubleshooting, see [/docs/teleoperation/overview](https://so101-nexus.github.io/docs/teleoperation/overview).

The CLI opens a Gradio UI in your browser where you configure all recording parameters:

- **Environment ID**: dropdown of all registered environments
- **Robot Type**: SO100 or SO101 (warns if it mismatches the selected environment)
- **Hugging Face Repo ID**: where the dataset will be stored
- **FPS, Camera Width/Height**: recording resolution and framerate
- **Number of Episodes, Action Space**: dataset structure
- **Max Episode Duration, Countdown**: recording timing

Click **Initialize Session** to connect the leader arm and create the environment and dataset.

### Recording flow

1. Click **Start Recording** (or press Enter). A countdown gives you time to grab the leader arm.
2. Teleoperate the simulated robot. The live wrist camera feed is shown in the UI.
3. Click **Stop Recording** (or wait for max duration). The episode ends.
4. **Review** the episode: video playback, joint state trajectory plot, task description, and metadata
5. **Approve** to save the episode to the dataset, or **Discard** to drop it
6. Repeat until all episodes are recorded
7. Optionally **Push to Hub** from the completion screen

#### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--leader-port` | `/dev/ttyACM0` | Serial port for the leader arm |
| `--leader-id` | `so101_leader` | Leader arm identifier |

---

## PPO Training

Use `examples/ppo_warp.py` for PPO baselines. It is the GPU-parallel CleanRL-style
recipe for `Warp*-v1` environments and keeps rollout collection on the CUDA device.
`examples/ppo.py` is the slower MuJoCo reference script and is not the recommended
path for RL baselines.

### Run one environment

```bash
uv run --extra warp --extra train python examples/ppo_warp.py \
  --env-id WarpTouch-v1 \
  --total-timesteps 5000000
```

Metrics stream to TensorBoard under `runs/`; view them with:

```bash
tensorboard --logdir runs
```

### List environment IDs

```bash
uv run python examples/list_envs.py
```

### Baseline hyperparameters

Common settings for the solved Warp baselines:

| Argument | Value |
|---|---:|
| `--num-envs` | `1024` |
| `--num-steps` | `16` |
| `--learning-rate` | `3e-4` |
| `--gamma` | `0.99` |
| `--gae-lambda` | `0.95` |
| `--num-minibatches` | `32` |
| `--update-epochs` | `10` |
| `--clip-coef` | `0.2` |
| `--ent-coef` | `0.005` |
| `--ent-coef-final` | `0.0` |
| `--vf-coef` | `0.5` |
| `--max-grad-norm` | `0.5` |
| `--target-kl` | `None` |
| `--hidden-dim` | `256` |
| `--control-mode` | `pd_joint_delta_pos` |
| `--episode-length` | `512` |

The entropy and learning-rate schedules run over `--total-timesteps`, so the step
budget is part of the recipe. PickLift uses a mild entropy warm-start and a
CleanRL-style update budget because the GPU Warp contact path is not bitwise
deterministic; the extra optimization turns early grasps into reliable lift
policies.

### Starting commands by environment

PickLift repeats the tuned defaults explicitly so copied runs keep the seed-validated
recipe even if script defaults change later.

```bash
uv run --extra warp --extra train python examples/ppo_warp.py \
  --env-id WarpTouch-v1 \
  --total-timesteps 5000000

uv run --extra warp --extra train python examples/ppo_warp.py \
  --env-id WarpLookAt-v1 \
  --total-timesteps 5000000

uv run --extra warp --extra train python examples/ppo_warp.py \
  --env-id WarpMove-v1 \
  --total-timesteps 5000000

uv run --extra warp --extra train python examples/ppo_warp.py \
  --env-id WarpPickLift-v1 \
  --total-timesteps 30000000 \
  --num-minibatches 32 \
  --update-epochs 10 \
  --ent-coef 0.005 \
  --ent-coef-final 0.0 \
  --max-grad-norm 0.5 \
  --target-kl None
```

### Results by environment

Success rate is the recent completed-episode success rate reported by the Warp
training rollout at the listed step budget. PickLift reports seed-validated results
from seeds 1, 2, and 3.

| env_id | steps | success rate | wall-clock |
|---|---:|---:|---:|
| `WarpTouch-v1` | 5.0M | 1.000 | 88 s |
| `WarpLookAt-v1` | 5.0M | 1.000 | 62 s |
| `WarpMove-v1` | 5.0M | 1.000 | 60 s |
| `WarpPickLift-v1` | 30.0M | 0.905 min, 0.952 mean, 0.993 max final | 24.5 min/run |
| `WarpPickAndPlace-v1` | pending | pending | pending |

`WarpPickAndPlace-v1` is intentionally excluded for now; the environment needs task
fixes before PPO baselines are meaningful.

### Evaluate a checkpoint

Evaluate the saved policy deterministically with no exploration noise:

```bash
uv run --extra warp python examples/eval_warp.py \
  --env-id WarpPickLift-v1 \
  --checkpoint "runs/WarpPickLift-v1__*/best_agent.pt"
```

### Train in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/johnsutor/so101-nexus/blob/main/examples/ppo_warp_colab.ipynb)

[`examples/ppo_warp_colab.ipynb`](ppo_warp_colab.ipynb) is a self-contained Colab
notebook: it installs the library, launches an embedded TensorBoard, trains,
evaluates, and displays a rollout video. Open it in Colab, select a GPU runtime,
and run all cells.
