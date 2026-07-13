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
| `--ent-coef` | `0.03` |
| `--ent-coef-final` | `0.005` |
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
  --ent-coef 0.03 \
  --ent-coef-final 0.005 \
  --max-grad-norm 0.5 \
  --target-kl None
```

### Results by environment

Success rate is the recent completed-episode success rate reported by the Warp
training rollout at the listed step budget. PickLift reports seed-validated results
from seeds 1, 2, 3, 4, and 5 (4 of 5 solved).
| env_id | steps | success rate | wall-clock |
|---|---:|---:|---:|
| `WarpTouch-v1` | 5.0M | 1.000 | 88 s |
| `WarpLookAt-v1` | 5.0M | 1.000 | 62 s |
| `WarpMove-v1` | 5.0M | 1.000 | 60 s |
| `WarpPickLift-v1` | 30.0M | 0.965 min, 0.973 mean, 0.985 max final | 24.5 min/run |
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

---

## BC-Seeded PPO Training

Use `examples/bc_ppo_warp.py` for demo-seeded PPO on `WarpPickLift-v1`: the exact
same GPU-parallel CleanRL PPO recipe as `ppo_warp.py` (fixed-horizon episodes,
CleanRL optimizer budget, entropy warm-start/anneal -- see that file's module doc
for why each is decisive), plus behavior-cloning (BC) seeding from the 10 successful
teleop episodes in
[`johnsutor/MuJoCoPickLift-v1`](https://huggingface.co/datasets/johnsutor/MuJoCoPickLift-v1):
the actor is BC-pretrained on the demos before online PPO starts, and an optional
persistent BC loss (`--bc-coef`, default on) keeps anchoring the actor mean toward
demo actions throughout training.

### Why this over `ppo_warp.py` alone

`ppo_warp.py` already solves this task through pure exploration (~97-100% success in
~30 min), but is exploration-luck seed-fragile: a 5-seed sweep at the current default
entropy schedule (`ent_coef=0.03 -> 0.005`) passed seeds 1-4 (final success 0.97,
0.985, 0.965, 0.97) but seed 5 got stuck at a grasp-hold-at-table local optimum and
never discovered the lift (`best_success=0.037`, `final_success=0.0`). Demo-seeding
directly targets that failure mode by starting the actor near the demos' actual
grasp-lift behavior instead of a random init, without touching PPO's own decisive
fixes or trading GPU-batch throughput away for a planning bottleneck (an MPC-planning
approach, TD-MPC2, was tried first and dropped for exactly that throughput reason --
see the CHANGELOG). An [RLPD](https://arxiv.org/abs/2302.02948)-style off-policy
alternative is designed but not yet built; see
[`docs/superpowers/specs/2026-07-11-rlpd-demo-augmented-sac-warp-design.md`](../docs/superpowers/specs/2026-07-11-rlpd-demo-augmented-sac-warp-design.md).

### Design notes

- **`pd_joint_delta_pos` control, unchanged from `ppo_warp.py`.** The demo dataset
  records absolute joint-position targets; rather than switch control modes (risking
  the proven recipe), demo actions are recomputed as the delta between consecutive
  recorded joint states (the realized per-step motion), normalized by the same
  `_DELTA_ACTION_SCALE` the env applies internally.
- **BC touches only the actor mean**, never the critic -- the demos have no
  associated value estimates under the online policy, so biasing the critic toward
  them would corrupt the advantage estimates PPO's own gradient relies on.
  `--bc-pretrain-updates` fits the actor alone (separate optimizer, `--bc-pretrain-lr`)
  before the PPO loop starts; the persistent `--bc-coef` term adds the same MSE loss
  into every PPO minibatch update afterward, optionally annealed via
  `--bc-anneal-steps`.
- **`--use-demos false` recovers `ppo_warp.py` exactly** (same Agent, same env
  helpers, same PPO loss) -- this file is a strict superset, not a fork with
  behavior drift.

### Run it

```bash
uv run --extra warp --extra train python examples/bc_ppo_warp.py
```

### Baseline hyperparameters

Same PPO baseline table as `ppo_warp.py`'s ["Baseline hyperparameters"](#baseline-hyperparameters)
above, plus:

| Argument | Value |
|---|---:|
| `--use-demos` | `true` |
| `--bc-pretrain-updates` | `2000` |
| `--bc-pretrain-lr` | `1e-3` |
| `--bc-coef` | `0.1` |
| `--bc-anneal-steps` | `0` (constant) |

### Results

Validated against the exact seed (5) that fails under `ppo_warp.py`'s current
default recipe, same `--total-timesteps 30000000`:

| Run | seed | best success | final success |
|---|---:|---:|---:|
| `ppo_warp.py` (no demos) | 5 | 0.037 | 0.000 |
| `bc_ppo_warp.py` (demo-seeded) | 5 | 0.993 | 0.983 |

Same `--total-timesteps 30000000`, same entropy/optimizer schedule -- demo-seeding is
the only difference, and it rescues the seed outright rather than nudging it. Success
climbed steadily from the BC-pretrained starting point (0.003 at 1.6M steps) through
0.807 at 6.6M, crossing 0.95+ by 8.2M and staying there through 30M steps. TensorBoard
under `runs/` shows the full curve for any local reproduction.
