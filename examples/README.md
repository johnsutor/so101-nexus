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

Basic PPO baselines for all SO101-Nexus registered environments, using CleanRL's continuous-action PPO implementation with minimal changes.

### Run one environment

```bash
uv run --extra train python examples/ppo.py --env-id MuJoCoPickLift-v1 --total-timesteps 200000
```

This PPO script vectorizes the MuJoCo environments through Gymnasium's `SyncVectorEnv` wrapper, controlled by `--num-envs`.

### List all environment IDs

```bash
uv run python examples/list_envs.py
```

### Run all baselines (one by one)

```bash
for env_id in $(uv run python examples/list_envs.py); do
  echo "Running $env_id"
  uv run --extra train python examples/ppo.py --env-id "$env_id" --total-timesteps 200000
done
```

### Run the Warp (GPU-parallel) backend

`examples/ppo_warp.py` trains on the batched `Warp*-v1` vector environments. Install the `warp` extra:

```bash
uv run --extra warp python examples/ppo_warp.py --env-id WarpTouch-v1 --num-envs 4096 --device cuda
```

### Results template

| env_id | total_timesteps | episodic_return (latest) | notes |
|---|---:|---:|---|
| MuJoCoPickLift-v1 | 200000 | | |
