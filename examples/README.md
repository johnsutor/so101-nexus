# Examples

## Teleoperation (Dataset Recording)

Record [LeRobot v3](https://huggingface.co/docs/lerobot/en/lerobot-dataset-v3) datasets by teleoperating an SO100 or SO101 leader arm to control a simulated follower in any SO101-Nexus environment. A Gradio web UI handles session configuration, live recording, and episode review.

### Prerequisites

Before using teleop, you need to set up and calibrate your physical SO101 (or SO100) leader arm:

- **SO-101 setup guide**: https://huggingface.co/docs/lerobot/en/so101
- **SO-100 setup guide**: https://huggingface.co/docs/lerobot/en/so100

Follow your robot's guide to assemble, flash firmware, and run calibration. This ensures your leader arm's joint readings are accurate.

**Find your leader arm's serial port** using the LeRobot port finder:

```bash
lerobot-find-port
```

This will list connected serial devices. Note the port (e.g. `/dev/ttyACM0` or `/dev/tty.usbmodemXXX`) for the `--leader-port` argument.

### Launch the recorder

```bash
uv run --package so101-nexus-mujoco --group teleop python examples/teleop.py \
  --leader-port /dev/ttyACM0
```

This opens a Gradio UI in your browser where you configure all recording parameters:

- **Environment ID** — dropdown of all registered environments
- **Robot Type** — SO100 or SO101 (warns if it mismatches the selected environment)
- **HuggingFace Repo ID** — where the dataset will be stored
- **FPS, Camera Width/Height** — recording resolution and framerate
- **Number of Episodes, Action Space** — dataset structure
- **Max Episode Duration, Countdown** — recording timing

Click **Initialize Session** to connect the leader arm and create the environment and dataset.

### Recording flow

1. Click **Start Recording** (or press Enter) — a countdown gives you time to grab the leader arm
2. Teleoperate the simulated robot — the live wrist camera feed is shown in the UI
3. Click **Stop Recording** (or wait for max duration) — the episode ends
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
uv run python examples/ppo.py --env-id MuJoCoPickLift-v1 --total-timesteps 200000
```

For ManiSkill environments:

```bash
uv run --package so101-nexus-maniskill --prerelease=allow python examples/ppo.py --env-id ManiSkillPickCubeLiftSO101-v1 --total-timesteps 200000
```

For MuJoCo environments:

```bash
uv run --package so101-nexus-mujoco python examples/ppo.py --env-id MuJoCoPickLift-v1 --total-timesteps 200000
```

This PPO script now uses ManiSkill's native batched env creation (`gym.make(..., num_envs=N)`) and the official `ManiSkillVectorEnv` adapter instead of custom ManiSkill wrappers.

### List all environment IDs

```bash
uv run python examples/list_envs.py
```

### Run all baselines (one by one)

```bash
for env_id in $(uv run python examples/list_envs.py); do
  echo "Running $env_id"
  uv run python examples/ppo.py --env-id "$env_id" --total-timesteps 200000
done
```

### Results template

| env_id | total_timesteps | episodic_return (latest) | notes |
|---|---:|---:|---|
| MuJoCoPickLift-v1 | 200000 | | |
