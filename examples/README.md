# Examples

Basic PPO baselines for all SO101-Nexus registered environments, using CleanRL's continuous-action PPO implementation with minimal changes.

## Run one environment

```bash
uv run python examples/ppo.py --env-id MuJoCoPickCubeGoal-v1 --total-timesteps 200000
```

For ManiSkill environments:

```bash
uv run --package so101-nexus-maniskill --prerelease=allow python examples/ppo.py --env-id ManiSkillPickCubeGoalSO101-v1 --total-timesteps 200000
```

For MuJoCo environments:

```bash
uv run --package so101-nexus-mujoco python examples/ppo.py --env-id MuJoCoPickCubeGoal-v1 --total-timesteps 200000
```

## List all environment IDs

```bash
uv run python examples/list_envs.py
```

## Run all baselines (one by one)

```bash
for env_id in $(uv run python examples/list_envs.py); do
  echo "Running $env_id"
  uv run python examples/ppo.py --env-id "$env_id" --total-timesteps 200000
done
```

## Results template

| env_id | total_timesteps | episodic_return (latest) | notes |
|---|---:|---:|---|
| MuJoCoPickCubeGoal-v1 | 200000 | | |
