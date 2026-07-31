---
license: apache-2.0
library_name: lerobot
tags:
  - robotics
  - lerobot
  - envhub
  - gymnasium
  - mujoco
  - so-101
  - so-100
  - simulation
---

# SO101-Nexus environments

Twelve SO-101 manipulation environments, loadable through
[LeRobot EnvHub](https://huggingface.co/docs/lerobot/en/envhub). Six tasks on a
CPU MuJoCo backend and the same six on a GPU-batched MuJoCo Warp backend, from
[so101-nexus](https://github.com/johnsutor/so101-nexus).

```python
from lerobot.envs.factory import make_env

envs = make_env(
    "johnsutor/so101-nexus-envs:envs/MuJoCoPickLift-v1.py",
    n_envs=4,
    trust_remote_code=True,
)
env = envs["MuJoCoPickLift-v1"][0]
obs, info = env.reset(seed=0)
```

Loading `env.py` at the repository root gives `MuJoCoPickLift-v1`. Every other
environment has its own file under `envs/`.

## Install

```bash
pip install "so101-nexus>=0.5.0"          # MuJoCo* environments
pip install "so101-nexus[warp]>=0.5.0"    # adds the Warp* environments (CUDA)
```

The environment code is the installed library, not this repository: these files
are thin shims over `so101_nexus.envhub`, so the physics, rewards, and
observation layouts are versioned and tested with the package.

## Environments

| File                          | Task                                | Steps | State dim | Backend |
| ----------------------------- | ----------------------------------- | ----- | --------- | ------- |
| `envs/MuJoCoTouch-v1.py`      | Touch the target object             | 512   | 31        | MuJoCo  |
| `envs/MuJoCoLookAt-v1.py`     | Point the wrist camera at an object | 256   | 23        | MuJoCo  |
| `envs/MuJoCoMove-v1.py`       | Move the end-effector a set offset  | 256   | 22        | MuJoCo  |
| `envs/MuJoCoPickLift-v1.py`   | Grasp and lift an object            | 1024  | 31        | MuJoCo  |
| `envs/MuJoCoPickAndPlace-v1.py` | Place an object on a goal disc    | 1024  | 43        | MuJoCo  |
| `envs/MuJoCoStackCube-v1.py`  | Stack one cube on another           | 1024  | 43        | MuJoCo  |
| `envs/WarpTouch-v1.py`        | Touch the target object             | 512   | 31        | Warp    |
| `envs/WarpLookAt-v1.py`       | Point the wrist camera at an object | 256   | 23        | Warp    |
| `envs/WarpMove-v1.py`         | Move the end-effector a set offset  | 256   | 22        | Warp    |
| `envs/WarpPickLift-v1.py`     | Grasp and lift an object            | 1024  | 31        | Warp    |
| `envs/WarpPickAndPlace-v1.py` | Place an object on a goal disc      | 1024  | 43        | Warp    |
| `envs/WarpStackCube-v1.py`    | Stack one cube on another           | 1024  | 43        | Warp    |

State dimensions are the default observation layout; they change with the
`observations` component list. Task semantics are identical across the two
backends. The MuJoCo backend vectorizes as `n_envs` independent processes or
threads; the Warp backend runs `n_envs` worlds inside one batched simulator.

## Observations and actions

`obs_type="state"` (the default) returns:

- `agent_pos`: `(n_envs, 6)` joint positions, in radians
- `environment_state`: `(n_envs, state_dim)` full state vector

`obs_type="pixels_agent_pos"` returns:

- `agent_pos`: `(n_envs, 6)` joint positions, in radians
- `pixels`: `{"wrist": ..., "overhead": ...}`, HWC uint8 images

LeRobot's `preprocess_observation` maps these to `observation.state`,
`observation.environment_state`, and `observation.images.<camera>`. The language
instruction for the current episode is read off `task_description`, and success
is reported in `info["final_info"]["is_success"]` on the terminating step.

Actions are `(n_envs, 6)` absolute joint targets in radians by default
(`control_mode="pd_joint_pos"`). Delta joint modes and end-effector modes
(`pd_joint_delta_pos`, `pd_ee_pose`, `pd_ee_delta_pose`) are selectable per
environment.

Units are the simulator's own. Datasets recorded through the library's LeRobot
follower adapter store LeRobot motor units instead (degrees, with the gripper in
`RANGE_0_100`); convert with `so101_nexus.dataset_row_to_sim_qpos`.

## Configuration

A `HubEnvConfig` selects the environment through its `task` field:

```python
from lerobot.envs.factory import make_env
from lerobot.envs.configs import HubEnvConfig

cfg = HubEnvConfig(hub_path="johnsutor/so101-nexus-envs", task="MuJoCoStackCube-v1")
envs = make_env(cfg, n_envs=2, trust_remote_code=True)
```

`obs_type`, `observation_width`, `observation_height` and `episode_length` are
read off the config too when it carries them (LeRobot's `LiberoEnv` and
`IsaaclabArenaEnv` do), as is a free-form `kwargs` dict. For the full option set
without a config class, call the library entry point directly:

```python
from so101_nexus.envhub import make_env

envs = make_env(
    n_envs=2,
    env_id="MuJoCoStackCube-v1",
    obs_type="pixels_agent_pos",
    observation_width=224,
    observation_height=224,
    episode_length=300,
    control_mode="pd_joint_delta_pos",
    render_mode="rgb_array",
)
```

Recognized options: `env_id`, `obs_type`, `observation_width`,
`observation_height`, `episode_length`, `control_mode`, `render_mode`,
`disable_env_checker`, `device` (Warp only), and `config` (a fully built
`so101_nexus` environment config, which overrides `obs_type` and the camera
resolution).

## Notes on the Warp backend

The Warp environments are natively batched on one device and speak torch
tensors. The EnvHub adapter converts to NumPy at the boundary because that is
what LeRobot's rollout consumes, which copies each observation to host memory
every step. For GPU-resident training loops, use
`gymnasium.make_vec("WarpPickLift-v1", num_envs=...)` directly. They also seed
one generator for the whole batch, so a per-world seed list collapses to its
first entry.

## Links

- Source: https://github.com/johnsutor/so101-nexus
- Documentation: https://so101-nexus.com/docs
- License: Apache-2.0
