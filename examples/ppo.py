"""Train PPO agents on SO101-Nexus environments.

This script adapts CleanRL's PPO training loop to the SO101-Nexus MuJoCo and
ManiSkill environments. It uses an MLP actor-critic over state observations
with continuous actions.
"""

import importlib
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import tyro
from torch import nn, optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


def import_backend_for_env_id(env_id: str) -> None:
    """Import the simulator package needed to register *env_id*."""
    if env_id.startswith("ManiSkill"):
        importlib.import_module("so101_nexus_maniskill")
        return
    if env_id.startswith("MuJoCo"):
        importlib.import_module("so101_nexus_mujoco")
        return
    raise ValueError(
        "env_id must start with 'ManiSkill' or 'MuJoCo' "
        "so the corresponding backend can be imported"
    )


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str | None = None
    """the entity (team) of wandb's project"""
    wandb_log_video: bool = True
    """if toggled, logs periodic rollout videos to wandb when tracking is enabled"""
    video_log_interval: int = 100000
    """log one rollout video every N environment steps (0 disables)"""
    video_fps: int = 10
    """fps used for logged wandb videos"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    env_id: str = "MuJoCoPickLift-v1"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float | None = None
    """the target KL divergence threshold"""

    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


_LAYER_INIT_STD = float(np.sqrt(2))


def is_maniskill_env(env_id: str) -> bool:
    """Return whether *env_id* targets a ManiSkill backend."""
    return env_id.startswith("ManiSkill")


def make_mujoco_env(env_id, idx, capture_video, run_name):
    """Build a thunk that creates one MuJoCo environment instance."""

    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def make_env(env_id, idx, capture_video, run_name, gamma):
    """Build an environment factory compatible with the selected backend."""
    del gamma

    if is_maniskill_env(env_id):

        def thunk():
            if capture_video and idx == 0:
                return gym.make(
                    env_id,
                    render_mode="rgb_array",
                    obs_mode="state",
                    control_mode="pd_joint_delta_pos",
                )
            return gym.make(env_id, obs_mode="state", control_mode="pd_joint_delta_pos")

        return thunk

    return make_mujoco_env(env_id, idx, capture_video, run_name)


def should_log_wandb_video(global_step: int, interval: int) -> bool:
    """Return whether the current step should trigger rollout video logging."""
    return interval > 0 and global_step > 0 and (global_step % interval == 0)


def frames_to_tchw(frames: np.ndarray) -> np.ndarray:
    """Convert NHWC image frames to the TCHW layout expected by Weights & Biases."""
    return np.transpose(frames, (0, 3, 1, 2))


def make_video_eval_env(env_id: str):
    """Create a single-environment evaluator configured for video rendering."""
    if is_maniskill_env(env_id):
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        env = gym.make(
            env_id,
            num_envs=1,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
            render_mode="rgb_array",
        )
        return ManiSkillVectorEnv(env, auto_reset=True, ignore_terminations=False)
    return gym.make(env_id, render_mode="rgb_array")


def maybe_log_wandb_rollout_video(  # noqa: C901
    *,
    args: Args,
    agent: "Agent",
    device: torch.device,
    wandb_module,
    global_step: int,
    run_seed: int,
) -> None:
    """Run a short evaluation rollout and upload it to Weights & Biases when enabled."""
    if wandb_module is None or not args.track or not args.wandb_log_video:
        return
    if not should_log_wandb_video(global_step=global_step, interval=args.video_log_interval):
        return

    eval_env = make_video_eval_env(args.env_id)
    maniskill = is_maniskill_env(args.env_id)
    frames: list[np.ndarray] = []

    def _to_frame(raw_frame):
        if isinstance(raw_frame, torch.Tensor):
            f = raw_frame.cpu().numpy()
            if f.ndim == 4:
                f = f[0]
            return f
        if isinstance(raw_frame, np.ndarray):
            return raw_frame
        return None

    obs, _ = eval_env.reset(seed=run_seed + global_step)
    first_frame = _to_frame(eval_env.render())
    if first_frame is not None:
        frames.append(first_frame)

    done = False
    max_steps = getattr(eval_env.spec, "max_episode_steps", 512) or 512
    steps = 0
    while not done and steps < max_steps:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(obs_tensor)
        step_action = action.cpu().numpy()
        if not maniskill:
            step_action = step_action[0]
        obs, _, terminated, truncated, _ = eval_env.step(step_action)
        term = bool(torch.as_tensor(terminated).bool().any().item())
        trunc = bool(torch.as_tensor(truncated).bool().any().item())
        done = term or trunc
        frame = _to_frame(eval_env.render())
        if frame is not None:
            frames.append(frame)
        steps += 1
    eval_env.close()

    if not frames:
        return
    frames_np = np.stack(frames)
    wandb_module.log(
        {
            "videos/policy_rollout": wandb_module.Video(
                frames_to_tchw(frames_np), fps=args.video_fps, format="mp4"
            )
        },
        step=global_step,
    )


def layer_init(layer, std=_LAYER_INIT_STD, bias_const=0.0):
    """Apply orthogonal initialization to a linear layer and return it."""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    """Simple MLP actor-critic used by the PPO training loop."""

    def __init__(self, envs):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(np.prod(envs.single_observation_space.shape), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
        )
        self.actor_mean = layer_init(
            nn.Linear(256, np.prod(envs.single_action_space.shape)), std=0.01
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))
        self.critic = layer_init(nn.Linear(256, 1), std=1)

    def get_value(self, x):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        hidden = self.network(x)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden)


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    wandb = None
    if args.track:
        wandb = importlib.import_module("wandb")
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n{}".format(
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])
        ),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    import_backend_for_env_id(args.env_id)
    if is_maniskill_env(args.env_id):
        from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

        maniskill_env = gym.make(
            args.env_id,
            num_envs=args.num_envs,
            obs_mode="state",
            control_mode="pd_joint_delta_pos",
        )
        envs = ManiSkillVectorEnv(
            maniskill_env,
            auto_reset=True,
            ignore_terminations=False,
            record_metrics=True,
        )
    else:
        envs = gym.vector.SyncVectorEnv(
            [
                make_mujoco_env(args.env_id, i, args.capture_video, run_name)
                for i in range(args.num_envs)
            ]
        )
    assert isinstance(envs.single_action_space, gym.spaces.Box), (
        "only continuous action space is supported"
    )
    assert envs.single_observation_space.shape is not None
    assert envs.single_action_space.shape is not None
    obs_shape = tuple(envs.single_observation_space.shape)
    action_shape = tuple(envs.single_action_space.shape)

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    obs = torch.zeros((args.num_steps, args.num_envs, *obs_shape)).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, *action_shape)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = torch.logical_or(
                torch.as_tensor(terminations, device=device),
                torch.as_tensor(truncations, device=device),
            )
            rewards[step] = torch.as_tensor(reward, dtype=torch.float32, device=device).view(-1)
            next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            next_done = next_done.to(dtype=torch.float32)

            if is_maniskill_env(args.env_id):
                if "final_info" in infos:
                    mask = infos["_final_info"]
                    ep = infos["final_info"]["episode"]
                    done_returns = ep["return"][mask]
                    done_lengths = ep["episode_len"][mask]
                    for ret, length in zip(done_returns, done_lengths, strict=True):
                        ep_return = float(ret.item())
                        ep_length = float(length.item())
                        print(f"global_step={global_step}, episodic_return={ep_return}")
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
            elif "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )

        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        b_obs = obs.reshape((-1, *obs_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1, *action_shape))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        b_inds = np.arange(args.batch_size)
        clipfracs = []
        approx_kl = old_approx_kl = pg_loss = v_loss = entropy_loss = torch.tensor(0.0)
        for _epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        maybe_log_wandb_rollout_video(
            args=args,
            agent=agent,
            device=device,
            wandb_module=wandb,
            global_step=global_step,
            run_seed=args.seed,
        )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        import importlib

        evaluate = importlib.import_module("cleanrl_utils.evals.ppo_eval").evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            import importlib

            push_to_hub = importlib.import_module("cleanrl_utils.huggingface").push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "PPO",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    envs.close()
    writer.close()
