import argparse
import functools
import os
import pickle
import time
from dataclasses import dataclass
from os.path import join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb
from brax.io import model
from brax.training.agents.ppo import train as ppo
from brax.training.agents.sac import train as sac

from train.envs import create_env
from train.render_policy import render_from_path


@dataclass
class ExperimentConfig:
    xml: str
    name: str
    render: bool = False
    log: bool = True
    algorithm: str = 'sac'
    timesteps: int = 1_000_000
    num_evals: int = 1
    terrain: str = 'flat'
    seed: int = 42


@dataclass
class Hyperparameters:
    lr: float = 0.001
    batch_size: int = 256
    discounting: float = 0.99
    grad_updates_per_step: int = 64
    min_replay_size: int = 5000
    max_replay_size: int = 4000000
    num_minibatches: int = 0
    entropy_cost: float = 0


EP_LENGTH = 1_000
NUM_ENVS = 4096
WANDB_ENTITY = os.environ.get('WANDB_ENTITY', None)
WANDB_PROJECT = os.environ.get('WANDB_PROJECT', 'robomorph')


def train_policy(config: ExperimentConfig, hyperparameters: Hyperparameters) -> float:
    """Train an RL agent on a robot XML and return forward velocity as fitness."""
    start_time = time.time()

    run_dir = join("runs", f"{config.name or 'unnamed_run'}_s_{config.seed}")
    params_path = join(run_dir, "params")
    args_path = join(run_dir, "args.pkl")
    plot_path = join(run_dir, "plot.png")

    os.makedirs(run_dir, exist_ok=True)

    env = create_env(config.terrain, config.xml)
    if config.algorithm == "ppo":
        train_fn = functools.partial(
            ppo.train,
            num_timesteps=config.timesteps,
            num_evals=config.num_evals,
            episode_length=EP_LENGTH,
            normalize_observations=True,
            num_updates_per_batch=hyperparameters.grad_updates_per_step,
            discounting=hyperparameters.discounting,
            learning_rate=hyperparameters.lr,
            num_envs=NUM_ENVS,
            batch_size=hyperparameters.batch_size,
            num_minibatches=hyperparameters.num_minibatches,
            entropy_cost=hyperparameters.entropy_cost,
            seed=config.seed,
        )
    else:
        train_fn = functools.partial(
            sac.train,
            num_timesteps=config.timesteps,
            num_evals=config.num_evals,
            episode_length=EP_LENGTH,
            normalize_observations=True,
            discounting=hyperparameters.discounting,
            learning_rate=hyperparameters.lr,
            num_envs=NUM_ENVS,
            batch_size=hyperparameters.batch_size,
            grad_updates_per_step=hyperparameters.grad_updates_per_step,
            max_replay_size=hyperparameters.max_replay_size,
            min_replay_size=hyperparameters.min_replay_size,
            seed=config.seed,
        )

    full_config = train_fn.keywords.copy()
    full_config["xml"] = config.xml
    full_config["algorithm"] = config.algorithm
    full_config["env"] = config.terrain

    wandb_mode = "online" if config.log and os.environ.get('WANDB_API_KEY') else "disabled"
    wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=config.name,
        config=full_config,
        mode=wandb_mode,
    )

    steps = []
    rewards = []
    rewards_forward = []

    def progress(num_steps, metrics):
        steps.append(num_steps)
        rewards.append(metrics['eval/episode_reward'])
        rewards_forward.append(metrics['eval/episode_reward_forward'])
        if config.log:
            wandb.log(metrics, step=num_steps)
        print(f"Step: {num_steps}, reward: {metrics['eval/episode_reward']:.2f}, "
              f"forward: {metrics['eval/episode_reward_forward']:.2f}")

    _, params, _ = train_fn(environment=env, progress_fn=progress)

    model.save_params(params_path, params)
    with open(args_path, "wb") as f:
        pickle.dump(full_config, f)

    print(f"Training finished in {time.time() - start_time:.1f}s, params at '{run_dir}'")

    fig = plt.figure()
    plt.plot(steps, rewards)
    plt.xlabel('# environment steps')
    plt.ylabel('reward per episode')
    plt.savefig(plot_path)
    plt.close(fig)

    if config.render:
        render = render_from_path(run_dir)
        wandb.log({"render": wandb.Html(render)})

    wandb.finish()
    return rewards_forward[-1] if rewards_forward else 0.0


def main():
    parser = argparse.ArgumentParser(description="Training script arguments")

    parser.add_argument("--xml", type=str, default=None)
    parser.add_argument("--algorithm", choices=["ppo", "sac"], default="sac")
    parser.add_argument("--terrain", choices=["flat", "ridged", "beams", "frozen_lake"], default="flat")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--render", default=False, action="store_true")
    parser.add_argument("--no_log", default=False, action="store_true")
    parser.add_argument("--timesteps", type=int, default=50_000_000, help="Training timesteps (default 50M for standalone, 1M during evolution)")
    parser.add_argument("--num_evals", default=1, type=int)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--grad_updates_per_step", type=int, default=64)
    parser.add_argument("--min_replay_size", type=int, default=5000)
    parser.add_argument("--max_replay_size", type=int, default=4000000)
    parser.add_argument("--num_minibatches", type=int, default=0)
    parser.add_argument("--entropy_cost", type=float, default=0)

    args = parser.parse_args()

    config = ExperimentConfig(
        xml=args.xml,
        name=args.name,
        render=args.render,
        log=not args.no_log,
        algorithm=args.algorithm,
        timesteps=args.timesteps,
        num_evals=args.num_evals,
        terrain=args.terrain,
        seed=args.seed,
    )

    hyperparameters = Hyperparameters(
        lr=args.lr,
        batch_size=args.batch_size,
        discounting=args.discounting,
        grad_updates_per_step=args.grad_updates_per_step,
        min_replay_size=args.min_replay_size,
        max_replay_size=args.max_replay_size,
        num_minibatches=args.num_minibatches,
        entropy_cost=args.entropy_cost,
    )

    reward = train_policy(config, hyperparameters)
    print(f"Reward: {reward} for: {config.name} experiment")


if __name__ == "__main__":
    main()
