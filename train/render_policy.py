import pickle
import sys
from os.path import join

import jax
from brax.io import html, model
from brax.training.acme import running_statistics
from brax.training.agents.ppo import train as ppo
import brax.training.agents.sac.networks as sac_net

from train.envs import create_env


def _collect_and_save_rollouts(env, inference_fn, run_folder_path, num_episodes=5, episode_length=1000):
    """Collect rollouts and save HTML render."""
    jit_env_reset = jax.jit(env.reset)
    jit_env_step = jax.jit(env.step)
    jit_inference_fn = jax.jit(inference_fn)
    rollouts = []
    for episode in range(num_episodes):
        rng = jax.random.PRNGKey(seed=episode)
        state = jit_env_reset(rng=rng)
        for i in range(episode_length):
            rollouts.append(state.pipeline_state)
            act_rng, rng = jax.random.split(rng)
            act, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_env_step(state, act)
    render = html.render(env.sys, rollouts)
    render_path = join(run_folder_path, "trained_render.html")
    with open(render_path, "w") as f:
        f.write(render)
    print(f"Rendered rollouts saved to '{render_path}'")
    return render


def render_from_path(run_folder_path):
    """Load a trained policy from disk and render it."""
    # Rendering with GPU acceleration enabled is very slow
    jax.config.update("jax_platform_name", "cpu")

    params_path = join(run_folder_path, "params")
    args_path = join(run_folder_path, "args.pkl")
    params = model.load_params(params_path)

    with open(args_path, "rb") as f:
        args = pickle.load(f)

    env = create_env(args['env'], args['xml'])

    if 'algorithm' in args and args["algorithm"] == "ppo":
        make_inference_fn, _, _ = ppo.train(environment=env, num_timesteps=0, episode_length=1000, normalize_observations=True)
    else:
        networks = sac_net.make_sac_networks(env.observation_size, env.action_size, preprocess_observations_fn=running_statistics.normalize)
        make_inference_fn = sac_net.make_inference_fn(networks)

    inference_fn = make_inference_fn(params)
    return _collect_and_save_rollouts(env, inference_fn, run_folder_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python render_policy.py path/to/run/folder/")
        sys.exit(1)

    run_folder_path = sys.argv[1]

    render_from_path(run_folder_path)
