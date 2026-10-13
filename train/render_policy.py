import pickle
import sys
from os.path import join

from pathlib import Path


def _collect_and_save_rollouts(env, inference_fn, run_folder_path, num_episodes=5, episode_length=1000):
    """Collect rollouts and save HTML render."""
    import jax
    from brax.io import html

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
    with open(render_path, "x") as f:
        f.write(render)
    print(f"Rendered rollouts saved to '{render_path}'")
    return render


def render_from_path(run_folder_path):
    """Load a trained policy from disk and render it without training."""
    output_path = Path(run_folder_path) / "trained_render.html"
    if output_path.exists():
        raise FileExistsError(f"Render already exists: {output_path}")

    import jax
    from brax.io import model
    from brax.training.acme import running_statistics
    from train.envs import create_env

    # Rendering with GPU acceleration enabled is very slow
    jax.config.update("jax_platform_name", "cpu")

    params_path = join(run_folder_path, "params")
    args_path = join(run_folder_path, "args.pkl")
    params = model.load_params(params_path)

    with open(args_path, "rb") as f:
        args = pickle.load(f)

    env = create_env(args['env'], args['xml'])

    if 'algorithm' in args and args["algorithm"] == "ppo":
        from brax.training.agents.ppo import networks as ppo_net
        networks = ppo_net.make_ppo_networks(
            env.observation_size, env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
        )
        make_inference_fn = ppo_net.make_inference_fn(networks)
    else:
        from brax.training.agents.sac import networks as sac_net
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
