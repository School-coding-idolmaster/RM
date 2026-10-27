"""Why do some wheeled designs terminate right after landing on CPU MuJoCo but not in SAC (MJX)?

For each design: drop from the initial pose with zero action and record the torso height (qpos z)
for 60 env steps under
  (a) CPU MuJoCo, XML solver settings (what mpc.mppi uses by default)
  (b) CPU MuJoCo, RoboMorph's MJX solver override
  (c) MJX through RoboMorph's own FlatEnv (what SAC trained on), same reset noise seed off
Prints min / final torso z and the first env step with z < 0.05 (RoboMorph's termination rule).

Usage (WSL, from D:\\RM\\robomorph):
    uv run --locked python -m mpc.diag_landing --xml_dir ../sac_runs/run1 --only 0,3,5,16,2
"""
import argparse
from pathlib import Path

import mujoco
import numpy as np

from mpc.mppi import load_model

N_ENV_STEPS = 60
N_FRAMES = 5


def cpu_trace(xml, mjx_solver):
    m = load_model(xml, "flat", mjx_solver)
    d = mujoco.MjData(m)
    zs = []
    for _ in range(N_ENV_STEPS):
        mujoco.mj_step(m, d, nstep=N_FRAMES)
        zs.append(float(d.qpos[2]))
    return np.array(zs)


def mjx_trace(xml_path):
    import jax
    from jax import numpy as jp
    from train.envs import create_env
    env = create_env("flat", str(xml_path), reset_noise_scale=0.0, terminate_when_unhealthy=False)
    state = jax.jit(env.reset)(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    act = jp.zeros(env.action_size)
    zs = []
    for _ in range(N_ENV_STEPS):
        state = step(state, act)
        zs.append(float(state.pipeline_state.x.pos[0, 2]))
    return np.array(zs)


def describe(name, zs):
    below = np.where(zs < 0.05)[0]
    first = int(below[0]) + 1 if len(below) else None
    print(f"  {name:22s} min z {zs.min():.4f}  final z {zs[-1]:.4f}  first env step z<0.05: {first}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml_dir", required=True)
    p.add_argument("--only", default="0,3,5,16,2")
    p.add_argument("--no_mjx", action="store_true")
    a = p.parse_args()
    for i in [int(s) for s in a.only.split(",")]:
        path = Path(a.xml_dir) / f"design_{i:02d}.xml"
        xml = path.read_text()
        print(f"design_{i:02d}")
        describe("CPU (XML solver)", cpu_trace(xml, False))
        describe("CPU (MJX solver opts)", cpu_trace(xml, True))
        if not a.no_mjx:
            describe("MJX (RoboMorph env)", mjx_trace(path))


if __name__ == "__main__":
    main()
