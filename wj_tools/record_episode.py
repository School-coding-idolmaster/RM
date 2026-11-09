"""Run one MPC episode of a Hyblock design and record the trajectory (qpos after every env step).

The episode loop is a copy of mpc.mppi.MPPIEvaluator.run_episode with one addition (qpos capture), so the
trajectory is the same one that evaluate_design(xml, episodes=1, seed=SEED) scores.

usage: python record_episode.py --src SRC_DIR --design "..." --out traj.npz [--seed 0] [--threads 2]
"""
import argparse
import json
import sys
import time

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--design", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--threads", type=int, default=2)
p.add_argument("--max_modules", type=int, default=8)
p.add_argument("--max_speed_deg_s", type=float, default=None, help="joint speed limit (None = modules.yaml, 1185)")
a = p.parse_args()
sys.path.insert(0, a.src)

import mujoco  # noqa: E402
from hyblock.mjcf import compile_design  # noqa: E402
from mpc.mppi import MPPIConfig, MPPIEvaluator, load_model  # noqa: E402

cfg = MPPIConfig(threads=a.threads)
xml = compile_design(a.design, max_modules=a.max_modules, max_speed_deg_s=a.max_speed_deg_s)
model = load_model(xml, "flat", cfg.mjx_solver, cfg.terrain_scale, cfg.min_solver_iterations)
ev = MPPIEvaluator(model, cfg, seed=a.seed)

t0 = time.time()
d = ev.reset(a.seed * 1000)  # evaluate_design: episode i uses seed*1000+i
plan = np.zeros((cfg.horizon, ev.nu))
history = np.zeros((0, ev.nu))
rewards, qpos, peak = [], [d.qpos.copy()], 0.0
terminated = unstable = False
badqacc = int(mujoco.mjtWarning.mjWARN_BADQACC)
for k in range(cfg.episode_control_steps):
    n = cfg.warmup_samples if k == 0 else cfg.num_samples
    plan = ev.update(d, plan, history, n)
    act = plan[0].copy()
    d.ctrl[:] = ev.scale(act)
    for _ in range(cfg.interval):
        x0, nwarn = d.qpos[0], d.warning[badqacc].number
        mujoco.mj_step(ev.m, d, nstep=ev.n_frames)
        r = -(d.qpos[0] - x0) / ev.dt_env
        if d.warning[badqacc].number > nwarn or not np.isfinite(r) or abs(r) > cfg.max_speed:
            unstable = terminated = True
            break
        rewards.append(r)
        qpos.append(d.qpos.copy())
        if ev.unhealthy(d.qpos[:7]):
            terminated = True
            break
    history = np.vstack([history, act[None]])[-cfg.horizon:]
    plan = np.vstack([plan[1:], np.zeros((1, ev.nu))])
    if terminated:
        break
    if (k + 1) % 50 == 0:
        print(f"ctrl step {k + 1}: score {sum(rewards):.1f}, {time.time() - t0:.0f}s", flush=True)
ev.close()

meta = dict(design=a.design, seed=a.seed, max_speed_deg_s=a.max_speed_deg_s, score=float(np.sum(rewards)), distance_m=float(np.sum(rewards) * ev.dt_env),
            env_steps=len(rewards), dt_env=ev.dt_env, terminated=terminated, unstable=unstable,
            peak_speed=float(np.max(np.abs(rewards))) if rewards else 0.0, seconds=time.time() - t0,
            mujoco=mujoco.__version__)
np.savez_compressed(a.out, qpos=np.array(qpos), rewards=np.array(rewards), xml=np.array(xml))
with open(a.out.replace(".npz", ".json"), "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
