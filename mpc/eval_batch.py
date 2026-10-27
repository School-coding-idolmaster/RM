"""Score a batch of RoboMorph XMLs with CPU MPPI and compare with the SAC scores.

Usage (WSL, from D:\\RM\\robomorph):
    uv run --locked python -m mpc.eval_batch --xml_dir ../sac_runs/run1 --out ../mpc_runs/run1 \
        --episodes 3 --threads 8
    uv run --locked python -m mpc.eval_batch --xml mpc/ant_test.xml --out ../mpc_runs/ant --steps 100

Outputs in --out: results.csv (one row per design), episodes.jsonl (per-episode details and the
per-env-step reward trace), summary.txt (timing + Spearman / top-k against the SAC results.csv).
Re-running skips designs already in results.csv, so an interrupted run can simply be restarted.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from mpc.mppi import MPPIConfig, evaluate_design


def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort()
    ranks = np.empty(len(a))
    ranks[order] = np.arange(len(a))
    for v in np.unique(a):                      # average ties
        idx = a == v
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def spearman_perm_p(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = abs(spearman(a, b))
    rb = rankdata(b)
    ra = rankdata(a)
    hits = sum(abs(np.corrcoef(ra, rng.permutation(rb))[0, 1]) >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


def topk_recall(score_ref, score_new, k):
    top_ref = set(np.argsort(score_ref)[::-1][:k])
    top_new = set(np.argsort(score_new)[::-1][:k])
    return len(top_ref & top_new) / k


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml_dir", help="directory with design_XX.xml (and optionally the SAC results.csv)")
    p.add_argument("--xml", help="single XML file instead of --xml_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--terrain", default="flat", choices=["flat", "ridged", "beams", "frozen_lake"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--steps", type=int, default=250, help="control steps per episode (250 = 1000 env steps)")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--mjx_solver", action="store_true", help="use RoboMorph's MJX solver override (unstable on CPU)")
    p.add_argument("--fall_penalty", type=float, default=100.0)
    p.add_argument("--noise_warm", type=float, default=0.5, help="noise std of the warm-start half")
    p.add_argument("--noise_hist", type=float, default=0.1, help="noise std of the history half")
    p.add_argument("--no_filter", action="store_true", help="disable the FIR low-pass on the sampling noise")
    p.add_argument("--max_speed", type=float, default=10.0, help="torso |v_x| above this [m/s] = physics glitch")
    p.add_argument("--min_iterations", type=int, default=20, help="raise the XML solver iterations to at least this")
    p.add_argument("--only", type=str, default=None, help="comma-separated design indices, e.g. 0,1,2")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.xml:
        xmls = [Path(a.xml)]
    else:
        xmls = sorted(Path(a.xml_dir).glob("design_*.xml"))
        if a.only:
            keep = {int(i) for i in a.only.split(",")}
            xmls = [x for x in xmls if int(x.stem.split("_")[1]) in keep]
    if not xmls:
        sys.exit("no XML files found")

    cfg = MPPIConfig(num_samples=a.samples, episode_control_steps=a.steps, threads=a.threads,
                     mjx_solver=a.mjx_solver, fall_penalty=a.fall_penalty,
                     noise_std_warm=a.noise_warm, noise_std_history=a.noise_hist,
                     filter_noise=not a.no_filter, max_speed=a.max_speed,
                     min_solver_iterations=a.min_iterations)
    with open(out / "config.json", "w") as f:
        json.dump({"args": vars(a), "mppi": cfg.__dict__}, f, indent=2, default=str)

    res_path = out / "results.csv"
    done = set()
    if res_path.exists():
        with open(res_path) as f:
            done = {row["design"] for row in csv.DictReader(f)}
    new_file = not res_path.exists()
    fields = ["design", "nu", "fitness", "fitness_std", "fitness_first200", "distance_m", "env_steps_mean",
              "terminated", "unstable", "seconds", "physics_steps_per_s"]
    with open(res_path, "a", newline="") as fres, open(out / "episodes.jsonl", "a") as fep:
        w = csv.DictWriter(fres, fieldnames=fields)
        if new_file:
            w.writeheader()
        for x in xmls:
            if x.stem in done:
                print(f"skip {x.stem} (already in results.csv)")
                continue
            t0 = time.time()
            r = evaluate_design(x.read_text(), a.terrain, a.episodes, a.seed, cfg)
            secs = time.time() - t0
            eps = r["episodes"]
            row = {
                "design": x.stem, "nu": r["nu"],
                "fitness": round(r["fitness"], 4),
                "fitness_std": round(float(np.std([e["score"] for e in eps])), 4),
                "fitness_first200": round(r["fitness_first200"], 4),
                "distance_m": round(float(np.mean([e["distance"] for e in eps])), 4),
                "env_steps_mean": round(float(np.mean([e["env_steps"] for e in eps])), 1),
                "terminated": sum(e["terminated"] for e in eps),
                "unstable": sum(e["unstable"] for e in eps),
                "seconds": round(secs, 1),
                "physics_steps_per_s": round(sum(e["physics_steps"] for e in eps) / secs),
            }
            w.writerow(row)
            fres.flush()
            fep.write(json.dumps({"design": x.stem, "episodes": eps, "rewards": r["rewards"]}) + "\n")
            fep.flush()
            print(f"{x.stem}: nu={row['nu']} fitness={row['fitness']:.2f} (+-{row['fitness_std']:.2f}) "
                  f"first200={row['fitness_first200']:.2f} dist={row['distance_m']:.2f} m "
                  f"term={row['terminated']} unstable={row['unstable']} {secs:.1f}s", flush=True)

    summarize(out, Path(a.xml_dir) if a.xml_dir else None)


def summarize(out: Path, xml_dir):
    with open(out / "results.csv") as f:
        mpc = {r["design"]: r for r in csv.DictReader(f)}
    lines = []
    secs = [float(r["seconds"]) for r in mpc.values()]
    lines.append(f"designs: {len(mpc)}; seconds per design: mean {np.mean(secs):.1f}, "
                 f"min {np.min(secs):.1f}, max {np.max(secs):.1f}")
    sac_csv = xml_dir / "results.csv" if xml_dir else None
    if sac_csv and sac_csv.exists():
        with open(sac_csv) as f:
            sac = {f"design_{int(r['idx']):02d}": float(r["reward"]) for r in csv.DictReader(f)}
        names = [n for n in sorted(mpc) if n in sac]
        if len(names) >= 5:
            s = np.array([sac[n] for n in names])
            for col in ("fitness_first200", "fitness"):
                m = np.array([float(mpc[n][col]) for n in names])
                rho = spearman(s, m)
                pval = spearman_perm_p(s, m)
                lines.append(f"SAC vs MPC[{col}] (n={len(names)}): Spearman {rho:.3f} (permutation p={pval:.4f}); "
                             f"top-5 recall {topk_recall(s, m, 5):.2f}; top-3 recall {topk_recall(s, m, 3):.2f}")
            lines.append("design, SAC, MPC_first200, MPC_full")
            for n in names:
                lines.append(f"{n}, {sac[n]:.2f}, {float(mpc[n]['fitness_first200']):.2f}, {float(mpc[n]['fitness']):.2f}")
    text = "\n".join(lines)
    print(text)
    (out / "summary.txt").write_text(text + "\n")


if __name__ == "__main__":
    main()
