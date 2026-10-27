"""汇总 RM 参照域 70 个设计的 SAC 与 MPC 排序一致性（事先定好 n = 70，跑完就用这个结果）。

读：../sac_runs/run1/results.csv（SAC，取 seed 0 / 50 万步 / 64 环境 / 回合 200 / 8 次更新、status=ok 的最后一行）
    ../mpc_runs/run2_x2/results.csv（MPC，噪声 ×2，3 回合）
写：../mpc_runs/run2_x2/summary_70.txt 与 paired_70.csv（两边都有分数的设计）

用法（在 robomorph 目录下）：
    uv run --locked python ../tools/rank70_summary.py
"""
import csv
import math
from pathlib import Path

import numpy as np

SAC_CSV = Path("../sac_runs/run1/results.csv")
MPC_DIR = Path("../mpc_runs/run2_x2")
CFG = {"sac_seed": "0", "steps": "500000", "envs": "64", "ep": "200", "updates": "8"}
N_PLANNED = 70


def rankdata(a):
    a = np.asarray(a, float)
    order = a.argsort()
    r = np.empty(len(a))
    r[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):
        idx = a == v
        r[idx] = r[idx].mean()
    return r


def spearman(a, b):
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def perm_p(a, b, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    ra, rb = rankdata(a), rankdata(b)
    obs = abs(np.corrcoef(ra, rb)[0, 1])
    hits = sum(abs(np.corrcoef(ra, rng.permutation(rb))[0, 1]) >= obs for _ in range(n))
    return (hits + 1) / (n + 1)


def topk(a, b, k):
    return len(set(np.argsort(a)[::-1][:k]) & set(np.argsort(b)[::-1][:k])) / k


def main():
    sac = {}
    with open(SAC_CSV, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok" or any(r.get(k) != v for k, v in CFG.items()):
                continue
            try:
                sac[f"design_{int(r['idx']):02d}"] = (float(r["reward"]), int(r["nu"]), float(r["seconds"]))
            except ValueError:
                continue
    mpc = {}
    with open(MPC_DIR / "results.csv", newline="") as f:
        for r in csv.DictReader(f):
            mpc[r["design"]] = (float(r["fitness"]), float(r["seconds"]), int(r["terminated"]))

    names = sorted(n for n in mpc if n in sac)
    s = np.array([sac[n][0] for n in names])
    m = np.array([mpc[n][0] for n in names])
    n = len(names)
    lines = [f"planned n = {N_PLANNED} (fixed in advance, 2026-10-26); paired designs with both scores: n = {n}",
             f"missing SAC: {sorted(set(mpc) - set(sac))}",
             f"missing MPC: {sorted(set(sac) - set(mpc))}"]
    if n >= 5:
        rho = spearman(s, m)
        z, se = math.atanh(rho), 1 / math.sqrt(n - 3)
        lo, hi = math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)
        lines.append(f"Spearman rho = {rho:.3f}  (95% CI Fisher z [{lo:.2f}, {hi:.2f}], permutation p = {perm_p(s, m):.4f})")
        for k in (3, 5, 10):
            lines.append(f"top-{k} recall = {topk(s, m, k):.2f}")
        old = [x for x in names if int(x.split('_')[1]) < 20]
        new = [x for x in names if int(x.split('_')[1]) >= 20]
        for tag, sub in (("designs 00-19", old), ("designs 20+", new)):
            if len(sub) >= 5:
                lines.append(f"  {tag}: n = {len(sub)}, Spearman {spearman([sac[x][0] for x in sub], [mpc[x][0] for x in sub]):.3f}")
        lines.append(f"mean seconds per design: SAC {np.mean([sac[x][2] for x in names]) / 60:.1f} min, "
                     f"MPC {np.mean([mpc[x][1] for x in names]):.1f} s")
        lines.append(f"MPC: designs with all 3 episodes terminated = {sum(mpc[x][2] == 3 for x in names)}")
    text = "\n".join(lines)
    print(text)
    (MPC_DIR / "summary_70.txt").write_text(text + "\n")
    with open(MPC_DIR / "paired_70.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "nu", "sac", "mpc", "mpc_terminated_of_3"])
        for x in names:
            w.writerow([x, sac[x][1], sac[x][0], mpc[x][0], mpc[x][2]])


if __name__ == "__main__":
    main()
