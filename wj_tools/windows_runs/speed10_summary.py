"""Summary after the 10-27 speed-guard change (max_speed 20 -> 10 m/s).

Reads (paths relative to robomorph/):
  ../sac_runs/run1/results.csv        SAC 500k (seed 0, 64 envs, ep 200, 8 updates, status ok)
  ../sac_runs/long1M/results.csv      SAC 1M for selected designs
  ../mpc_runs/run3_v10/results.csv    MPC, max_speed 10 (this rerun)
  ../mpc_runs/run3_v10/episodes.jsonl per-step rewards (peak speed check)
  ../mpc_runs/run2_x2/results.csv     MPC, max_speed 20 (old)
Writes ../mpc_runs/run3_v10/summary_speed10.txt and paired_speed10.csv
"""
import csv
import json
import math
from pathlib import Path

import numpy as np

SAC = Path("../sac_runs/run1/results.csv")
LONG = Path("../sac_runs/long1M/results.csv")
NEW = Path("../mpc_runs/run3_v10")
OLD = Path("../mpc_runs/run2_x2/results.csv")
CFG = {"sac_seed": "0", "steps": "500000", "envs": "64", "ep": "200", "updates": "8"}
SELECTED = [20, 21, 57, 66, 2, 19, 25]          # fixed on 10-27 08:06 (top-10 on one side, rank > 20 on the other)
OLD_1M = [8, 18, 14, 7, 6]                      # run on 10-26


def rankdata(a):
    a = np.asarray(a, float)
    order = (-a).argsort(kind="stable")
    r = np.empty(len(a))
    r[order] = np.arange(1, len(a) + 1)
    for v in np.unique(a):
        idx = a == v
        r[idx] = r[idx].mean()
    return r                                     # 1 = best


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


def ci(rho, n):
    z, se = math.atanh(rho), 1 / math.sqrt(n - 3)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    sac = {f"design_{int(r['idx']):02d}": (float(r["reward"]), int(r["nu"])) for r in read(SAC)
           if r.get("status") == "ok" and all(r.get(k) == v for k, v in CFG.items())}
    new = {r["design"]: r for r in read(NEW / "results.csv")}
    old = {r["design"]: float(r["fitness"]) for r in read(OLD)}
    long1m = {r["design"]: float(r["reward_1M"]) for r in read(LONG)} if LONG.exists() else {}
    names = sorted(n for n in new if n in sac)
    s = np.array([sac[n][0] for n in names])
    m = np.array([float(new[n]["fitness"]) for n in names])
    L = []
    L.append(f"MPC max_speed = 10 m/s (run3_v10) vs SAC 500k: paired n = {len(names)}")
    rho = spearman(s, m)
    lo, hi = ci(rho, len(names))
    L.append(f"  Spearman rho = {rho:.3f}  95% CI [{lo:.2f}, {hi:.2f}]  permutation p = {perm_p(s, m):.4f}")
    L.append(f"  top-3 / 5 / 10 recall = {topk(s, m, 3):.2f} / {topk(s, m, 5):.2f} / {topk(s, m, 10):.2f}")
    L.append(f"  MPC unstable episodes = {sum(int(new[n]['unstable']) for n in names)}; "
             f"designs with all 3 episodes terminated = {sum(int(new[n]['terminated']) == 3 for n in names)}; "
             f"mean seconds/design = {np.mean([float(new[n]['seconds']) for n in names]):.1f}")
    peaks = []
    with open(NEW / "episodes.jsonl") as f:
        for line in f:
            d = json.loads(line)
            for r in d["rewards"]:
                if r:
                    peaks.append(max(r))
    L.append(f"  peak forward speed over {len(peaks)} executed episodes: max {max(peaks):.2f} m/s; "
             f"> 6 m/s: {sum(p > 6 for p in peaks)}")
    both = [n for n in names if n in old]
    L.append(f"  MPC old (20 m/s) vs new (10 m/s), all {len(both)}: Spearman "
             f"{spearman([old[n] for n in both], [float(new[n]['fitness']) for n in both]):.3f}")
    glitch = {"design_21", "design_29", "design_52", "design_57"}
    rest = [n for n in both if n not in glitch]
    L.append(f"  same, excluding the 4 glitch designs (n = {len(rest)}): Spearman "
             f"{spearman([old[n] for n in rest], [float(new[n]['fitness']) for n in rest]):.3f}")
    for n in sorted(glitch):
        L.append(f"    {n}: MPC old {old[n]:.1f} -> new {float(new[n]['fitness']):.1f} "
                 f"(unstable {new[n]['unstable']}/3, terminated {new[n]['terminated']}/3); SAC 500k {sac[n][0]:.1f}")

    rs, rm = rankdata(s), rankdata(m)
    rank_s = dict(zip(names, rs))
    rank_m = dict(zip(names, rm))
    L.append("")
    L.append("SAC 500k -> 1M (selection fixed in advance; ranks: 1 = best among 70)")
    L.append("design, nu, SAC500k, SAC1M, ratio, SACrank500k, MPCrank(v10), group")
    for idx in SELECTED + OLD_1M:
        n = f"design_{idx:02d}"
        grp = ("MPC-favoured" if idx in (20, 21, 57, 66) else "SAC-favoured" if idx in (2, 19, 25)
               else "10-26 run")
        if n in long1m:
            L.append(f"{n}, {sac[n][1]}, {sac[n][0]:.1f}, {long1m[n]:.1f}, {long1m[n] / max(sac[n][0], 1e-6):.2f}, "
                     f"{rank_s[n]:.0f}, {rank_m[n]:.0f}, {grp}")
        else:
            L.append(f"{n}, {sac[n][1]}, {sac[n][0]:.1f}, (missing), , {rank_s[n]:.0f}, {rank_m[n]:.0f}, {grp}")
    mix = np.array([long1m.get(n, sac[n][0]) for n in names])
    rho2 = spearman(mix, m)
    L.append(f"mixed budget (1M where available, n_1M = {sum(n in long1m for n in names)}): Spearman {rho2:.3f}; "
             f"top-5 / 10 recall {topk(mix, m, 5):.2f} / {topk(mix, m, 10):.2f}  (reference only)")
    text = "\n".join(L)
    print(text)
    (NEW / "summary_speed10.txt").write_text(text + "\n")
    with open(NEW / "paired_speed10.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["design", "nu", "sac500k", "sac1M", "mpc_v10", "mpc_v20", "mpc_terminated_of_3", "mpc_unstable"])
        for n in names:
            w.writerow([n, sac[n][1], sac[n][0], long1m.get(n, ""), new[n]["fitness"], old.get(n, ""),
                        new[n]["terminated"], new[n]["unstable"]])


if __name__ == "__main__":
    main()
