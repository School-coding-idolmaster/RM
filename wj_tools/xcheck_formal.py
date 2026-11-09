"""跨机器抽查（工作流规则 2；WJ 11-04 决定做）：从 Mac 的 formal_flat 每个运行里固定随机抽 5% 的 MPC 评估
（去掉配对组复制来的重复行），在 Windows 上用完全相同的设置重评（2 回合，mpc_seed 0，平地），比较得分。
不要求逐位相同（不同机器本来就不同），看统计上是否一致：Spearman、配对差、相对误差。
用法（WSL）：python wj_tools/xcheck_formal.py [--dry]；可中断，再运行从断点继续。结果 D:\\RM\\win_runs\\xcheck_formal_flat\\"""
import argparse, csv, json, random, time
from pathlib import Path

import os
SRC = Path(os.environ.get("XCHECK_SRC", "/mnt/d/RM/from_mac/mac_runs/formal_flat"))
OUT = Path("/mnt/d/RM/win_runs/xcheck_formal_flat")
FRAC = 0.05


def sample():
    seen, picks = set(), []
    for d in sorted(p for p in SRC.iterdir() if (p / "DONE").exists()):
        rows = [r for r in csv.DictReader(open(d / "evals.csv", encoding="utf-8")) if r["status"] == "mpc"]
        own = [r for r in rows if (r["time"], r["design"]) not in seen]
        for r in rows:
            seen.add((r["time"], r["design"]))
        rng = random.Random(f"xcheck|{d.name}")
        k = max(1, round(FRAC * len(rows)))
        for r in rng.sample(own, min(k, len(own))):
            picks.append({"run": d.name, "gen": r["gen"], "idx": r["idx"], "design": r["design"],
                          "mac_fitness": float(r["fitness"]), "mac_eps": r["ep_scores"]})
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    a = p.parse_args()
    picks = sample()
    print(f"抽样 {len(picks)} 个设计（{len({x['run'] for x in picks})} 个运行）")
    if a.dry:
        return
    from hyblock.mjcf import compile_design
    from mpc.mppi import MPPIConfig, evaluate_design
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "xcheck.csv"
    done = {(r["run"], r["gen"], r["idx"]) for r in csv.DictReader(open(out, encoding="utf-8"))} if out.exists() else set()
    cfg = MPPIConfig(threads=a.threads, episode_control_steps=250, terrain_scale=1.0)
    fields = ["run", "gen", "idx", "design", "mac_fitness", "mac_eps", "win_fitness", "win_eps", "seconds"]
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for i, x in enumerate(picks):
            if (x["run"], x["gen"], x["idx"]) in done:
                continue
            t = time.time()
            r = evaluate_design(compile_design(x["design"], max_modules=8), "flat", 2, 0, cfg)
            x.update(win_fitness=round(r["fitness"], 3), win_eps=json.dumps([round(e["score"], 2) for e in r["episodes"]]),
                     seconds=round(time.time() - t, 1))
            w.writerow(x); f.flush()
            print(f"[{i + 1}/{len(picks)}] {x['run']} {x['gen']}.{x['idx']}: Mac {x['mac_fitness']:.1f}  Windows {x['win_fitness']:.1f}", flush=True)
    report(out)


def report(out):
    import numpy as np
    from scipy.stats import spearmanr, pearsonr
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    m = np.array([float(r["mac_fitness"]) for r in rows]); w = np.array([float(r["win_fitness"]) for r in rows])
    d = w - m
    rng = np.random.default_rng(0)
    bs = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(10000)]
    L = ["# 跨机器抽查：formal_flat（Mac）× Windows 重评", "",
         f"- 设计数 {len(rows)}；Spearman {spearmanr(m, w)[0]:.3f}；Pearson {pearsonr(m, w)[0]:.3f}",
         f"- Windows − Mac 均值 {d.mean():+.1f}（bootstrap 95% [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]），中位数 {np.median(d):+.1f}",
         f"- |差| 中位数 {np.median(np.abs(d)):.1f}；|差| ≤ 20% 的比例（Mac ≥ 50 的设计）{np.mean(np.abs(d[m >= 50]) <= 0.2 * m[m >= 50]):.2f}",
         f"- 能走（≥ 20）的判定一致率 {np.mean((m >= 20) == (w >= 20)):.2f}", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
