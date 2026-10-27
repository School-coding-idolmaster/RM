"""并行方式测速（双机方案第 5 节第 8 步）：不同的 进程数 × 线程数 下，每小时能评估多少个 Hyblock 回合。

用法（仓库根目录）：
    uv run --locked python wj_tools/bench_parallel.py --out ../mac_runs/bench
每个配置同时起 P 个进程，每个进程用 T 个线程把下面 3 个设计（关节 4 / 6 / 10，取自平地标定）各跑 1 个完整回合
（种子 0，结果与线程数无关，所以各配置的工作量完全相同）。指标：回合 / 小时 = P × 3 × 3600 / 墙钟时间。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DESIGNS = {
    "j4_rand_002": "h0=root; h1=h0.xm@270; h2=h0.ym@270; m3=h0.zp@0 q90/30 mirror; foot=h2.zm; foot=m3.P2",
    "j6_rand_029": "h0=root; m1=h0.ym@0 x2 q90/150; h2=m1.P2@270; m3=h0.zm@270 q0 mirror; foot=m3.P2",
    "j10_rand_014": "m0=root q-30; m1=m0.P1@0 q30/60; m2=m1.P2@180 q30/150; m3=m0.P2@0 q150/30; h4=m2.P2@90; m5=m3.P2@90",
}
CONFIGS = [(1, 4), (1, 6), (1, 8), (1, 12), (2, 6), (3, 4), (4, 3)]
REPO = Path(__file__).resolve().parents[1]


def worker(threads: int) -> None:
    sys.path.insert(0, str(REPO))
    from hyblock.mjcf import compile_design
    from mpc.mppi import MPPIConfig, evaluate_design
    out = {}
    for name, text in DESIGNS.items():
        t0 = time.time()
        r = evaluate_design(compile_design(text, max_modules=6), "flat", 1, 0, MPPIConfig(threads=threads))
        out[name] = {"score": r["fitness"], "seconds": time.time() - t0,
                     "env_steps": r["episodes"][0]["env_steps"]}
    print("RESULT " + json.dumps(out), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out")
    a = p.parse_args()
    if a.worker:
        return worker(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_meta import start, done
    start(out, threads=None, note="parallel benchmark: processes x threads, 3 Hyblock designs, 1 episode each")
    rows = []
    ref_scores = None
    for P, T in CONFIGS:
        t0 = time.time()
        procs = [subprocess.Popen([sys.executable, __file__, "--worker", "--threads", str(T)], cwd=REPO,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True) for _ in range(P)]
        results = []
        for pr in procs:
            text, _ = pr.communicate()
            line = [l for l in text.splitlines() if l.startswith("RESULT ")]
            results.append(json.loads(line[-1][7:]) if line else None)
        wall = time.time() - t0
        ok = all(results)
        scores = {k: v["score"] for k, v in results[0].items()} if ok else None
        if ref_scores is None:
            ref_scores = scores
        same = ok and all({k: v["score"] for k, v in r.items()} == ref_scores for r in results)
        per = {k: round(sum(r[k]["seconds"] for r in results) / P, 2) for k in DESIGNS} if ok else {}
        row = {"processes": P, "threads": T, "wall_s": round(wall, 1),
               "episodes_per_hour": round(P * len(DESIGNS) * 3600 / wall, 1) if ok else 0,
               "scores_identical": same, **{f"s_{k}": v for k, v in per.items()}}
        rows.append(row)
        print(row, flush=True)
    with open(out / "bench.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best = max(rows, key=lambda r: r["episodes_per_hour"])
    single = max((r for r in rows if r["processes"] == 1), key=lambda r: r["episodes_per_hour"])
    lines = [f"best overall: {best['processes']} proc x {best['threads']} threads, "
             f"{best['episodes_per_hour']} episodes/h",
             f"best single process: {single['threads']} threads, {single['episodes_per_hour']} episodes/h, "
             + ", ".join(f"{k} {single['s_' + k]} s" for k in DESIGNS),
             f"reference scores (seed 0): {ref_scores}"]
    (out / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    done(out, best=best, best_single=single)


if __name__ == "__main__":
    main()
