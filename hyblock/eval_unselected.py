"""② 挑选的诊断（第 10 周，10-31）：被刷掉的候选如果评估了会怎样？

从 --select 运行里按固定随机数抽 --per_run 个 unselected 候选（均匀抽，不偏向任何一类），用与搜索完全相同的评估器
（① 已通过；MPC 2 回合、同一 MPC 种子规则）补跑 MPC，与同一代被挑中（top）的候选比较：
  - 平均 / 90 分位 / 最高分、≥ 400 的比例
  - "错过的突破"：补跑得分超过该代之前迄今最好得分的被刷掉候选
  - 按修改大类（selection.family）分开
补跑结果不写回原运行（原运行保持不变），写到 --out/unselected_evals.csv 与 report.md。

用法（仓库根目录）：
  uv run --locked python -m hyblock.eval_unselected --runs ../mac_runs/base_flat8/ga_sel_llm_s0 ... --per_run 15 --out <目录>
可中断后重新运行（已补跑的跳过）。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from hyblock.mjcf import compile_design
from hyblock.screen import ScreenConfig, postcheck
from hyblock.selection import family
from mpc.mppi import MPPIConfig, evaluate_design

FIELDS = ["run", "gen", "idx", "source", "family", "sel_rank", "fitness", "ep_scores", "best_before", "beats_best",
          "modules", "mpc_s"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--per_run", type=int, default=15)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--steps", type=int, default=None, help="tests only (default: the run's own setting)")
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "unselected_evals.csv"
    done = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            done = {(r["run"], r["gen"], r["idx"]) for r in csv.DictReader(f)}
    t0 = time.time()
    check(Path(a.runs[0]), a)
    for run in map(Path, a.runs):
        meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))["args"]
        with open(run / "evals.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        uns = [r for r in rows if r["status"] == "unselected"]
        pick = random.Random(f"eval-unselected|{run.name}").sample(uns, min(a.per_run, len(uns)))
        cfg = MPPIConfig(threads=a.threads, episode_control_steps=a.steps or meta["steps"],
                         terrain_scale=meta["terrain_scale"])
        scfg = ScreenConfig()
        for r in sorted(pick, key=lambda r: (int(r["gen"]), int(r["idx"]))):
            if (run.name, r["gen"], r["idx"]) in done:
                continue
            g, i = int(r["gen"]), int(r["idx"])
            before = [float(x["fitness"]) for x in rows if x["status"] == "mpc" and int(x["gen"]) < g]
            xml = compile_design(r["design"], max_modules=meta["max_modules"])
            seed = meta["mpc_seed"] + 7919 * g + i if meta.get("vary_mpc_seed") else meta["mpc_seed"]
            t1 = time.time()
            res = evaluate_design(xml, meta["terrain"], meta["episodes"], seed, cfg)
            postcheck(res, scfg)
            best = max(before) if before else float("-inf")
            row = {"run": run.name, "gen": g, "idx": i, "source": r["source"], "family": family(r["source"]),
                   "sel_rank": r["sel_rank"], "fitness": round(res["fitness"], 3),
                   "ep_scores": json.dumps([round(e["score"], 2) for e in res["episodes"]]),
                   "best_before": round(best, 3), "beats_best": int(res["fitness"] > best), "modules": r["modules"],
                   "mpc_s": round(time.time() - t1, 2)}
            new = not path.exists()
            with open(path, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"  {run.name} 代 {g} #{i} {row['family']:9s} {row['fitness']:7.1f}（迄今最好 {best:.1f}）"
                  f"{'  <<< 超过迄今最好' if row['beats_best'] else ''}  {time.time() - t0:.0f} s", flush=True)
    report(a.runs, path, out)


def check(run: Path, a) -> None:
    """自检：用同一评估器重跑该运行里第一个跑过 MPC 的设计，得分应与原记录逐位相同（同一台机器）。"""
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8"))["args"]
    with open(run / "evals.csv", newline="", encoding="utf-8") as f:
        r = next(x for x in csv.DictReader(f) if x["status"] == "mpc")
    cfg = MPPIConfig(threads=a.threads, episode_control_steps=a.steps or meta["steps"], terrain_scale=meta["terrain_scale"])
    g, i = int(r["gen"]), int(r["idx"])
    seed = meta["mpc_seed"] + 7919 * g + i if meta.get("vary_mpc_seed") else meta["mpc_seed"]
    res = evaluate_design(compile_design(r["design"], max_modules=meta["max_modules"]), meta["terrain"], meta["episodes"],
                          seed, cfg)
    same = round(res["fitness"], 3) == float(r["fitness"])
    print(f"自检：{run.name} 代 {g} #{i} 原得分 {float(r['fitness']):.3f}，重跑 {res['fitness']:.3f} → "
          f"{'一致' if same else '!!! 不一致（不同机器时只会统计等价，同一台机器应逐位相同）'}", flush=True)


def report(runs, path: Path, out: Path) -> None:
    with open(path, newline="", encoding="utf-8") as f:
        ev = list(csv.DictReader(f))
    L = [f"# 被刷掉的候选补跑 MPC（{time.strftime('%Y-%m-%d %H:%M')}）", "",
         "每个运行均匀抽 unselected 候选补跑（同一评估器），与同一批代里被挑中（top）的候选比较。", "",
         "| 运行 | 组 | 个数 | 均值 | 中位数 | 90 分位 | 最高 | ≥400 | 超过迄今最好 |", "|---|---|---|---|---|---|---|---|---|"]

    def stats(f):
        f = np.array(f, float)
        return (f"{len(f)} | {f.mean():.0f} | {np.median(f):.0f} | {np.percentile(f, 90):.0f} | {f.max():.0f} | "
                f"{np.mean(f >= 400):.2f}") if len(f) else "0 | – | – | – | – | –"

    fams_all = {}
    for run in map(Path, runs):
        mine = [r for r in ev if r["run"] == run.name]
        if not mine:
            continue
        gens = {r["gen"] for r in mine}
        with open(run / "evals.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        top = [r for r in rows if r.get("sel_slot") == "top" and r["status"] == "mpc" and r["gen"] in gens]
        bt = 0
        for r in top:
            before = [float(x["fitness"]) for x in rows if x["status"] == "mpc" and int(x["gen"]) < int(r["gen"])]
            bt += float(r["fitness"]) > (max(before) if before else float("-inf"))
        L.append(f"| {run.name} | 被刷掉（补跑） | {stats([float(r['fitness']) for r in mine])} | "
                 f"{sum(int(r['beats_best']) for r in mine)} / {len(mine)} |")
        L.append(f"| | 被挑中（同代） | {stats([float(r['fitness']) for r in top])} | {bt} / {len(top)} |")
        for r in mine:
            fams_all.setdefault(r["family"], []).append(float(r["fitness"]))
    L += ["", "## 被刷掉的候选按修改大类", "", "| 大类 | 个数 | 均值 | 中位数 | 90 分位 | 最高 | ≥400 |", "|---|---|---|---|---|---|---|"]
    for k in sorted(fams_all):
        L.append(f"| {k} | {stats(fams_all[k])} |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
