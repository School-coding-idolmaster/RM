"""最优设计重评（第 15 周，追加 C；claude/正式实验计划.md 变更记录第 19 条；结果出来之前写定）。

问题：评估噪声大（两回合均值的重测信度约 0.68），"最终最好"= 200 个带噪声分数的最大值，带正向选择偏差。
做法：每个已完成运行（有 DONE）的最终最好设计（evals.csv 里 fitness 最大的 MPC 行；并列取最早的），
     用同一评估器、同一物理参数（取自 run_meta.json 的 args）、新的回合种子（--seed 1：回合种子 1000–1009；
     原评估都是 mpc_seed 0：回合种子 0、1）重跑 10 回合。同一实验根目录里设计与物理参数都相同的只算一次。
只做描述（第 19 条）：原得分 vs 重评得分（回归幅度、Spearman）；组间"最终最好"配对差在重评前后的对比。

用法：python -m hyblock.reeval_best run --roots <根目录>... --out <mac_runs/reeval_best> [--threads 12] [--dry]
      python -m hyblock.reeval_best report --out <mac_runs/reeval_best>
可中断，再运行从断点继续（reeval.csv 逐行追加）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np

EPISODES = 10
SEED = 1
FIELDS = ["root", "run", "group", "seed", "gen", "idx", "design", "terrain", "physics", "orig_fitness", "orig_eps",
          "re_fitness", "re_eps", "re_sd", "seconds", "reused"]
# 事先定的配对比较（根目录名 → [(组 x, 组 y, 说明)]；组 = 运行目录名去掉 _s<种子>）
PAIRS = {
    "formal_flat": [("ga_llmtyped", "ga", "A4 − A1 LLM 变异 − GA"), ("ga_llmtyped_sel_llmEv", "ga", "A5 − A1 组合 − GA"),
                    ("ga_llmtyped_sel_llmEv", "ga_llmtyped", "A5 − A4 ② 在 LLM 变异之上"),
                    ("ga_sel_llmEv", "ga", "A2 − A1 ② − GA")],
    "replicate_flat": [("ga_llmtyped_sel_llmEv", "ga_llmtyped", "A5 − A4（复现）")],
    "formal_beams": [("ga_llmtyped_sel_llmEv", "ga", "B6 − B1 组合 − GA"), ("ga_llmtyped", "ga", "B5 − B1 LLM 变异 − GA"),
                     ("ga_sel_llmEv", "ga", "B2 − B1 ② − GA"), ("ga_init_transfer", "ga", "B3 − B1 平地迁移 − GA"),
                     ("ga_init_llmtransfer", "ga_init_transfer", "B4 − B3 LLM 改写迁移 − 原样迁移"),
                     ("ga_llmtyped_sel_llmEv", "ga_llmtyped", "B6 − B5 ② 在 LLM 变异之上")],
    "p14": [("ga_llmtyped", "ga", "A4 − A1"), ("ga_llmtyped_sel_llmEv", "ga", "A5 − A1")],
    "m15": [("ga_llmtyped", "ga", "A4 − A1"), ("ga_llmtyped_sel_llmEv", "ga", "A5 − A1")],
    "v370": [("ga_llmtyped", "ga", "A4 − A1"), ("ga_llmtyped_sel_llmEv", "ga", "A5 − A1")],
}
PHYS = ("pressure", "mass_scale", "max_speed_deg")
DEFAULT = {"pressure": 8.0, "mass_scale": 1.0}       # 与 search.physics_kw 相同：默认值不传


def best_row(d: Path) -> dict:
    rows = [r for r in csv.DictReader(open(d / "evals.csv", encoding="utf-8")) if r["status"] == "mpc"]
    rows.sort(key=lambda r: (-float(r["fitness"]), int(r["gen"]), int(r["idx"])))
    return rows[0]


def run_args(d: Path) -> dict:
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8"))
    return meta.get("args") or {}


def targets(roots: list[Path]) -> list[dict]:
    out = []
    for root in roots:
        for d in sorted(p for p in root.iterdir() if p.is_dir() and (p / "DONE").exists()):
            m = re.match(r"^(.*)_s(\d+)$", d.name)
            if not m:
                continue
            a, r = run_args(d), best_row(d)
            phys = {k: a[k] for k in PHYS if a.get(k) is not None and a.get(k) != DEFAULT.get(k)}
            out.append({"root": root.name, "run": d.name, "group": m.group(1), "seed": m.group(2), "gen": r["gen"],
                        "idx": r["idx"], "design": r["design"], "terrain": a.get("terrain", "flat"),
                        "terrain_scale": float(a.get("terrain_scale", 1.0)), "max_modules": int(a.get("max_modules", 8)),
                        "steps": int(a.get("steps", 250)), "physics": json.dumps(phys, sort_keys=True),
                        "orig_fitness": float(r["fitness"]), "orig_eps": r["ep_scores"]})
    return out


def physics_kw(phys: dict) -> dict:
    kw = {}
    if "pressure" in phys:
        kw["pressure_MPa"] = float(phys["pressure"])
    if "mass_scale" in phys:
        kw["mass_scale"] = float(phys["mass_scale"])
    if "max_speed_deg" in phys:
        kw["max_speed_deg_s"] = float(phys["max_speed_deg"])
    return kw


def cmd_run(a):
    T = targets([Path(r) for r in a.roots])
    keys = {(t["root"], t["design"], t["terrain"], t["physics"]) for t in T}
    print(f"重评 {len(T)} 个运行的最终最好设计（不同的 {len(keys)} 个），每个 {EPISODES} 回合，回合种子 {SEED}")
    if a.dry:
        for root in sorted({t["root"] for t in T}):
            print(f"  {root}: {sum(t['root'] == root for t in T)} 个运行")
        return
    from hyblock.mjcf import compile_design
    from mpc.mppi import MPPIConfig, evaluate_design
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "reeval.csv"
    have = {}
    if path.exists():
        for r in csv.DictReader(open(path, encoding="utf-8")):
            have[(r["root"], r["run"])] = r
    cache = {(r["root"], r["design"], r["terrain"], r["physics"]): r for r in have.values()}
    new = not path.exists()
    from wj_tools.run_meta import done, start
    start(out, threads=a.threads, note=f"hyblock.reeval_best {EPISODES} episodes seed {SEED}",
          args={"roots": a.roots, "episodes": EPISODES, "seed": SEED, "threads": a.threads})
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, t in enumerate(T):
            if (t["root"], t["run"]) in have:
                continue
            k = (t["root"], t["design"], t["terrain"], t["physics"])
            if k in cache:
                c = cache[k]
                t.update(re_fitness=c["re_fitness"], re_eps=c["re_eps"], re_sd=c["re_sd"], seconds=0, reused=c["run"])
            else:
                t0 = time.time()
                cfg = MPPIConfig(threads=a.threads, episode_control_steps=t["steps"], terrain_scale=t["terrain_scale"])
                xml = compile_design(t["design"], max_modules=t["max_modules"], **physics_kw(json.loads(t["physics"])))
                r = evaluate_design(xml, t["terrain"], EPISODES, SEED, cfg)
                eps = [round(e["score"], 2) for e in r["episodes"]]
                t.update(re_fitness=round(r["fitness"], 3), re_eps=json.dumps(eps), re_sd=round(float(np.std(eps, ddof=1)), 2),
                         seconds=round(time.time() - t0, 1), reused="")
            w.writerow(t)
            f.flush()
            cache[k] = t
            have[(t["root"], t["run"])] = t
            print(f"[{i + 1}/{len(T)}] {t['root']}/{t['run']}: 原 {t['orig_fitness']:.1f} → 重评 {float(t['re_fitness']):.1f}"
                  f"（{t['seconds']} s{'，沿用 ' + t['reused'] if t['reused'] else ''}）", flush=True)
    if len(have) >= len(T):
        done(out, n=len(T))
    (out / "report.md").write_text(report(path), encoding="utf-8")
    print(f"写好了 {out / 'report.md'}")


def report(path: Path) -> str:
    from scipy.stats import spearmanr
    from hyblock.formal_analysis import boot_ci
    R = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in R:
        r["o"], r["n"] = float(r["orig_fitness"]), float(r["re_fitness"])
    L = ["# 最优设计重评（描述性）", "",
         "登记：`claude/正式实验计划.md` 变更记录第 19 条。脚本 `hyblock/reeval_best.py`（结果出来之前写定）。",
         f"每个运行的最终最好设计，同一评估器与物理参数，新的回合种子（{EPISODES} 回合，种子 {SEED}）。", "",
         "## 原得分 vs 重评得分", "",
         "| 实验 | 运行 | 原得分均值 | 重评均值 | 回归（原 − 重评）均值 [95%] | 重评 / 原 | Spearman（原, 重评） | 重评回合 SD 中位数 |",
         "|---|---|---|---|---|---|---|---|"]
    roots = list(dict.fromkeys(r["root"] for r in R))
    for root in roots + ["全部"]:
        S = R if root == "全部" else [r for r in R if r["root"] == root]
        o, n = np.array([r["o"] for r in S]), np.array([r["n"] for r in S])
        lo, hi = boot_ci(o - n)
        rho = spearmanr(o, n)[0] if len(S) >= 5 else float("nan")
        sd = np.median([float(r["re_sd"]) for r in S])
        L.append(f"| {root} | {len(S)} | {o.mean():.0f} | {n.mean():.0f} | {np.mean(o - n):+.0f} [{lo:+.0f}, {hi:+.0f}] | "
                 f"{n.mean() / o.mean():.2f} | {rho:.2f} | {sd:.0f} |")
    L += ["", "## 各组（种子间均值）", "", "| 实验 | 组 | 种子 | 原最终最好 | 重评 | 回归 |", "|---|---|---|---|---|---|"]
    for root in roots:
        for g in dict.fromkeys(r["group"] for r in R if r["root"] == root):
            S = [r for r in R if r["root"] == root and r["group"] == g]
            o, n = np.mean([r["o"] for r in S]), np.mean([r["n"] for r in S])
            L.append(f"| {root} | {g} | {len(S)} | {o:.0f} | {n:.0f} | {o - n:+.0f} |")
    L += ["", "## 组间\"最终最好\"配对差：重评前后（只描述）", "",
          "| 实验 | 比较 | 种子 | 原得分差 [95%]（为正） | 重评差 [95%]（为正） |", "|---|---|---|---|---|"]
    for root in roots:
        for x, y, desc in PAIRS.get(root, []):
            X = {r["seed"]: r for r in R if r["root"] == root and r["group"] == x}
            Y = {r["seed"]: r for r in R if r["root"] == root and r["group"] == y}
            ss = sorted(set(X) & set(Y))
            if len(ss) < 2:
                continue
            do = np.array([X[s]["o"] - Y[s]["o"] for s in ss])
            dn = np.array([X[s]["n"] - Y[s]["n"] for s in ss])
            (a0, a1), (b0, b1) = boot_ci(do), boot_ci(dn)
            L.append(f"| {root} | {desc} | {len(ss)} | {do.mean():+.0f} [{a0:+.0f}, {a1:+.0f}]（{int((do > 0).sum())} / {len(ss)}） | "
                     f"{dn.mean():+.0f} [{b0:+.0f}, {b1:+.0f}]（{int((dn > 0).sum())} / {len(ss)}） |")
    L += ["", "- 回归 = 原得分（2 回合，选出来的最大值）− 重评（10 回合，新种子）；正值 = 原得分被噪声抬高",
          "- 沿用：同一实验里设计与物理参数都相同的（例如配对组复制来的前 5 代里的最好设计）只评一次"]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--roots", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--threads", type=int, default=12)
    r.add_argument("--dry", action="store_true")
    q = sub.add_parser("report")
    q.add_argument("--out", required=True)
    a = p.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
    else:
        out = Path(a.out)
        (out / "report.md").write_text(report(out / "reeval.csv"), encoding="utf-8")
        print(f"写好了 {out / 'report.md'}")


if __name__ == "__main__":
    main()
