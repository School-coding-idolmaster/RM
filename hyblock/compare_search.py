"""比较 hyblock/search.py 的多次运行（第 8 周起）：曲线、终点、无效率、重复率、多样性、用时、LLM 成本。

用法（Claude 的虚拟机或 WSL，需要 matplotlib 才出图）：
    python -m hyblock.compare_search ../from_mac/mac_runs/base_flat8 --out ../from_mac/mac_runs/base_flat8/compare
输入目录下每个 <方法>_s<种子>/evals.csv 是一次运行（没有 DONE 的也读，标出未完成）。
输出：report.md、curves.csv（每种方法每个评估次数的均值 / 最小 / 最大）、curves.svg / curves.png。
统计：种子少时只报告均值、各种子的值和 bootstrap 区间，不做显著性检验（正式实验按 pipeline 用 IQM + 分层 bootstrap）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}   # USD / MTok（10-28 官网）
EXAMPLE_PREFIX = "h0=root; m1=h0.zp@90 q60/120 mirror"
WALK = 20.0
CHECKPOINTS = (25, 50, 100, 150, 200, 300, 400)


def load(run: Path) -> dict:
    with open(run / "evals.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    m = re.match(r"(.+)_s(\d+)$", run.name)
    meta = json.loads((run / "run_meta.json").read_text(encoding="utf-8")) if (run / "run_meta.json").exists() else {}
    used = [r for r in rows if r["status"] in ("mpc", "duplicate")]
    curve = np.maximum.accumulate([float(r["fitness"]) for r in used]) if used else np.array([])
    return {"name": run.name, "method": m.group(1) if m else run.name, "seed": int(m.group(2)) if m else 0,
            "rows": rows, "used": used, "mpc": [r for r in rows if r["status"] == "mpc"], "curve": curve,
            "done": (run / "DONE").exists(), "budget": (meta.get("args") or {}).get("budget"),
            "model": (meta.get("args") or {}).get("model")}


def boot_ci(x, n=5000, seed=0):
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return tuple(np.percentile(m, [2.5, 97.5]))


def run_stats(r: dict) -> dict:
    rows, mpc, used = r["rows"], r["mpc"], r["used"]
    f = np.array([float(x["fitness"]) for x in mpc]) if mpc else np.array([0.0])
    non_dup = [x for x in rows if x["status"] != "duplicate"]
    kinds = Counter(f"{x['status']}:{x['kind']}" for x in rows if x["status"] in ("invalid", "screened", "no_design"))
    st = {"evals": len(used), "unique": len(mpc), "dup_rate": 1 - len(mpc) / max(1, len(used)),
          "invalid_rate": sum(x["status"] in ("invalid", "no_design") for x in non_dup) / max(1, len(non_dup)),
          "screened_rate": sum(x["status"] == "screened" for x in non_dup) / max(1, len(non_dup)),
          "kinds": kinds, "best": float(r["curve"][-1]) if len(r["curve"]) else float("nan"),
          "median": float(np.median(f)), "walk": float(np.mean(f >= WALK)),
          "top10": float(np.mean(np.sort(f)[-10:])),
          "mpc_h": sum(float(x["mpc_s"] or 0) for x in mpc) / 3600,
          "s_per_design": sum(float(x["mpc_s"] or 0) for x in mpc) / max(1, len(mpc)),
          "modules": float(np.mean([int(x["modules"]) for x in mpc])) if mpc else float("nan"),
          "joints": float(np.mean([int(x["joints"]) for x in mpc])) if mpc else float("nan"),
          "hub_root": float(np.mean([x["root"] == "hub" for x in mpc])) if mpc else float("nan"),
          "copies_example": float(np.mean([x["design"].startswith(EXAMPLE_PREFIX) for x in mpc])) if mpc else 0.0,
          "fast": sum(int(x["fast_episodes"] or 0) for x in mpc)}
    for k in CHECKPOINTS:
        st[f"best@{k}"] = float(r["curve"][k - 1]) if len(r["curve"]) >= k else float("nan")
    calls = [x for x in rows if x["source"] == "llm"]
    if calls:
        tin = sum(int(x["llm_in"] or 0) for x in calls)
        tout = sum(int(x["llm_out"] or 0) for x in calls)
        pin, pout = PRICE.get(r["model"] or "claude-haiku-5-5", PRICE["claude-haiku-5-5"])
        st.update(llm_calls=len(calls), llm_usd=tin / 1e6 * pin + tout / 1e6 * pout,
                  llm_s=float(np.mean([float(x["llm_s"] or 0) for x in calls])))
    return st


def fmt(x, nd=1):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", help="directory containing <method>_s<seed>/ runs")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "compare"
    out.mkdir(parents=True, exist_ok=True)
    runs = [load(d) for d in sorted(root.iterdir()) if (d / "evals.csv").exists()]
    by = defaultdict(list)
    for r in runs:
        r["stats"] = run_stats(r)
        by[r["method"]].append(r)
    order = [m for m in ("random", "ga", "llm") if m in by] + [m for m in by if m not in ("random", "ga", "llm")]

    L = [f"# 搜索对比：{root.name}", "", f"运行 {len(runs)} 个：" + "，".join(
        f"{r['name']}{'' if r['done'] else '（未完成）'}" for r in runs), ""]
    L += ["## 每个运行", "", "| 运行 | 评估 | 不同设计 | 重复率 | 无效率 | ① 淘汰率 | 最好 | @50 | @100 | @200 | "
          "中位数 | 能走 | 前10均值 | 模块均值 | 照抄示例 | MPC h | s/设计 |",
          "|" + "---|" * 17]
    for m in order:
        for r in sorted(by[m], key=lambda r: r["seed"]):
            s = r["stats"]
            L.append(f"| {r['name']} | {s['evals']} | {s['unique']} | {s['dup_rate']:.2f} | {s['invalid_rate']:.2f} | "
                     f"{s['screened_rate']:.2f} | {fmt(s['best'])} | {fmt(s['best@50'])} | {fmt(s['best@100'])} | "
                     f"{fmt(s['best@200'])} | {fmt(s['median'])} | {s['walk']:.2f} | {fmt(s['top10'])} | "
                     f"{fmt(s['modules'])} | {s['copies_example']:.2f} | {s['mpc_h']:.2f} | {s['s_per_design']:.1f} |")
    L += ["", "## 按方法（种子间均值，括号 = bootstrap 95% 区间；种子少时区间很宽，只作参考）", "",
          "| 方法 | 种子 | 最好 | @50 | @100 | 中位数 | 能走 | 前10均值 | 不同设计 | 无效率 | 重复率 | MPC h |",
          "|" + "---|" * 12]
    for m in order:
        S = [r["stats"] for r in by[m]]
        def agg(k):
            v = [s[k] for s in S if not np.isnan(s[k])]
            if not v:
                return "—"
            lo, hi = boot_ci(v)
            return f"{np.mean(v):.1f}" + (f"（{lo:.0f}–{hi:.0f}）" if len(v) > 1 else "")
        L.append(f"| {m} | {len(S)} | {agg('best')} | {agg('best@50')} | {agg('best@100')} | {agg('median')} | "
                 f"{np.mean([s['walk'] for s in S]):.2f} | {agg('top10')} | {np.mean([s['unique'] for s in S]):.0f} | "
                 f"{np.mean([s['invalid_rate'] for s in S]):.2f} | {np.mean([s['dup_rate'] for s in S]):.2f} | "
                 f"{sum(s['mpc_h'] for s in S):.1f} |")
    L += ["", "## 无效与淘汰原因（各方法合计）", ""]
    for m in order:
        k = sum((r["stats"]["kinds"] for r in by[m]), Counter())
        L.append(f"- {m}：" + ("，".join(f"{a} {b}" for a, b in k.most_common()) or "无"))
    llm = [r for r in runs if "llm_calls" in r["stats"]]
    if llm:
        L += ["", "## LLM", ""]
        for r in llm:
            s = r["stats"]
            L.append(f"- {r['name']}：调用 {s['llm_calls']} 次，约 ${s['llm_usd']:.2f}，平均 {s['llm_s']:.1f} s/次；"
                     f"评估过的设计里照抄示例开头的比例 {s['copies_example']:.2f}")
        L.append(f"- 合计约 ${sum(r['stats']['llm_usd'] for r in llm):.2f}")
    L += ["", "## 各方法最好的设计", ""]
    for m in order:
        best = max((x for r in by[m] for x in r["mpc"]), key=lambda x: float(x["fitness"]), default=None)
        if best:
            L.append(f"- {m}：{float(best['fitness']):.1f}（{best['source']}，模块 {best['modules']}）`{best['design']}`")
    fast = sum(r["stats"]["fast"] for r in runs)
    L += ["", f"峰值速度 > 6 m/s 的回合（全部运行）：{fast}"]

    # curves
    maxn = max((len(r["curve"]) for r in runs), default=0)
    with open(out / "curves.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "n_eval", "mean", "min", "max", "n_runs"])
        for m in order:
            for n in range(1, maxn + 1):
                v = [r["curve"][n - 1] for r in by[m] if len(r["curve"]) >= n]
                if v:
                    w.writerow([m, n, round(float(np.mean(v)), 2), round(float(min(v)), 2), round(float(max(v)), 2),
                                len(v)])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = {"random": "#7f7f7f", "ga": "#1f77b4", "llm": "#d62728"}
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for m in order:
            for r in by[m]:
                ax.plot(np.arange(1, len(r["curve"]) + 1), r["curve"], color=colors.get(m), alpha=0.25, lw=1)
            n = min(len(r["curve"]) for r in by[m])
            if n:
                mean = np.mean([r["curve"][:n] for r in by[m]], axis=0)
                ax.plot(np.arange(1, n + 1), mean, color=colors.get(m), lw=2.2, label=f"{m} (n={len(by[m])})")
        ax.set_xlabel("evaluations (MPC budget)")
        ax.set_ylabel("best score so far")
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
        ax.set_title(f"Hyblock flat, {root.name}")
        fig.tight_layout()
        fig.savefig(out / "curves.svg")
        fig.savefig(out / "curves.png", dpi=150)
        L += ["", "![curves](curves.png)"]
    except ImportError:
        L += ["", "（没有 matplotlib，未出图；数据在 curves.csv）"]
    text = "\n".join(L) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
