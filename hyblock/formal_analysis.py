"""正式实验（平地）的预先登记分析（claude/正式实验计划.md 第 3 节；第 12 周写，结果出来之前）。

用法：python -m hyblock.formal_analysis <mac_runs/formal_flat> [--out <目录>] [--seeds 100-109]
只用有 DONE 的运行；配对比较只用两组都完成的种子（缺的列出来）。不在这里改方法或加分析口径：
指标定义与 base_flat8/compare_week11b/analysis_2x2.py 相同。

  H1（唯一的确认性检验）：A2 − A1 的后期中位数，配对 t（双侧 α = 0.05），同时报 Wilcoxon 与配对 bootstrap 95%
  次要：效应大小 + bootstrap 95%（不做显著性主张）；描述性：其余
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

import numpy as np

GROUPS = {   # 代号: (目录名, 说明)
    "A1": ("ga", "普通 GA"),
    "A2": ("ga_sel_llmEv", "GA + ② LLM Ev"),
    "A3": ("ga_sel_knnEv", "GA + ② knn Ev"),
    "A4": ("ga_llmtyped", "GA + LLM 变异"),
    "A5": ("ga_llmtyped_sel_llmEv", "GA + LLM 变异 + ② LLM Ev（本方案）"),
    "A6": ("ga_sel_llmEv_sonnet", "GA + ② LLM Ev（Sonnet）"),
    "R1": ("random", "随机搜索"),
    "R2": ("llm", "RM 式 LLM"),
}
LATE_GEN = 5
B = 10000                                   # bootstrap 次数
PRICE = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.0, 10.0)}    # $/百万 token（10-31 官网）


# ---------------------------------------------------------------- 每个运行的指标

def run_metrics(d: Path) -> dict:
    rows = list(csv.DictReader(open(d / "evals.csv", encoding="utf-8")))
    mpc = sorted((r for r in rows if r["status"] == "mpc"), key=lambda r: (int(r["gen"]), int(r["idx"])))
    f = [float(r["fitness"]) for r in mpc]
    late = [float(r["fitness"]) for r in mpc if int(r["gen"]) >= LATE_GEN]
    best_curve = np.maximum.accumulate(f)

    def reach(t):
        i = np.nonzero(best_curve >= t)[0]
        return int(i[0]) + 1 if len(i) else None

    m = dict(n_mpc=len(f), best=max(f), auc=float(np.mean(best_curve)), med=st.median(late),
             ge400=sum(v >= 400 for v in late) / len(late), lt20=sum(v < 20 for v in late) / len(late),
             r441=reach(441), r600=reach(600),
             mpc_h=None)
    meta0 = json.loads((d / "run_meta.json").read_text(encoding="utf-8")) if (d / "run_meta.json").exists() else {}
    a0 = meta0.get("args") or {}
    g_own = int(a0.get("sel_from", 1)) if a0.get("init_from") else 0     # 复制来的前几代不算本运行的 MPC 时间
    m["mpc_h"] = sum(float(r["mpc_s"] or 0) for r in mpc if int(r["gen"]) >= g_own) / 3600
    m["mpc_h_note"] = f"第 {g_own} 代起" if g_own else "全部"
    # ② 随机名额上的 Spearman（预测 vs 真实；只看第 5 代起有 sel_pred 的随机名额，按代合并）
    rnd = [(float(r["sel_pred"]), float(r["fitness"])) for r in mpc
           if r.get("sel_slot") == "rand" and r.get("sel_pred") not in ("", None)]
    if len(rnd) >= 5:
        from scipy.stats import spearmanr
        m["rand_rho"] = float(spearmanr(*zip(*rnd))[0])
    # token（evals.csv 里 RM 式 / ② 的调用；LLM 变异在 llm_mut.csv）
    tin = sum(float(r["llm_in"] or 0) for r in rows)
    tout = sum(float(r["llm_out"] or 0) for r in rows)
    lm = d / "llm_mut.csv"
    if lm.exists():
        mr = list(csv.DictReader(open(lm, encoding="utf-8")))
        fresh = [r for r in mr if r.get("cached") != "True"]
        tin += sum(float(r["llm_in"] or 0) for r in fresh)
        tout += sum(float(r["llm_out"] or 0) for r in fresh)
        req = {}
        for r in mr:
            req.setdefault((r["gen"], r["idx"]), []).append(r["outcome"])
        m["mut_ok_first"] = sum(v[0] == "ok" for v in req.values()) / max(1, len(req))
        m["mut_ok_any"] = sum("ok" in v for v in req.values()) / max(1, len(req))
        m["mut_requests"] = len(req)
    sel = d / "select.csv"
    if sel.exists():
        sr = [r for r in csv.DictReader(open(sel, encoding="utf-8")) if r.get("cached") != "True"]
        tin += sum(float(r.get("llm_in") or 0) for r in sr)
        tout += sum(float(r.get("llm_out") or 0) for r in sr)
    meta = json.loads((d / "run_meta.json").read_text(encoding="utf-8")) if (d / "run_meta.json").exists() else {}
    model = (meta.get("args") or {}).get("model", "claude-haiku-5-5")
    pi, po = PRICE.get(model, (math.nan, math.nan))
    m.update(tok_in=tin, tok_out=tout, usd=(tin * pi + tout * po) / 1e6)
    return m


def load(root: Path, seeds) -> tuple[dict, list]:
    D, missing = {}, []
    for g, (name, _) in GROUPS.items():
        for s in seeds:
            d = root / f"{name}_s{s}"
            if (d / "DONE").exists():
                D[(g, s)] = run_metrics(d)
            elif d.exists():
                missing.append(f"{g} 种子 {s}（没有 DONE）")
    return D, missing


# ---------------------------------------------------------------- 统计

def boot_ci(x, stat=np.mean, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    v = np.array([stat(x[rng.integers(0, len(x), len(x))]) for _ in range(B)])
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def iqm(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    lo, hi = int(math.floor(n * 0.25)), int(math.ceil(n * 0.75))
    return float(np.mean(x[lo:hi])) if hi > lo else float(np.mean(x))


def clopper_pearson(k, n, a=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return lo, hi


def paired(D, x, y, key, seeds):
    s_ok = [s for s in seeds if (x, s) in D and (y, s) in D]
    d = [D[(x, s)][key] - D[(y, s)][key] for s in s_ok]
    return s_ok, d


def fmt(v, nd=0):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:+.{nd}f}" if nd >= 0 else str(v)


# ---------------------------------------------------------------- 报告

def report(root: Path, seeds) -> str:
    from scipy.stats import ttest_rel, wilcoxon
    D, missing = load(root, seeds)
    L = [f"# 正式实验（平地）分析：{root.name}", "",
         "预先登记：`claude/正式实验计划.md` 第 3 节。脚本 `hyblock/formal_analysis.py`（结果出来之前写定）。", ""]
    have = {g: [s for s in seeds if (g, s) in D] for g in GROUPS}
    L.append("完成的运行：" + "，".join(f"{g} {len(v)}" for g, v in have.items()))
    if missing:
        L.append("未完成：" + "；".join(missing))
    L.append("")

    # ---- H1
    s_ok, d = paired(D, "A2", "A1", "med", seeds)
    L += ["## H1：② LLM Ev 提高后期质量（A2 − A1，后期中位数）", ""]
    if len(d) >= 3:
        t = ttest_rel([D[("A2", s)]["med"] for s in s_ok], [D[("A1", s)]["med"] for s in s_ok])
        try:
            w = wilcoxon(d)
            wp = f"{w.pvalue:.3f}"
        except ValueError:
            wp = "—"
        lo, hi = boot_ci(d)
        L += [f"- 种子 {len(s_ok)} 个：配对差均值 **{np.mean(d):+.1f}**，SD {np.std(d, ddof=1):.1f}，"
              f"bootstrap 95% [{lo:+.1f}, {hi:+.1f}]",
              f"- 配对 t：t = {t.statistic:.2f}，p = {t.pvalue:.4f}（双侧）；Wilcoxon 符号秩 p = {wp}；"
              f"为正的种子 {sum(v > 0 for v in d)} / {len(d)}",
              f"- 各种子：" + " ".join(f"{s}:{v:+.0f}" for s, v in zip(s_ok, d)),
              f"- **判定**：{'H1 成立（p < 0.05 且差 > 0）' if t.pvalue < 0.05 and np.mean(d) > 0 else 'H1 不成立'}"
              + ("" if len(s_ok) == len(seeds) else f"（**只有 {len(s_ok)} / {len(seeds)} 个种子，还不是最终判定**）"), ""]
    else:
        L += ["- 完成的配对种子不足 3 个，暂不计算", ""]

    # ---- 次要
    L += ["## 次要（效应大小 + 配对 bootstrap 95%，不做显著性主张）", "",
          "| 比较 | 指标 | 种子 | 配对差均值 | 95% 区间 | 为正 | 各种子 |", "|---|---|---|---|---|---|---|"]
    sec = [("② 的最终最好（A2 − A1）", "A2", "A1", "best"),
           ("② 在 LLM 变异下（A5 − A4）", "A5", "A4", "med"), ("", "A5", "A4", "best"),
           ("LLM 与 knn（A2 − A3）", "A2", "A3", "med"), ("", "A2", "A3", "best"), ("", "A2", "A3", "lt20"),
           ("换 Sonnet（A6 − A2）", "A6", "A2", "med"), ("", "A6", "A2", "best"),
           ("LLM 变异（A4 − A1）", "A4", "A1", "best"), ("", "A4", "A1", "auc"), ("", "A4", "A1", "med")]
    for name, x, y, k in sec:
        s_ok, d = paired(D, x, y, k, seeds)
        if len(d) < 2:
            L.append(f"| {name} | {k} | {len(d)} | — | — | — | — |")
            continue
        nd = 2 if k == "lt20" else 0
        lo, hi = boot_ci(d)
        L.append(f"| {name} | {k} | {len(d)} | {np.mean(d):+.{nd}f} | [{lo:+.{nd}f}, {hi:+.{nd}f}] | "
                 f"{sum(v > 0 for v in d)} / {len(d)} | {' '.join(f'{v:+.{nd}f}' for v in d)} |")
    # 交互作用
    s_ok = [s for s in seeds if all((g, s) in D for g in ("A1", "A2", "A4", "A5"))]
    if len(s_ok) >= 2:
        for k in ("best", "med"):
            it = [(D[("A5", s)][k] - D[("A4", s)][k]) - (D[("A2", s)][k] - D[("A1", s)][k]) for s in s_ok]
            lo, hi = boot_ci(it)
            L.append(f"| 交互作用 (A5 − A4) − (A2 − A1) | {k} | {len(it)} | {np.mean(it):+.0f} | [{lo:+.0f}, {hi:+.0f}] | "
                     f"{sum(v > 0 for v in it)} / {len(it)} | {' '.join(f'{v:+.0f}' for v in it)} |")
    L.append("")

    # LLM 变异：达到 600 的种子比例、IQM
    L += ["### LLM 变异 vs GA（A4 vs A1）", "", "| 组 | 种子 | 达到 600 的种子 | 95%（精确二项） | 最终最好 IQM | 均值 | 最好曲线均值 |",
          "|---|---|---|---|---|---|---|"]
    for g in ("A1", "A4", "A2", "A5"):
        ss = have[g]
        if not ss:
            continue
        k = sum(D[(g, s)]["best"] >= 600 for s in ss)
        lo, hi = clopper_pearson(k, len(ss))
        b = [D[(g, s)]["best"] for s in ss]
        L.append(f"| {g} {GROUPS[g][1]} | {len(ss)} | {k} / {len(ss)} | [{lo:.2f}, {hi:.2f}] | {iqm(b):.0f} | {np.mean(b):.0f} | "
                 f"{np.mean([D[(g, s)]['auc'] for s in ss]):.0f} |")
    L.append("")

    # ---- 描述性：每组
    L += ["## 各组（种子间均值；括号 = 种子间 bootstrap 95%）", "",
          "| 组 | 种子 | 最终最好 | 后期中位数 | 最好曲线均值 | 后期 ≥ 400 | 后期 < 20 | 到 441（中位） | 到 600（达到/中位） | 随机名额 rho | 本运行 MPC h（配对组不含复制的前 5 代） | $ |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, (name, desc) in GROUPS.items():
        ss = have[g]
        if not ss:
            continue
        v = lambda k: [D[(g, s)][k] for s in ss]
        def mci(k, nd=0):
            x = v(k)
            if len(x) < 2:
                return f"{np.mean(x):.{nd}f}"
            lo, hi = boot_ci(x)
            return f"{np.mean(x):.{nd}f}（{lo:.{nd}f}–{hi:.{nd}f}）"
        r441 = [x for x in v("r441") if x is not None]
        r600 = [x for x in v("r600") if x is not None]
        rho = [D[(g, s)].get("rand_rho") for s in ss if D[(g, s)].get("rand_rho") is not None]
        L.append(f"| {g} {desc} | {len(ss)} | {mci('best')} | {mci('med')} | {mci('auc')} | {np.mean(v('ge400')):.2f} | "
                 f"{np.mean(v('lt20')):.2f} | {len(r441)}/{len(ss)}，{st.median(r441) if r441 else '—'} | "
                 f"{len(r600)}/{len(ss)}，{st.median(r600) if r600 else '—'} | {np.mean(rho):.2f} | " if rho else
                 f"| {g} {desc} | {len(ss)} | {mci('best')} | {mci('med')} | {mci('auc')} | {np.mean(v('ge400')):.2f} | "
                 f"{np.mean(v('lt20')):.2f} | {len(r441)}/{len(ss)}，{st.median(r441) if r441 else '—'} | "
                 f"{len(r600)}/{len(ss)}，{st.median(r600) if r600 else '—'} | — | ")
        L[-1] += f"{np.mean(v('mpc_h')):.2f} | {sum(v('usd')):.2f} |"
    L.append("")
    mut = [(g, s) for g in ("A4", "A5") for s in have[g] if "mut_ok_first" in D[(g, s)]]
    if mut:
        L.append("LLM 变异：一次通过 " + "，".join(
            f"{g} {np.mean([D[(g, s)]['mut_ok_first'] for s in have[g]]):.2f}" for g in ("A4", "A5") if have[g])
                 + "；重问后通过 " + "，".join(
            f"{g} {np.mean([D[(g, s)]['mut_ok_any'] for s in have[g]]):.2f}" for g in ("A4", "A5") if have[g]))
        L.append("")

    # ---- 每个运行
    L += ["## 每个运行", "", "| 组 | 种子 | 最终最好 | 后期中位数 | 最好曲线均值 | ≥ 400 | < 20 | 到 441 | 到 600 | MPC h |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for g in GROUPS:
        for s in have[g]:
            x = D[(g, s)]
            L.append(f"| {g} | {s} | {x['best']:.0f} | {x['med']:.0f} | {x['auc']:.0f} | {x['ge400']:.2f} | {x['lt20']:.2f} | "
                     f"{x['r441'] or '—'} | {x['r600'] or '—'} | {x['mpc_h']:.2f} |")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--out", default=None)
    p.add_argument("--seeds", default="100-109", help="例如 100-109 或 0-2")
    p.add_argument("--groups", default=None, help="自检用：代号=目录名,... 覆盖默认目录名")
    a = p.parse_args(argv)
    lo, hi = (int(v) for v in a.seeds.split("-"))
    if a.groups:
        for kv in a.groups.split(","):
            g, n = kv.split("=")
            GROUPS[g] = (n, GROUPS[g][1])
    root = Path(a.root)
    txt = report(root, list(range(lo, hi + 1)))
    out = Path(a.out) if a.out else root / "formal_analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
