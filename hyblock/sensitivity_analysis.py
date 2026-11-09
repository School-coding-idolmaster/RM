"""第 14 周敏感性分析（claude/正式实验计划.md 变更记录第 16 条；结果出来之前写定，11-09）。只做描述，不做显著性检验。

问题：平地上两条结论在物理参数改变时方向是否一致——
  (a) LLM 变异提高种群整体质量：A4 − A1 的后期中位数 > 0
  (b) GA + LLM 变异 + ② 的整体质量高于 GA：A5 − A1 的后期中位数 > 0
  同时报告 A5 − A4（② 在 LLM 变异之上）的后期中位数，以及三组的最终最好、最好曲线均值。
条件：p14（压力 14 MPa）、m15（质量 ×1.5）、v370（速度上限 370 deg/s），种子 300–302；
参照：默认条件 = formal_flat 的 A1 / A4 / A5（种子 100–109，同一台 Mac）。
判定（描述性，事先定）：某条结论在某个条件下"方向一致" = 3 个种子的配对差均值 > 0；
  三个条件都一致 → "对这三项参数不敏感"；两个一致 → "大体一致"；一个或零个 → "受参数影响"。
用法：python -m hyblock.sensitivity_analysis --sen <mac_runs/sensitivity> --flat <mac_runs/formal_flat> --out <目录>
      自检：--conds 名=目录,... 覆盖条件目录（例如用 formal_flat 代替）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hyblock.formal_analysis import boot_ci, run_metrics

GROUPS = {"A1": "ga", "A4": "ga_llmtyped", "A5": "ga_llmtyped_sel_llmEv"}
DESC = {"A1": "普通 GA", "A4": "GA + LLM 变异", "A5": "GA + LLM 变异 + ②"}
CONDS = {"p14": "压力 14 MPa", "m15": "质量 ×1.5", "v370": "速度上限 370 deg/s"}
CLAIMS = [("A4", "A1", "med", "(a) LLM 变异 − GA，后期中位数"), ("A5", "A1", "med", "(b) GA + LLM 变异 + ② − GA，后期中位数"),
          ("A5", "A4", "med", "② 在 LLM 变异之上，后期中位数")]
OTHER = [("A4", "A1", "best", "LLM 变异 − GA，最终最好"), ("A5", "A1", "best", "GA + LLM 变异 + ② − GA，最终最好"),
         ("A4", "A1", "auc", "LLM 变异 − GA，最好曲线均值"), ("A5", "A1", "auc", "GA + LLM 变异 + ② − GA，最好曲线均值")]


def load(root: Path, seeds) -> tuple[dict, list]:
    D, missing = {}, []
    for g, name in GROUPS.items():
        for s in seeds:
            d = root / f"{name}_s{s}"
            if (d / "DONE").exists():
                D[(g, s)] = run_metrics(d)
            elif d.exists():
                missing.append(f"{g} 种子 {s}")
    return D, missing


def diffs(D, x, y, key, seeds):
    ss = [s for s in seeds if (x, s) in D and (y, s) in D]
    return ss, [D[(x, s)][key] - D[(y, s)][key] for s in ss]


def fmt(d):
    if not d:
        return "—", "—", "—"
    if len(d) >= 2:
        lo, hi = boot_ci(d)
        ci = f"[{lo:+.0f}, {hi:+.0f}]"
    else:
        ci = "—"
    return f"{np.mean(d):+.0f}", ci, f"{sum(v > 0 for v in d)} / {len(d)}"


def report(conds: dict, flat: Path | None, seeds, flat_seeds) -> str:
    data = {c: load(p, seeds) for c, p in conds.items()}
    ref = load(flat, flat_seeds) if flat else ({}, [])
    L = ["# 第 14 周敏感性分析（描述性）", "",
         "登记：`claude/正式实验计划.md` 变更记录第 16 条。脚本 `hyblock/sensitivity_analysis.py`（结果出来之前写定）。", ""]
    for c, (D, miss) in data.items():
        have = {g: sum((g, s) in D for s in seeds) for g in GROUPS}
        L.append(f"- {c}（{CONDS.get(c, c)}）完成：" + "，".join(f"{g} {n}" for g, n in have.items())
                 + (f"；未完成：{'、'.join(miss)}" if miss else ""))
    L += ["", "## 两条结论在各条件下的方向（后期中位数配对差；默认条件 = formal_flat 种子 100–109）", "",
          "| 比较 | 默认（参照） | " + " | ".join(f"{c} {CONDS.get(c, c)}" for c in conds) + " | 三个条件合计 | 判定 |",
          "|---|---|" + "---|" * len(conds) + "---|---|"]
    for x, y, key, desc in CLAIMS:
        _, dref = diffs(ref[0], x, y, key, flat_seeds)
        m, ci, pos = fmt(dref)
        cells, pooled, ok, n_full = [], [], 0, 0
        for c, (D, _) in data.items():
            ss, d = diffs(D, x, y, key, seeds)
            pooled += d
            mm, _, pp = fmt(d)
            cells.append(f"{mm}（{pp}；" + " ".join(f"{v:+.0f}" for v in d) + "）" if d else "—")
            if len(d) == len(seeds):
                n_full += 1
                ok += np.mean(d) > 0
        pm, pci, ppos = fmt(pooled)
        if x == "A5" and y == "A4":
            verdict = "（只描述）"
        elif n_full < len(conds):
            verdict = f"条件未全部完成（{n_full} / {len(conds)}）"
        else:
            verdict = {3: "对这三项参数不敏感", 2: "大体一致"}.get(ok, "受参数影响") + f"（{ok} / 3 个条件 > 0）"
        L.append(f"| {desc} | {m} {ci}（{pos}） | " + " | ".join(cells) + f" | {pm} {pci}（{ppos}） | {verdict} |")
    L += ["", "## 其他指标（描述）", "", "| 比较 | 默认（参照） | " + " | ".join(conds) + " | 三个条件合计 |",
          "|---|---|" + "---|" * len(conds) + "---|"]
    for x, y, key, desc in OTHER:
        _, dref = diffs(ref[0], x, y, key, flat_seeds)
        m, ci, pos = fmt(dref)
        cells, pooled = [], []
        for c, (D, _) in data.items():
            _, d = diffs(D, x, y, key, seeds)
            pooled += d
            mm, _, pp = fmt(d)
            cells.append(f"{mm}（{pp}）")
        pm, pci, ppos = fmt(pooled)
        L.append(f"| {desc} | {m} {ci}（{pos}） | " + " | ".join(cells) + f" | {pm} {pci}（{ppos}） |")
    L += ["", "## 各组（种子间均值）", "", "| 条件 | 组 | 种子 | 最终最好 | 后期中位数 | 最好曲线均值 | 后期 < 20 | MPC h | $ |",
          "|---|---|---|---|---|---|---|---|---|"]
    rows = [("默认（formal_flat）", ref[0], flat_seeds)] + [(f"{c} {CONDS.get(c, c)}", D, seeds) for c, (D, _) in data.items()]
    for name, D, ss_all in rows:
        for g in GROUPS:
            ss = [s for s in ss_all if (g, s) in D]
            if not ss:
                continue
            v = lambda k: [D[(g, s)][k] for s in ss]
            L.append(f"| {name} | {g} {DESC[g]} | {len(ss)} | {np.mean(v('best')):.0f} | {np.mean(v('med')):.0f} | "
                     f"{np.mean(v('auc')):.0f} | {np.mean(v('lt20')):.2f} | {np.mean(v('mpc_h')):.2f} | {sum(v('usd')):.2f} |")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sen", default=None, help="mac_runs/sensitivity（下面有 p14、m15、v370）")
    p.add_argument("--flat", default=None, help="mac_runs/formal_flat（默认条件的参照）")
    p.add_argument("--seeds", default="300-302")
    p.add_argument("--flat_seeds", default="100-109")
    p.add_argument("--conds", default=None, help="自检用：名=目录,... 覆盖条件目录")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rng = lambda t: list(range(int(t.split("-")[0]), int(t.split("-")[1]) + 1))
    if a.conds:
        conds = {k: Path(v) for k, v in (kv.split("=") for kv in a.conds.split(","))}
    else:
        conds = {c: Path(a.sen) / c for c in CONDS}
    txt = report(conds, Path(a.flat) if a.flat else None, rng(a.seeds), rng(a.flat_seeds))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
