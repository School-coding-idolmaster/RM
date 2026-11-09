"""第 13 周的预先登记分析（claude/正式实验计划.md 变更记录第 10、12、13 条；结果出来之前写定，11-04）。

  H3（平地复现，种子 200–209）：A5 − A4 的最终最好 > 0           mac_runs/replicate_flat/
  H2（横梁，种子 100–109）：B4 − B3 的后期中位数 > 0               mac_runs/formal_beams/
  H4（横梁，种子 100–109）：B6 − B5 的最终最好 > 0
  H5（横梁，种子 100–109）：B6 − B1 的后期中位数 > 0（变更记录第 15 条，11-05 追加，B5 / B6 有数据之前）
  每个：10 个种子配对 t（双侧 α = 0.05）+ Wilcoxon + 配对 bootstrap 95%；四个 p 值另做 Holm 校正（修论以校正后为准）。
  次要（效应大小 + bootstrap 95%，不做显著性主张）：见第 12 条。指标定义与 hyblock/formal_analysis.py 相同。

用法：python -m hyblock.followup_analysis --rep <replicate_flat> --beams <formal_beams> [--out 目录]
      自检：--rep_seeds / --beams_seeds、--beams_groups B1=ga,... 覆盖目录名（例如用开发数据）
只用有 DONE 的运行；配对只用两组都完成的种子（缺的列出来，判定标注"还不是最终"）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hyblock.formal_analysis import boot_ci, run_metrics

REP = {"A4": "ga_llmtyped", "A5": "ga_llmtyped_sel_llmEv"}
BEAMS = {"B1": "ga", "B2": "ga_sel_llmEv", "B3": "ga_init_transfer", "B4": "ga_init_llmtransfer",
         "B5": "ga_llmtyped", "B6": "ga_llmtyped_sel_llmEv"}
DESC = {"A4": "GA + LLM 变异", "A5": "本方案（LLM 变异 + ②）", "B1": "普通 GA", "B2": "GA + ② LLM Ev",
        "B3": "平地迁移", "B4": "LLM 改写后迁移", "B5": "GA + 地形版 LLM 变异", "B6": "B5 + ② LLM Ev"}
CONFIRM = [("H3", "rep", "A5", "A4", "best", "平地复现：本方案 − LLM 变异，最终最好"),
           ("H2", "beams", "B4", "B3", "med", "横梁：LLM 改写后迁移 − 平地迁移，后期中位数"),
           ("H4", "beams", "B6", "B5", "best", "横梁：组合效应（B6 − B5），最终最好"),
           ("H5", "beams", "B6", "B1", "med", "横梁：GA + LLM 变异 + ② − 普通 GA，后期中位数")]
SECONDARY = [("beams", "B3", "B1", "auc", "迁移效果：最好曲线均值"), ("beams", "B3", "B1", "best", "迁移效果：最终最好"),
             ("beams", "B3", "B1", "g0", "迁移效果：第 0 代中位数"),
             ("beams", "B4", "B3", "best", "LLM 改写后迁移：最终最好"), ("beams", "B4", "B3", "auc", "LLM 改写后迁移：最好曲线均值"),
             ("beams", "B2", "B1", "med", "② 在横梁上：后期中位数"), ("beams", "B2", "B1", "best", "② 在横梁上：最终最好"),
             ("beams", "B5", "B1", "med", "地形版 LLM 变异：后期中位数"), ("beams", "B5", "B1", "best", "地形版 LLM 变异：最终最好"),
             ("rep", "A5", "A4", "med", "平地复现：本方案 − LLM 变异，后期中位数")]


def load(root: Path, groups: dict, seeds) -> tuple[dict, list]:
    import csv
    D, missing = {}, []
    for g, name in groups.items():
        for s in seeds:
            d = root / f"{name}_s{s}"
            if (d / "DONE").exists():
                m = run_metrics(d)
                rows = [r for r in csv.DictReader(open(d / "evals.csv", encoding="utf-8"))
                        if r["status"] == "mpc" and r["gen"] == "0"]
                m["g0"] = float(np.median([float(r["fitness"]) for r in rows])) if rows else float("nan")
                ip = d / "init.csv"                          # --init llm / llm_transfer：第 0 代的 LLM 调用（Haiku 价格）
                if ip.exists():
                    seen = set()
                    for r in csv.DictReader(open(ip, encoding="utf-8")):
                        key = (r.get("call"), r.get("attempt"))
                        if r.get("call") == "final" or r.get("cached") == "True" or key in seen:
                            continue
                        seen.add(key)                       # llm_init 一次调用对应多行（每个设计一行），只算一次
                        m["usd"] += (float(r.get("llm_in") or 0) * 0.10 + float(r.get("llm_out") or 0) * 0.50) / 1e6
                D[(g, s)] = m
            elif d.exists():
                missing.append(f"{g} 种子 {s}")
    return D, missing


def paired(D, x, y, key, seeds):
    ss = [s for s in seeds if (x, s) in D and (y, s) in D]
    return ss, [D[(x, s)][key] - D[(y, s)][key] for s in ss]


def holm(ps: dict) -> dict:
    items = sorted((p, k) for k, p in ps.items() if p == p)
    m, out, run = len(items), {}, 0.0
    for i, (p, k) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = run
    return out


def report(rep_root, beams_root, rep_seeds, beams_seeds, beams_groups) -> str:
    from scipy.stats import ttest_rel, wilcoxon
    data = {}
    data["rep"] = load(rep_root, REP, rep_seeds) if rep_root else ({}, [])
    data["beams"] = load(beams_root, beams_groups, beams_seeds) if beams_root else ({}, [])
    seeds = {"rep": list(rep_seeds), "beams": list(beams_seeds)}
    L = ["# 第 13 周预先登记分析：平地复现（H3）与横梁正式实验（H2、H4、H5）", "",
         "登记：`claude/正式实验计划.md` 变更记录第 10、12、13、15 条（H5 于 11-05 追加）。脚本 `hyblock/followup_analysis.py`（结果出来之前写定）。", ""]
    for part, (D, miss) in data.items():
        groups = REP if part == "rep" else beams_groups
        have = {g: sum((g, s) in D for s in seeds[part]) for g in groups}
        L.append(f"{'平地复现' if part == 'rep' else '横梁'}完成的运行：" + "，".join(f"{g} {n}" for g, n in have.items())
                 + (f"；未完成：{'、'.join(miss)}" if miss else ""))
    L += ["", "## 确认性假设", "",
          "| 假设 | 比较 | 种子 | 配对差均值 | SD | bootstrap 95% | 配对 t | p | Wilcoxon p | 为正 | Holm p | 判定 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    ps, rows = {}, {}
    for h, part, x, y, key, desc in CONFIRM:
        D, _ = data[part]
        ss, d = paired(D, x, y, key, seeds[part])
        if len(d) < 3:
            rows[h] = f"| {h} | {desc} | {len(d)} | — | — | — | — | — | — | — | — | 种子不足 |"
            continue
        t = ttest_rel([D[(x, s)][key] for s in ss], [D[(y, s)][key] for s in ss])
        try:
            wp = wilcoxon(d).pvalue
        except ValueError:
            wp = float("nan")
        lo, hi = boot_ci(d)
        ps[h] = float(t.pvalue)
        rows[h] = (h, desc, ss, d, t, wp, lo, hi)
    hp = holm(ps)
    for h, part, x, y, key, desc in CONFIRM:
        r = rows[h]
        if isinstance(r, str):
            L.append(r)
            continue
        _, desc, ss, d, t, wp, lo, hi = r
        full = len(ss) == len(seeds[part])
        ok_raw = t.pvalue < 0.05 and np.mean(d) > 0
        ok_holm = hp[h] < 0.05 and np.mean(d) > 0
        verdict = ("成立（Holm 后）" if ok_holm else ("未校正成立、Holm 后不成立" if ok_raw else "不成立"))
        if not full:
            verdict += f"（只有 {len(ss)} / {len(seeds[part])} 个种子，还不是最终）"
        L.append(f"| {h} | {desc} | {len(ss)} | {np.mean(d):+.1f} | {np.std(d, ddof=1):.1f} | [{lo:+.1f}, {hi:+.1f}] | "
                 f"{t.statistic:.2f} | {t.pvalue:.4f} | {wp:.3f} | {sum(v > 0 for v in d)} / {len(d)} | {hp[h]:.4f} | **{verdict}** |")
    L.append("")
    for h, part, x, y, key, desc in CONFIRM:
        r = rows[h]
        if not isinstance(r, str):
            L.append(f"- {h} 各种子：" + " ".join(f"{s}:{v:+.0f}" for s, v in zip(r[2], r[3])))
    L += ["", "## 次要比较（效应大小 + 配对 bootstrap 95%，不做显著性主张）", "",
          "| 比较 | 种子 | 配对差均值 | 95% | 为正 | 各种子 |", "|---|---|---|---|---|---|"]
    for part, x, y, key, desc in SECONDARY:
        D, _ = data[part]
        ss, d = paired(D, x, y, key, seeds[part])
        if len(d) < 2:
            L.append(f"| {desc}（{x} − {y}） | {len(d)} | — | — | — | — |")
            continue
        lo, hi = boot_ci(d)
        L.append(f"| {desc}（{x} − {y}） | {len(d)} | {np.mean(d):+.0f} | [{lo:+.0f}, {hi:+.0f}] | {sum(v > 0 for v in d)} / {len(d)} | "
                 + " ".join(f"{v:+.0f}" for v in d) + " |")
    D, _ = data["beams"]
    ss = [s for s in seeds["beams"] if all((g, s) in D for g in ("B1", "B2", "B5", "B6"))]
    if len(ss) >= 2:
        it = [(D[("B6", s)]["best"] - D[("B5", s)]["best"]) - (D[("B2", s)]["best"] - D[("B1", s)]["best"]) for s in ss]
        lo, hi = boot_ci(it)
        L.append(f"| 横梁交互作用 (B6 − B5) − (B2 − B1)，最终最好 | {len(it)} | {np.mean(it):+.0f} | [{lo:+.0f}, {hi:+.0f}] | "
                 f"{sum(v > 0 for v in it) } / {len(it)} | " + " ".join(f"{v:+.0f}" for v in it) + " |")
    L += ["", "## 各组（种子间均值；括号 = bootstrap 95%）", "",
          "| 组 | 种子 | 最终最好 | 后期中位数 | 最好曲线均值 | 第 0 代中位数 | 后期 < 20 | 随机名额 rho | MPC h | $ |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for part, groups in (("rep", REP), ("beams", beams_groups)):
        D, _ = data[part]
        for g in groups:
            ss = [s for s in seeds[part] if (g, s) in D]
            if not ss:
                continue
            v = lambda k: [D[(g, s)][k] for s in ss]

            def mci(k, nd=0):
                x = v(k)
                if len(x) < 2:
                    return f"{np.mean(x):.{nd}f}"
                lo, hi = boot_ci(x)
                return f"{np.mean(x):.{nd}f}（{lo:.{nd}f}–{hi:.{nd}f}）"
            rho = [D[(g, s)].get("rand_rho") for s in ss if D[(g, s)].get("rand_rho") is not None]
            L.append(f"| {g} {DESC[g]} | {len(ss)} | {mci('best')} | {mci('med')} | {mci('auc')} | {np.nanmean(v('g0')):.0f} | "
                     f"{np.mean(v('lt20')):.2f} | {(f'{np.mean(rho):.2f}' if rho else '—')} | {np.mean(v('mpc_h')):.2f} | {sum(v('usd')):.2f} |")
    L.append("")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rep", default=None)
    p.add_argument("--beams", default=None)
    p.add_argument("--rep_seeds", default="200-209")
    p.add_argument("--beams_seeds", default="100-109")
    p.add_argument("--beams_groups", default=None, help="自检用：B1=ga,B3=... 覆盖目录名（只列出的组）")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    rng = lambda t: range(int(t.split("-")[0]), int(t.split("-")[1]) + 1)
    groups = dict(BEAMS)
    if a.beams_groups:
        groups = dict(kv.split("=") for kv in a.beams_groups.split(","))
    txt = report(Path(a.rep) if a.rep else None, Path(a.beams) if a.beams else None, rng(a.rep_seeds), rng(a.beams_seeds), groups)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
