"""第 15 周的预先登记分析（claude/正式实验计划.md 变更记录第 18、20 条；结果出来之前写定，11-11）。

追加 A：H6 = H5 在新种子上的独立确认（横梁，mac_runs/confirm_beams/，种子 400–423）
  H6：B6 − B1 的后期中位数 > 0。配对 t（双侧 α = 0.05），单一假设，不和 H2–H5 合并校正；
      同时报 Wilcoxon、配对 bootstrap 95%、为正的种子数。成立 = p < 0.05 且均值 > 0。
  次要（只描述）：B6 − B1 的最终最好、最好曲线均值；B5 − B1、B6 − B5 的后期中位数与最终最好；
      H5 + H6 合并（formal_beams 100–109 + confirm_beams 400–423）的 B6 − B1 后期中位数。
追加 D：v370 + 更新提示词（mac_runs/sensitivity/v370_prompt370/，种子 300–302；A1 沿用 sensitivity/v370/ga，
      GA 不用 LLM，同一台 Mac 上逐位相同）
  只描述：A4′ − A1、A5′ − A1、A4′ − A4、A5′ − A5（后期中位数；另报最终最好、后期 < 20 的比例）。
  事先定的读法（3 个种子均值）：A4′ − A4 与 A5′ − A5 都 > 0，且 A4′ − A1 与 A5′ − A1 都 > 0 →"提示词不符的解释得到支持"；
      A4′ − A4 与 A5′ − A5 都 ≤ 0 →"不支持"；其余 →"不确定"。3 个种子只能给方向，不做显著性主张。
指标定义与 hyblock/formal_analysis.py 相同（后期 = 第 5 代起）。只用有 DONE 的运行；配对只用两组都完成的种子。

用法：python -m hyblock.week15_analysis --confirm <confirm_beams> --beams <formal_beams>
          --v370 <sensitivity/v370> --v370p <sensitivity/v370_prompt370> --out <目录>
      自检：--confirm_seeds、--v370p 指向别的目录（例如用 formal_beams 100–109 代替 confirm_beams）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hyblock.followup_analysis import load, paired
from hyblock.formal_analysis import boot_ci

BG = {"B1": "ga", "B5": "ga_llmtyped", "B6": "ga_llmtyped_sel_llmEv"}
VG = {"A1": "ga", "A4": "ga_llmtyped", "A5": "ga_llmtyped_sel_llmEv"}
DESC = {"B1": "普通 GA", "B5": "GA + 地形版 LLM 变异", "B6": "B5 + ② LLM Ev",
        "A1": "普通 GA", "A4": "GA + LLM 变异（提示词 1200）", "A5": "A4 + ②（提示词 1200）",
        "A4p": "GA + LLM 变异（提示词 370）", "A5p": "A4′ + ②（提示词 370）"}


def rng(t):
    a, b = t.split("-")
    return range(int(a), int(b) + 1)


def line(desc, d, nd=1):
    if len(d) < 2:
        return f"| {desc} | {len(d)} | — | — | — | — |"
    lo, hi = boot_ci(d)
    ns = max(0, nd - 1) if nd < 2 else nd
    return (f"| {desc} | {len(d)} | {np.mean(d):+.{nd}f} | [{lo:+.{nd}f}, {hi:+.{nd}f}] | {sum(v > 0 for v in d)} / {len(d)} | "
            + " ".join(f"{v:+.{ns}f}" for v in d) + " |")


def part_h6(confirm: Path, beams: Path | None, seeds, beams_seeds) -> list[str]:
    from scipy.stats import ttest_rel, wilcoxon
    D, miss = load(confirm, BG, seeds)
    have = {g: sum((g, s) in D for s in seeds) for g in BG}
    L = ["## 追加 A：H6（横梁，B6 − B1 的后期中位数）", "",
         "完成的运行：" + "，".join(f"{g} {n}" for g, n in have.items()) + (f"；未完成：{'、'.join(miss)}" if miss else ""), "",
         "| 假设 | 种子 | 配对差均值 | SD | bootstrap 95% | 配对 t | p | Wilcoxon p | 为正 | 判定 |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    ss, d = paired(D, "B6", "B1", "med", seeds)
    if len(d) >= 3:
        t = ttest_rel([D[("B6", s)]["med"] for s in ss], [D[("B1", s)]["med"] for s in ss])
        try:
            wp = wilcoxon(d).pvalue
        except ValueError:
            wp = float("nan")
        lo, hi = boot_ci(d)
        verdict = "成立" if (t.pvalue < 0.05 and np.mean(d) > 0) else "不成立"
        if len(ss) < len(seeds):
            verdict += f"（只有 {len(ss)} / {len(seeds)} 个种子，还不是最终）"
        L.append(f"| H6 | {len(ss)} | {np.mean(d):+.1f} | {np.std(d, ddof=1):.1f} | [{lo:+.1f}, {hi:+.1f}] | {t.statistic:.2f} | "
                 f"{t.pvalue:.4f} | {wp:.3f} | {sum(v > 0 for v in d)} / {len(d)} | **{verdict}** |")
        L += ["", "- 各种子：" + " ".join(f"{s}:{v:+.0f}" for s, v in zip(ss, d))]
    else:
        L.append(f"| H6 | {len(d)} | — | — | — | — | — | — | — | 种子不足 |")
    L += ["", "### 次要（只描述）", "", "| 比较 | 种子 | 配对差均值 | 95% | 为正 | 各种子 |", "|---|---|---|---|---|---|"]
    for x, y, key, desc in [("B6", "B1", "best", "B6 − B1 最终最好"), ("B6", "B1", "auc", "B6 − B1 最好曲线均值"),
                            ("B5", "B1", "med", "B5 − B1 后期中位数"), ("B5", "B1", "best", "B5 − B1 最终最好"),
                            ("B6", "B5", "med", "B6 − B5 后期中位数"), ("B6", "B5", "best", "B6 − B5 最终最好")]:
        L.append(line(desc, paired(D, x, y, key, seeds)[1]))
    if beams is not None:
        D0, _ = load(beams, {"B1": "ga", "B6": "ga_llmtyped_sel_llmEv"}, beams_seeds)
        d0 = paired(D0, "B6", "B1", "med", beams_seeds)[1]
        L.append(line(f"H5 + H6 合并：B6 − B1 后期中位数（{len(d0)} + {len(d)} 个种子）", list(d0) + list(d)))
    L += ["", "### 各组（种子间均值）", "", "| 组 | 种子 | 最终最好 | 后期中位数 | 最好曲线均值 | 后期 < 20 | MPC h | $ |",
          "|---|---|---|---|---|---|---|---|"]
    for g in BG:
        S = [D[(g, s)] for s in seeds if (g, s) in D]
        if S:
            L.append(f"| {g} {DESC[g]} | {len(S)} | {np.mean([m['best'] for m in S]):.0f} | {np.mean([m['med'] for m in S]):.0f} | "
                     f"{np.mean([m['auc'] for m in S]):.0f} | {np.mean([m['lt20'] for m in S]):.2f} | "
                     f"{np.mean([m['mpc_h'] for m in S]):.2f} | {sum(m['usd'] for m in S):.2f} |")
    return L + [""]


def part_d(v370: Path, v370p: Path, seeds) -> list[str]:
    D, miss = load(v370, VG, seeds)
    Dp, missp = load(v370p, {"A4p": VG["A4"], "A5p": VG["A5"]}, seeds)
    D.update(Dp)
    have = {g: sum((g, s) in D for s in seeds) for g in ("A1", "A4", "A5", "A4p", "A5p")}
    L = ["## 追加 D：v370 + 提示词写 370 deg/s（平地，只描述）", "",
         "完成的运行：" + "，".join(f"{g} {n}" for g, n in have.items())
         + (f"；未完成：{'、'.join(miss + missp)}" if miss + missp else ""), "",
         "| 比较 | 种子 | 配对差均值 | 95% | 为正 | 各种子 |", "|---|---|---|---|---|---|"]
    mean = {}
    for x, y, key, desc in [("A4p", "A1", "med", "A4′ − A1 后期中位数"), ("A5p", "A1", "med", "A5′ − A1 后期中位数"),
                            ("A4p", "A4", "med", "A4′ − A4 后期中位数（改提示词的作用）"),
                            ("A5p", "A5", "med", "A5′ − A5 后期中位数（改提示词的作用）"),
                            ("A4", "A1", "med", "参照：A4 − A1 后期中位数（第 14 周）"), ("A5", "A1", "med", "参照：A5 − A1 后期中位数（第 14 周）"),
                            ("A4p", "A1", "best", "A4′ − A1 最终最好"), ("A5p", "A1", "best", "A5′ − A1 最终最好"),
                            ("A4p", "A1", "lt20", "A4′ − A1 后期 < 20 的比例"), ("A5p", "A1", "lt20", "A5′ − A1 后期 < 20 的比例")]:
        ss, d = paired(D, x, y, key, seeds)
        mean[(x, y, key)] = (np.mean(d) if d else float("nan"), len(d))
        L.append(line(desc, d, 3 if key == "lt20" else 1))
    full = all(mean[k][1] == len(seeds) for k in [("A4p", "A4", "med"), ("A5p", "A5", "med"), ("A4p", "A1", "med"), ("A5p", "A1", "med")])
    a = mean[("A4p", "A4", "med")][0], mean[("A5p", "A5", "med")][0]
    b = mean[("A4p", "A1", "med")][0], mean[("A5p", "A1", "med")][0]
    if all(v > 0 for v in a) and all(v > 0 for v in b):
        verdict = "提示词不符的解释得到支持"
    elif all(v <= 0 for v in a):
        verdict = "不支持提示词不符的解释"
    else:
        verdict = "不确定"
    if not full:
        verdict += "（种子未齐，还不是最终）"
    L += ["", f"**事先定的读法：{verdict}**（A4′ − A4 {a[0]:+.1f}、A5′ − A5 {a[1]:+.1f}；A4′ − A1 {b[0]:+.1f}、A5′ − A1 {b[1]:+.1f}）", "",
          "| 组 | 种子 | 最终最好 | 后期中位数 | 后期 < 20 | MPC h | $ |", "|---|---|---|---|---|---|---|"]
    for g in ("A1", "A4", "A5", "A4p", "A5p"):
        S = [D[(g, s)] for s in seeds if (g, s) in D]
        if S:
            L.append(f"| {g} {DESC[g]} | {len(S)} | {np.mean([m['best'] for m in S]):.0f} | {np.mean([m['med'] for m in S]):.0f} | "
                     f"{np.mean([m['lt20'] for m in S]):.2f} | {np.mean([m['mpc_h'] for m in S]):.2f} | {sum(m['usd'] for m in S):.2f} |")
    return L + [""]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--confirm", default=None)
    p.add_argument("--beams", default=None, help="formal_beams（H5 + H6 合并的描述）")
    p.add_argument("--confirm_seeds", default="400-423")
    p.add_argument("--beams_seeds", default="100-109")
    p.add_argument("--v370", default=None)
    p.add_argument("--v370p", default=None)
    p.add_argument("--v_seeds", default="300-302")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    L = ["# 第 15 周预先登记分析：H6（横梁确认）与 v370 + 更新提示词", "",
         "登记：`claude/正式实验计划.md` 变更记录第 18、20 条。脚本 `hyblock/week15_analysis.py`（结果出来之前写定）。", ""]
    if a.confirm:
        L += part_h6(Path(a.confirm), Path(a.beams) if a.beams else None, rng(a.confirm_seeds), rng(a.beams_seeds))
    if a.v370 and a.v370p:
        L += part_d(Path(a.v370), Path(a.v370p), rng(a.v_seeds))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    txt = "\n".join(L)
    (out / "report.md").write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
