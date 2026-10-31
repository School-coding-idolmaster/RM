"""两个模型在 ② 离线回放上的逐题配对比较（第 11 周；只读两边的 llm_cache，不调用 API）。

同样的题目、同样的提示词（经验池、排序方式、票数、打乱顺序都相同），只换模型 → 逐题配对。
指标（每道题）：
  rho        预测与真实得分的 Spearman（--votes 票与 1 票）
  gain       预测前一半的平均真实得分 − 全部平均
  bot_prec   预测后 1/4 里真是后 1/4 的比例（q = 候选数 // 4）
  veto_gain  只淘汰预测最差 1/4 后的平均真实得分 − 全部平均
  veto_best  真实最好的被淘汰（1/0）
另：knn、parent 作参照；两个模型预测之间的 Spearman。差值 = new − base：题目 bootstrap 与
运行两级 bootstrap（先抽运行，再在运行内抽题）的 95% 区间，以及每个运行的平均差。

用法（仓库根目录）：
  uv run --locked python -m hyblock.replay_paired --runs ../mac_runs/base_flat8/ga_s0 ... \
      --base claude-haiku-5-5=../mac_runs/select_replay --new claude-sonnet-5-5=../mac_runs/select_replay_sonnet \
      --out ../mac_runs/select_replay_sonnet/paired.txt
"""
from __future__ import annotations

import argparse
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr

from hyblock import selection
from hyblock import select_replay as sr


def make_llm(model: str, cache: Path, max_modules: int):
    from hyblock.search import LLM
    la = SimpleNamespace(model=model, llm_cache=str(cache), replay=True, seed=0, max_modules=max_modules)
    llm = LLM.__new__(LLM)                               # 与 select_replay 相同：只设 call() 用到的
    llm.a, llm.cache, llm.lock = la, Path(la.llm_cache), threading.Lock()
    return llm


def llm_preds(llm, pr, pool, mode, votes, max_modules):
    """-> {票数: 预测} ；读不到的回复跳过，全部读不到时返回 None。缓存缺失抛 RuntimeError。"""
    sel = selection.LLMSel(f"replay|{pr['run']}", llm, max_modules, pool, mode, votes, votes)
    n = len(pr["cands"])
    m = max(1, n // 2)
    P = []
    for v, (perm, user) in enumerate(sel.prompts(pr["gen"], pr["cands"], pr["hist"], m)):
        rec = llm.call(user, pr["gen"], v, system=sel.system, tag=f"sel{v}")
        got, _, _ = selection.parse_answer(rec.get("response", ""), n, mode, m)
        P.append(selection.positions(got, perm, n) if got else None)
    out = {}
    for nv in sorted({1, votes}):
        ok = [x for x in P[:nv] if x is not None]
        out[nv] = -np.mean(ok, axis=0) if ok else None
    return out


def per_problem(pred, truth, key, gen):
    mm = sr.metrics(pred, truth, key, gen)
    n = len(truth)
    q = max(1, n // 4)
    order = selection.order_of(pred, key, gen)
    bot = set(order[-q:])
    tbot = set(np.argsort(truth, kind="stable")[:q])
    keep = [i for i in range(n) if i not in bot]
    return {"rho": mm["rho"], "gain": mm["gain"], "bot_prec": len(bot & tbot) / q,
            "veto_gain": float(truth[keep].mean() - truth.mean()), "veto_best": float(int(np.argmax(truth)) in bot)}


def boot2(runs_vals: dict, n=4000, seed=0):
    """运行两级 bootstrap 的 95% 区间（runs_vals: 运行 -> 每题的值）。"""
    rng = np.random.default_rng(seed)
    keys = list(runs_vals)
    arr = [np.asarray(runs_vals[k], float) for k in keys]
    means = []
    for _ in range(n):
        pick = rng.integers(0, len(arr), len(arr))
        vals = np.concatenate([rng.choice(arr[i], len(arr[i])) for i in pick])
        means.append(vals.mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--base", required=True, help="模型=回放目录（含 llm_cache）")
    p.add_argument("--new", required=True, help="模型=回放目录（含 llm_cache）")
    p.add_argument("--pool", type=int, default=24)
    p.add_argument("--mode", default="rank", choices=["rank", "half"])
    p.add_argument("--votes", type=int, default=3)
    p.add_argument("--max_modules", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    (bm, bd), (nm, nd) = a.base.split("=", 1), a.new.split("=", 1)
    llms = {"base": make_llm(bm, Path(bd) / "llm_cache", a.max_modules),
            "new": make_llm(nm, Path(nd) / "llm_cache", a.max_modules)}
    probs = sr.load_problems([Path(r) for r in a.runs])

    res = {}                                           # 方法 -> list of (run, metrics)
    agree, missing, nopred = [], {"base": 0, "new": 0}, {"base": 0, "new": 0}
    for pr in probs:
        key = f"replay|{pr['run']}"
        t = pr["truth"]
        preds = {}
        for side, llm in llms.items():
            try:
                pv = llm_preds(llm, pr, a.pool, a.mode, a.votes, a.max_modules)
            except RuntimeError:
                missing[side] += 1
                pv = None
            for nv in sorted({1, a.votes}):
                x = None if pv is None else pv[nv]
                if x is None:
                    nopred[side] += nv == a.votes
                    x = selection.RandomSel(key).predict(pr["gen"], pr["cands"], [])[0]   # 与 select_replay 相同的退路
                preds[f"{side}_v{nv}"] = x
        preds["knn"] = selection.KnnSel(key).predict(pr["gen"], pr["cands"], pr["hist"])[0]
        preds["parent"] = selection.ParentSel(key).predict(pr["gen"], pr["cands"], pr["hist"])[0]
        for k, x in preds.items():
            res.setdefault(k, []).append((pr["run"], per_problem(x, t, key, pr["gen"])))
        r = spearmanr(preds[f"base_v{a.votes}"], preds[f"new_v{a.votes}"]).statistic
        agree.append(0.0 if np.isnan(r) else float(r))

    names = {"base": bm, "new": nm}
    lab = lambda k: k.replace("base", names["base"]).replace("new", names["new"])
    X = ("rho", "gain", "bot_prec", "veto_gain", "veto_best")
    L = [f"② 回放的模型配对比较：{len(probs)} 道题（{len(a.runs)} 个运行）；经验池 {a.pool}、{a.mode}、{a.votes} 票",
         f"base = {bm}（{bd}），new = {nm}（{nd}）",
         f"缓存缺失的题：base {missing['base']}、new {missing['new']}；{a.votes} 票全部读不到（退回随机）：base {nopred['base']}、new {nopred['new']}",
         "", "## 各方法（题目 bootstrap 95%）", "",
         "| 方法 | " + " | ".join(X) + " |", "|---|" + "---|" * len(X)]
    for k, rows in res.items():
        cells = []
        for x in X:
            mu, lo, hi = sr.boot([m[x] for _, m in rows])
            cells.append(f"{mu:+.1f}（{lo:+.1f}–{hi:+.1f}）" if "gain" in x else f"{mu:.2f}（{lo:.2f}–{hi:.2f}）")
        L.append(f"| {lab(k)} | " + " | ".join(cells) + " |")
    L += ["", f"## 配对差 new − base（{a.votes} 票，1 票另列）", "",
          "| 指标 | 均值 | 题目 bootstrap 95% | 运行两级 95% | 每个运行的平均差 | 为正的运行 |", "|---|---|---|---|---|---|"]
    for nv in sorted({1, a.votes}, reverse=True):
        B, N = res[f"base_v{nv}"], res[f"new_v{nv}"]
        for x in X:
            d = [n_[1][x] - b_[1][x] for b_, n_ in zip(B, N)]
            mu, lo, hi = sr.boot(d)
            by = {}
            for (run, _), v in zip(B, d):
                by.setdefault(run, []).append(v)
            lo2, hi2 = boot2(by)
            per = [np.mean(by[r]) for r in by]
            L.append(f"| {x}（{nv} 票） | {mu:+.3f} | [{lo:+.3f}, {hi:+.3f}] | [{lo2:+.3f}, {hi2:+.3f}] | "
                     + " ".join(f"{v:+.2f}" for v in per) + f" | {sum(v > 0 for v in per)}/{len(per)} |")
    L += ["", f"两个模型 {a.votes} 票预测之间的 Spearman（逐题平均）：{np.mean(agree):.2f}",
          "运行顺序：" + " ".join(dict.fromkeys(pr["run"] for pr in probs))]
    text = "\n".join(L) + "\n"
    Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
