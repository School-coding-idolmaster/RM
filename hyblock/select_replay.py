"""② 挑选的离线回放（第 10 周第 1 步；不跑 MPC，只调用 LLM）。

用已经跑完的 GA 系运行（每代 16 个候选都有 MPC 得分）重放挑选：第 g 代（g ≥ 1）的题目 = 本代通过 ① 并跑了 MPC 的候选，
挑选器只能看到第 g 代以前的记录，给这些候选排序，再与真实得分比较。这样不花 MPC 就能比较挑选器与 LLM 的设计选择：
经验池大小（--pools）、全部排序 / 只挑前一半（--modes）、1 票 / --votes 票。

挑选器：random（200 个随机顺序的平均）、parent、regress、knn（hyblock/selection.py），llm_p<池>_<方式>_v<票>
指标（每道题；m = 候选数的一半）：
  rho     预测与真实得分的 Spearman
  recall  预测前 m 个里有多少是真实前 m 个（/ m）
  best    真实最好的候选是否在预测前 m 个里
  gain    预测前 m 个的平均真实得分 − 全部候选的平均（= ② 每代能把送评估的设计平均提高多少分）

用法（仓库根目录）：
  uv run --locked python -m hyblock.select_replay --runs ../mac_runs/base_flat8/ga_s0 ... --out ../mac_runs/select_replay
  --no_llm：只算不用 LLM 的挑选器；--smoke：只做第一道题（检查回复格式，ok 比例 < 0.8 时退出码 1）
  --replay：只用 llm_cache 里的回复
输出：problems.csv（每题 × 挑选器）、calls.csv（每次 LLM 调用）、report.md。
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr

from hyblock import selection

PRICE_IN, PRICE_OUT = 0.10, 0.50                  # $/M token（llm_mutation 的冒烟检查用同一估计）


def load_problems(runs: list[Path], min_cands: int = 4) -> list[dict]:
    probs = []
    for run in runs:
        with open(run / "evals.csv", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        gens = sorted({int(r["gen"]) for r in rows})
        for g in gens:
            if g == 0:
                continue
            hist = [r for r in rows if int(r["gen"]) < g]
            fit = selection.fit_map(hist)
            cur = [r for r in rows if int(r["gen"]) == g and r["status"] == "mpc"]
            if len(cur) < min_cands:
                continue
            cands = [selection.candidate(int(r["idx"]), r["design"], r["source"], r["parents"], fit) for r in cur]
            probs.append({"run": run.name, "gen": g, "hist": hist, "cands": cands,
                          "truth": np.array([float(r["fitness"]) for r in cur])})
    return probs


def metrics(pred: np.ndarray, truth: np.ndarray, key: str, gen: int) -> dict:
    n = len(truth)
    m = max(1, n // 2)
    order = selection.order_of(pred, key, gen)
    top = set(order[:m])
    true_top = set(np.argsort(-truth, kind="stable")[:m])
    rho = spearmanr(pred, truth).statistic if np.ptp(pred) > 0 else 0.0
    return {"rho": float(0.0 if np.isnan(rho) else rho), "recall": len(top & true_top) / m,
            "best": float(int(np.argmax(truth)) in top), "gain": float(truth[list(top)].mean() - truth.mean())}


def random_metrics(truth: np.ndarray, gen: int, k: int = 200) -> dict:
    rng = np.random.default_rng(gen)
    acc = [metrics(rng.random(len(truth)), truth, f"r{j}", gen) for j in range(k)]
    return {x: float(np.mean([a[x] for a in acc])) for x in acc[0]}


def boot(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, (n, len(x))).mean(1)
    return float(x.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pools", type=int, nargs="+", default=[8, 16, 24])
    p.add_argument("--modes", nargs="+", default=["rank", "half"], choices=["rank", "half"])
    p.add_argument("--votes", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--max_modules", type=int, default=8)
    p.add_argument("--no_llm", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--min_ok", type=float, default=0.8)
    a = p.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    probs = load_problems([Path(r) for r in a.runs])
    if a.smoke:
        probs = probs[:1]
    print(f"{len(probs)} 道题（{len(a.runs)} 个运行）", flush=True)

    res = []                                            # (run, gen, method, metrics)
    for pr in probs:
        key = f"replay|{pr['run']}"
        res.append((pr, "random", random_metrics(pr["truth"], pr["gen"])))
        for cls in (selection.ParentSel, selection.RegressSel, selection.KnnSel):
            pred, _ = cls(key).predict(pr["gen"], pr["cands"], pr["hist"])
            res.append((pr, cls.name, metrics(pred, pr["truth"], key, pr["gen"])))

    calls_all = []
    if not a.no_llm:
        if not a.replay:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=Path.cwd() / ".env")
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise SystemExit("ANTHROPIC_API_KEY missing (.env)")
        from hyblock.search import LLM
        la = SimpleNamespace(model=a.model, llm_cache=str(out / "llm_cache"), replay=a.replay, seed=0,
                             max_modules=a.max_modules)
        la_llm = LLM.__new__(LLM)                       # LLM.__init__ needs the full search args; set what call() uses
        la_llm.a, la_llm.cache = la, Path(la.llm_cache)
        la_llm.cache.mkdir(parents=True, exist_ok=True)
        import threading
        la_llm.lock = threading.Lock()
        jobs = [(pr, K, mode) for pr in probs for K in a.pools for mode in a.modes]
        t0 = time.time()

        def work(job):
            pr, K, mode = job
            sel = selection.LLMSel(f"replay|{pr['run']}", la_llm, a.max_modules, K, mode, a.votes, a.votes)
            n = len(pr["cands"])
            m = max(1, n // 2)
            scores_v, calls = [], []
            P, recs = [], sel.prompts(pr["gen"], pr["cands"], pr["hist"], m)
            for v, (perm, user) in enumerate(recs):
                rec = la_llm.call(user, pr["gen"], v, system=sel.system, tag=f"sel{v}")
                got, oc, ana = selection.parse_answer(rec.get("response", ""), n, mode, m)
                P.append(selection.positions(got, perm, n) if got else None)
                u = rec.get("usage", {})
                calls.append({"run": pr["run"], "gen": pr["gen"], "pool": K, "mode": mode, "vote": v, "n_cand": n,
                              "outcome": oc, "n_parsed": len(got), "llm_in": u.get("input_tokens", 0),
                              "llm_out": u.get("output_tokens", 0), "llm_s": rec.get("seconds", ""),
                              "cached": rec.get("cached", ""), "analysis": ana})
            out_m = {}
            for nv in sorted({1, a.votes}):
                ok = [x for x in P[:nv] if x is not None]
                if ok:
                    pred = -np.mean(ok, axis=0)
                else:
                    pred = selection.RandomSel(f"replay|{pr['run']}").predict(pr["gen"], pr["cands"], [])[0]
                out_m[nv] = metrics(pred, pr["truth"], f"replay|{pr['run']}", pr["gen"])
            return pr, K, mode, out_m, calls

        with ThreadPoolExecutor(max_workers=max(1, -(-a.workers // a.votes))) as ex:
            for k, (pr, K, mode, out_m, calls) in enumerate(ex.map(work, jobs), 1):
                for nv, mm in out_m.items():
                    res.append((pr, f"llm_p{K}_{mode}_v{nv}", mm))
                calls_all += calls
                if k % 10 == 0 or k == len(jobs):
                    print(f"  {k}/{len(jobs)} 组（{time.time() - t0:.0f} s）", flush=True)

    with open(out / "problems.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "gen", "n_cand", "method", "rho", "recall", "best", "gain"])
        for pr, meth, mm in res:
            w.writerow([pr["run"], pr["gen"], len(pr["cands"]), meth] + [round(mm[x], 4) for x in ("rho", "recall", "best", "gain")])
    if calls_all:
        with open(out / "calls.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(calls_all[0]))
            w.writeheader()
            w.writerows(calls_all)

    methods = list(dict.fromkeys(m for _, m, _ in res))
    groups = [("全部", lambda pr: True)] + [(g, (lambda g: lambda pr: pr["run"].rsplit("_s", 1)[0] == g)(g))
                                            for g in sorted({pr["run"].rsplit("_s", 1)[0] for pr in probs})]
    L = [f"# ② 挑选的离线回放（{time.strftime('%Y-%m-%d %H:%M')}）", "",
         f"运行：{', '.join(Path(r).name for r in a.runs)}；题目 {len(probs)} 道（每道 = 一代通过 ① 并跑了 MPC 的候选，"
         f"平均 {np.mean([len(p['cands']) for p in probs]):.1f} 个）。m = 候选数的一半。",
         f"LLM：{a.model}，effort {os.environ.get('LLM_EFFORT', 'high')}；系统提示 sha256 "
         f"{selection.prompt_sha256(a.max_modules)[:12]}…", "",
         "指标：rho = Spearman；recall = 预测前 m 个中真实前 m 个的比例；best = 真实最好的在预测前 m 个里的比例；"
         "gain = 预测前 m 个的平均真实得分 − 全部候选平均。括号 = 题目 bootstrap 95% 区间。", ""]
    for gname, gf in groups:
        L += [f"## {gname}", "", "| 挑选器 | 题 | rho | recall | best | gain |", "|---|---|---|---|---|---|"]
        for meth in methods:
            sub = [mm for pr, mt, mm in res if mt == meth and gf(pr)]
            if not sub:
                continue
            cells = []
            for x in ("rho", "recall", "best", "gain"):
                mu, lo, hi = boot([s[x] for s in sub])
                cells.append(f"{mu:.2f}（{lo:.2f}–{hi:.2f}）" if x != "gain" else f"{mu:+.1f}（{lo:+.1f}–{hi:+.1f}）")
            L.append(f"| {meth} | {len(sub)} | " + " | ".join(cells) + " |")
        L.append("")
    if calls_all:
        L += ["## LLM 调用", "", "| 设置 | 调用 | ok | partial | fail | 每次输入 / 输出 token | 费用（约） |", "|---|---|---|---|---|---|---|"]
        for K in a.pools:
            for mode in a.modes:
                cs = [c for c in calls_all if c["pool"] == K and c["mode"] == mode]
                tin = sum(int(c["llm_in"] or 0) for c in cs)
                tout = sum(int(c["llm_out"] or 0) for c in cs)
                oc = {o: sum(c["outcome"] == o for c in cs) for o in ("ok", "partial", "fail")}
                L.append(f"| p{K} {mode} | {len(cs)} | {oc['ok']} | {oc['partial']} | {oc['fail']} | "
                         f"{tin / max(1, len(cs)):.0f} / {tout / max(1, len(cs)):.0f} | "
                         f"${(tin * PRICE_IN + tout * PRICE_OUT) / 1e6:.2f} |")
        L.append("")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    if a.smoke and calls_all:
        ok = sum(c["outcome"] == "ok" for c in calls_all) / len(calls_all)
        print(f"冒烟检查：{len(calls_all)} 次调用，ok 比例 {ok:.2f}")
        if ok < a.min_ok:
            print(f"!!! ok 比例 {ok:.2f} < {a.min_ok}：回复格式或提示词有问题，停止")
            sys.exit(1)


if __name__ == "__main__":
    main()
