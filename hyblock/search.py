"""Hyblock 域的搜索驱动（第 8 周）：随机搜索、进化算法、RM 式 LLM 循环共用一套评估、预算、日志与续跑。

每一代提出 --batch 个候选 → 合法性（含自碰撞）→ ① 筛选 → ③ MPC（每设计 --episodes 回合）。
**预算 = MPC 次数**：无效设计、被 ① 淘汰的设计、与已评估设计完全相同的设计（直接沿用旧分数）都不占预算。
预算用完即停（--max_gens 防止 LLM 一直给无效设计时死循环）。

方法
  random  每代 --batch 个文法随机设计（hyblock/generator.py，上限 --max_modules）
  ga      第 0 代与 random 同一批（同一种子）；之后从已评估的前 --pop 名里锦标赛选亲本，以 --p_cross 做子树交换
          交叉，否则（以及交叉后以 --p_mut）做一次随机变异；与已评估设计重复的子代重抽
  llm     RoboMorph 式：系统提示 = 冻结的 Hyblock 提示词；用户提示 = 任务描述 + 迄今最好的 --top_k 个设计
          （RM 的 population 文本格式，含得分）；每代 --batch 次独立调用（--llm_workers 并发）

续跑：evals.csv 每评估一个候选写一行，键 = (代, 序号)。随机数按 (种子, 代) 生成，LLM 回复存在 llm_cache/
（键含模型、提示词、种子、代、序号），所以中断后重新运行会得到同样的候选并跳过已完成的行。
--replay：只用 llm_cache 里的回复，不调用 API（复现实验用）。

用法（仓库根目录）：
    uv run --locked python -m hyblock.search --method ga --budget 200 --seed 0 --out ../mac_runs/base/ga_s0
输出：evals.csv、summary.txt、curve.csv（预算 → 迄今最好得分）、run_meta.json、DONE；llm 另有 llm_cache/。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from hyblock.design import DesignError, parse, summary as design_summary, to_text
from hyblock.generator import crossover, mutate, random_design
from hyblock.mjcf import compile_design
from hyblock.screen import VERSION as SCREEN_VERSION, ScreenConfig, postcheck, screen
from mpc.mppi import MPPIConfig, evaluate_design

FIELDS = ["gen", "idx", "method", "source", "parents", "design", "status", "kind", "detail", "fitness",
          "ep_scores", "distance_m", "terminated", "env_steps", "peak_speed", "fast_episodes", "modules", "hubs",
          "joints", "feet", "root", "n_mpc", "best_so_far", "screen_s", "mpc_s", "llm_in", "llm_out",
          "llm_s", "rules", "time"]
# status: mpc（跑了 MPC，占预算）| invalid（DesignError；kind = 原因）| screened（① 淘汰；kind = 原因）
#         | duplicate（与已评估设计相同，沿用分数，不占预算）| no_design（LLM 回复里没有 DESIGN 行）


# ---------------------------------------------------------------- 记录

class Log:
    def __init__(self, out: Path):
        self.path = out / "evals.csv"
        self.rows: list[dict] = []
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                self.rows = list(csv.DictReader(f))
        self.done = {(int(r["gen"]), int(r["idx"])) for r in self.rows}

    def add(self, row: dict) -> None:
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in FIELDS})
        row = {k: ("" if row.get(k) is None else str(row.get(k, ""))) for k in FIELDS}
        self.rows.append(row)
        self.done.add((int(row["gen"]), int(row["idx"])))

    # derived views
    def scored(self, before_gen: int | None = None) -> list[dict]:
        return [r for r in self.rows if r["status"] == "mpc" and (before_gen is None or int(r["gen"]) < before_gen)]

    def n_mpc(self) -> int:
        return sum(r["status"] == "mpc" for r in self.rows)

    def score_of(self, text: str) -> float | None:
        for r in self.rows:
            if r["status"] == "mpc" and r["design"] == text:
                return float(r["fitness"])
        return None

    def best(self) -> float:
        s = [float(r["fitness"]) for r in self.rows if r["status"] == "mpc"]
        return max(s) if s else float("nan")


# ---------------------------------------------------------------- 提出候选

def _rng(seed: int, gen: int, tag: str = "") -> random.Random:
    return random.Random(f"hyblock-search|{seed}|{gen}|{tag}")


def propose_random(a, gen: int, log: Log) -> list[dict]:
    rng = _rng(a.seed, gen, "random")
    return [{"source": "random", "design": to_text(random_design(rng, max_modules=a.max_modules))}
            for _ in range(a.batch)]


def propose_ga(a, gen: int, log: Log) -> list[dict]:
    if gen == 0:
        return propose_random(a, 0, log)
    rng = _rng(a.seed, gen, "ga")
    pool = sorted(log.scored(before_gen=gen), key=lambda r: -float(r["fitness"]))[:a.pop]
    if len(pool) < 2:                                   # nothing to breed from (e.g. all screened): random
        return propose_random(a, gen, log)
    seen = {r["design"] for r in log.rows if int(r["gen"]) < gen}     # gen < current: identical after a restart

    def pick():
        cand = rng.sample(pool, min(a.tournament, len(pool)))
        return max(cand, key=lambda r: float(r["fitness"]))

    out = []
    for _ in range(a.batch):
        for _try in range(20):
            pa, pb = pick(), pick()
            da = parse(pa["design"])
            child, ops = None, []
            if rng.random() < a.p_cross and pa is not pb:
                child = crossover(da, parse(pb["design"]), rng, max_modules=a.max_modules)
                if child is not None:
                    ops.append("crossover")
            if child is None or rng.random() < a.p_mut:
                child, name = mutate(child or da, rng, max_modules=a.max_modules)
                ops.append(f"mutate:{name}")
            text = to_text(child)
            if text not in seen:
                break
        seen.add(text)
        parents = f"{pa['gen']}.{pa['idx']}" + (f"+{pb['gen']}.{pb['idx']}" if "crossover" in ops else "")
        out.append({"source": "+".join(ops), "parents": parents, "design": text})
    return out


class LLM:
    """RM 式生成：冻结提示词 + 迄今最好 top_k 个设计；回复缓存在磁盘上（续跑与重放）。"""

    def __init__(self, a, out: Path):
        from hyblock import llm_io
        self.a = a
        self.io = llm_io
        self.system = llm_io.system_prompt(a.max_modules)
        self.cache = Path(a.llm_cache) if a.llm_cache else out / "llm_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def user_prompt(self, log: Log, gen: int) -> str:
        best = sorted(log.scored(before_gen=gen), key=lambda r: -float(r["fitness"]))[:self.a.top_k]
        text = self.a.problem
        if best:                                          # robomorph/evolution/utils.py write_txt format
            lines = ["Here are some examples of designs and their associated fitness values. "
                     "How can you design a better robot with a higher fitness?"]
            for i, r in enumerate(best):
                lines += [f"Example {i}:", f"Structural: {r['rules'] or 'None'}", f"Components: {r['design']}",
                          f"Fitness: {float(r['fitness'])}", ""]
            text += " " + "\n".join(lines) + "\n"
        return text

    def _key(self, user: str, gen: int, idx: int) -> str:
        blob = json.dumps({"model": self.a.model, "effort": os.environ.get("LLM_EFFORT", "high"),
                           "system": self.system, "user": user, "seed": self.a.seed, "gen": gen, "idx": idx},
                          sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def call(self, user: str, gen: int, idx: int) -> dict:
        key = self._key(user, gen, idx)
        path = self.cache / f"{key}.json"
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            rec["cached"] = True
            return rec
        if self.a.replay:
            raise RuntimeError(f"--replay: no cached response for gen {gen} idx {idx}")
        from robomorph.evolution import llm
        t0 = time.time()
        text, usage = llm.anthropic_chat_full(self.system, user, self.a.model)
        rec = {"key": key, "seed": self.a.seed, "gen": gen, "idx": idx, "model": self.a.model,
               "effort": os.environ.get("LLM_EFFORT", "high"), "user": user, "response": text, "usage": usage,
               "seconds": round(time.time() - t0, 2), "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(path)
        rec["cached"] = False
        return rec

    def propose(self, gen: int, log: Log) -> list[dict]:
        user = self.user_prompt(log, gen)
        todo = [i for i in range(self.a.batch) if (gen, i) not in log.done]
        with ThreadPoolExecutor(max_workers=max(1, self.a.llm_workers)) as ex:
            recs = dict(zip(todo, ex.map(lambda i: self.call(user, gen, i), todo)))
        out = []
        for i in range(self.a.batch):
            if i not in recs:
                out.append(None)                         # already in evals.csv
                continue
            r = recs[i]
            text = self.io.extract_design(r["response"])
            out.append({"source": "llm", "design": text, "rules": self.io.extract_rules(r["response"]),
                        "llm_in": r["usage"].get("input_tokens", ""), "llm_out": r["usage"].get("output_tokens", ""),
                        "llm_s": r.get("seconds", "")})
        return out


# ---------------------------------------------------------------- 评估

def evaluate(a, gen: int, idx: int, prop: dict, log: Log, cfg: MPPIConfig, scfg: ScreenConfig) -> None:
    row = {"gen": gen, "idx": idx, "method": a.method, **{k: v for k, v in prop.items() if k in FIELDS},
           "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
    text = prop.get("design")
    if not text:
        row.update(status="no_design", kind="no_design")
        log.add(row)
        return
    try:
        text = to_text(parse(text))                    # canonical form (LLM spacing, option order, ...)
    except DesignError as e:
        row.update(status="invalid", kind=e.kind, detail=str(e)[:200])
        log.add(row)
        return
    row["design"] = text
    old = log.score_of(text)
    if old is not None:
        row.update(status="duplicate", fitness=old, n_mpc=log.n_mpc(), best_so_far=log.best())
        log.add(row)
        return
    t0 = time.time()
    try:
        xml = compile_design(text, max_modules=a.max_modules)
        s = design_summary(text)
        row.update(modules=s["modules"], hubs=s["hubs"], joints=s["joints"], feet=s["feet"], root=s["root"])
    except DesignError as e:
        row.update(status="invalid", kind=e.kind, detail=str(e)[:200])
        log.add(row)
        return
    if a.screen == "on":
        sr = screen(text, scfg, max_modules=a.max_modules, xml=xml)
        row["screen_s"] = round(time.time() - t0, 3)
        if not sr.passed:
            row.update(status="screened", kind=sr.reason, detail=sr.detail[:200])
            log.add(row)
            return
    t1 = time.time()
    r = evaluate_design(xml, a.terrain, a.episodes, a.mpc_seed + 7919 * gen + idx if a.vary_mpc_seed else a.mpc_seed,
                        cfg)
    pc = postcheck(r, scfg)
    eps = r["episodes"]
    row.update(status="mpc", fitness=round(r["fitness"], 3),
               ep_scores=json.dumps([round(e["score"], 2) for e in eps]),
               distance_m=round(float(np.mean([e["distance"] for e in eps])), 3),
               terminated=sum(e["terminated"] for e in eps), env_steps=json.dumps([e["env_steps"] for e in eps]),
               peak_speed=json.dumps(pc["peak_speed"]), fast_episodes=pc["fast_episodes"],
               mpc_s=round(time.time() - t1, 2))
    best = max(log.best(), r["fitness"]) if log.n_mpc() else r["fitness"]
    row.update(n_mpc=log.n_mpc() + 1, best_so_far=round(best, 3))
    log.add(row)


def run(a, out: Path) -> Log:
    log = Log(out)
    cfg = MPPIConfig(threads=a.threads, episode_control_steps=a.steps, terrain_scale=a.terrain_scale)
    scfg = ScreenConfig()
    llm = LLM(a, out) if a.method == "llm" else None
    for gen in range(a.max_gens):
        if log.n_mpc() >= a.budget:
            break
        if all((gen, i) in log.done for i in range(a.batch)):
            continue
        t0 = time.time()
        if a.method == "random":
            props = propose_random(a, gen, log)
        elif a.method == "ga":
            props = propose_ga(a, gen, log)
        else:
            props = llm.propose(gen, log)
        t_prop = time.time() - t0
        for idx, prop in enumerate(props):
            if prop is None or (gen, idx) in log.done:
                continue
            if log.n_mpc() >= a.budget:
                break
            evaluate(a, gen, idx, prop, log, cfg, scfg)
        rows = [r for r in log.rows if int(r["gen"]) == gen]
        st = {k: sum(r["status"] == k for r in rows) for k in ("mpc", "invalid", "screened", "duplicate", "no_design")}
        print(f"[{a.method} s{a.seed}] gen {gen}: " + " ".join(f"{k} {v}" for k, v in st.items() if v)
              + f" | MPC {log.n_mpc()}/{a.budget}, best {log.best():.1f} | propose {t_prop:.0f}s, "
              f"gen {time.time() - t0:.0f}s", flush=True)
    return log


# ---------------------------------------------------------------- 汇总

def write_summary(out: Path, a, log: Log) -> str:
    rows = log.rows
    mpc = [r for r in rows if r["status"] == "mpc"]
    L = [f"Hyblock 搜索（{time.strftime('%Y-%m-%d %H:%M')}）：{a.method}，种子 {a.seed}，预算 {a.budget} 次 MPC，"
         f"每代 {a.batch}，{a.terrain} ×{a.terrain_scale}，每设计 {a.episodes} 回合，筛选 {a.screen}（{SCREEN_VERSION}）"]
    if a.method == "llm":
        L.append(f"LLM：{a.model}，effort {os.environ.get('LLM_EFFORT', 'high')}，top_k {a.top_k}，问题 {a.problem!r}")
    if a.method == "ga":
        L.append(f"GA：pop {a.pop}，锦标赛 {a.tournament}，p_cross {a.p_cross}，p_mut {a.p_mut}")
    gens = 1 + max((int(r["gen"]) for r in rows), default=-1)
    L += ["", f"代数 {gens}，候选 {len(rows)}：" + "，".join(
        f"{k} {sum(r['status'] == k for r in rows)}" for k in ("mpc", "invalid", "screened", "duplicate", "no_design"))]
    kinds: dict[str, int] = {}
    for r in rows:
        if r["status"] in ("invalid", "screened"):
            kinds[f"{r['status']}:{r['kind']}"] = kinds.get(f"{r['status']}:{r['kind']}", 0) + 1
    if kinds:
        L.append("无效 / 淘汰原因：" + "，".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    n_prop = sum(r["status"] != "duplicate" for r in rows)
    if n_prop:
        L.append(f"无效设计率 {sum(r['status'] in ('invalid', 'no_design') for r in rows) / n_prop:.3f}（不含重复）；"
                 f"重复率 {sum(r['status'] == 'duplicate' for r in rows) / len(rows):.3f}")
    if mpc:
        f = np.array([float(r["fitness"]) for r in mpc])
        L.append(f"MPC {len(mpc)} 次：最好 {f.max():.1f}，中位数 {np.median(f):.1f}，能走（≥ 20）{np.mean(f >= 20):.2f}；"
                 f"峰值速度 > 6 m/s 的回合 {sum(int(r['fast_episodes'] or 0) for r in mpc)}")
        for k in (50, 100, 150, 200, 300, 400):
            if len(mpc) >= k:
                L.append(f"  前 {k} 次 MPC 的最好得分 {max(float(r['fitness']) for r in mpc[:k]):.1f}")
        top = sorted(mpc, key=lambda r: -float(r["fitness"]))[:5]
        L.append("最好的 5 个：")
        for r in top:
            L.append(f"  {float(r['fitness']):7.1f}  代 {r['gen']} #{r['idx']} {r['source']}  "
                     f"模块 {r['modules']} 关节 {r['joints']}  {r['design']}")
        uniq = len({r["design"] for r in mpc})
        L.append(f"多样性：评估过的不同设计 {uniq}；模块数分布 "
                 + str(dict(sorted({m: sum(r['modules'] == m for r in mpc) for m in {r['modules'] for r in mpc}}.items()))))
        sec = sum(float(r["mpc_s"] or 0) for r in mpc)
        L.append(f"MPC 用时 {sec / 3600:.2f} h（{sec / len(mpc):.1f} s/设计）")
    if a.method == "llm":
        tin = sum(int(r["llm_in"] or 0) for r in rows)
        tout = sum(int(r["llm_out"] or 0) for r in rows)
        calls = sum(r["source"] == "llm" for r in rows)
        L.append(f"LLM 调用 {calls} 次，输入 {tin} token，输出 {tout} token（每次平均 {tin / max(1, calls):.0f} / "
                 f"{tout / max(1, calls):.0f}）")
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    with open(out / "curve.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n_mpc", "best_so_far", "fitness"])
        best = -np.inf
        for k, r in enumerate(mpc, 1):
            best = max(best, float(r["fitness"]))
            w.writerow([k, round(best, 3), r["fitness"]])
    return text


def build_parser():
    p = argparse.ArgumentParser(description="Hyblock search baselines (random / ga / llm)")
    p.add_argument("--method", choices=["random", "ga", "llm"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--budget", type=int, default=200, help="number of MPC evaluations")
    p.add_argument("--batch", type=int, default=16, help="candidates per generation")
    p.add_argument("--max_gens", type=int, default=None, help="default: 4 x budget / batch")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_modules", type=int, default=6)
    p.add_argument("--terrain", default="flat")
    p.add_argument("--terrain_scale", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--steps", type=int, default=250, help="control steps per episode")
    p.add_argument("--mpc_seed", type=int, default=0)
    p.add_argument("--vary_mpc_seed", action="store_true", help="different MPC seed per candidate (default: same)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--screen", choices=["on", "off"], default="on")
    p.add_argument("--pop", type=int, default=16, help="ga: parents pool = best `pop` evaluated designs")
    p.add_argument("--tournament", type=int, default=2)
    p.add_argument("--p_cross", type=float, default=0.5)
    p.add_argument("--p_mut", type=float, default=0.5, help="ga: mutate after a successful crossover")
    p.add_argument("--model", default="claude-haiku-5-5")
    p.add_argument("--problem", default=None, help="llm: task text (default: robomorph/prompts/user.txt)")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--llm_workers", type=int, default=8)
    p.add_argument("--llm_cache", default=None)
    p.add_argument("--replay", action="store_true", help="llm: use cached responses only")
    p.add_argument("--summary_only", action="store_true")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.max_gens is None:
        a.max_gens = max(1, 4 * a.budget // a.batch)
    if a.problem is None:
        from robomorph.evolution.utils import prompts_path, read_txt
        a.problem = read_txt(prompts_path("user.txt")).strip()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.method == "llm" and not a.replay and not a.summary_only:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=Path.cwd() / ".env")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY missing (.env) — run copy_env_to_mac.bat on Windows")
    if a.summary_only:
        print(write_summary(out, a, Log(out)))
        return
    from wj_tools.run_meta import done, start
    start(out, threads=a.threads, note=f"hyblock.search {a.method} seed {a.seed} budget {a.budget}",
          screen_version=SCREEN_VERSION, args=vars(a))
    log = run(a, out)
    print(write_summary(out, a, log), flush=True)
    if log.n_mpc() >= a.budget:
        done(out, n_mpc=log.n_mpc(), best=log.best())
    else:
        print(f"预算没用完（{log.n_mpc()}/{a.budget}，到达 --max_gens {a.max_gens}）：不写 DONE")


if __name__ == "__main__":
    main()
