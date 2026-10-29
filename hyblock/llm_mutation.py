"""GA 的 LLM 变异算子（第 9 周；pipeline v4.6 的 2 × 2 设计中\"变异方式 = LLM\"一项）。

`hyblock/search.py --method ga --mutation llm`：GA 照常从前 --pop 名里锦标赛选亲本、做子树交换交叉；凡是普通 GA
要做一次随机变异（generator.mutate）的地方，改为让 LLM 对这个设计做\"一处有针对性的修改\"。其余（选择、交叉概率、
预算、① 筛选、MPC）与普通 GA 完全相同，第 0 代也是同一批随机设计（配对比较）。

LLM 看到的内容（系统提示 = 提示词 v2 的零件 / 规则 / 上限 / 穿透 / 设计格式部分 + prompts/mutate.txt；
用户提示每次调用不同，只用本代以前的记录，所以续跑时提示词不变、缓存命中）：
  - 要修改的设计：亲本（得分、排名、位移、提前终止回合数、结构指标），或交叉得到、还没评估的子代（两个亲本的得分）
  - 迄今最好的 --mut_good 个设计（得分）
  - 仿真里失败的设计（每回合都提前终止，或得分 < 20）--mut_poor 个、被拒绝的设计（以前 LLM 提出的无效设计及原因、① 淘汰）
    --mut_failed 个：都取最近的
  - 与要修改的设计最相似的 --mut_similar 个已评估设计（零件项 Jaccard；即\"已经试过的改法\"及其得分）
回复：ANALYSIS / CHANGE / DESIGN 三行。检查：能解析、与输入不同、与已评估 / 本代已提出的设计都不同、合法
（模块数上限、自碰撞等）。不通过时带着原因（英文，含模块数、穿透深度与零件名）重问 --mut_retries 次；仍不通过则
退回一次随机变异（来源记为 llm_fail>mutate:…），保证每代候选数与普通 GA 相同。

记录：<out>/llm_mut.csv 每次 LLM 调用一行（代、序号、第几次、输入、结果、原因、token）；evals.csv 的 source 为
llm_mut / llm_mut_retry / llm_fail>mutate:…（交叉后的为 crossover+…），llm_in / llm_out / llm_s / llm_calls 为该候选
所有调用之和。
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from hyblock.design import DesignError, build_robot, parse, to_text
from hyblock.llm_io import PROMPT_DIR, extract_design
from hyblock.model import load_spec

MARKER = "You are to apply rules sequentially"          # system.txt: generation instructions start here
ATTEMPT_FIELDS = ["gen", "idx", "attempt", "input_kind", "parents", "input_fitness", "input_design", "outcome",
                  "kind", "reason", "change", "design", "llm_in", "llm_out", "llm_s", "cached"]
# outcome: ok | invalid（kind = DesignError 类别）| duplicate | unchanged | no_design

KIND_TEXT = {
    "syntax": "the design line could not be parsed",
    "duplicate_name": "two parts have the same name",
    "root": "there must be exactly one root part",
    "unknown_parent": "a part or foot refers to a parent that does not exist or is not connected to the root",
    "unknown_port": "a port name does not exist on that part",
    "roll": "a roll is not 0, 90, 180 or 270",
    "joint_range": "an initial joint angle is outside -30 to 150 deg",
    "parallel": "x2 is only allowed on modules",
    "mirror": "mirror is only allowed on a part attached to hub face zp or zm",
    "port_conflict": "a port is used by more than one part or foot",
    "foot": "a foot must be on a free port",
    "hub_cap": "more than 4 hubs",
    "invalid": "the design breaks a connection rule",
}


def system_prompt(max_modules: int) -> str:
    base = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
    common = base[:base.index(MARKER)].rstrip() + "\n\n"
    text = common + (PROMPT_DIR / "mutate.txt").read_text(encoding="utf-8")
    return text.replace("{max_modules}", str(max_modules))


def prompt_sha256(max_modules: int = 8) -> str:
    return hashlib.sha256(system_prompt(max_modules).encode("utf-8")).hexdigest()


def extract_change(text: str) -> str:
    hits = re.findall(r"^\W*CHANGE\W*:\s*(.+)$", text or "", flags=re.M | re.I)
    return hits[-1].strip()[:300] if hits else ""


# ---------------------------------------------------------------- 原因（给 LLM 看的英文）

def _body_label(name: str, labels: list[str]) -> str:
    m = re.match(r"n(\d+)_(\w+)$", name or "")
    if not m or int(m.group(1)) >= len(labels):
        return name
    lab, body = labels[int(m.group(1))], m.group(2)
    where = {"block": "block", "P1": "plate P1", "P2": "plate P2"}.get(body, "" if body == "hub" else body)
    copy = " (the mirror copy)" if lab.endswith("'") else ""
    return f"{lab.rstrip(chr(39))}{copy}" + (f" {where}" if where else "")


def reason_text(text: str, err: DesignError, max_modules: int, spec=None) -> str:
    kind = err.kind
    if kind == "module_cap":
        try:
            _, ex = build_robot(text, spec, max_modules=10 ** 6, max_hubs=10 ** 6)
            return (f"too many modules: {ex.module_count()} after expanding mirror and x2 (limit {max_modules}; "
                    "every module below a mirrored part counts twice)")
        except DesignError:
            return f"too many modules (limit {max_modules})"
    if kind == "self_collision":
        m = re.search(r"(\d+) mm（(.+?)\|(.+?)）", str(err))
        if m:
            try:
                _, ex = build_robot(text, spec, max_modules=10 ** 6, max_hubs=10 ** 6)
                a, b = _body_label(m.group(2), ex.labels), _body_label(m.group(3), ex.labels)
            except DesignError:
                a, b = m.group(2), m.group(3)
            return f"parts overlap in the initial pose: {a} and {b} intersect by {m.group(1)} mm"
        return "parts overlap in the initial pose"
    return KIND_TEXT.get(kind, kind)


# ---------------------------------------------------------------- 提示词

def _items(text: str) -> set:
    return {x.strip() for x in (text or "").split(";") if x.strip()}


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


def _fmt_row(r: dict, with_score: bool = True) -> str:
    s = f"score {float(r['fitness']):.1f} | " if with_score else ""
    return f"{s}{r['modules']} modules, {r['joints']} joints, {r['feet']} feet | {r['design']}"


def _fail_text(r: dict) -> str:
    t = int(r["terminated"] or 0)
    if t:
        return (f"score {float(r['fitness']):.1f}, {t} of 2 episodes ended early "
                "(torso touched the ground or the robot flipped)")
    return f"score {float(r['fitness']):.1f}, hardly moved forward"


class Context:
    """本代以前的记录（evals.csv 的行 + llm_mut.csv 的行）→ 提示词的各部分。"""

    def __init__(self, rows: list[dict], attempts: list[dict], gen: int, a):
        self.a = a
        self.rows = [r for r in rows if int(r["gen"]) < gen]
        self.scored = sorted([r for r in self.rows if r["status"] == "mpc"], key=lambda r: -float(r["fitness"]))
        self.rank = {r["design"]: k + 1 for k, r in enumerate(self.scored)}
        self.attempts = [r for r in attempts if int(r["gen"]) < gen]

    def describe_input(self, job: dict) -> str:
        text = job["design"]
        if job["kind"] == "parent":
            r = job["parent"]
            return (f"DESIGN: {text}\n"
                    f"Score {float(r['fitness']):.1f} (rank {self.rank.get(r['design'], '?')} of {len(self.scored)} "
                    f"evaluated designs; best so far {float(self.scored[0]['fitness']):.1f}). Forward distance "
                    f"{float(r['distance_m']):.1f} m per episode; {int(r['terminated'] or 0)} of 2 episodes ended "
                    f"early. {r['modules']} modules, {r['joints']} joints, {r['hubs']} hubs, {r['feet']} feet.")
        pa, pb = job["parent"], job["other"]
        return (f"DESIGN: {text}\n"
                f"Not evaluated yet: it was made by crossover, grafting part of design B (score "
                f"{float(pb['fitness']):.1f}) onto design A (score {float(pa['fitness']):.1f}).\n"
                f"Design A: {pa['design']}\nDesign B: {pb['design']}")

    def user_prompt(self, job: dict) -> str:
        a, text = self.a, job["design"]
        skip = {text, job["parent"]["design"]} | ({job["other"]["design"]} if job.get("other") else set())
        L = ["## Design to modify", self.describe_input(job), ""]
        best = [r for r in self.scored if r["design"] not in skip][:a.mut_good]
        if best:
            L += ["## Best designs evaluated so far"] + [_fmt_row(r) for r in best] + [""]
        poor = [r for r in self.rows if r["status"] == "mpc" and (int(r["terminated"] or 0) >= a.episodes
                                                                   or float(r["fitness"]) < 20)]
        poor = [r for r in poor if r["design"] not in skip][-a.mut_poor:][::-1] if a.mut_poor else []
        shown = skip | {r["design"] for r in best} | {r["design"] for r in poor}
        if poor:
            L += ["## Designs that failed in simulation (most recent first)"]
            L += [f"{_fail_text(r)} | {r['design']}" for r in poor] + [""]
        rejected = [(int(r["gen"]), int(r["idx"]), int(r["attempt"]), r["reason"], r["design"])
                    for r in self.attempts if r["outcome"] in ("invalid", "no_design") and r["design"]]
        rejected += [(int(r["gen"]), int(r["idx"]), -1,
                      "the torso is the lowest part and would touch the ground (rejected before simulation)",
                      r["design"]) for r in self.rows if r["status"] == "screened"]
        rejected = sorted(rejected)[-a.mut_failed:][::-1] if a.mut_failed else []
        if rejected:
            L += ["## Rejected designs, not simulated (most recent first)"]
            L += [f"{why} | {d[:400]}" for *_, why, d in rejected] + [""]
        mine = _items(text)
        sim = sorted((r for r in self.scored if r["design"] not in shown),
                     key=lambda r: (-_jaccard(mine, _items(r["design"])), -int(r["gen"]), -int(r["idx"])))
        sim = sim[:a.mut_similar]
        if sim:
            L += ["## Other evaluated designs most similar to the design to modify (already tried; do not repeat them)"]
            L += [f"score {float(r['fitness']):.1f} | {r['design']}" for r in sim] + [""]
        L.append("Make one targeted modification of the design to modify so that the robot walks faster. "
                 "The new design must not be identical to any design above.")
        return "\n".join(L)


def retry_block(design: str | None, reason: str) -> str:
    shown = design if design else "(no DESIGN line)"
    return ("\n\n## Your previous answer was rejected\n"
            f"You proposed: DESIGN: {shown[:400]}\nReason: {reason}.\n"
            "Make a different targeted modification that avoids this problem.")


# ---------------------------------------------------------------- 变异算子

class LLMMutator:
    def __init__(self, a, out: Path, llm):
        self.a = a
        self.llm = llm                                   # search.LLM: cache, replay, API call
        self.spec = load_spec()
        self.system = system_prompt(a.max_modules)
        self.path = out / "llm_mut.csv"
        self.attempts: list[dict] = []
        if self.path.exists():
            with open(self.path, newline="", encoding="utf-8") as f:
                self.attempts = list(csv.DictReader(f))
        self.logged = {(int(r["gen"]), int(r["idx"]), int(r["attempt"])) for r in self.attempts}

    def _log(self, row: dict) -> None:
        key = (int(row["gen"]), int(row["idx"]), int(row["attempt"]))
        if key in self.logged:                           # resume: recomputed from cache, already written
            return
        new = not self.path.exists()
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ATTEMPT_FIELDS)
            if new:
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in ATTEMPT_FIELDS})
        self.attempts.append({k: ("" if row.get(k) is None else str(row.get(k, ""))) for k in ATTEMPT_FIELDS})
        self.logged.add(key)

    def check(self, raw: str | None, job: dict, seen: set, scores: dict) -> tuple[str | None, str, str, str]:
        """-> (规范化设计或 None, outcome, kind, 英文原因)。"""
        if not raw:
            return None, "no_design", "no_design", "no DESIGN line was found in your answer"
        try:
            text = to_text(parse(raw))
        except DesignError as e:
            return None, "invalid", e.kind, reason_text(raw, e, self.a.max_modules, self.spec)
        if text == job["design"]:
            return None, "unchanged", "unchanged", "it is identical to the design you were asked to modify"
        if text in seen:
            if text in scores:
                why = f"it is identical to an already evaluated design (score {scores[text]:.1f})"
            else:
                why = "it is identical to a design that was already proposed or rejected"
            return None, "duplicate", "duplicate", why
        try:
            build_robot(text, self.spec, max_modules=self.a.max_modules)
            if self.spec.sim.get("self_collision", False):
                from hyblock.mjcf import check_self_collision
                check_self_collision(text, self.spec, max_modules=self.a.max_modules)
        except DesignError as e:
            return None, "invalid", e.kind, reason_text(text, e, self.a.max_modules, self.spec)
        return text, "ok", "", ""

    def mutate_batch(self, gen: int, jobs: dict, out: list, seen: set, rows: list[dict]) -> None:
        """jobs: {序号: {design, kind: parent|crossover, parent, other, ops, parents}}；结果写进 out[序号]。"""
        from hyblock.generator import mutate
        ctx = Context(rows, self.attempts, gen, self.a)
        scores = {r["design"]: float(r["fitness"]) for r in ctx.scored}
        base = {i: ctx.user_prompt(j) for i, j in jobs.items()}
        users = dict(base)
        tok = {i: [0, 0, 0.0, 0] for i in jobs}          # in, out, seconds, calls
        pending = sorted(jobs)
        for attempt in range(1 + max(0, self.a.mut_retries)):
            if not pending:
                break
            with ThreadPoolExecutor(max_workers=max(1, self.a.llm_workers)) as ex:
                recs = dict(zip(pending, ex.map(
                    lambda i: self.llm.call(users[i], gen, i, system=self.system, tag=f"mut{attempt}"), pending)))
            still = []
            for i in pending:                              # sequential: later slots see earlier accepted designs
                rec, job = recs[i], jobs[i]
                u = rec.get("usage", {})
                t = tok[i]
                t[0] += int(u.get("input_tokens") or 0)
                t[1] += int(u.get("output_tokens") or 0)
                t[2] += float(rec.get("seconds") or 0)
                t[3] += 1
                raw = extract_design(rec.get("response", ""))
                text, outcome, kind, why = self.check(raw, job, seen, scores)
                self._log({"gen": gen, "idx": i, "attempt": attempt, "input_kind": job["kind"],
                           "parents": job["parents"],
                           "input_fitness": job["parent"]["fitness"] if job["kind"] == "parent" else "",
                           "input_design": job["design"], "outcome": outcome, "kind": kind, "reason": why,
                           "change": extract_change(rec.get("response", "")), "design": text or raw or "",
                           "llm_in": u.get("input_tokens", ""), "llm_out": u.get("output_tokens", ""),
                           "llm_s": rec.get("seconds", ""), "cached": rec.get("cached", "")})
                if text is not None:
                    seen.add(text)
                    out[i]["design"] = text
                    out[i]["source"] = "+".join(job["ops"] + ["llm_mut" if attempt == 0 else "llm_mut_retry"])
                else:
                    users[i] = base[i] + retry_block(text or raw, why)
                    still.append(i)
            pending = still
        for i in pending:                                  # LLM failed every attempt: one random mutation
            rng = random.Random(f"hyblock-search|{self.a.seed}|{gen}|llm-fallback|{i}")
            d = parse(jobs[i]["design"])
            for _ in range(20):
                child, name = mutate(d, rng, max_modules=self.a.max_modules)
                text = to_text(child)
                if text not in seen:
                    break
            seen.add(text)
            out[i]["design"] = text
            out[i]["source"] = "+".join(jobs[i]["ops"] + [f"llm_fail>mutate:{name}"])
        for i, t in tok.items():
            out[i].update(llm_in=t[0], llm_out=t[1], llm_s=round(t[2], 2), llm_calls=t[3])

    def gen_stats(self, gen: int) -> str:
        rows = [r for r in self.attempts if int(r["gen"]) == gen]
        if not rows:
            return ""
        first = [r for r in rows if r["attempt"] == "0"]
        ok0 = sum(r["outcome"] == "ok" for r in first)
        ok1 = sum(r["outcome"] == "ok" for r in rows if r["attempt"] != "0")
        slots = {r["idx"] for r in rows}
        fb = len(slots) - ok0 - ok1
        return f" | LLM mut {len(slots)}: ok {ok0}, retry ok {ok1}, fallback {fb}"


# ---------------------------------------------------------------- 冒烟检查（Mac 任务开头用）

def main(argv=None):
    """用一个已有运行的前几代记录，对下一代做一次 LLM 变异（不跑 MPC），检查回复格式与通过率。

        python -m hyblock.llm_mutation --history ../mac_runs/base_flat8/ga_s0/evals.csv --gen 4 --out <目录>
    --dry：只打印提示词，不调用 API。第 1 次就通过的比例低于 --min_ok 时退出码为 1（任务脚本据此停止）。"""
    import argparse
    import os
    import sys
    from hyblock import search
    p = argparse.ArgumentParser()
    p.add_argument("--history", required=True, help="evals.csv of a finished ga run")
    p.add_argument("--gen", type=int, default=4, help="mutate generation GEN using generations < GEN")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min_ok", type=float, default=0.3)
    p.add_argument("--dry", action="store_true")
    a0 = p.parse_args(argv)
    out = Path(a0.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(a0.history, newline="", encoding="utf-8") as f:
        hist = [r for r in csv.DictReader(f) if int(r["gen"]) < a0.gen]
    a = search.build_parser().parse_args(["--method", "ga", "--mutation", "llm", "--out", str(out),
                                          "--seed", str(a0.seed)])
    log = search.Log(out / "no_such_dir")
    log.rows = hist
    if a0.dry:
        class Dry:
            def call(self, user, gen, idx, system=None, tag=""):
                print(f"===== slot {idx} {tag}\n{user}\n")
                return {"response": "", "usage": {}, "seconds": 0, "cached": False}
        a.mut_retries = 0
        search.propose_ga(a, a0.gen, log, LLMMutator(a, out, Dry()))
        return
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path.cwd() / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY missing (.env)")
    mut = LLMMutator(a, out, search.LLM(a, out))
    t0 = time.time()
    props = search.propose_ga(a, a0.gen, log, mut)
    att = [r for r in mut.attempts if int(r["gen"]) == a0.gen]
    first = [r for r in att if r["attempt"] == "0"]
    ok0 = sum(r["outcome"] == "ok" for r in first) / max(1, len(first))
    print(f"LLM 变异冒烟检查：{len(first)} 个子代，{len(att)} 次调用，{time.time() - t0:.0f} s")
    for r in att:
        print(f"  #{r['idx']} 第{int(r['attempt']) + 1}次 {r['outcome']:9s} {r['kind']:14s} {r['change'][:90]}"
              + (f"  [{r['reason'][:80]}]" if r["reason"] else ""))
    print(f"  第 1 次就通过 {ok0:.2f}{mut.gen_stats(a0.gen)}；来源 " + ", ".join(p["source"] for p in props))
    tin = sum(int(r["llm_in"] or 0) for r in att)
    tout = sum(int(r["llm_out"] or 0) for r in att)
    print(f"  token 输入 {tin} / 输出 {tout}（Haiku 5.5 约 ${tin / 1e6 * 0.10 + tout / 1e6 * 0.50:.3f}）")
    if ok0 < a0.min_ok:
        print(f"!!! 第 1 次通过率 {ok0:.2f} < {a0.min_ok}：回复格式或提示词有问题，停止")
        sys.exit(1)


if __name__ == "__main__":
    main()
