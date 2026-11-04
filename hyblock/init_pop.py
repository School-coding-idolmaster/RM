"""第 0 代的来源（第 12 周，横梁；第12周记录 第 2b 节）。--init：

  random        随机设计（默认；与之前完全相同）
  llm           I1：LLM 按地形描述一次提出 batch 个不同设计（prompts/init.txt）；不合法 / 重复的带原因重问一次，
                仍不行用随机设计补（source = llm_init_fail>random）
  transfer      I2：--init_source（同种子平地 GA 的 evals.csv）得分最高的 batch 个不同设计，原样拿来
  llm_transfer  I3：I2 的每个设计各一次调用（prompts/transfer.txt：平地得分 + 地形描述），LLM 决定改不改、怎么改；
                不合法 / 重复的带原因重问一次，仍不行用原设计（source = llm_transfer_fail>transfer）

系统提示 = 提示词 v2 的零件 / 规则 / 上限 / 穿透 / 设计格式部分 + 地形段（hyblock/terrain_text.py，平地没有）+ 任务。
回复缓存在 search.LLM 的 llm_cache（键里有 tag "init" / "transfer<k>"），续跑与重放结果相同。
记录：<out>/init.csv 每次调用、每个设计一行。平地搜索的评估次数不计入本运行的预算。
"""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

from hyblock.design import DesignError, parse, to_text
from hyblock.generator import random_design
from hyblock.llm_io import PROMPT_DIR
from hyblock.llm_mutation import MARKER, reason_text
from hyblock.mjcf import compile_design
from hyblock.terrain_text import NAME, TERRAIN

FIELDS = ["init", "call", "attempt", "slot", "input_fitness", "input_design", "outcome", "kind", "reason", "idea",
          "design", "final_source", "llm_in", "llm_out", "llm_s", "cached"]
NAMES = {"flat": "flat ground", **NAME}


def system_prompt(task: str, terrain: str, max_modules: int, n: int = 16) -> str:
    base = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
    common = base[:base.index(MARKER)].rstrip() + "\n\n"
    if terrain != "flat":
        common += f"### TERRAIN ###\n{TERRAIN[terrain]}\n\n"
    body = (PROMPT_DIR / f"{task}.txt").read_text(encoding="utf-8")
    return (common + body).replace("{max_modules}", str(max_modules)).replace("{name}", NAMES[terrain]) \
        .replace("{n}", str(n))


def example_design() -> str:
    base = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
    return to_text(parse(re.findall(r"^DESIGN:\s*(.+)$", base, flags=re.M)[-1]))


def check(text: str | None, max_modules: int, taken: set) -> tuple[str | None, str, str]:
    """-> (规范化设计或 None, outcome, 给 LLM 的原因)。"""
    if not text:
        return None, "no_design", "no design was found in the response"
    try:
        canon = to_text(parse(text))
        compile_design(canon, max_modules=max_modules)
    except DesignError as e:
        return None, f"invalid:{e.kind}", reason_text(text, e, max_modules)
    if canon in taken:
        return None, "duplicate", "it is the same as another design (or as the example design)"
    return canon, "ok", ""


def _usage(rec):
    u = rec.get("usage", {})
    return dict(llm_in=u.get("input_tokens", ""), llm_out=u.get("output_tokens", ""), llm_s=rec.get("seconds", ""),
                cached=rec.get("cached", False))


def parse_numbered(text: str) -> tuple[dict, dict]:
    des = {int(k): v.strip().strip("`* ").strip() for k, v in
           re.findall(r"^\W*DESIGN\s*(\d+)\W*:\s*(.+)$", text or "", flags=re.M | re.I)}
    ideas = {int(k): v.strip()[:300] for k, v in re.findall(r"^\W*IDEA\s*(\d+)\W*:\s*(.+)$", text or "", flags=re.M | re.I)}
    return des, ideas


def top_flat(path: str | Path, n: int) -> list[dict]:
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8")) if r["status"] == "mpc"]
    rows.sort(key=lambda r: (-float(r["fitness"]), int(r["gen"]), int(r["idx"])))
    out, seen = [], set()
    for r in rows:
        t = to_text(parse(r["design"]))
        if t not in seen:
            seen.add(t)
            out.append({"design": t, "fitness": float(r["fitness"]), "ref": f"{r['gen']}.{r['idx']}"})
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"--init_source {path}: only {len(out)} distinct evaluated designs")
    return out


class Init:
    def __init__(self, a, out: Path, llm=None):
        self.a, self.llm, self.out = a, llm, out
        self.log: list[dict] = []

    def _write(self):
        with open(self.out / "init.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.log)

    def _rand(self, k):
        rng = random.Random(f"hyblock-init|{self.a.seed}|{k}")
        return to_text(random_design(rng, max_modules=self.a.max_modules))

    # ------------------------------------------------------------ I1
    def llm_init(self, n: int) -> list[dict]:
        a = self.a
        system = system_prompt("init", a.terrain, a.max_modules, n)
        taken = {example_design()}
        user = f"Propose the {n} designs now."
        rec = self.llm.call(user, 0, 0, system=system, tag="init")
        des, ideas = parse_numbered(rec["response"])
        got, bad = {}, {}
        for k in range(1, n + 1):
            canon, outcome, reason = check(des.get(k), a.max_modules, taken)
            self.log.append(dict(init="llm", call=0, attempt=0, slot=k, outcome=outcome, reason=reason,
                                 idea=ideas.get(k, ""), design=des.get(k, ""), **_usage(rec)))
            if canon:
                got[k] = canon
                taken.add(canon)
            else:
                bad[k] = (des.get(k, ""), reason)
        if bad:                                           # one retry for all failed slots, with the reasons
            lines = [f"Your designs {', '.join(map(str, bad))} could not be used:"]
            for k, (d, r) in bad.items():
                lines.append(f"DESIGN {k}: {d or '(missing)'} -> {r}")
            lines.append("Designs already accepted (do not repeat them):")
            lines += [f"DESIGN {k}: {t}" for k, t in sorted(got.items())]
            lines.append(f"Give a new valid design for each of {', '.join(map(str, bad))}, "
                         "in the same response format with the same numbers.")
            rec2 = self.llm.call(user + "\n\n" + "\n".join(lines), 0, 1, system=system, tag="init")
            des2, ideas2 = parse_numbered(rec2["response"])
            for k in bad:
                canon, outcome, reason = check(des2.get(k), a.max_modules, taken)
                self.log.append(dict(init="llm", call=1, attempt=1, slot=k, outcome=outcome, reason=reason,
                                     idea=ideas2.get(k, ""), design=des2.get(k, ""), **_usage(rec2)))
                if canon:
                    got[k] = canon
                    taken.add(canon)
        props = []
        for k in range(1, n + 1):
            if k in got:
                props.append({"source": "llm_init", "design": got[k]})
            else:
                t = self._rand(k)
                props.append({"source": "llm_init_fail>random", "design": t})
        for i, p in enumerate(props):
            self.log.append(dict(init="llm", call="final", slot=i + 1, outcome="final", design=p["design"],
                                 final_source=p["source"]))
        self._write()
        return props

    # ------------------------------------------------------------ I2 / I3
    def transfer(self, n: int, use_llm: bool) -> list[dict]:
        a = self.a
        src = top_flat(a.init_source, n)
        if not use_llm:
            props = [{"source": "transfer", "design": s["design"], "parents": f"flat:{s['ref']}"} for s in src]
            for i, p in enumerate(props):
                self.log.append(dict(init="transfer", slot=i + 1, outcome="final", input_fitness=src[i]["fitness"],
                                     input_design=src[i]["design"], design=p["design"], final_source="transfer"))
            self._write()
            return props
        system = system_prompt("transfer", a.terrain, a.max_modules)
        taken_all = {s["design"] for s in src}

        def one(k):
            s = src[k]
            user = (f"Design (score {s['fitness']:.1f} on level ground):\n{s['design']}\n\n"
                    f"Adapt it for {NAMES[a.terrain]}.")
            rec = self.llm.call(user, 0, k, system=system, tag=f"transfer{k}")
            return k, user, rec

        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max(1, a.llm_workers)) as ex:
            first = list(ex.map(one, range(n)))
        props, chosen = [None] * n, set()
        from hyblock.llm_io import extract_design
        from hyblock.llm_mutation import extract_change
        retry = []
        for k, user, rec in first:
            d = extract_design(rec["response"])
            others = (taken_all - {src[k]["design"]}) | chosen
            canon, outcome, reason = check(d, a.max_modules, others)
            if canon == src[k]["design"]:
                outcome = "unchanged"
            self.log.append(dict(init="llm_transfer", call=k, attempt=0, slot=k + 1, input_fitness=src[k]["fitness"],
                                 input_design=src[k]["design"], outcome=outcome, reason=reason,
                                 idea=extract_change(rec["response"]), design=d or "", **_usage(rec)))
            if canon:
                props[k] = canon
                chosen.add(canon)
            else:
                retry.append((k, user, rec, d, reason))

        def again(item):
            k, user, rec, d, reason = item
            u2 = (user + f"\n\nYour previous answer could not be used:\nDESIGN: {d or '(missing)'}\n-> {reason}\n"
                  "Give a corrected design in the same response format.")
            return k, self.llm.call(u2, 0, k, system=system, tag=f"transfer{k}_retry")

        with ThreadPoolExecutor(max_workers=max(1, a.llm_workers)) as ex:
            second = list(ex.map(again, retry))
        for k, rec in second:
            d = extract_design(rec["response"])
            others = (taken_all - {src[k]["design"]}) | chosen
            canon, outcome, reason = check(d, a.max_modules, others)
            if canon == src[k]["design"]:
                outcome = "unchanged"
            self.log.append(dict(init="llm_transfer", call=k, attempt=1, slot=k + 1, input_fitness=src[k]["fitness"],
                                 input_design=src[k]["design"], outcome=outcome, reason=reason,
                                 idea=extract_change(rec["response"]), design=d or "", **_usage(rec)))
            if canon:
                props[k] = canon
                chosen.add(canon)
        out = []
        for k in range(n):
            if props[k] is None:
                out.append({"source": "llm_transfer_fail>transfer", "design": src[k]["design"],
                            "parents": f"flat:{src[k]['ref']}"})
            else:
                same = props[k] == src[k]["design"]
                out.append({"source": "llm_transfer_none" if same else "llm_transfer", "design": props[k],
                            "parents": f"flat:{src[k]['ref']}"})
            self.log.append(dict(init="llm_transfer", call="final", slot=k + 1, outcome="final",
                                 input_fitness=src[k]["fitness"], input_design=src[k]["design"],
                                 design=out[-1]["design"], final_source=out[-1]["source"]))
        self._write()
        return out


def propose_init(a, out: Path, llm=None) -> list[dict]:
    n = a.batch
    it = Init(a, out, llm)
    if a.init == "llm":
        return it.llm_init(n)
    if a.init == "transfer":
        return it.transfer(n, use_llm=False)
    if a.init == "llm_transfer":
        return it.transfer(n, use_llm=True)
    raise ValueError(a.init)
