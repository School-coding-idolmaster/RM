"""② 挑选（第 10 周；pipeline v4.7 的 2 × 2 设计中"送评估的方式"一项）。

hyblock/search.py --select <挑选器>：第 1 代起每代提出 round(--batch × --oversample) 个候选，全部先做合法性 / 重复 /
① 筛选（都很便宜）；通过的候选里只把 --batch 个送 MPC：--rand_slots 个从通过的候选里均匀随机抽（也用来测排序质量），
其余按挑选器的排序从前往后取。没被挑中的写进 evals.csv，status = unselected，不占预算。
第 0 代不挑选（与普通 GA 同一批，配对比较）。

挑选器（都只用本代以前的记录）
  random   随机排序（= 普通 GA 多生成后随机送评估；对照）
  parent   按亲本得分（交叉取较好的亲本）排序：GA 里不花成本的最简单预测（对照）
  regress  岭回归：结构特征（模块、分支件、关节、脚、mirror、x2、深度、根类型、质量、模块数²）→ 得分（对照）
  knn      零件项 Jaccard 最相似的 5 个已评估设计的加权平均得分（对照）
  llm      LLM：系统提示 = 提示词 v2 的零件 / 规则部分 + prompts/select.txt；用户提示 = 经验池（--sel_pool 个"设计 + 得分"：
           最好的一半、与候选最相似的四分之一、其余按得分名次均匀取）+ 顺序打乱的候选（怎么来的、亲本得分、结构指标）
           → RANKING（全部排序，--sel_mode rank）或 SELECTED（只挑前 m 个，--sel_mode half）。--sel_votes 次调用
           （每次打乱的顺序不同），按平均名次合并。回复缓存在 llm_cache/（tag = sel<票>），续跑与重放与 LLM 变异相同
--sel_strata family（10-31）：top 名额按修改种类分层——候选按怎么来的分为 交叉 / 删减（去掉零件、去掉脚）/ 增加（加模块、
  加分支件、加脚、x2、mirror）/ 微调（初始角、滚转）/ 其他，各类按其候选数的比例分到名额（最大余数法），类内按挑选器的排序取；
  挑选器与提示词不变。原因：第 10 周在线试跑中挑选器偏爱"增加 / 微调"、压低"交叉 / 删减"，把种群带离好区域（第四阶段（下）记录 第 5 节）
记录：select.csv 每次 LLM 调用一行；evals.csv 的 sel_rank（挑选器给的名次，1 = 最好）、sel_slot（top / rand / all）、
sel_pred（预测值：llm 为负的平均名次）。离线回放见 hyblock/select_replay.py。
"""
from __future__ import annotations

import csv
import hashlib
import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from hyblock.design import summary
from hyblock.llm_io import PROMPT_DIR
from hyblock.llm_mutation import MARKER, _items, _jaccard

SELECTORS = ("random", "parent", "regress", "knn", "llm")
CALL_FIELDS = ["gen", "vote", "mode", "pool", "n_cand", "m", "labels", "parsed", "outcome", "n_parsed",
               "llm_in", "llm_out", "llm_s", "cached", "analysis"]
# outcome: ok（全部 / 恰好 m 个）| partial（缺一些，缺的并列排在后面）| fail（一个也没读到）

OPS = {"add_module": "added a module", "add_hub": "added a hub", "remove_part": "removed a part",
       "roll": "changed a roll", "q0": "changed initial joint angles", "parallel": "added or removed x2",
       "mirror": "added or removed mirror", "add_foot": "added a foot", "remove_foot": "removed a foot"}


def system_prompt(max_modules: int) -> str:
    base = (PROMPT_DIR / "system.txt").read_text(encoding="utf-8")
    text = base[:base.index(MARKER)].rstrip() + "\n\n" + (PROMPT_DIR / "select.txt").read_text(encoding="utf-8")
    return text.replace("{max_modules}", str(max_modules))


def prompt_sha256(max_modules: int = 8) -> str:
    return hashlib.sha256(system_prompt(max_modules).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 候选与特征

def fit_map(rows: list[dict]) -> dict:
    return {f"{r['gen']}.{r['idx']}": float(r["fitness"]) for r in rows if r["status"] == "mpc"}


def candidate(idx: int, design: str, source: str, parents: str, fit: dict) -> dict:
    pf = [fit[p] for p in (parents or "").split("+") if p in fit]
    return {"idx": idx, "design": design, "source": source or "", "parents": parents or "", "pfit": pf}


_FEAT: dict[str, tuple] = {}
_SUM: dict[str, dict] = {}


def _summary(design: str) -> dict:
    if design not in _SUM:
        _SUM[design] = summary(design)
    return _SUM[design]


def features(design: str) -> tuple:
    if design not in _FEAT:
        s = _summary(design)
        _FEAT[design] = (s["modules"], s["hubs"], s["joints"], s["feet"], s["mirrored"], s["parallel"],
                         s["max_depth"], float(s["root"] == "hub"), s["mass_kg"], s["modules"] ** 2)
    return _FEAT[design]


def origin_text(c: dict) -> str:
    src, pf = c["source"], c["pfit"]
    kinds = [OPS[k] for k in re.findall(r"(?:mutate:|\[)(\w+)", src) if k in OPS]
    kind = kinds[0] if kinds else ("a targeted modification" if "llm_mut" in src else "")
    if src.startswith("crossover") and len(pf) >= 2:
        return f"crossover of designs scored {pf[0]:.1f} and {pf[1]:.1f}" + (f", then {kind}" if kind else "")
    if pf:
        return f"from a design scored {pf[0]:.1f}: {kind or 'modified'}"
    return "new random design"


# ---------------------------------------------------------------- 挑选器

def _scored(hist: list[dict]) -> list[dict]:
    return sorted([r for r in hist if r["status"] == "mpc"], key=lambda r: -float(r["fitness"]))


class Base:
    name = "base"

    def __init__(self, key: str):
        self.key = key                                   # 随机数的键（在线 = 种子，回放 = 运行名）

    def predict(self, gen: int, cands: list[dict], hist: list[dict]) -> tuple[np.ndarray, list[dict]]:
        raise NotImplementedError


class RandomSel(Base):
    name = "random"

    def predict(self, gen, cands, hist):
        rng = random.Random(f"{self.key}|{gen}|sel-random")
        return np.array([rng.random() for _ in cands]), []


class ParentSel(Base):
    name = "parent"

    def predict(self, gen, cands, hist):
        s = _scored(hist)
        med = float(np.median([float(r["fitness"]) for r in s])) if s else 0.0
        return np.array([max(c["pfit"]) if c["pfit"] else med for c in cands]), []


class RegressSel(Base):
    name = "regress"
    lam = 1.0

    def predict(self, gen, cands, hist):
        s = _scored(hist)
        if len(s) < 8:
            return RandomSel(self.key).predict(gen, cands, hist)
        X = np.array([features(r["design"]) for r in s], dtype=float)
        y = np.array([float(r["fitness"]) for r in s])
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = np.c_[np.ones(len(X)), (X - mu) / sd]
        reg = self.lam * np.eye(Z.shape[1])
        reg[0, 0] = 0.0
        w = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        C = np.array([features(c["design"]) for c in cands], dtype=float)
        return np.c_[np.ones(len(C)), (C - mu) / sd] @ w, []


class KnnSel(Base):
    name = "knn"
    k = 5

    def predict(self, gen, cands, hist):
        s = _scored(hist)
        if not s:
            return RandomSel(self.key).predict(gen, cands, hist)
        items = [_items(r["design"]) for r in s]
        y = np.array([float(r["fitness"]) for r in s])
        out = []
        for c in cands:
            ci = _items(c["design"])
            j = np.array([_jaccard(ci, it) for it in items])
            top = np.argsort(-j, kind="stable")[:self.k]
            w = j[top] + 1e-6
            out.append(float(np.sum(w * y[top]) / np.sum(w)))
        return np.array(out), []


def pool_rows(scored: list[dict], cands: list[dict], K: int) -> list[dict]:
    """经验池：最好的一半 → 与候选最相似的（轮流取，到 3/4）→ 其余按名次均匀取；按得分从高到低。"""
    if len(scored) <= K:
        return list(scored)
    chosen, used = [], set()

    def add(r):
        if r["design"] not in used and len(chosen) < K:
            used.add(r["design"])
            chosen.append(r)

    for r in scored[:(K + 1) // 2]:
        add(r)
    items = [_items(r["design"]) for r in scored]
    lists = []
    for c in cands:
        ci = _items(c["design"])
        js = [_jaccard(ci, it) for it in items]
        lists.append(sorted(range(len(scored)), key=lambda k: (-js[k], k)))
    target, ptr = (K * 3) // 4, [0] * len(lists)
    while len(chosen) < target and any(p < len(scored) for p in ptr):
        for li, order in enumerate(lists):
            while ptr[li] < len(order) and scored[order[ptr[li]]]["design"] in used:
                ptr[li] += 1
            if ptr[li] < len(order) and len(chosen) < target:
                add(scored[order[ptr[li]]])
                ptr[li] += 1
    rest = [r for r in scored if r["design"] not in used]
    need = K - len(chosen)
    if need > 0 and rest:
        for k in sorted(set(int(round(x)) for x in np.linspace(0, len(rest) - 1, need))):
            add(rest[k])
    return sorted(chosen, key=lambda r: -float(r["fitness"]))


def _cand_line(label: str, c: dict) -> str:
    s = _summary(c["design"])
    return f"{label} | {origin_text(c)} | {s['modules']} modules, {s['joints']} joints, {s['feet']} feet | {c['design']}"


def user_prompt(pool: list[dict], n_scored: int, best: float, cands: list[dict], perm: list[int],
                mode: str, m: int) -> str:
    L = [f"## Designs already simulated ({len(pool)} of the {n_scored} simulated so far; best score {best:.1f})"]
    L += [f"score {float(r['fitness']):.1f} | {r['modules']} modules, {r['joints']} joints, {r['feet']} feet | "
          f"{r['design']}" for r in pool]
    L += ["", f"## New candidates, not simulated yet ({len(cands)})"]
    L += [_cand_line(f"C{k + 1}", cands[p]) for k, p in enumerate(perm)]
    L += ["", "## Your task"]
    if mode == "rank":
        L += [f"Rank all {len(cands)} candidates by their predicted score, best first. Answer with exactly these two lines:",
              "ANALYSIS: what separates high-scoring from low-scoring bodies here (1-3 sentences)",
              "RANKING: every candidate label exactly once, best first, separated by commas (for example: C3, C1, C2, ...)"]
    else:
        L += [f"Select the {m} candidates you predict will score highest. Answer with exactly these two lines:",
              "ANALYSIS: what separates high-scoring from low-scoring bodies here (1-3 sentences)",
              f"SELECTED: exactly {m} candidate labels, best first, separated by commas"]
    return "\n".join(L)


def parse_answer(text: str, n: int, mode: str, m: int) -> tuple[list[int], str, str]:
    """-> (读到的标签序号 0..n-1，按回复的顺序；outcome；ANALYSIS)。"""
    key = "RANKING" if mode == "rank" else "SELECTED"
    hits = re.findall(rf"^\W*{key}\W*:\s*(.+)$", text or "", flags=re.M | re.I)
    ana = re.findall(r"^\W*ANALYSIS\W*:\s*(.+)$", text or "", flags=re.M | re.I)
    got = []
    if hits:
        for x in re.findall(r"C\s*(\d+)", hits[-1], flags=re.I):
            k = int(x) - 1
            if 0 <= k < n and k not in got:
                got.append(k)
    if mode == "half":
        got = got[:m]
    want = n if mode == "rank" else m
    outcome = "ok" if len(got) == want else ("partial" if got else "fail")
    return got, outcome, (ana[-1].strip()[:300] if ana else "")


def positions(got_labels: list[int], perm: list[int], n: int) -> np.ndarray:
    """一次回复 → 每个候选（原序号）的名次（0 = 最好）；没列出的并列排在后面（取平均名次）。"""
    pos = np.full(n, np.nan)
    for r, k in enumerate(got_labels):
        pos[perm[k]] = r
    miss = np.isnan(pos)
    if miss.any():
        pos[miss] = (len(got_labels) + n - 1) / 2.0
    return pos


class LLMSel(Base):
    name = "llm"

    def __init__(self, key: str, llm, max_modules: int = 8, pool: int = 16, mode: str = "rank", votes: int = 3,
                 workers: int = 8, m: int | None = None, log_path: Path | None = None):
        super().__init__(key)
        self.llm, self.pool, self.mode, self.votes, self.workers, self.m = llm, pool, mode, votes, workers, m
        self.system = system_prompt(max_modules)
        self.log_path = log_path
        self.logged = set()
        if log_path and log_path.exists():
            with open(log_path, newline="", encoding="utf-8") as f:
                self.logged = {(r["gen"], r["vote"]) for r in csv.DictReader(f)}

    def prompts(self, gen: int, cands: list[dict], hist: list[dict], m: int) -> list[tuple[list[int], str]]:
        s = _scored(hist)
        pool = pool_rows(s, cands, self.pool)
        best = float(s[0]["fitness"]) if s else 0.0
        out = []
        for v in range(self.votes):
            perm = list(range(len(cands)))
            random.Random(f"{self.key}|{gen}|sel-shuffle|{v}").shuffle(perm)
            out.append((perm, user_prompt(pool, len(s), best, cands, perm, self.mode, m)))
        return out

    def predict(self, gen, cands, hist, m: int | None = None):
        n = len(cands)
        m = m or self.m or max(1, n // 2)
        jobs = self.prompts(gen, cands, hist, m)
        with ThreadPoolExecutor(max_workers=max(1, min(self.workers, len(jobs)))) as ex:
            recs = list(ex.map(lambda vj: self.llm.call(vj[1][1], gen, vj[0], system=self.system, tag=f"sel{vj[0]}"),
                               enumerate(jobs)))
        P, calls = [], []
        for v, ((perm, _), rec) in enumerate(zip(jobs, recs)):
            got, outcome, ana = parse_answer(rec.get("response", ""), n, self.mode, m)
            if got:
                P.append(positions(got, perm, n))
            u = rec.get("usage", {})
            calls.append({"gen": gen, "vote": v, "mode": self.mode, "pool": self.pool, "n_cand": n, "m": m,
                          "labels": " ".join(str(cands[p]["idx"]) for p in perm),
                          "parsed": " ".join(str(cands[perm[k]]["idx"]) for k in got), "outcome": outcome,
                          "n_parsed": len(got), "llm_in": u.get("input_tokens", ""), "llm_out": u.get("output_tokens", ""),
                          "llm_s": rec.get("seconds", ""), "cached": rec.get("cached", ""), "analysis": ana})
        self._log(calls)
        if not P:                                         # every vote failed: random order (reported in select.csv)
            return RandomSel(self.key).predict(gen, cands, hist)[0], calls
        return -np.mean(P, axis=0), calls

    def _log(self, calls: list[dict]) -> None:
        if not self.log_path:
            return
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CALL_FIELDS, extrasaction="ignore")
            if new:
                w.writeheader()
            for c in calls:
                if (str(c["gen"]), str(c["vote"])) in self.logged:
                    continue
                w.writerow(c)
                self.logged.add((str(c["gen"]), str(c["vote"])))


def family(source: str) -> str:
    """候选怎么来的 → 修改的大类（--sel_strata family 与分析用）。"""
    if (source or "").startswith("crossover"):
        return "crossover"
    k = re.findall(r"mutate:(\w+)|\[(\w+)\]", source or "")
    k = (k[0][0] or k[0][1]) if k else ""
    if k in ("remove_part", "remove_foot"):
        return "remove"
    if k in ("add_module", "add_hub", "add_foot", "parallel", "mirror"):
        return "add"
    if k in ("q0", "roll"):
        return "tweak"
    return "other"


def quotas(counts: dict, n: int) -> dict:
    """n 个名额按 counts 的比例分（最大余数法；余数相同时按类名）。"""
    tot = sum(counts.values())
    if tot == 0 or n <= 0:
        return {k: 0 for k in counts}
    raw = {k: n * c / tot for k, c in counts.items()}
    q = {k: int(v) for k, v in raw.items()}
    for k in sorted(counts, key=lambda k: (-(raw[k] - q[k]), k))[:n - sum(q.values())]:
        q[k] += 1
    return q


def make_selector(a, out: Path, llm=None) -> Base:
    key = f"hyblock-search|{a.seed}"
    if a.select == "llm":
        return LLMSel(key, llm, a.max_modules, a.sel_pool, a.sel_mode, a.sel_votes, a.llm_workers,
                      log_path=out / "select.csv")
    return {"random": RandomSel, "parent": ParentSel, "regress": RegressSel, "knn": KnnSel}[a.select](key)


def order_of(scores: np.ndarray, key: str, gen: int) -> list[int]:
    """预测值从高到低；并列时按固定的随机次序（不偏向小序号）。"""
    rng = random.Random(f"{key}|{gen}|sel-tie")
    tie = [rng.random() for _ in range(len(scores))]
    return sorted(range(len(scores)), key=lambda k: (-float(scores[k]), tie[k]))


def choose(a, gen: int, cands: list[dict], hist: list[dict], selector: Base, n_eval: int) -> tuple[dict, list]:
    """-> ({候选位置: (名次, 名额, 预测值)}：只含送 MPC 的；另含 'all' 键 = 所有候选的 (名次, 预测值)），LLM 调用记录。"""
    n = len(cands)
    if n <= n_eval:
        return {"picked": {p: ("", "all", "") for p in range(n)}, "all": {}}, []
    if isinstance(selector, LLMSel):
        scores, calls = selector.predict(gen, cands, hist, m=max(1, n_eval - a.rand_slots))
    else:
        scores, calls = selector.predict(gen, cands, hist)
    order = order_of(scores, selector.key, gen)
    rank = {p: k + 1 for k, p in enumerate(order)}
    R = min(a.rand_slots, n_eval)
    rand = random.Random(f"hyblock-search|{a.seed}|{gen}|sel-rand").sample(range(n), R)
    picked = {p: (rank[p], "rand", round(float(scores[p]), 3)) for p in rand}
    if getattr(a, "sel_strata", "none") == "family":
        rest = [p for p in order if p not in picked]
        fams = {p: family(cands[p].get("source", "")) for p in rest}
        q = quotas({f: sum(fams[p] == f for p in rest) for f in set(fams.values())}, n_eval - len(picked))
        for p in rest:                                   # within each family: best predicted first
            if q.get(fams[p], 0) > 0:
                picked[p] = (rank[p], "top", round(float(scores[p]), 3))
                q[fams[p]] -= 1
    for p in order:
        if len(picked) >= n_eval:
            break
        if p not in picked:
            picked[p] = (rank[p], "top", round(float(scores[p]), 3))
    return {"picked": picked, "all": {p: (rank[p], round(float(scores[p]), 3)) for p in range(n)}}, calls
