"""Hyblock 设计的随机生成器与变异（随机搜索基线、进化算法的变异、模型验证的抽样都用它）。

生成过程按文法规则逐步加零件（与 prompts/system.txt 的规则一一对应）：
  r0 根零件（模块或分支件）；r1 串联（空闲板口 ← 新模块）；r2 加分支件（任意空闲口 ← 新分支件）；
  r3 分支（分支件的空闲面 ← 新模块）；r4 镜像（分支件 zp / zm 面，另一侧自动对称）；r5 并联 x2；r6 被动脚。
每一步都用 design.build_robot 检查，无效就重抽。
"""
from __future__ import annotations

import random

from hyblock.design import (DEFAULT_Q, MIRROR_PORTS, PORTS, ROLLS, Design, DesignError, PartDecl, build_robot,
                            mirror_port)
from hyblock.model import ModuleSpec, load_spec

Q_CHOICES = (-30, 0, 30, 60, 90, 120, 150)


def _attach_port(kind: str) -> str:
    return "P1" if kind == "module" else "xm"


def free_ports(d: Design, name: str) -> list[str]:
    """声明层面的空闲口（不含镜像出来的实例；最终由 build_robot 检查）。"""
    p = next(x for x in d.parts if x.name == name)
    used = set()
    if p.parent is not None:
        used.add(_attach_port(p.kind))
    for c in d.parts:
        if c.parent == name:
            used.add(c.parent_port)
            if c.mirror:
                used.add(mirror_port(c.parent_port))
    used |= {port for n, port in d.feet if n == name}
    return [x for x in PORTS[p.kind] if x not in used]


def _next_name(d: Design, prefix: str) -> str:
    return f"{prefix}{max(int(p.name[1:]) for p in d.parts) + 1}"


def _valid(d: Design, spec: ModuleSpec, max_modules: int) -> bool:
    try:
        build_robot(d, spec, max_modules=max_modules)
        return True
    except DesignError:
        return False


def _counts(d: Design, spec: ModuleSpec) -> tuple[int, int]:
    _, ex = build_robot(d, spec, max_modules=10 ** 6, max_hubs=10 ** 6)
    return ex.module_count(), ex.hub_count()


def _rand_q(rng: random.Random) -> tuple:
    return (float(rng.choice(Q_CHOICES)), float(rng.choice(Q_CHOICES)))


def add_part(d: Design, rng: random.Random, spec: ModuleSpec, max_modules: int, kind: str | None = None,
             p_hub: float = 0.25, p_mirror: float = 0.7, p_par: float = 0.1) -> Design | None:
    """r1–r5：在随机一个零件的空闲口上接一个新零件（模块或分支件）。返回新设计（无效则 None）。"""
    out = d.copy()
    parent = rng.choice(out.parts)
    free = free_ports(out, parent.name)
    if not free:
        return None
    kind = kind or ("hub" if rng.random() < p_hub else "module")
    port = rng.choice(free)
    decl = PartDecl(_next_name(out, "m" if kind == "module" else "h"), parent.name, port, rng.choice(ROLLS))
    if kind == "module":
        decl.q0_deg = _rand_q(rng)
        decl.n_par = 2 if rng.random() < p_par else 1
    decl.mirror = (parent.kind == "hub" and port in MIRROR_PORTS and mirror_port(port) in free
                   and rng.random() < p_mirror)
    out.parts.append(decl)
    return out if _valid(out, spec, max_modules) else None


def add_foot(d: Design, rng: random.Random, spec: ModuleSpec, max_modules: int) -> Design | None:
    """r6：在随机一个空闲口上装被动脚。"""
    out = d.copy()
    p = rng.choice(out.parts)
    free = free_ports(out, p.name)
    if not free:
        return None
    out.feet.append((p.name, rng.choice(free)))
    return out if _valid(out, spec, max_modules) else None


def random_design(rng: random.Random | int | None = None, max_modules: int = 6, spec: ModuleSpec | None = None,
                  n_modules: int | None = None, p_hub_root: float = 0.5, p_foot: float = 0.7,
                  max_tries: int = 300) -> Design:
    """随机设计：目标模块数在 1…max_modules 中均匀抽（镜像和并联按实际模块计；分支件不计入），
    然后给末端空闲口（模块的 P2 / 根模块的板、分支件的空闲面）以概率 p_foot 装脚。"""
    spec = spec or load_spec()
    rng = rng if isinstance(rng, random.Random) else random.Random(rng)
    target = n_modules or rng.randint(1, max_modules)
    if rng.random() < p_hub_root:
        d = Design([PartDecl("h0")])
    else:
        d = Design([PartDecl("m0", q0_deg=_rand_q(rng))])
    tries = 0
    while _counts(d, spec)[0] < target and tries < max_tries:
        tries += 1
        nd = add_part(d, rng, spec, max_modules)
        if nd is not None and _counts(nd, spec)[0] <= target:
            d = nd
    for p in list(d.parts):
        ports = free_ports(d, p.name)
        if p.kind == "module":
            cand_ports = [x for x in ports if x in ("P1", "P2")]
        else:
            cand_ports = [x for x in ports if x != "yp"]          # rarely put a foot on a hub's top face
        for port in cand_ports:
            if rng.random() < (p_foot if p.kind == "module" else p_foot / 3):
                cand = d.copy()
                cand.feet.append((p.name, port))
                if _valid(cand, spec, max_modules):
                    d = cand
    return d


# ---------------------------------------------------------------- 变异（进化算法用）

def _remove_leaf(d: Design, rng: random.Random, **_) -> Design | None:
    parents = {p.parent for p in d.parts}
    leaves = [p for p in d.parts if p.parent is not None and p.name not in parents]
    if not leaves:
        return None
    victim = rng.choice(leaves)
    out = d.copy()
    out.parts = [p for p in out.parts if p.name != victim.name]
    out.feet = [(n, port) for n, port in out.feet if n != victim.name]
    return out


def _tweak(field: str):
    def op(d: Design, rng: random.Random, **_) -> Design | None:
        out = d.copy()
        if field == "roll":
            cands = [p for p in out.parts if p.parent is not None]
        elif field in ("q0", "parallel"):
            cands = [p for p in out.parts if p.kind == "module"]
        else:                                                   # mirror
            kinds = {p.name: p.kind for p in out.parts}
            cands = [p for p in out.parts if p.parent is not None and kinds[p.parent] == "hub"
                     and p.parent_port in MIRROR_PORTS]
        if not cands:
            return None
        p = rng.choice(cands)
        if field == "roll":
            p.roll = rng.choice([r for r in ROLLS if r != p.roll])
        elif field == "q0":
            q = list(p.q0_deg)
            k = rng.randrange(2)
            q[k] = float(rng.choice([x for x in Q_CHOICES if x != q[k]]))
            p.q0_deg = tuple(q)
        elif field == "parallel":
            p.n_par = 3 - p.n_par
        else:
            p.mirror = not p.mirror
        return out
    return op


def _remove_foot(d: Design, rng: random.Random, **_) -> Design | None:
    if not d.feet:
        return None
    out = d.copy()
    out.feet.pop(rng.randrange(len(out.feet)))
    return out


MUTATIONS = {
    "add_module": lambda d, rng, spec, max_modules: add_part(d, rng, spec, max_modules, kind="module"),
    "add_hub": lambda d, rng, spec, max_modules: add_part(d, rng, spec, max_modules, kind="hub"),
    "remove_part": _remove_leaf,
    "roll": _tweak("roll"),
    "q0": _tweak("q0"),
    "parallel": _tweak("parallel"),
    "mirror": _tweak("mirror"),
    "add_foot": lambda d, rng, spec, max_modules: add_foot(d, rng, spec, max_modules),
    "remove_foot": _remove_foot,
}


def mutate(d: Design, rng: random.Random | int | None = None, max_modules: int = 6,
           spec: ModuleSpec | None = None, max_tries: int = 50) -> tuple[Design, str]:
    """随机选一种变异，直到得到一个合法的新设计。返回 (新设计, 变异名)。"""
    spec = spec or load_spec()
    rng = rng if isinstance(rng, random.Random) else random.Random(rng)
    names = list(MUTATIONS)
    for _ in range(max_tries):
        name = rng.choice(names)
        out = MUTATIONS[name](d, rng=rng, spec=spec, max_modules=max_modules)
        if out is not None and _valid(out, spec, max_modules):
            return out, name
    return d.copy(), "none"
