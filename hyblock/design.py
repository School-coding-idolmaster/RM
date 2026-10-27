"""Hyblock 设计的文本格式、镜像展开与合法性检查（v5：水力君模块 + 被动分支件 + 被动脚）。

一行文本描述一个设计（LLM 输出、随机生成器、进化变异、日志共用），各项用 \";\" 分隔：

    h0=root; m1=h0.zp@90 q60/120 mirror; m2=h0.xp@0; m3=m2.P2@90 q120/90; foot=m1.P2; foot=m3.P2

- 名字的首字母决定零件类型：``m…`` = 水力君模块（2 个关节），``h…`` = 被动分支件（立方体，无关节）
- ``m0=root`` / ``h0=root``：根零件（躯干）。模块作根时方块底面朝下、P1 朝前；分支件作根时 ym 面朝下、xp 朝前
- ``m<k>=<父>.<口>@<滚转>``：模块 k 用它的 P1 板面对面接到父零件的 <口> 上，绕法向滚转 0/90/180/270°；
  它的 P2 板留给后续连接。可选项：``q<a>/<b>``（两个关节的初始角，默认 90/90，范围 −30…150；
  ``q60`` = 两个都是 60）、``x2``（两个模块同向并联）、``mirror``（见下）
- ``h<k>=<父>.<口>@<滚转>``：分支件 k 用它的 xm 面接到父零件上。可选 ``mirror``
- ``mirror``：只能用在父口为分支件的 zp / zm 面时；在同一分支件的另一侧面自动接一份镜像的整棵子树（含其上的脚）
- ``foot=<零件>.<口>``：在空闲口上装被动脚（终端零件）
口：模块 P1、P2（两端板的外面；子模块的 P1 已用于连接父零件）；分支件 xp、xm、yp、ym、zp、zm（子分支件的 xm 已占用）。
零件编号只是名字，展开后按\"父在前\"重新编号；镜像出来的零件名字加撇号（m1'）。
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace

import numpy as np

from hyblock.model import ModuleSpec, Node, Robot, load_spec

KINDS = {"m": "module", "h": "hub"}
PORTS = {"module": ("P1", "P2"), "hub": ("xp", "xm", "yp", "ym", "zp", "zm")}
_PORT_LOOKUP = {p.lower(): p for ps in PORTS.values() for p in ps}
ROLLS = (0, 90, 180, 270)
MIRROR_PORTS = {"zp": "zm", "zm": "zp"}
DEFAULT_Q = 90.0


class DesignError(ValueError):
    """设计无效。``kind`` 是无效原因的类别，用于统计无效设计率。"""

    def __init__(self, kind: str, message: str):
        super().__init__(f"{kind}: {message}")
        self.kind = kind


@dataclass
class PartDecl:
    name: str
    parent: str | None = None
    parent_port: str = ""
    roll: int = 0
    n_par: int = 1
    q0_deg: tuple = (DEFAULT_Q, DEFAULT_Q)
    mirror: bool = False

    @property
    def kind(self) -> str:
        return KINDS[self.name[0]]


@dataclass
class Design:
    parts: list[PartDecl]
    feet: list[tuple[str, str]] = field(default_factory=list)

    def copy(self) -> "Design":
        return Design([replace(p) for p in self.parts], list(self.feet))


@dataclass
class Expanded:
    """镜像展开后的设计：可直接交给 Robot（静力学）和 MJCF 生成器。"""
    nodes: list[Node]
    feet: list[tuple[int, str]]
    labels: list[str]
    parity: list[int]

    def module_count(self) -> int:
        return sum(n.n_par for n in self.nodes if n.kind == "module")

    def hub_count(self) -> int:
        return sum(n.kind == "hub" for n in self.nodes)

    def joint_count(self) -> int:
        return 2 * sum(n.kind == "module" for n in self.nodes)


# ---------------------------------------------------------------- 文本格式

_ITEM_ROOT = re.compile(r"^([mh]\d+)\s*=\s*root\b(.*)$", re.I)
_ITEM_PART = re.compile(r"^([mh]\d+)\s*=\s*([mh]\d+)\s*\.\s*(\w+)\s*(?:@\s*(-?\d+))?(.*)$", re.I)
_ITEM_FOOT = re.compile(r"^foot\s*=\s*([mh]\d+)\s*\.\s*(\w+)\s*$", re.I)


def _port(name: str) -> str:
    p = _PORT_LOOKUP.get(name.strip().lower())
    if p is None:
        raise DesignError("unknown_port", f"未知口 {name!r}")
    return p


def _options(text: str, decl: PartDecl) -> None:
    for tok in text.replace(",", " ").split():
        t = tok.lower()
        if t in ("x2", "par2", "parallel"):
            decl.n_par = 2
        elif t in ("x1", "par1"):
            decl.n_par = 1
        elif t == "mirror":
            decl.mirror = True
        elif m := re.fullmatch(r"q(-?\d+(?:\.\d+)?)(?:/(-?\d+(?:\.\d+)?))?(?:deg)?", t):
            a = float(m.group(1))
            decl.q0_deg = (a, float(m.group(2)) if m.group(2) is not None else a)
        else:
            raise DesignError("syntax", f"无法识别的选项 {tok!r}")


def parse(text: str) -> Design:
    if text is None or not text.strip():
        raise DesignError("syntax", "空设计")
    text = text.strip().strip("`").strip()
    if text.upper().startswith("DESIGN:"):
        text = text[7:]
    parts, feet, names = [], [], set()
    for raw in text.split(";"):
        item = raw.strip()
        if not item:
            continue
        if m := _ITEM_FOOT.match(item):
            feet.append((m.group(1).lower(), _port(m.group(2))))
            continue
        if m := _ITEM_ROOT.match(item):
            decl = PartDecl(m.group(1).lower())
            _options(m.group(2), decl)
        elif m := _ITEM_PART.match(item):
            roll = int(m.group(4)) if m.group(4) is not None else 0
            decl = PartDecl(m.group(1).lower(), m.group(2).lower(), _port(m.group(3)), roll % 360)
            _options(m.group(5), decl)
        else:
            raise DesignError("syntax", f"无法解析 {item!r}")
        if decl.name in names:
            raise DesignError("duplicate_name", f"零件名重复 {decl.name}")
        names.add(decl.name)
        parts.append(decl)
    return Design(parts, feet)


def _fmt(x: float) -> str:
    return f"{int(x)}" if float(x).is_integer() else f"{x:g}"


def to_text(d: Design) -> str:
    items = []
    for p in d.parts:
        opts = []
        if p.kind == "module":
            if p.n_par == 2:
                opts.append("x2")
            if tuple(p.q0_deg) != (DEFAULT_Q, DEFAULT_Q):
                a, b = p.q0_deg
                opts.append(f"q{_fmt(a)}" if a == b else f"q{_fmt(a)}/{_fmt(b)}")
        if p.mirror:
            opts.append("mirror")
        head = f"{p.name}=root" if p.parent is None else f"{p.name}={p.parent}.{p.parent_port}@{p.roll}"
        items.append(" ".join([head] + opts))
    items += [f"foot={n}.{port}" for n, port in d.feet]
    return "; ".join(items)


# ---------------------------------------------------------------- 镜像

_M = np.diag([1.0, 1.0, -1.0, 1.0])
_MIRROR_CACHE: dict = {}


def mirror_port(port: str) -> str:
    return MIRROR_PORTS.get(port, port)


def mirror_connection(spec: ModuleSpec, parent_kind: str, parent_port: str, child_kind: str, roll: int):
    """关于父刚体 xy 平面（z → −z）的镜像连接 -> (镜像口, 镜像滚转)。零件都关于自己的 xy 平面对称，
    关节轴沿 z，所以镜像后关节角不变。"""
    key = (id(spec), parent_kind, parent_port, child_kind, roll)
    if key not in _MIRROR_CACHE:
        target = _M @ spec.connection_transform(parent_kind, parent_port, child_kind, roll) @ _M
        pm = mirror_port(parent_port)
        hit = [r2 for r2 in ROLLS
               if np.allclose(spec.connection_transform(parent_kind, pm, child_kind, r2), target, atol=1e-9)]
        if len(hit) != 1:
            raise RuntimeError(f"no unique mirror image for {key}")
        _MIRROR_CACHE[key] = (pm, hit[0])
    return _MIRROR_CACHE[key]


# ---------------------------------------------------------------- 展开与检查


def expand(d: Design, spec: ModuleSpec | None = None) -> Expanded:
    spec = spec or load_spec()
    if not d.parts:
        raise DesignError("syntax", "没有零件")
    by_name = {p.name: p for p in d.parts}
    roots = [p for p in d.parts if p.parent is None]
    if len(roots) != 1:
        raise DesignError("root", f"必须恰好有一个根零件（现在 {len(roots)} 个）")
    children: dict[str, list[PartDecl]] = {}
    for p in d.parts:
        if p.parent is not None:
            if p.parent not in by_name:
                raise DesignError("unknown_parent", f"{p.name} 的父零件 {p.parent} 不存在")
            if p.parent_port not in PORTS[by_name[p.parent].kind]:
                raise DesignError("unknown_port", f"{p.parent} 没有口 {p.parent_port}")
            children.setdefault(p.parent, []).append(p)
        if p.roll not in ROLLS:
            raise DesignError("roll", f"{p.name} 的滚转 {p.roll} 不是 0/90/180/270")
        if p.kind == "module":
            for q in p.q0_deg:
                if not spec.range_min - 1e-9 <= math.radians(q) <= spec.range_max + 1e-9:
                    raise DesignError("joint_range", f"{p.name} 的初始角 {q} 超出关节范围")
            if not 1 <= p.n_par <= spec.max_parallel:
                raise DesignError("parallel", f"{p.name} 并联数 {p.n_par} 超限")
        elif p.n_par != 1:
            raise DesignError("parallel", f"分支件 {p.name} 不能并联")
        if p.mirror and (p.parent is None or by_name.get(p.parent) is None
                         or by_name[p.parent].kind != "hub" or p.parent_port not in MIRROR_PORTS):
            raise DesignError("mirror", f"{p.name}: mirror 只能用在分支件的 zp / zm 面上")
    feet_by: dict[str, list[str]] = {}
    for name, port in d.feet:
        if name not in by_name:
            raise DesignError("unknown_parent", f"脚装在不存在的零件 {name} 上")
        if port not in PORTS[by_name[name].kind]:
            raise DesignError("unknown_port", f"{name} 没有口 {port}")
        feet_by.setdefault(name, []).append(port)

    nodes, feet, labels, parity = [], [], [], []
    seen: set[str] = set()

    def inst(p: PartDecl, parent_idx: int, par: int, label: str):
        seen.add(p.name)
        if len(nodes) > 64:
            raise DesignError("module_cap", "展开后零件过多")
        q0 = tuple(math.radians(q) for q in p.q0_deg) if p.kind == "module" else ()
        if p.parent is None:
            node = Node(p.kind, -1, "", 0, p.n_par, q0)
        else:
            port, roll = p.parent_port, p.roll
            if par:
                port, roll = mirror_connection(spec, nodes[parent_idx].kind, port, p.kind, roll)
            node = Node(p.kind, parent_idx, port, roll, p.n_par, q0)
        idx = len(nodes)
        nodes.append(node)
        labels.append(label)
        parity.append(par)
        for port in feet_by.get(p.name, []):
            feet.append((idx, mirror_port(port) if par else port))
        for ch in children.get(p.name, []):
            inst(ch, idx, par, ch.name + ("'" if par else ""))
            if ch.mirror:
                inst(ch, idx, par ^ 1, ch.name + ("" if par else "'"))

    inst(roots[0], -1, 0, roots[0].name)
    if len(seen) != len(d.parts):
        raise DesignError("unknown_parent", "有零件没有连到根上")
    return Expanded(nodes, feet, labels, parity)


def build_robot(d: Design | str, spec: ModuleSpec | None = None, max_modules: int | None = None,
                **robot_kw) -> tuple[Robot, Expanded]:
    """解析 + 展开 + 用静力学模型做合法性检查（口冲突、零件数上限、脚的位置）。"""
    spec = spec or load_spec()
    if isinstance(d, str):
        d = parse(d)
    ex = expand(d, spec)
    try:
        robot = Robot(spec, ex.nodes, feet=ex.feet, max_modules=max_modules, **robot_kw)
    except ValueError as e:
        msg = str(e)
        kind = ("module_cap" if "模块数" in msg else "hub_cap" if "分支件数" in msg else "foot" if "脚" in msg
                else "port_conflict" if "口" in msg else "invalid")
        raise DesignError(kind, msg) from None
    return robot, ex


def summary(d: Design | str, spec: ModuleSpec | None = None) -> dict:
    """设计的简单结构指标（用于多样性统计和日志）。"""
    robot, ex = build_robot(d, spec, max_modules=10 ** 6, max_hubs=10 ** 6)
    depth = [0] * len(ex.nodes)
    for i, n in enumerate(ex.nodes):
        if n.parent >= 0:
            depth[i] = depth[n.parent] + 1
    return {"modules": ex.module_count(), "hubs": ex.hub_count(), "joints": ex.joint_count(),
            "feet": len(ex.feet), "mirrored": int(sum(ex.parity)),
            "parallel": sum(n.n_par == 2 for n in ex.nodes), "max_depth": max(depth),
            "root": ex.nodes[0].kind, "mass_kg": robot.total_mass()}
