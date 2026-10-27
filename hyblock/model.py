"""Hyblock 仿真零件库与静力学模型（v5：水力君双连接件模块 + 被动分支件 + 被动脚）。

零件 = 若干刚体 + 连接刚体的转动关节 + 带坐标系的连接口：
  - 水力君模块（module）：中间方块 block，两端各一块铰接板 P1 / P2（各一个液压关节 a / b），只有两块板的外面能连接；
    作为子零件时用 P1 接父零件
  - 被动分支件（hub）：一个立方体，六个面（xp/xm/yp/ym/zp/zm）都能连接；作为子零件时用 xm 接父零件
  - 被动脚（foot）：装在空闲口上的终端球，只作为点质量计入
连接：两口面对面（法向相反、中心重合），绕法向滚转 0/90/180/270°；所有连接都是刚性的。

本模块做三件事（不依赖 MuJoCo）：
  1. 正运动学：每个零件每个刚体的世界位姿
  2. 静态关节力矩（自重 + 被动脚 + 末端负载），用于 ① 筛选和与 MuJoCo 重力项的交叉核对
  3. 力矩上限 = 压力 × 缸面积 × 常数等效力臂（伸出/缩回不对称）
关节约定：T_v = T_u · F_u · Rz(θ − θ_ref) · F_v⁻¹（u = 方块，v = 板）；θ 的广义力 = 作用在板上、绕铰接轴 +z 的力矩，
正方向 = 伸出侧。参数见 modules.yaml。
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

G = 9.81
GRAVITY = np.array([0.0, 0.0, -G])
DEFAULT_YAML = Path(__file__).with_name("modules.yaml")

# ---------------------------------------------------------------- 变换工具


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[1:3, 1:3] = [[c, -s], [s, c]]
    return T


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    T = np.eye(4)
    T[0, 0], T[0, 2], T[2, 0], T[2, 2] = c, s, -s, c
    return T


def trans(p) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = p
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R, p = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def port_frame(pos, normal, x_dir) -> np.ndarray:
    """口坐标系：z 轴 = 外法向，x 轴 = 切向（滚转 0° 的基准），原点 = 口中心（在刚体系中表示）。"""
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    x = np.asarray(x_dir, float)
    x = x - x.dot(n) * n
    x = x / np.linalg.norm(x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(n, x), n, pos
    return T


def ground_frame(x_dir=(1.0, 0.0, 0.0)) -> np.ndarray:
    return port_frame([0, 0, 0], [0, 0, 1], list(x_dir))


def _pt(T: np.ndarray, p) -> np.ndarray:
    return (T @ np.append(np.asarray(p, float), 1.0))[:3]


def _box_inertia(m: float, size) -> np.ndarray:
    dx, dy, dz = size
    return m / 12.0 * np.array([dy * dy + dz * dz, dx * dx + dz * dz, dx * dx + dy * dy])


# ---------------------------------------------------------------- 零件类型


@dataclass
class BodySpec:
    mass: float
    com: np.ndarray
    inertia: np.ndarray                       # 绕质心的主惯量（刚体坐标轴方向）
    boxes: list                               # 碰撞盒 [(center(3), half_size(3)), ...]


@dataclass
class JointSpec:
    name: str                                 # "a" / "b"
    u: str                                    # 方块
    v: str                                    # 板
    F_u: np.ndarray                           # 铰接坐标系在 u 中的位姿（z = 转轴）
    F_v: np.ndarray                           # 铰接坐标系在 v 中的位姿


@dataclass
class PartType:
    kind: str
    bodies: dict[str, BodySpec]
    joints: list[JointSpec]
    ports: dict[str, tuple[str, np.ndarray]]  # 口名 -> (刚体, 口坐标系)
    attach_port: str                          # 作为子零件时接父零件用的口
    root_body: str                            # 作为根时的基体（带自由关节，= 躯干）
    down_frame: np.ndarray                    # 作为根时朝下贴地的面（root_body 坐标系）
    mirror_ports: dict[str, str] = field(default_factory=dict)   # 关于 z → −z 镜像时口名的对应


class ModuleSpec:
    """零件库（名字沿用第一阶段的 ModuleSpec）。"""

    def __init__(self, cfg: dict, range_min_deg: float | None = None, range_max_deg: float | None = None):
        self.cfg = cfg
        jt = cfg["joint"]
        self.l_eff = jt["effective_lever_m"]
        self.theta_ref = math.radians(jt.get("theta_ref_deg", 90.0))
        self.range_min = math.radians(jt["range_min_deg"] if range_min_deg is None else range_min_deg)
        self.range_max = math.radians(jt["range_max_deg"] if range_max_deg is None else range_max_deg)
        cy = cfg["cylinder"]
        self.A_ext = math.pi * cy["bore_m"] ** 2 / 4
        self.A_ret = self.A_ext - math.pi * cy["rod_m"] ** 2 / 4
        par = cfg.get("parallel", {})
        self.max_parallel = par.get("max_count", 2)
        self.par_eta = par.get("eta", 0.8)
        foot = cfg.get("foot", {})
        self.foot_radius = foot.get("radius_m", 0.04)
        self.foot_mass = foot.get("mass_kg", 0.3)
        self.foot_offset = foot.get("center_offset_m", 0.0)
        self.actuator = cfg.get("actuator", {})
        self.sim = cfg.get("sim", {})
        self.tiers = cfg.get("tiers", {})
        self.max_hubs = cfg.get("hub", {}).get("max_count", 4)
        self.parts = {"module": self._module(cfg["module"]), "hub": self._hub(cfg["hub"])}

    # ---- 水力君模块
    def _module(self, c: dict) -> PartType:
        blk, pl = c["block"], c["plate"]
        size = np.array(blk["size_m"], float)
        half = size / 2
        y0, y1 = pl["y_range_m"]
        t = pl["thickness_m"]
        pl_size = np.array([t, y1 - y0, size[2]])
        bodies = {
            "block": BodySpec(blk["mass_kg"], np.array(blk["com_m"], float), _box_inertia(blk["mass_kg"], size),
                              [(np.zeros(3), half)]),
        }
        hinge = np.array(c["hinge_m"], float)
        F_front = trans(hinge)                                           # 板 +x = 方块 +x（向外）
        F_rear = trans([-hinge[0], hinge[1], hinge[2]]) @ rot_y(math.pi)  # 板 +x = 方块 −x（向外）
        ports = {}
        joints = []
        for name, plate, F in (("a", "P1", F_front), ("b", "P2", F_rear)):
            bodies[plate] = BodySpec(pl["mass_kg"], np.array(pl["com_m"], float), _box_inertia(pl["mass_kg"], pl_size),
                                     [(np.array([t / 2, (y0 + y1) / 2, 0.0]), pl_size / 2)])
            joints.append(JointSpec(name, "block", plate, F, np.eye(4)))
            ports[plate] = (plate, port_frame([t, (y0 + y1) / 2, 0.0], [1, 0, 0], [0, 1, 0]))
        down = port_frame([0, -half[1], 0], [0, -1, 0], [1, 0, 0])
        return PartType("module", bodies, joints, ports, "P1", "block", down)

    # ---- 被动分支件
    def _hub(self, c: dict) -> PartType:
        a = c["size_m"]
        h = a / 2
        body = BodySpec(c["mass_kg"], np.zeros(3), _box_inertia(c["mass_kg"], [a, a, a]),
                        [(np.zeros(3), np.array([h, h, h]))])
        ports = {
            "xp": ("hub", port_frame([h, 0, 0], [1, 0, 0], [0, 1, 0])),
            "xm": ("hub", port_frame([-h, 0, 0], [-1, 0, 0], [0, 1, 0])),
            "yp": ("hub", port_frame([0, h, 0], [0, 1, 0], [1, 0, 0])),
            "ym": ("hub", port_frame([0, -h, 0], [0, -1, 0], [1, 0, 0])),
            "zp": ("hub", port_frame([0, 0, h], [0, 0, 1], [1, 0, 0])),
            "zm": ("hub", port_frame([0, 0, -h], [0, 0, -1], [1, 0, 0])),
        }
        down = ports["ym"][1]
        return PartType("hub", {"hub": body}, [], ports, "xm", "hub", down, {"zp": "zm", "zm": "zp"})

    @classmethod
    def from_yaml(cls, path: str | Path, **kw) -> "ModuleSpec":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f), **kw)

    def tau_limits(self, pressure_pa: float) -> tuple[float, float]:
        """单个关节的 (伸出侧, 缩回侧) 力矩上限幅值。"""
        return pressure_pa * self.A_ext * self.l_eff, pressure_pa * self.A_ret * self.l_eff

    def connection_transform(self, parent_kind: str, parent_port: str, child_kind: str, roll_deg: int) -> np.ndarray:
        """子零件\"接口所在刚体\"相对父零件\"口所在刚体\"的位姿。"""
        _, P_p = self.parts[parent_kind].ports[parent_port]
        child = self.parts[child_kind]
        _, P_c = child.ports[child.attach_port]
        return P_p @ rot_x(math.pi) @ rot_z(math.radians(roll_deg)) @ inv_T(P_c)

    def foot_center(self, kind: str, port: str) -> tuple[str, np.ndarray]:
        body, P = self.parts[kind].ports[port]
        return body, P[:3, 3] + P[:3, 2] * self.foot_offset


# ---------------------------------------------------------------- 设计（零件树）


@dataclass
class Node:
    kind: str                     # "module" / "hub"
    parent: int                   # 父零件编号；-1 = 根
    parent_port: str = ""
    roll_deg: int = 0
    n_par: int = 1                # 模块并联数（1 或 2）
    q0: tuple = ()                # 初始关节角（rad），每个关节一个


class Robot:
    def __init__(self, spec: ModuleSpec, nodes: list[Node], pressure_MPa: float = 8.0, mass_scale: float = 1.0,
                 par_eta: float | None = None, feet: list[tuple[int, str]] | None = None,
                 ground_x=(1.0, 0.0, 0.0), max_modules: int | None = None, max_hubs: int | None = None,
                 tip: tuple[int, str] | None = None):
        self.spec, self.nodes = spec, nodes
        self.P = pressure_MPa * 1e6
        self.mass_scale = mass_scale
        self.par_eta = spec.par_eta if par_eta is None else par_eta
        self.feet = list(feet or [])
        self.ground_x = ground_x
        self.max_modules = spec.tiers.get("max_modules", 10 ** 9) if max_modules is None else max_modules
        self.max_hubs = spec.max_hubs if max_hubs is None else max_hubs
        self.tip = tip                                     # (节点, 口)：末端负载作用点
        self._validate()
        self.joint_index = [(i, js) for i, nd in enumerate(nodes) for js in spec.parts[nd.kind].joints]
        self._build_graph()

    @property
    def n_joints(self) -> int:
        return len(self.joint_index)

    def module_count(self) -> int:
        return sum(nd.n_par for nd in self.nodes if nd.kind == "module")

    def hub_count(self) -> int:
        return sum(nd.kind == "hub" for nd in self.nodes)

    # ---- 合法性检查
    def _validate(self) -> None:
        sp, nodes = self.spec, self.nodes
        if not nodes:
            raise ValueError("空设计")
        if self.module_count() > self.max_modules:
            raise ValueError("模块数超过上限")
        if self.hub_count() > self.max_hubs:
            raise ValueError("分支件数超过上限")
        used: set[tuple[int, str]] = set()
        for i, nd in enumerate(nodes):
            if nd.kind not in sp.parts:
                raise ValueError(f"未知零件 {nd.kind}")
            part = sp.parts[nd.kind]
            if (nd.parent < 0) != (i == 0):
                raise ValueError("必须且只能有节点 0 为根")
            if nd.parent >= i:
                raise ValueError("节点必须按拓扑序排列（父在前）")
            if nd.roll_deg not in (0, 90, 180, 270):
                raise ValueError("roll 只能取 0/90/180/270")
            if nd.kind == "module" and not 1 <= nd.n_par <= sp.max_parallel:
                raise ValueError("并联数超限")
            if nd.kind == "hub" and nd.n_par != 1:
                raise ValueError("分支件不能并联")
            if len(nd.q0) != len(part.joints):
                raise ValueError("初始关节角个数不对")
            for q in nd.q0:
                if not sp.range_min - 1e-9 <= q <= sp.range_max + 1e-9:
                    raise ValueError("初始关节角超出范围")
            if nd.parent >= 0:
                pk = nodes[nd.parent].kind
                if nd.parent_port not in sp.parts[pk].ports:
                    raise ValueError(f"未知口 {nd.parent_port}")
                for key in ((nd.parent, nd.parent_port), (i, part.attach_port)):
                    if key in used:
                        raise ValueError("同一个口不能接两个零件")
                    used.add(key)
        for (j, port) in self.feet:
            if not 0 <= j < len(nodes):
                raise ValueError("被动脚挂在不存在的零件上")
            if port not in sp.parts[nodes[j].kind].ports:
                raise ValueError(f"未知口 {port}")
            if (j, port) in used:
                raise ValueError("被动脚只能装在空闲口上")
            used.add((j, port))

    # ---- 刚体图（用于找关节下游）
    def _build_graph(self) -> None:
        sp = self.spec
        adj: dict[tuple, list] = {}

        def link(a, b, tag):
            adj.setdefault(a, []).append((b, tag))
            adj.setdefault(b, []).append((a, tag))

        for i, nd in enumerate(self.nodes):
            part = sp.parts[nd.kind]
            for b in part.bodies:
                adj.setdefault((i, b), [])
            for js in part.joints:
                link((i, js.u), (i, js.v), ("joint", i, js.name))
            if nd.parent >= 0:
                pk = self.nodes[nd.parent].kind
                pbody = sp.parts[pk].ports[nd.parent_port][0]
                link((nd.parent, pbody), (i, part.ports[part.attach_port][0]), ("attach", i))
        self.adj = adj
        self.feet_on: dict[tuple[int, str], list[str]] = {}
        for (j, port) in self.feet:
            body = sp.parts[self.nodes[j].kind].ports[port][0]
            self.feet_on.setdefault((j, body), []).append(port)

    def _side(self, start, blocked_tag) -> list:
        seen, dq = {start}, deque([start])
        while dq:
            a = dq.popleft()
            for b, tag in self.adj[a]:
                if tag == blocked_tag or b in seen:
                    continue
                seen.add(b)
                dq.append(b)
        return list(seen)

    def base_body(self, i: int) -> str:
        part = self.spec.parts[self.nodes[i].kind]
        return part.root_body if self.nodes[i].parent < 0 else part.ports[part.attach_port][0]

    # ---- 正运动学
    def fk(self, q) -> dict:
        """返回 {(节点, 刚体): 世界位姿}。q 按 joint_index 排列。"""
        sp = self.spec
        T: dict[tuple[int, str], np.ndarray] = {}
        k = 0
        for i, nd in enumerate(self.nodes):
            part = sp.parts[nd.kind]
            if nd.parent < 0:
                T[(i, part.root_body)] = ground_frame(self.ground_x) @ rot_x(math.pi) @ inv_T(part.down_frame)
            else:
                pk = self.nodes[nd.parent].kind
                pbody = sp.parts[pk].ports[nd.parent_port][0]
                T[(i, part.ports[part.attach_port][0])] = (
                    T[(nd.parent, pbody)] @ sp.connection_transform(pk, nd.parent_port, nd.kind, nd.roll_deg))
            pending = list(part.joints)
            while pending:
                for js in list(pending):
                    th = q[k + part.joints.index(js)] - sp.theta_ref
                    if (i, js.u) in T and (i, js.v) not in T:
                        T[(i, js.v)] = T[(i, js.u)] @ js.F_u @ rot_z(th) @ inv_T(js.F_v)
                    elif (i, js.v) in T and (i, js.u) not in T:
                        T[(i, js.u)] = T[(i, js.v)] @ js.F_v @ rot_z(-th) @ inv_T(js.F_u)
                    else:
                        continue
                    pending.remove(js)
            k += len(part.joints)
        return T

    # ---- 质量点
    def _point_masses(self, T, node: int, body: str):
        sp, nd = self.spec, self.nodes[node]
        b = sp.parts[nd.kind].bodies[body]
        out = [(b.mass * self.mass_scale * nd.n_par, _pt(T[(node, body)], b.com))]
        for port in self.feet_on.get((node, body), []):
            out.append((sp.foot_mass, _pt(T[(node, body)], sp.foot_center(nd.kind, port)[1])))
        return out

    def total_mass(self) -> float:
        sp = self.spec
        m = sum(sum(b.mass for b in sp.parts[nd.kind].bodies.values()) * self.mass_scale * nd.n_par
                for nd in self.nodes)
        return m + sp.foot_mass * len(self.feet)

    def _tip_point(self, T):
        if self.tip is None:
            return None
        j, port = self.tip
        body, P = self.spec.parts[self.nodes[j].kind].ports[port]
        return _pt(T[(j, body)], P[:3, 3])

    def potential_energy(self, q, m_load: float = 0.0) -> float:
        T = self.fk(q)
        U = sum(m * G * c[2] for key in T for m, c in self._point_masses(T, *key))
        tip = self._tip_point(T)
        return U + (m_load * G * tip[2] if tip is not None else 0.0)

    # ---- 静态关节力矩
    def static_torques(self, q, m_load: float = 0.0) -> np.ndarray:
        """每个关节坐标 θ 上需要的静态广义力（N·m）= 维持姿态所需的执行器力矩。"""
        T = self.fk(q)
        tip = self._tip_point(T)
        root = (0, self.base_body(0))
        tau = np.zeros(self.n_joints)
        for k, (i, js) in enumerate(self.joint_index):
            W = T[(i, js.u)] @ js.F_u
            z_w, p_w = W[:3, 2], W[:3, 3]
            tag = ("joint", i, js.name)
            v_side = self._side((i, js.v), tag)
            moving_v = root not in v_side
            down = v_side if moving_v else self._side((i, js.u), tag)
            M = np.zeros(3)
            for key in down:
                for m, c in self._point_masses(T, *key):
                    M += np.cross(c - p_w, m * GRAVITY)
            if tip is not None and (self.tip[0], self.spec.parts[self.nodes[self.tip[0]].kind]
                                    .ports[self.tip[1]][0]) in down:
                M += np.cross(tip - p_w, m_load * GRAVITY)
            hold = -M @ z_w                                   # 作用在运动侧、绕 +z 的保持力矩
            tau[k] = hold if moving_v else -hold
        return tau

    # ---- 力矩上限
    def limits(self) -> tuple[np.ndarray, np.ndarray]:
        lp, ln = self.spec.tau_limits(self.P)
        pos, neg = np.zeros(self.n_joints), np.zeros(self.n_joints)
        for k, (i, _) in enumerate(self.joint_index):
            n = self.nodes[i].n_par
            f = n * (self.par_eta if n > 1 else 1.0)
            pos[k], neg[k] = lp * f, ln * f
        return pos, neg

    def check(self, q, m_load: float = 0.0) -> dict:
        q = np.asarray(q, float)
        in_range = bool(np.all((q >= self.spec.range_min - 1e-12) & (q <= self.spec.range_max + 1e-12)))
        tau = self.static_torques(q, m_load)
        lp, ln = self.limits()
        util = np.where(tau >= 0, tau / lp, -tau / ln)
        return {"tau": tau, "limit_pos": lp, "limit_neg": ln, "utilization": util,
                "in_range": in_range, "feasible": in_range and bool(np.all(util <= 1.0))}

    def q0(self) -> np.ndarray:
        return np.array([q for nd in self.nodes for q in nd.q0], float)


def load_spec(path: str | Path | None = None, **kw) -> ModuleSpec:
    """kw 可覆盖关节范围：range_min_deg / range_max_deg。"""
    return ModuleSpec.from_yaml(path or DEFAULT_YAML, **kw)
