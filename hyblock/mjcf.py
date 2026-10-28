"""Hyblock 设计 -> MuJoCo MJCF（v5：水力君模块 + 被动分支件 + 被动脚）。

- 每个零件按 model.py 的刚体/关节展开：接父零件的刚体是\"基体\"（根零件为方块或分支件本身，带自由关节 = 躯干），
  其余刚体经 hinge 逐级挂在下面；零件之间刚性连接（父子 body）
- 关节坐标 = 论文关节角 θ（hinge 的 ref = 初始角，range −30…150°），正方向 = 伸出侧，
  与 model.Robot.static_torques 的符号一致
- 执行器：general，ctrl/force 单位 N·m，范围 [−缩回上限, +伸出上限]（并联 ×2×η）；dyntype=filterexact（一阶滞后）。
  MPPI 的 (−1, 1) 动作在 mpc/mppi.py 里以 0 为中心分段映射到这个范围
- 速度上限：关节阻尼 b = 伸出侧力矩上限 / 最大速度（线性力矩–速度曲线，相当于阀控缸的流量限制）
- 被动脚：无关节子 body，球几何
- 根零件朝下的面贴地，+x 指向世界 −x（前进方向，与 RM 的奖励一致）
- 自碰撞（10-28 起，modules.yaml sim.self_collision）：机器人几何 contype=1, conaffinity=1，不相邻零件之间碰撞，
  父子刚体（直接相连的方块、板、脚）由 MuJoCo filterparent 排除；关掉时与 RM 相同（conaffinity=0，零件之间不碰撞）。
  初始姿态下零件互相穿透超过 sim.self_collision_tol_m 的设计无效（DesignError kind = self_collision）
- 地面 priority=1，接触摩擦由地面（地形）决定
- <custom> 里写入 hyblock_* 数值，mpc/mppi.py 据此切换到 Hyblock 域的动作映射、终止条件和初始化
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from hyblock.design import Design, DesignError, Expanded, build_robot
from hyblock.model import ModuleSpec, ground_frame, inv_T, load_spec, rot_x, rot_z

GROUND_X = (-1.0, 0.0, 0.0)        # 根零件的 +x 指向世界 −x（前进方向）
RGBA = {"block": "0.85 0.55 0.2 1", "P1": "0.3 0.3 0.35 1", "P2": "0.3 0.3 0.35 1", "hub": "0.5 0.6 0.7 1"}


def _f(v) -> str:
    return " ".join(f"{x:.10g}" for x in np.atleast_1d(v))


def _quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).reshape(-1))
    return q


def torso_box(spec: ModuleSpec, kind: str) -> list[float]:
    """躯干（根零件基体）碰撞盒的包围范围（基体坐标系）：xmin xmax ymin ymax zmin zmax。"""
    part = spec.parts[kind]
    lo = np.min([c - h for c, h in part.bodies[part.root_body].boxes], axis=0)
    hi = np.max([c + h for c, h in part.bodies[part.root_body].boxes], axis=0)
    return [lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]]


class _Builder:
    def __init__(self, spec, ex: Expanded, pressure_MPa, mass_scale, par_eta, max_speed_deg_s,
                 time_constant_s, armature, fixed_base, floor_friction, self_collision=None):
        self.spec, self.ex = spec, ex
        self.P = pressure_MPa * 1e6
        self.mass_scale, self.fixed_base = mass_scale, fixed_base
        self.par_eta = spec.par_eta if par_eta is None else par_eta
        act = spec.actuator
        self.wmax = math.radians(max_speed_deg_s if max_speed_deg_s is not None else act.get("max_speed_deg_s", 1185.0))
        self.tc = time_constant_s if time_constant_s is not None else act.get("time_constant_s", 0.03)
        self.armature = armature if armature is not None else act.get("armature", 0.3)
        self.floor_friction = floor_friction if floor_friction is not None else spec.sim.get("floor_friction", 1.0)
        self.self_collision = bool(spec.sim.get("self_collision", False) if self_collision is None else self_collision)
        self.children: dict[tuple[int, str], list[int]] = {}
        for j, n in enumerate(ex.nodes):
            if n.parent >= 0:
                pk = ex.nodes[n.parent].kind
                self.children.setdefault((n.parent, spec.parts[pk].ports[n.parent_port][0]), []).append(j)
        self.feet_on: dict[tuple[int, str], list[str]] = {}
        for j, port in ex.feet:
            self.feet_on.setdefault((j, spec.parts[ex.nodes[j].kind].ports[port][0]), []).append(port)
        self.actuators = []

    def _body_content(self, el, i: int, body: str):
        sp, nd = self.spec, self.ex.nodes[i]
        b = sp.parts[nd.kind].bodies[body]
        k_m = self.mass_scale * nd.n_par
        ET.SubElement(el, "inertial", pos=_f(b.com), mass=_f(b.mass * k_m), diaginertia=_f(b.inertia * k_m))
        for k, (c, h) in enumerate(b.boxes):
            ET.SubElement(el, "geom", name=f"n{i}_{body}_box{k}", type="box", pos=_f(c), size=_f(h),
                          rgba=RGBA.get(body, "0.6 0.6 0.6 1"))
        for port in self.feet_on.get((i, body), []):
            _, c = sp.foot_center(nd.kind, port)
            fb = ET.SubElement(el, "body", name=f"foot_n{i}_{port}", pos=_f(c))
            r = sp.foot_radius
            ET.SubElement(fb, "inertial", pos="0 0 0", mass=_f(sp.foot_mass),
                          diaginertia=_f([0.4 * sp.foot_mass * r * r] * 3))
            ET.SubElement(fb, "geom", name=f"foot_n{i}_{port}", type="sphere", size=_f(r), rgba="0.1 0.1 0.1 1")
        for j in self.children.get((i, body), []):
            self._part(el, j)

    def _part(self, parent_el, i: int, world_pose: np.ndarray | None = None):
        sp, nd = self.spec, self.ex.nodes[i]
        part = sp.parts[nd.kind]
        if world_pose is not None:
            base, T = part.root_body, world_pose
        else:
            base = part.ports[part.attach_port][0]
            T = sp.connection_transform(self.ex.nodes[nd.parent].kind, nd.parent_port, nd.kind, nd.roll_deg)
        els = {base: ET.SubElement(parent_el, "body", name=f"n{i}_{base}", pos=_f(T[:3, 3]),
                                   quat=_f(_quat(T[:3, :3])))}
        if world_pose is not None and not self.fixed_base:
            ET.SubElement(els[base], "freejoint", name="root")
        lp, ln = sp.tau_limits(self.P)
        f = nd.n_par * (self.par_eta if nd.n_par > 1 else 1.0)
        tau_ext, tau_ret = lp * f, ln * f
        pending = list(enumerate(part.joints))
        while pending:
            for k, js in list(pending):
                q0 = nd.q0[k]
                if js.u in els and js.v not in els:          # child = plate v (positive θ turns v about +z)
                    rel = js.F_u @ rot_z(q0 - sp.theta_ref) @ inv_T(js.F_v)
                    parent, child, Fc, sign = js.u, js.v, js.F_v, 1.0
                elif js.v in els and js.u not in els:        # child = block u (base is a plate)
                    rel = js.F_v @ rot_z(-(q0 - sp.theta_ref)) @ inv_T(js.F_u)
                    parent, child, Fc, sign = js.v, js.u, js.F_u, -1.0
                else:
                    continue
                pending.remove((k, js))
                cel = ET.SubElement(els[parent], "body", name=f"n{i}_{child}", pos=_f(rel[:3, 3]),
                                    quat=_f(_quat(rel[:3, :3])))
                ET.SubElement(cel, "joint", name=f"j{i}_{js.name}", type="hinge", pos=_f(Fc[:3, 3]),
                              axis=_f(sign * Fc[:3, 2]),
                              range=f"{math.degrees(sp.range_min):.10g} {math.degrees(sp.range_max):.10g}",
                              ref=f"{math.degrees(q0):.10g}", limited="true",
                              damping=_f(tau_ext / self.wmax), armature=_f(self.armature * nd.n_par))
                self.actuators.append((f"j{i}_{js.name}", -tau_ret, tau_ext))
                els[child] = cel
        for body, el in els.items():
            self._body_content(el, i, body)
        return els[base]

    def build(self, root_z: float) -> str:
        sp = self.spec
        root_kind = self.ex.nodes[0].kind
        root = ET.Element("mujoco", model="hyblock")
        ET.SubElement(root, "compiler", angle="degree", coordinate="local", inertiafromgeom="auto")
        ET.SubElement(root, "option", timestep=_f(sp.sim.get("timestep_s", 0.01)),
                      iterations=str(int(sp.sim.get("solver_iterations", 20))))
        custom = ET.SubElement(root, "custom")
        ET.SubElement(custom, "numeric", name="hyblock_torso_box", data=_f(torso_box(sp, root_kind)))
        ET.SubElement(custom, "numeric", name="hyblock_torso_touch", data=_f(sp.sim.get("torso_touch_m", 0.005)))
        ET.SubElement(custom, "numeric", name="hyblock_init_clearance", data=_f(sp.sim.get("init_clearance_m", 0.02)))
        ET.SubElement(custom, "numeric", name="hyblock_n_frames", data=_f(sp.sim.get("n_frames", 5)))
        default = ET.SubElement(root, "default")
        ET.SubElement(default, "geom", contype="1", conaffinity="1" if self.self_collision else "0", condim="3")
        asset = ET.SubElement(root, "asset")
        ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="512", height="512",
                      rgb1=".1 .2 .3", rgb2=".2 .3 .4")
        ET.SubElement(asset, "material", name="grid", texture="grid", texrepeat="1 1", texuniform="true",
                      reflectance=".2")
        wb = ET.SubElement(root, "worldbody")
        ET.SubElement(wb, "light", directional="true", diffuse="1 1 1", dir="0 0 -1.3", pos="0 0 1.3")
        ET.SubElement(wb, "geom", name="floor", type="plane", size="20 20 .01", material="grid",
                      contype="0", conaffinity="1", priority="1", friction=f"{self.floor_friction:.6g} 0.005 0.0001")
        T0 = ground_frame(GROUND_X) @ rot_x(math.pi) @ inv_T(sp.parts[root_kind].down_frame)
        T0[2, 3] += root_z
        torso = self._part(wb, 0, world_pose=T0)
        if not self.fixed_base:
            ET.SubElement(torso, "camera", name="track", mode="trackcom", pos="0 -3 0.6", xyaxes="1 0 0 0 0 1")
        acts = ET.SubElement(root, "actuator")
        for joint, lo, hi in self.actuators:
            ET.SubElement(acts, "general", name=f"a_{joint}", joint=joint, ctrllimited="true",
                          ctrlrange=f"{lo:.10g} {hi:.10g}", forcelimited="true", forcerange=f"{lo:.10g} {hi:.10g}",
                          gainprm="1", biasprm="0 0 0", dyntype="filterexact", dynprm=f"{self.tc:.10g}")
        ET.indent(root)
        return ET.tostring(root, encoding="unicode")


def joint_qpos_index(model: mujoco.MjModel, robot) -> np.ndarray:
    """qpos 地址，按 Robot.joint_index 的顺序（MuJoCo 的关节顺序是 body 树的深度优先顺序，与零件编号不同）。"""
    out = []
    for i, js in robot.joint_index:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j{i}_{js.name}")
        out.append(model.jnt_qposadr[j])
    return np.array(out, int)


def lowest_point(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    """机器人所有几何（不含世界 body 上的地面/地形）的最低点 z。"""
    zmin = math.inf
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        c = data.geom_xpos[g]
        t = model.geom_type[g]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            ext = np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g]
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ext = model.geom_size[g][0]
        else:
            ext = model.geom_rbound[g]
        zmin = min(zmin, c[2] - ext)
    return zmin


def self_penetration(model: mujoco.MjModel, data: mujoco.MjData | None = None) -> tuple[float, str]:
    """零件之间的最大穿透深度（m）与对应的刚体对（\"名|名\"）。不看地面；父子刚体（filterparent）不算。
    不改变 model（临时打开机器人几何之间的碰撞，结束时恢复）。data 给定时用它的姿态，否则用 qpos0。"""
    m = model
    keep = (m.geom_contype.copy(), m.geom_conaffinity.copy(), m.body_contype.copy(), m.body_conaffinity.copy())
    robot = m.geom_bodyid != 0
    try:
        m.geom_contype[:] = np.where(robot, 2, 0)
        m.geom_conaffinity[:] = np.where(robot, 2, 0)
        m.body_contype[:] = np.where(np.arange(m.nbody) > 0, 2, 0)     # the broadphase filters on these too
        m.body_conaffinity[:] = m.body_contype
        d = mujoco.MjData(m)
        if data is not None:
            d.qpos[:] = data.qpos
        mujoco.mj_forward(m, d)
        worst, pair = 0.0, ""
        for c in d.contact[:d.ncon]:
            if -c.dist > worst:
                worst = float(-c.dist)
                b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
                pair = (f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b1)}|"
                        f"{mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b2)}")
        return worst, pair
    finally:
        m.geom_contype[:], m.geom_conaffinity[:], m.body_contype[:], m.body_conaffinity[:] = keep


def check_self_collision(design: Design | str, spec: ModuleSpec | None = None, max_modules: int | None = None,
                         tol: float | None = None) -> float:
    """初始姿态零件穿透深度（m）；超过 tol（默认 modules.yaml sim.self_collision_tol_m）抛 DesignError。
    只做合法性检查时用（生成器、变异）；compile_design 默认也会检查。"""
    spec = spec or load_spec()
    _, ex = build_robot(design, spec, max_modules=max_modules)
    m = mujoco.MjModel.from_xml_string(_Builder(spec, ex, pressure_MPa=8.0, mass_scale=1.0, par_eta=None, max_speed_deg_s=None,
                                                   time_constant_s=None, armature=None, fixed_base=False,
                                                   floor_friction=None).build(0.0))
    pen, pair = self_penetration(m)
    tol = spec.sim.get("self_collision_tol_m", 0.005) if tol is None else tol
    if pen > tol:
        raise DesignError("self_collision", f"初始姿态零件互相穿透 {pen * 1000:.0f} mm（{pair}）")
    return pen


def compile_design(design: Design | str, spec: ModuleSpec | None = None, *, pressure_MPa: float = 8.0,
                   mass_scale: float = 1.0, par_eta: float | None = None, max_speed_deg_s: float | None = None,
                   time_constant_s: float | None = None, armature: float | None = None,
                   fixed_base: bool = False, floor_friction: float | None = None,
                   max_modules: int | None = None, max_hubs: int | None = None,
                   self_collision: bool | None = None, check_self_collision: bool = True) -> str:
    """设计文本 / Design -> MJCF 字符串。无效设计抛 DesignError（带 kind）。

    浮动基座：躯干高度自动定为\"初始姿态下最低点离地 init_clearance\"。fixed_base=True 时根零件焊在地面上
    （用于和静力学模型比较重力项）。
    self_collision：仿真里零件之间是否碰撞（None = modules.yaml sim.self_collision）。check_self_collision：初始姿态
    穿透超过 sim.self_collision_tol_m 时抛 DesignError(self_collision)（只在打开自碰撞时检查）。"""
    spec = spec or load_spec()
    _, ex = build_robot(design, spec, max_modules=max_modules, max_hubs=max_hubs, mass_scale=mass_scale,
                        par_eta=par_eta, pressure_MPa=pressure_MPa, ground_x=GROUND_X)
    kw = dict(pressure_MPa=pressure_MPa, mass_scale=mass_scale, par_eta=par_eta, max_speed_deg_s=max_speed_deg_s,
              time_constant_s=time_constant_s, armature=armature, fixed_base=fixed_base, floor_friction=floor_friction,
              self_collision=False if fixed_base else self_collision)
    # fixed_base: the root is welded to the world, so MuJoCo's filterparent no longer excludes the root's own
    # plates (weld parent = world); that mode is only for the gravity comparison, so no self-collision there
    builder = _Builder(spec, ex, **kw)
    xml = builder.build(0.0)
    if builder.self_collision and check_self_collision:
        pen, pair = self_penetration(mujoco.MjModel.from_xml_string(xml))
        if pen > spec.sim.get("self_collision_tol_m", 0.005):
            raise DesignError("self_collision", f"初始姿态零件互相穿透 {pen * 1000:.0f} mm（{pair}）")
    if fixed_base:
        return xml
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lift = spec.sim.get("init_clearance_m", 0.02) - lowest_point(m, d)
    return _Builder(spec, ex, **kw).build(lift)
