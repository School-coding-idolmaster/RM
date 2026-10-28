"""① 筛选（Hyblock 域，第 7 周）：在 MPC 之前淘汰明显不可能的设计。保守：只淘汰物理上造不出来的设计，和
\"MPC 也几乎必然 ≈ 0 分\"的设计；其余全部放行。

检查（按顺序；前一项淘汰就不再做后面的）：
  1. 编译与合法性（hyblock.design / hyblock.mjcf，DesignError 的 kind 原样作为淘汰原因）。其中包括自碰撞
     self_collision：初始姿态下零件之间穿透 > modules.yaml sim.self_collision_tol_m（参照 RoboGrammar 的
     presimulate；10-28 起仿真里不相邻零件之间也碰撞）。属于\"造不出来\"，计入约束违反 / 无效设计率
  2. torso_ground：初始姿态（与 MPC 的 reset 相同：关节角加 U(−0.1, 0.1) rad 噪声后整体抬到最低点离地 2 cm）
     下，躯干的最低角就是整个机器人的最低点（差 ≤ gap_tol），且 n_noise 个噪声样本里全都如此。这种设计落地第一步
     就触发\"躯干触地\"终止。10-26 标定的 63 个设计里有 6 个满足，MPC 3 回合全部在第 1–2 个 env step 终止（得分 0）；
     其余 57 个没有一个满足（包括只在个别噪声样本下躯干最低、MPC 照样能走的 hand_inchworm、rand_051）
  3. cannot_stand：PD 以最大力矩保持初始关节角 hold_s 秒，躯干触地或翻倒，且触地前关节饱和的 env step 比例
     ≥ sat_frac（腿使出全力也撑不住）。只是\"保持初始姿态失败\"不淘汰：标定里很多能走的设计（rand_039 570 分）
     在 PD 保持下也会倒，MPC 会主动换姿态
  记录（不作淘汰条件）：hyblock/model.py 的静力矩利用率（躯干固定、腿悬空时保持初始姿态所需力矩 / 上限），作为
  站立检查的交叉核对。标定里利用率 > 1 的 4 个设计全部能走（57–240 分），所以不能用来淘汰
地形相关的检查（冰面摩擦、沟坎高度、横梁高度）在第 19 周定下主实验地形后再加。

MPC 之后的检查（postcheck）：峰值速度 > peak_speed_flag 的回合计数；得分 > score_bound 时标为物理漏洞
（score_bound 默认 None = 只监控，阈值按实测得分分布定）。
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import mujoco
import numpy as np

from hyblock.design import DesignError, build_robot, summary
from hyblock.mjcf import compile_design, lowest_point, self_penetration

VERSION = "screen-v1 (2026-10-28)"


@dataclass
class ScreenConfig:
    gap_tol: float = 0.001              # m：躯干最低角与机器人最低点之差 ≤ 它 = 躯干就是最低的零件
    n_noise: int = 16                   # 初始关节噪声样本数（与 MPC reset 同分布）
    reset_noise: float = 0.1            # rad（mpc/mppi.py MPPIConfig.reset_noise）
    hold_s: float = 1.0                 # 站立检查时长
    sat_frac: float = 0.8               # 触地前\"有关节饱和\"的 env step 比例
    kp: float = 3000.0                  # N·m/rad；0.06 rad 误差就到伸出侧上限
    kd: float = 30.0                    # N·m·s/rad
    check_torso_ground: bool = True
    check_stand: bool = False           # 10-28：开自碰撞后 PD 会顶着别的零件饱和（rand_012 误杀），先只记录
    peak_speed_flag: float = 6.0        # m/s，postcheck 监控
    score_bound: float | None = None    # postcheck：得分上界（None = 只监控）


@dataclass
class ScreenResult:
    passed: bool
    reason: str | None                 # None = 通过；否则为淘汰原因（kind）
    detail: str = ""
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _numeric(model, name):
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if i < 0:
        return None
    adr, size = model.numeric_adr[i], model.numeric_size[i]
    return np.array(model.numeric_data[adr:adr + size])


class _Torso:
    """与 mpc/mppi.py MPPIEvaluator.unhealthy 相同的终止判据（Hyblock 域）。"""

    def __init__(self, model):
        x0, x1, y0, y1, z0, z1 = _numeric(model, "hyblock_torso_box")
        self.corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
        self.touch = float(_numeric(model, "hyblock_torso_touch")[0])

    def state(self, qpos7) -> tuple[bool, float]:
        z = qpos7[2]
        w, x, y, zq = qpos7[3:7]
        r3 = np.array([2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)])
        cz = float(z + (r3 @ self.corners.T).min())
        return bool(cz < self.touch or r3[1] < 0 or z > 5.0), cz


def _reset(model, rng: np.random.Generator | None, noise: float) -> mujoco.MjData:
    """与 MPPIEvaluator.reset 相同（rng=None 时不加噪声）；速度为 0。"""
    m = model
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    if rng is not None:
        d.qpos[7:] += rng.uniform(-noise, noise, m.nq - 7)
        d.qpos[7:] = np.clip(d.qpos[7:], m.jnt_range[1:, 0], m.jnt_range[1:, 1])
    mujoco.mj_forward(m, d)
    d.qpos[2] += float(_numeric(m, "hyblock_init_clearance")[0]) - lowest_point(m, d)
    mujoco.mj_forward(m, d)
    return d


def torso_gap(model, cfg: ScreenConfig, seed: int = 0) -> tuple[float, float]:
    """(无噪声时, n_noise 个噪声样本中最大的) 躯干最低角 − 机器人最低点（m）。"""
    tor = _Torso(model)
    clear = float(_numeric(model, "hyblock_init_clearance")[0])
    rng = np.random.default_rng(seed)
    gaps = [tor.state(_reset(model, r, cfg.reset_noise).qpos[:7])[1] - clear
            for r in [None] + [rng] * cfg.n_noise]
    return float(gaps[0]), float(max(gaps))


def stand_test(model, cfg: ScreenConfig) -> dict:
    """PD 以最大力矩保持初始关节角 hold_s 秒（初始化与 MPC 的 reset 相同，不加噪声）。"""
    m = model
    tor = _Torso(m)
    nf = int(_numeric(m, "hyblock_n_frames")[0])
    d = _reset(m, None, 0.0)
    qadr = m.jnt_qposadr[m.actuator_trnid[:, 0]]
    vadr = m.jnt_dofadr[m.actuator_trnid[:, 0]]
    q_ref = m.qpos0[qadr].copy()
    lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
    n_env = max(1, int(round(cfg.hold_s / (m.opt.timestep * nf))))
    sat_steps, term_step, min_corner, unstable = 0, None, math.inf, False
    for k in range(n_env):
        for _ in range(nf):
            d.ctrl[:] = np.clip(cfg.kp * (q_ref - d.qpos[qadr]) - cfg.kd * d.qvel[vadr], lo, hi)
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)):
            unstable, term_step = True, k + 1
            break
        f = d.actuator_force
        sat_steps += bool(np.any((f >= 0.98 * hi) | (f <= 0.98 * lo)))
        bad, cz = tor.state(d.qpos[:7])
        min_corner = min(min_corner, cz)
        if bad:
            term_step = k + 1
            break
    return {"hold_term_step": term_step, "hold_sat_frac": round(sat_steps / (term_step or n_env), 3),
            "hold_min_corner_m": round(min_corner, 4), "hold_unstable": unstable}


def static_utilization(design_text: str, max_modules: int | None) -> float:
    """躯干固定时保持初始姿态所需静力矩 / 上限 的最大值（>1 = 腿撑不起自己的重量；只作交叉核对）。"""
    robot, _ = build_robot(design_text, max_modules=max_modules)
    return float(np.max(robot.check(robot.q0())["utilization"])) if robot.n_joints else 0.0


def screen(design_text: str, cfg: ScreenConfig = ScreenConfig(), max_modules: int | None = 6,
           xml: str | None = None) -> ScreenResult:
    """筛选一个设计。xml 给定时跳过编译（但合法性检查要求 design_text 仍能解析）。"""
    met: dict = {}
    try:
        s = summary(design_text)
        met.update(modules=s["modules"], hubs=s["hubs"], joints=s["joints"], feet=s["feet"], root=s["root"])
        if xml is None:
            xml = compile_design(design_text, max_modules=max_modules)
        met["static_util"] = round(static_utilization(design_text, max_modules), 3)
    except DesignError as e:
        return ScreenResult(False, e.kind, str(e), met)
    model = mujoco.MjModel.from_xml_string(xml)
    if model.nu == 0:
        return ScreenResult(False, "no_actuator", "设计没有关节", met)
    pen, pair = self_penetration(model)
    met.update(self_penetration_m=round(pen, 4))
    g0, gmax = torso_gap(model, cfg)
    met.update(torso_gap_m=round(g0, 4), torso_gap_max_m=round(gmax, 4))
    if cfg.check_torso_ground and gmax <= cfg.gap_tol:
        return ScreenResult(False, "torso_ground", "躯干是机器人最低的零件（初始姿态及关节噪声下都是），落地即触发躯干触地终止", met)
    st = stand_test(model, cfg)
    met.update(st)
    if cfg.check_stand and st["hold_term_step"] is not None:
        if st["hold_unstable"]:
            return ScreenResult(False, "unstable", "站立检查中仿真发散", met)
        if st["hold_sat_frac"] >= cfg.sat_frac:
            return ScreenResult(False, "cannot_stand",
                                f"最大力矩保持初始姿态时躯干在 {st['hold_term_step']} 个 env step 触地"
                                f"（关节饱和 {st['hold_sat_frac']:.0%}）", met)
    return ScreenResult(True, None, "", met)


def postcheck(mpc_result: dict, cfg: ScreenConfig = ScreenConfig()) -> dict:
    """MPC 之后：峰值速度监控与得分上界。mpc_result = mpc.mppi.evaluate_design 的返回值。"""
    peaks = [max(r) if r else 0.0 for r in mpc_result.get("rewards", [])]
    out = {"peak_speed": [round(p, 3) for p in peaks],
           "fast_episodes": int(sum(p > cfg.peak_speed_flag for p in peaks)),
           "over_bound": bool(cfg.score_bound is not None and mpc_result["fitness"] > cfg.score_bound)}
    out["suspect"] = bool(out["fast_episodes"] or out["over_bound"])
    return out
