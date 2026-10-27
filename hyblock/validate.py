"""Hyblock 仿真模型验证（pipeline 3.5 的第 1–4 项；第 5 项\"能走\"在 hyblock/calibrate.py 里用完整 MPC 回合做）。

用法（WSL，在 D:\\RM\\robomorph 下）：
    uv run --locked python -m hyblock.validate --out ../hyblock_runs/validate
输出 report.txt（人读）和 report.json。几分钟内跑完，不需要 MPC。
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import time
from pathlib import Path

import mujoco
import numpy as np

from hyblock.design import DesignError, build_robot, to_text
from hyblock.generator import random_design
from hyblock.mjcf import GROUND_X, compile_design, joint_qpos_index
from hyblock.model import load_spec
from mpc.mppi import apply_terrain

QUAD = ("h0=root; m1=h0.zp@90 q90/150 mirror; m2=h0.xm@0 q90; h3=m2.P2@0; m4=h3.zp@90 q90/150 mirror; "
        "foot=m1.P2; foot=m4.P2")


def check_gravity(n_designs: int, seed: int) -> dict:
    """1. 静力矩一致：固定基座下 MuJoCo 的 qfrc_bias 与静力学模型逐关节比较。"""
    rng = random.Random(seed)
    worst, rel, count = 0.0, 0.0, 0
    for k in range(n_designs):
        d = random_design(rng, max_modules=8)
        m = mujoco.MjModel.from_xml_string(compile_design(d, fixed_base=True, max_modules=8))
        data = mujoco.MjData(m)
        robot, _ = build_robot(d, ground_x=GROUND_X, max_modules=8)
        idx = joint_qpos_index(m, robot)
        for t in range(5):
            q = np.random.default_rng(1000 * k + t).uniform(-0.5, 2.6, len(idx))
            data.qpos[idx] = q
            mujoco.mj_forward(m, data)
            ref = robot.static_torques(q)
            err = np.abs(data.qfrc_bias[idx] - ref)
            worst = max(worst, err.max())
            rel = max(rel, err.max() / max(1e-9, np.abs(ref).max()))
            count += len(idx)
    return {"designs": n_designs, "joint_checks": count, "max_abs_err_Nm": worst, "max_rel_err": rel}


def check_constraints() -> dict:
    """2. 约束生效：单模块（固定基座）在满指令下的力矩、速度、关节范围与执行器滞后。"""
    spec = load_spec()
    out = {}
    m = mujoco.MjModel.from_xml_string(compile_design("m0=root q-30", fixed_base=True))   # joint a of one module
    m.body_pos[1][2] += 1.0                                # lift the welded module off the floor
    d = mujoco.MjData(m)
    lo, hi = m.actuator_ctrlrange[0]
    wmax = math.radians(spec.actuator["max_speed_deg_s"])
    tc = spec.actuator["time_constant_s"]
    # step response of the actuator state (joint held at the lower limit by a negative command first)
    d.ctrl[0] = hi
    t63 = None
    qv, q, f = [], [], []
    for k in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        if t63 is None and d.act[0] >= (1 - math.exp(-1)) * hi:
            t63 = d.time
        qv.append(d.qvel[0])
        q.append(d.qpos[0])
        f.append(d.actuator_force[0])
    d.ctrl[0] = lo
    for k in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
        qv.append(d.qvel[0])
        q.append(d.qpos[0])
        f.append(d.actuator_force[0])
    d.ctrl[0] = 10 * hi                                    # beyond ctrlrange: must be clipped
    for k in range(20):
        mujoco.mj_step(m, d)
        f.append(d.actuator_force[0])
    qv, q, f = map(np.array, (qv, q, f))
    rmin, rmax = math.degrees(spec.range_min), math.degrees(spec.range_max)
    out["torque_limits_Nm"] = [float(lo), float(hi)]
    out["actuator_force_range_seen_Nm"] = [float(f.min()), float(f.max())]
    out["max_speed_deg_s"] = float(np.degrees(np.abs(qv).max()))
    out["max_speed_limit_deg_s"] = spec.actuator["max_speed_deg_s"]
    out["range_deg"] = [rmin, rmax]
    out["joint_seen_deg"] = [float(np.degrees(q.min())), float(np.degrees(q.max()))]
    out["range_violation_deg"] = float(max(0.0, np.degrees(q.max()) - rmax, rmin - np.degrees(q.min())))
    out["lag_t63_s"] = t63
    out["lag_time_constant_s"] = tc
    return out


def check_contact() -> dict:
    """3. 接触：四足在平地/冰面上落地静止后，接触摩擦系数与给定初速度的滑行减速度。"""
    out = {}
    for terrain, mu in (("flat", 1.0), ("frozen_lake", 0.05)):
        m = mujoco.MjModel.from_xml_string(apply_terrain(compile_design(QUAD), terrain))
        d = mujoco.MjData(m)
        for _ in range(200):                                 # settle 2 s, actuators off
            mujoco.mj_step(m, d)
        ncon = int(d.ncon)
        frics = sorted({round(float(x), 6) for x in d.contact.friction[:d.ncon, 0]})
        robot_geoms = sorted({mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                              for c in range(d.ncon) for g in d.contact.geom[c] if m.geom_bodyid[g] > 0})
        d.qvel[0] = -1.0                                     # 1 m/s forward
        v0 = d.qvel[0]
        steps = 5                                            # 0.05 s: before the robot stops on flat ground
        for _ in range(steps):
            mujoco.mj_step(m, d)
        dv = abs(d.qvel[0] - v0)
        out[terrain] = {"contacts": ncon, "contact_friction": frics, "touching": robot_geoms,
                        "decel_m_s2": float(dv / (steps * m.opt.timestep)), "mu_g": mu * 9.81,
                        "torso_z": float(d.qpos[2])}
    return out


def check_compile(n: int, seed: int) -> dict:
    """4. 编译：随机设计的编译、加载、1 s 被动仿真是否成功；随机\"变异式\"文本的无效原因分布。"""
    rng = random.Random(seed)
    ok, fail, unstable = 0, collections.Counter(), 0
    mods = collections.Counter()
    t0 = time.time()
    for _ in range(n):
        d = random_design(rng, max_modules=8)
        try:
            m = mujoco.MjModel.from_xml_string(compile_design(d, max_modules=8))
        except Exception as e:                               # noqa: BLE001
            fail[type(e).__name__] += 1
            continue
        data = mujoco.MjData(m)
        for _ in range(100):
            mujoco.mj_step(m, data)
        if not np.all(np.isfinite(data.qpos)) or data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
            unstable += 1
        ok += 1
        mods[build_robot(d, max_modules=8)[1].module_count()] += 1
    # corrupted texts: what the checker reports when a random edit breaks a valid design
    kinds = collections.Counter()
    rng2 = random.Random(seed + 1)
    for _ in range(n):
        t = to_text(random_design(rng2, max_modules=8)).split("; ")
        i = rng2.randrange(len(t))
        t[i] = rng2.choice([t[i].replace("@0", "@45"), t[i].replace("P2", "zp"), t[i] + " mirror",
                            t[i].replace("0.", "7."), t[i].replace("q", "q1")])
        try:
            build_robot("; ".join(t), max_modules=8)
            kinds["still_valid"] += 1
        except DesignError as e:
            kinds[e.kind] += 1
    return {"designs": n, "compiled": ok, "compile_errors": dict(fail), "passive_unstable": unstable,
            "module_count_hist": dict(sorted(mods.items())), "seconds": time.time() - t0,
            "corrupted_text_kinds": dict(kinds)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = {"gravity": check_gravity(a.n, a.seed), "constraints": check_constraints(),
           "contact": check_contact(), "compile": check_compile(a.n, a.seed)}
    (out / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    g, c, k, cp = rep["gravity"], rep["constraints"], rep["contact"], rep["compile"]
    lines = [
        "Hyblock 仿真模型验证（pipeline 3.5）",
        f"1 静力矩一致：{g['designs']} 个随机设计 × 5 个姿态（{g['joint_checks']} 个关节值），"
        f"最大误差 {g['max_abs_err_Nm']:.2e} N·m（相对 {g['max_rel_err']:.1e}）",
        f"2 约束：力矩上限 {c['torque_limits_Nm'][0]:.1f} / +{c['torque_limits_Nm'][1]:.1f} N·m，"
        f"实测执行器力 {c['actuator_force_range_seen_Nm'][0]:.1f} … {c['actuator_force_range_seen_Nm'][1]:.1f}（指令 10 倍时也被截断）；"
        f"最大速度 {c['max_speed_deg_s']:.0f} deg/s（上限 {c['max_speed_limit_deg_s']:.0f}）；"
        f"关节 {c['joint_seen_deg'][0]:.1f}° … {c['joint_seen_deg'][1]:.1f}°（范围 {c['range_deg']}，越界 {c['range_violation_deg']:.2f}°）；"
        f"执行器 63% 上升时间 {c['lag_t63_s']} s（时间常数 {c['lag_time_constant_s']} s）",
    ]
    for terr, v in k.items():
        lines.append(f"3 接触（{terr}）：{v['contacts']} 个接触，摩擦系数 {v['contact_friction']}，接触几何 {v['touching']}；"
                     f"1 m/s 滑行减速度 {v['decel_m_s2']:.2f} m/s²（μg = {v['mu_g']:.2f}）")
    lines.append(f"4 编译：{cp['compiled']}/{cp['designs']} 编译并加载成功，错误 {cp['compile_errors']}，"
                 f"被动仿真 1 s 发散 {cp['passive_unstable']}；模块数分布 {cp['module_count_hist']}；"
                 f"随机改坏的文本的无效原因 {cp['corrupted_text_kinds']}")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
