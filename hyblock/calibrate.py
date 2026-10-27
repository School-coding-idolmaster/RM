"""Hyblock 域的标定批量（第 5–6 周）：用随机设计定模块数上限、重复评估次数、地形缩放比例，并测 MPC 用时。

阶段 A（平地）：3 个手工设计 + N 个随机设计（模块数上限 8，模块数在 1…8 均匀抽；其中 ≤4、≤6 个模块的子集
  恰好是\"上限 4 / 6\"的随机分布），每个设计 MPC 3 回合。
阶段 B（地形）：阶段 A 里平地上能走的设计（3 回合平均位移 ≥ 1 m），在沟坎 / 横梁 × 缩放 {1.5, 2, 3} 上各 1 回合，
  记录最远位移。走不动的设计记为通不过（平地都走不了，地形更走不了）。
中断后重跑会跳过已完成的行。最后写 summary.txt。
第 7 周补测（Mac）：--scales 1.0,1.25,1.5,2,3 --n 120 在同一台机器上把五个比例一起重测（summary 按 terrain.csv 里出现的比例汇总）。

用法（WSL，在 D:\\RM\\robomorph 下）：
    uv run --locked python -m hyblock.calibrate --out ../hyblock_runs/calib --n 60
    uv run --locked python -m hyblock.calibrate --out ../hyblock_runs/calib --summary_only
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from hyblock.design import build_robot, summary, to_text
from hyblock.generator import random_design
from hyblock.mjcf import compile_design
from mpc.mppi import MPPIConfig, evaluate_design

HAND = {
    "hand_quad": ("h0=root; m1=h0.zp@90 q90/150 mirror; m2=h0.xm@0 q90; h3=m2.P2@0; "
                  "m4=h3.zp@90 q90/150 mirror; foot=m1.P2; foot=m4.P2"),
    "hand_quad_yaw": ("h0=root; m1=h0.zp@0 q90/150 mirror; m2=h0.xm@0 q90; h3=m2.P2@0; "
                      "m4=h3.zp@0 q90/150 mirror; foot=m1.P2; foot=m4.P2"),
    "hand_inchworm": "m0=root; m1=m0.P1@0; m2=m0.P2@0; foot=m1.P2; foot=m2.P2",
}
CAP = 8
CAPS = (4, 6, 8)
SCALES = (1.5, 2.0, 3.0)
TERRAINS = ("ridged", "beams")
WALK_M = 1.0          # \"能走\"：3 回合平均位移 ≥ 1 m（50 s 内）；得分 = 位移 / 0.05，所以 = 得分 ≥ 20
PASS_MARGIN_M = 0.5   # 通过第一个障碍：躯干原点越过它的远侧边缘 0.5 m


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def designs(n: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    out = list(HAND.items())
    out += [(f"rand_{i:03d}", to_text(random_design(rng, max_modules=CAP))) for i in range(n)]
    return out


def phase_a(out: Path, items, episodes: int, threads: int, seed: int, steps: int) -> None:
    path = out / "flat.csv"
    done = {r["name"] for r in _read(path)}
    cfg = MPPIConfig(threads=threads, episode_control_steps=steps)
    for name, text in items:
        if name in done:
            continue
        s = summary(text)
        t0 = time.time()
        r = evaluate_design(compile_design(text, max_modules=CAP), "flat", episodes, seed, cfg)
        eps = r["episodes"]
        dist = [e["distance"] for e in eps]
        row = {"name": name, "design": text, "modules": s["modules"], "hubs": s["hubs"], "joints": s["joints"],
               "feet": s["feet"], "root": s["root"],
               "mass_kg": round(s["mass_kg"], 2), "fitness": round(r["fitness"], 3),
               "ep_scores": json.dumps([round(e["score"], 2) for e in eps]),
               "distance_m": round(float(np.mean(dist)), 3), "terminated": sum(e["terminated"] for e in eps),
               "unstable": sum(e["unstable"] for e in eps), "env_steps": json.dumps([e["env_steps"] for e in eps]),
               "seconds": round(time.time() - t0, 2),
               "physics_steps": sum(e["physics_steps"] for e in eps)}
        _append(path, row)
        print(f"[A] {name} modules={row['modules']} hubs={row['hubs']} nu={row['joints']} fitness={row['fitness']:.1f} "
              f"dist={row['distance_m']:.2f} m term={row['terminated']} {row['seconds']:.0f}s", flush=True)


def phase_b(out: Path, threads: int, seed: int, steps: int, scales=SCALES) -> None:
    flat = _read(out / "flat.csv")
    walkers = [r for r in flat if float(r["distance_m"]) >= WALK_M]
    path = out / "terrain.csv"
    done = {(r["name"], r["terrain"], r["scale"]) for r in _read(path)}
    for r in walkers:
        for terrain in TERRAINS:
            for scale in scales:
                key = (r["name"], terrain, f"{scale}")
                if key in done:
                    continue
                cfg = MPPIConfig(threads=threads, episode_control_steps=steps, terrain_scale=scale)
                t0 = time.time()
                res = evaluate_design(compile_design(r["design"], max_modules=CAP), terrain, 1, seed, cfg)
                e = res["episodes"][0]
                far_edge = (1.0 + 0.2) * scale                      # first obstacle: x = -1.0 s, radius 0.2 s
                need = far_edge + PASS_MARGIN_M + e["x_start"]       # displacement needed (x_start < 0 = ahead)
                row = {"name": r["name"], "modules": r["modules"], "terrain": terrain, "scale": f"{scale}",
                       "fitness": round(res["fitness"], 3), "max_distance_m": round(res["max_distance"], 3),
                       "need_m": round(need, 3), "passed": int(res["max_distance"] >= need),
                       "terminated": int(e["terminated"]), "unstable": int(e["unstable"]),
                       "seconds": round(time.time() - t0, 2)}
                _append(path, row)
                print(f"[B] {r['name']} {terrain} x{scale}: max {row['max_distance_m']:.2f} m "
                      f"(need {row['need_m']:.2f}) passed={row['passed']} {row['seconds']:.0f}s", flush=True)


def _cv(scores):
    s = np.array(scores, float)
    return float(s.std() / s.mean()) if s.mean() > 0 else float("nan")


def write_summary(out: Path) -> str:
    flat = _read(out / "flat.csv")
    terr = _read(out / "terrain.csv")
    rnd = [r for r in flat if r["name"].startswith("rand_")]
    lines = [f"Hyblock 域标定（{time.strftime('%Y-%m-%d %H:%M')}）", ""]
    lines.append("== 手工设计（平地） ==")
    for r in flat:
        if r["name"].startswith("hand_"):
            lines.append(f"  {r['name']}: 得分 {float(r['fitness']):.1f}（回合 {r['ep_scores']}），平均位移 "
                         f"{float(r['distance_m']):.2f} m，终止 {r['terminated']}/3，用时 {float(r['seconds']):.0f} s")
    for cap in CAPS:
        sub = [r for r in rnd if int(r["modules"]) <= cap]
        if not sub:
            continue
        d = np.array([float(r["distance_m"]) for r in sub])
        term = np.array([int(r["terminated"]) for r in sub])
        lines += ["", f"== 随机设计，模块数 ≤ {cap}（{len(sub)} 个）=="]
        for th in (0.5, 1.0, 2.0, 5.0):
            lines.append(f"  平均位移 ≥ {th} m 的比例：{np.mean(d >= th):.2f}")
        lines.append(f"  位移中位数 {np.median(d):.2f} m，最大 {d.max():.2f} m；3 回合全部提前终止的比例 {np.mean(term == 3):.2f}")
        walk = [r for r in sub if float(r["distance_m"]) >= WALK_M]
        cvs = [_cv(json.loads(r["ep_scores"])) for r in walk]
        if cvs:
            lines.append(f"  能走的设计（{len(walk)} 个）3 回合得分变异系数：中位数 {np.nanmedian(cvs):.2f}，"
                         f"75% 分位 {np.nanpercentile(cvs, 75):.2f}，最大 {np.nanmax(cvs):.2f}")
        for k in (1, 2, 3):                                   # ranking stability if only k episodes were used
            if len(walk) >= 5:
                full = np.array([float(r["fitness"]) for r in sub])
                part = np.array([np.mean(json.loads(r["ep_scores"])[:k]) for r in sub])
                rf, rp = np.argsort(np.argsort(full)), np.argsort(np.argsort(part))
                lines.append(f"  只用前 {k} 回合时与 3 回合平均的 Spearman：{np.corrcoef(rf, rp)[0, 1]:.3f}")
    if flat:
        sec_ep = np.array([float(r["seconds"]) / 3 for r in flat])
        nu = np.array([int(r["joints"]) for r in flat])
        steps_ep = np.array([sum(json.loads(r["env_steps"])) / 3 for r in flat])
        full = steps_ep >= 999
        lines += ["", "== 用时 =="]
        lines.append(f"  每回合平均 {sec_ep.mean():.1f} s（未终止的完整回合 {sec_ep[full].mean() if full.any() else float('nan'):.1f} s）；"
                     f"每设计 3 回合平均 {3 * sec_ep.mean():.0f} s")
        for lo, hi in ((2, 4), (6, 8), (10, 12), (14, 16)):
            m = full & (nu >= lo) & (nu <= hi)
            if m.any():
                lines.append(f"  关节数 {lo}–{hi}：完整回合 {sec_ep[m].mean():.1f} s（{m.sum()} 个设计）")
        uns = sum(int(r["unstable"]) for r in flat)
        lines.append(f"  物理故障（unstable）回合数：{uns}")
    scales = sorted({float(t["scale"]) for t in terr}) if terr else []
    if terr:
        lines += ["", "== 地形：通过第一个障碍的比例（分母 = 全部随机设计，平地走不动的算不通过）=="]
        for cap in CAPS:
            n_all = len([r for r in rnd if int(r["modules"]) <= cap])
            for terrain in TERRAINS:
                parts = []
                for scale in scales:
                    rows = [t for t in terr if t["terrain"] == terrain and t["scale"] == f"{scale}"
                            and t["name"].startswith("rand_") and int(t["modules"]) <= cap]
                    passed = sum(int(t["passed"]) for t in rows)
                    parts.append(f"×{scale}: {passed}/{n_all} = {passed / max(1, n_all):.2f}（已测 {len(rows)}）")
                lines.append(f"  ≤{cap} 模块，{terrain}：" + "；".join(parts))
        for name in HAND:
            rows = [t for t in terr if t["name"] == name]
            if rows:
                lines.append(f"  {name}：" + "，".join(f"{t['terrain']}×{t['scale']} {float(t['max_distance_m']):.1f}/"
                                                      f"{float(t['need_m']):.1f} m" for t in rows))
    text = "\n".join(lines) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=60, help="number of random designs")
    p.add_argument("--seed", type=int, default=0, help="design sampling seed")
    p.add_argument("--mpc_seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--steps", type=int, default=250, help="control steps per episode")
    p.add_argument("--phase", choices=["A", "B", "all"], default="all")
    p.add_argument("--scales", default=",".join(str(s) for s in SCALES),
                   help="comma-separated terrain scales for phase B, e.g. 1.0,1.25,1.5,2,3")
    p.add_argument("--summary_only", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        items = designs(a.n, a.seed)
        (out / "designs.txt").write_text("\n".join(f"{n}\t{t}" for n, t in items) + "\n", encoding="utf-8")
        if a.phase in ("A", "all"):
            phase_a(out, items, a.episodes, a.threads, a.mpc_seed, a.steps)
            print(write_summary(out), flush=True)
        if a.phase in ("B", "all"):
            phase_b(out, a.threads, a.mpc_seed, a.steps, tuple(float(x) for x in a.scales.split(",")))
    print(write_summary(out))


if __name__ == "__main__":
    main()
