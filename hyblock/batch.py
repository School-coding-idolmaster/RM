"""Hyblock 设计的批量评估：① 筛选 + ③ MPC（第 7 周）。中断后重新运行会跳过 results.csv 里已有的设计（断点续跑），
开始写 run_meta.json，正常结束写 DONE。

设计来源（可组合）：--designs FILE（每行\"名字<TAB>设计文本\"或只有设计文本）、--random N（文法随机生成器，
模块数上限 --max_modules）、--hand（hyblock/calibrate.py 的 3 个手工设计）。

--screen：
  on      筛掉的设计不跑 MPC（正式搜索里的用法）
  record  全部设计都跑 MPC，只记录筛选结论 → 误杀率（被淘汰但 MPC 能走的比例）
  off     不筛选（只做合法性检查）
无效设计（DesignError，包括初始姿态自碰撞）永远不跑 MPC。

用法（仓库根目录）：
    uv run --locked python -m hyblock.batch --random 200 --seed 1 --hand --screen record --out ../mac_runs/screen_audit
    uv run --locked python -m hyblock.batch --out ../mac_runs/screen_audit --summary_only
输出：results.csv（每个设计一行）、summary.txt、run_meta.json、DONE。
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np

from hyblock.design import DesignError, summary as design_summary, to_text
from hyblock.generator import random_design
from hyblock.mjcf import compile_design
from hyblock.screen import VERSION as SCREEN_VERSION, ScreenConfig, postcheck, screen
from mpc.mppi import MPPIConfig, evaluate_design

WALK = 20.0          # 得分 ≥ 20 = 50 s 平均位移 ≥ 1 m（\"能走\"，与 calibrate.py 相同）
FIELDS = ["name", "design", "modules", "hubs", "joints", "feet", "root", "valid", "screen", "screen_detail",
          "static_util", "torso_gap_m", "torso_gap_max_m", "hold_term_step", "hold_sat_frac", "self_penetration_m",
          "evaluated", "fitness", "ep_scores", "distance_m", "terminated", "unstable", "env_steps", "peak_speed",
          "fast_episodes", "screen_s", "mpc_s"]


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _append(path: Path, row: dict) -> None:
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def load_designs(a) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    if a.hand:
        from hyblock.calibrate import HAND
        items += list(HAND.items())
    if a.designs:
        for k, line in enumerate(Path(a.designs).read_text(encoding="utf-8").splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, text = line.split("\t", 1) if "\t" in line else (f"d{k:04d}", line)
            items.append((name.strip(), text.strip()))
    if a.random:
        rng = random.Random(a.seed)
        items += [(f"rand_{i:03d}", to_text(random_design(rng, max_modules=a.max_modules))) for i in range(a.random)]
    names = [n for n, _ in items]
    if len(set(names)) != len(names):
        raise SystemExit("duplicate design names")
    return items


def run(a, out: Path, items) -> None:
    path = out / "results.csv"
    done = {r["name"] for r in _read(path)}
    cfg = MPPIConfig(threads=a.threads, episode_control_steps=a.steps, terrain_scale=a.terrain_scale)
    scfg = ScreenConfig()
    for name, text in items:
        if name in done:
            continue
        row = {"name": name, "design": text}
        t0 = time.time()
        try:
            xml = compile_design(text, max_modules=a.max_modules)
            s = design_summary(text)
            row.update(modules=s["modules"], hubs=s["hubs"], joints=s["joints"], feet=s["feet"], root=s["root"],
                       valid=1)
        except DesignError as e:
            row.update(valid=0, screen=e.kind, screen_detail=str(e), evaluated=0)
            _append(path, row)
            print(f"{name}: invalid ({e.kind})", flush=True)
            continue
        if a.screen != "off":
            sr = screen(text, scfg, max_modules=a.max_modules, xml=xml)
            row.update(screen=sr.reason or "pass", screen_detail=sr.detail,
                       **{k: sr.metrics.get(k, "") for k in ("static_util", "torso_gap_m", "torso_gap_max_m",
                                                             "hold_term_step", "hold_sat_frac",
                                                             "self_penetration_m")})
            killed = not sr.passed
        else:
            row["screen"], killed = "off", False
        row["screen_s"] = round(time.time() - t0, 3)
        if killed and a.screen == "on":
            row["evaluated"] = 0
            _append(path, row)
            print(f"{name}: screened out ({row['screen']})", flush=True)
            continue
        t1 = time.time()
        r = evaluate_design(xml, a.terrain, a.episodes, a.mpc_seed, cfg)
        eps = r["episodes"]
        pc = postcheck(r, scfg)
        row.update(evaluated=1, fitness=round(r["fitness"], 3), ep_scores=json.dumps([round(e["score"], 2) for e in eps]),
                   distance_m=round(float(np.mean([e["distance"] for e in eps])), 3),
                   terminated=sum(e["terminated"] for e in eps), unstable=sum(e["unstable"] for e in eps),
                   env_steps=json.dumps([e["env_steps"] for e in eps]), peak_speed=json.dumps(pc["peak_speed"]),
                   fast_episodes=pc["fast_episodes"], mpc_s=round(time.time() - t1, 2))
        _append(path, row)
        print(f"{name}: screen={row['screen']} nu={row['joints']} fitness={row['fitness']:.1f} "
              f"term={row['terminated']} {row['mpc_s']:.0f}s", flush=True)


def _f(r, k, default=np.nan):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return default


def write_summary(out: Path, a) -> str:
    rows = _read(out / "results.csv")
    L = [f"Hyblock 批量评估（{time.strftime('%Y-%m-%d %H:%M')}，筛选 {SCREEN_VERSION}，--screen {a.screen}，"
         f"{a.terrain} ×{a.terrain_scale}，每设计 {a.episodes} 回合）", ""]
    if not rows:
        return "\n".join(L + ["（没有结果）"]) + "\n"
    n = len(rows)
    valid = [r for r in rows if r["valid"] == "1"]
    L.append(f"设计 {n} 个：有效 {len(valid)}，无效 {n - len(valid)}")
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["screen"]] = kinds.get(r["screen"], 0) + 1
    L.append("筛选结论：" + "，".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])))
    ev = [r for r in rows if r["evaluated"] == "1"]
    if ev:
        fit = np.array([_f(r, "fitness") for r in ev])
        L.append(f"跑了 MPC 的 {len(ev)} 个：能走（得分 ≥ {WALK:.0f}）{np.mean(fit >= WALK):.2f}，得分中位数 "
                 f"{np.median(fit):.1f}，最高 {fit.max():.1f}；回合全部提前终止 "
                 f"{np.mean([int(r['terminated']) == a.episodes for r in ev]):.2f}；"
                 f"unstable 回合 {sum(int(r['unstable']) for r in ev)}；峰值速度 > 6 m/s 的回合 "
                 f"{sum(int(r['fast_episodes']) for r in ev)}")
        sec = np.array([_f(r, "mpc_s") for r in ev])
        L.append(f"MPC 用时：平均 {sec.mean():.1f} s/设计，合计 {sec.sum() / 3600:.2f} h；筛选用时平均 "
                 f"{np.nanmean([_f(r, 'screen_s') for r in rows]):.3f} s/设计")
    if a.screen == "record" and ev:
        L += ["", "== 误杀检查（record：被淘汰的设计也跑了 MPC）=="]
        rules = {"torso_ground（启用）": lambda r: r["screen"] == "torso_ground",
                 "cannot_stand（只记录：PD 保持 1 s 内触地且饱和 ≥ 80%）":
                     lambda r: r["hold_term_step"] not in ("", None) and _f(r, "hold_sat_frac", 0) >= 0.8,
                 "PD 保持 1 s 内触地（任意饱和）": lambda r: r["hold_term_step"] not in ("", None),
                 "静力矩利用率 > 1": lambda r: _f(r, "static_util", 0) > 1.0}
        for label, rule in rules.items():
            hit = [r for r in ev if rule(r)]
            if not hit:
                L.append(f"  {label}：0 个")
                continue
            f = np.array([_f(r, "fitness") for r in hit])
            bad = [r for r in hit if _f(r, "fitness") >= WALK]
            L.append(f"  {label}：{len(hit)} 个（占 {len(hit) / len(ev):.2f}），其中能走 {len(bad)} 个 → 误杀率 "
                     f"{len(bad) / len(hit):.2f}；得分最高 {f.max():.1f}"
                     + (f"（{', '.join(r['name'] for r in sorted(bad, key=lambda r: -_f(r, 'fitness'))[:5])}）"
                        if bad else ""))
        passed = [r for r in ev if r["screen"] == "pass"]
        if passed:
            f = np.array([_f(r, "fitness") for r in passed])
            L.append(f"  通过筛选的 {len(passed)} 个：能走 {np.mean(f >= WALK):.2f}，得分中位数 {np.median(f):.1f}")
    walkers = [r for r in ev if _f(r, "fitness") >= WALK]
    if walkers and a.episodes >= 2:
        cvs = []
        for r in walkers:
            s = np.array(json.loads(r["ep_scores"]))
            cvs.append(s.std() / s.mean() if s.mean() > 0 else np.nan)
        L.append(f"能走设计回合间变异系数：中位数 {np.nanmedian(cvs):.2f}，75% 分位 {np.nanpercentile(cvs, 75):.2f}")
    text = "\n".join(L) + "\n"
    (out / "summary.txt").write_text(text, encoding="utf-8")
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--designs", help="file: one design per line, optionally 'name<TAB>design'")
    p.add_argument("--random", type=int, default=0, help="number of random designs")
    p.add_argument("--seed", type=int, default=0, help="random-design seed")
    p.add_argument("--hand", action="store_true", help="include the hand-made designs of hyblock/calibrate.py")
    p.add_argument("--max_modules", type=int, default=6)
    p.add_argument("--screen", choices=["on", "record", "off"], default="on")
    p.add_argument("--terrain", default="flat")
    p.add_argument("--terrain_scale", type=float, default=1.0)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--mpc_seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--steps", type=int, default=250, help="control steps per episode")
    p.add_argument("--summary_only", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.summary_only:
        from wj_tools.run_meta import done, start
        start(out, threads=a.threads, note=f"hyblock.batch screen={a.screen} {a.terrain}x{a.terrain_scale}",
              screen_version=SCREEN_VERSION, args=vars(a))
        items = load_designs(a)
        (out / "designs.txt").write_text("\n".join(f"{n}\t{t}" for n, t in items) + "\n", encoding="utf-8")
        run(a, out, items)
        print(write_summary(out, a), flush=True)
        done(out, n_designs=len(items))
    else:
        print(write_summary(out, a))


if __name__ == "__main__":
    main()
