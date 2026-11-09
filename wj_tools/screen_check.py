"""① 筛选的误杀率检验（pipeline 第 4 节检验表"LLM 设计上的误杀率待查"；WJ 11-05 决定在 Windows 上跑）：
把 formal_flat 与 replicate_flat 里被 ① 筛掉（status = screened）的设计（按设计文本去重）全部送 MPC，
设置与正式实验相同（平地、2 回合、mpc_seed 0），看有没有其实能走的（得分 ≥ 20 = 误杀）。
按来源分两类报告：LLM 变异产生的（source 含 llm_mut）与其他（随机第 0 代、随机变异、交叉）。
"设计没有关节"的设计也照样送 MPC（预计 0 分；编译或评估报错的记为 error）。
用法（WSL）：python wj_tools/screen_check.py [--dry] [--threads 6]；可中断，再运行从断点继续。结果 D:\\RM\\win_runs\\screen_check_flat\\"""
import argparse, csv, json, time
from pathlib import Path

import os
SRC = Path(os.environ.get("SCREEN_SRC", "/mnt/d/RM/from_mac/mac_runs"))
EXPS = ["formal_flat", "replicate_flat"]
OUT = Path("/mnt/d/RM/win_runs/screen_check_flat")


def collect():
    u = {}
    for e in EXPS:
        for f in sorted((SRC / e).glob("*/evals.csv")):
            for r in csv.DictReader(open(f, encoding="utf-8")):
                if r["status"] != "screened":
                    continue
                x = u.setdefault(r["design"], {"design": r["design"], "first": f"{e}/{f.parent.name}/{r['gen']}.{r['idx']}",
                                               "source": r["source"], "detail": r["detail"], "n_seen": 0})
                x["n_seen"] += 1
    picks = list(u.values())
    for x in picks:
        x["origin"] = "llm" if "llm_mut" in x["source"] else "other"
        x["reason"] = "no_joints" if "没有关节" in x["detail"] else ("torso_ground" if "躯干" in x["detail"] else "other")
    return picks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry", action="store_true")
    p.add_argument("--threads", type=int, default=6)
    a = p.parse_args()
    picks = collect()
    print(f"被 ① 筛掉的不同设计 {len(picks)} 个：LLM 变异 {sum(x['origin'] == 'llm' for x in picks)}，其他 {sum(x['origin'] == 'other' for x in picks)}")
    if a.dry:
        return
    from hyblock.mjcf import compile_design
    from mpc.mppi import MPPIConfig, evaluate_design
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "screen_check.csv"
    done = {r["design"] for r in csv.DictReader(open(out, encoding="utf-8"))} if out.exists() else set()
    cfg = MPPIConfig(threads=a.threads, episode_control_steps=250, terrain_scale=1.0)
    fields = ["first", "origin", "reason", "source", "n_seen", "fitness", "eps", "error", "seconds", "design"]
    new = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, x in enumerate(picks):
            if x["design"] in done:
                continue
            t = time.time()
            try:
                r = evaluate_design(compile_design(x["design"], max_modules=8), "flat", 2, 0, cfg)
                x.update(fitness=round(r["fitness"], 3), eps=json.dumps([round(e["score"], 2) for e in r["episodes"]]), error="")
            except Exception as ex:  # 没有关节的设计可能无法评估
                x.update(fitness="", eps="", error=f"{type(ex).__name__}: {ex}"[:200])
            x["seconds"] = round(time.time() - t, 1)
            w.writerow(x); f.flush()
            print(f"[{i + 1}/{len(picks)}] {x['origin']:5s} {x['reason']:12s} {x['first']}: "
                  f"{x['fitness'] if x['error'] == '' else 'error ' + x['error'][:60]}", flush=True)
    report(out, len(picks))


def report(out, total):
    import statistics as st
    rows = list(csv.DictReader(open(out, encoding="utf-8")))
    L = ["# ① 筛选的误杀率检验（平地；formal_flat + replicate_flat 被筛掉的设计，Windows MPC 重评）", "",
         f"- 设计 {len(rows)} / {total}（按设计文本去重）；设置：平地、2 回合、mpc_seed 0；误杀 = MPC 得分 ≥ 20", "",
         "| 来源 | 筛掉的原因 | 设计数 | 评估出错 | 得分最大 | 得分中位数 | 误杀（≥ 20） |", "|---|---|---|---|---|---|---|"]
    for o in ["llm", "other"]:
        for rs in ["torso_ground", "no_joints", "other"]:
            g = [r for r in rows if r["origin"] == o and r["reason"] == rs]
            if not g:
                continue
            ok = [float(r["fitness"]) for r in g if r["error"] == ""]
            L.append(f"| {'LLM 变异' if o == 'llm' else '其他'} | {rs} | {len(g)} | {len(g) - len(ok)} | "
                     f"{max(ok):.1f} | {st.median(ok):.1f} | {sum(v >= 20 for v in ok)} |" if ok else
                     f"| {'LLM 变异' if o == 'llm' else '其他'} | {rs} | {len(g)} | {len(g)} | — | — | — |")
    ok = [float(r["fitness"]) for r in rows if r["error"] == ""]
    L += ["", f"- 合计误杀 {sum(v >= 20 for v in ok)} / {len(ok)}（可评估的设计）", ""]
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
