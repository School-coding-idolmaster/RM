"""10-27: does the RM-domain glitch disappear with 20 Newton iterations (instead of the XML's 4)?
Re-scores the 4 glitch designs and 3 normal top designs with iterations=20, everything else identical to
run2_x2 (noise x2, 3 episodes, seed 0) and max_speed 20 (so a glitch is not hidden by the 10 m/s guard).
Compares with run2_x2 (iterations 4, max_speed 20). Run from robomorph/:
    uv run --locked python ../tools/solver_test.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from mpc.mppi import MPPIConfig, evaluate_design  # noqa: E402

DESIGNS = [21, 29, 52, 57, 8, 15, 31]          # 4 glitch designs + 3 normal top designs (controls)
OUT = Path("../mpc_runs/solver_test")
OUT.mkdir(parents=True, exist_ok=True)

old = {}
with open("../mpc_runs/run2_x2/episodes.jsonl") as f:
    for line in f:
        d = json.loads(line)
        old[d["design"]] = d["rewards"]

rows = []
for i in DESIGNS:
    name = f"design_{i:02d}"
    xml = Path(f"../sac_runs/run1/{name}.xml").read_text()
    assert xml.count('iterations="4"') == 1, name
    xml20 = xml.replace('iterations="4"', 'iterations="20"')
    t = time.time()
    r = evaluate_design(xml20, "flat", 3, 0, MPPIConfig(max_speed=20.0))
    sec = time.time() - t
    peaks = [round(max(x), 2) if x else 0.0 for x in r["rewards"]]
    oldr = old[name]
    row = {"design": name, "fitness_it20": round(r["fitness"], 1),
           "fitness_it4": round(sum(sum(x) for x in oldr) / len(oldr), 1),
           "peaks_it20": peaks, "peaks_it4": [round(max(x), 2) if x else 0.0 for x in oldr],
           "terminated_it20": sum(e["terminated"] for e in r["episodes"]),
           "unstable_it20": sum(e["unstable"] for e in r["episodes"]), "seconds_it20": round(sec, 1)}
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
    with open(OUT / "results.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    with open(OUT / "episodes.jsonl", "a") as f:
        f.write(json.dumps({"design": name, "episodes": r["episodes"], "rewards": r["rewards"]}) + "\n")

print("\ndesign | MPC iter4 -> iter20 | peak speed iter4 -> iter20 (m/s)")
for r in rows:
    print(f"{r['design']} | {r['fitness_it4']} -> {r['fitness_it20']} | {max(r['peaks_it4'])} -> {max(r['peaks_it20'])}")
