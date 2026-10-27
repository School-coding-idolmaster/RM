"""Train SAC for longer (default 1M steps) on selected designs of sac_runs/run1, same reduced config as
run1 / long1M (64 envs, episode 200, 8 grad updates, SAC seed 0, flat). Resumable: designs already in
<out>/results.csv are skipped. Appends to <out>/results.csv (design,reward_1M,seconds) and <out>/run_log.txt.

Usage (from robomorph/):
    uv run --locked python ../tools/sac_long.py --designs 20,21,57,66,2,19,25
"""
import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", required=True, help="comma-separated design indices")
    p.add_argument("--xml_dir", default="../sac_runs/run1")
    p.add_argument("--out", default="../sac_runs/long1M")
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--num_evals", type=int, default=10)
    a = p.parse_args()

    out = Path(a.out).resolve()
    (out / "runs").mkdir(parents=True, exist_ok=True)
    res = out / "results.csv"
    done = set()
    if res.exists():
        with open(res, newline="") as f:
            done = {r["design"] for r in csv.DictReader(f)}
    for idx in [int(x) for x in a.designs.split(",")]:
        name = f"design_{idx:02d}"
        if name in done:
            print(f"[{name}] already in {res.name}, skip", flush=True)
            continue
        xml = (Path(a.xml_dir) / f"{name}.xml").resolve()
        if not xml.exists():
            sys.exit(f"missing {xml}")
        stale = out / "runs" / f"{name}_s_0"
        if stale.exists():
            shutil.rmtree(stale)
        cmd = [sys.executable, str(HERE / "sac_batch.py"), "--worker", "--xml", str(xml), "--name", name,
               "--run_dir", str(out / "runs"), "--steps", str(a.steps), "--envs", "64", "--ep", "200",
               "--updates", "8", "--num_evals", str(a.num_evals), "--terrain", "flat", "--sac_seed", "0"]
        print(f"\n[{name}] SAC {a.steps} steps ...", flush=True)
        t, reward = time.time(), None
        with open(out / f"{name}.log", "w") as lf, open(out / "run_log.txt", "a") as rl:
            rl.write(f"=== {name}  start {time.strftime('%H:%M')}\n")
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout:
                lf.write(line)
                if line.startswith(("Step:", "RESULT")):
                    print(f"  {line.rstrip()}", flush=True)
                    rl.write(line)
                    rl.flush()
                elif line.startswith("Traceback") or "Error" in line:
                    print(f"  {line.rstrip()}", flush=True)
                if line.startswith("RESULT"):
                    reward = float(line.split()[1])
            proc.wait()
        sec = time.time() - t
        if proc.returncode == 0 and reward is not None:
            new_file = not res.exists()
            with open(res, "a", newline="") as f:
                w = csv.writer(f)
                if new_file:
                    w.writerow(["design", "reward_1M", "seconds"])
                w.writerow([name, reward, round(sec)])
            print(f"[{name}] done: reward={reward:.2f}, {sec / 60:.1f} min", flush=True)
        else:
            print(f"[{name}] FAILED (see {name}.log); continuing", flush=True)


if __name__ == "__main__":
    main()
