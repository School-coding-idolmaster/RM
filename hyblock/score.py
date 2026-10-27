"""给指定的 Hyblock 设计做 MPC 打分（平地），用于参照线，例如系统提示里的示例设计。

用法（WSL，在 D:\\RM\\robomorph 下）：
    uv run --locked python -m hyblock.score --from_prompt --out ../hyblock_runs/example
    uv run --locked python -m hyblock.score --design "h0=root; m1=h0.zp@90 mirror; foot=m1.P2" --out ../hyblock_runs/x
输出 result.json（每回合得分、位移、是否终止、用时）和 prompt_sha256.txt（--from_prompt 时，冻结提示词版本用）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from hyblock.design import summary
from hyblock.llm_io import PROMPT_DIR, extract_design
from hyblock.mjcf import compile_design
from mpc.mppi import MPPIConfig, evaluate_design


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--design", help="design text")
    p.add_argument("--from_prompt", action="store_true", help="score the example DESIGN in hyblock/prompts/system.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--max_modules", type=int, default=8)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.from_prompt:
        raw = (PROMPT_DIR / "system.txt").read_bytes()
        (out / "prompt_sha256.txt").write_text(hashlib.sha256(raw).hexdigest() + "  hyblock/prompts/system.txt\n")
        text = extract_design(raw.decode("utf-8"))
    else:
        text = a.design
    if not text:
        p.error("no design given")
    t0 = time.time()
    r = evaluate_design(compile_design(text, max_modules=a.max_modules), "flat", a.episodes, a.seed,
                        MPPIConfig(threads=a.threads))
    res = {"design": text, "summary": summary(text), "fitness": r["fitness"],
           "fitness_first2": float(np.mean([e["score"] for e in r["episodes"][:2]])),
           "distance_m": float(np.mean([e["distance"] for e in r["episodes"]])),
           "episodes": r["episodes"], "seconds": time.time() - t0}
    (out / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{text}\nfitness {res['fitness']:.1f} (episodes {[round(e['score'], 1) for e in r['episodes']]}), "
          f"distance {res['distance_m']:.2f} m, {res['seconds']:.0f} s")


if __name__ == "__main__":
    main()
