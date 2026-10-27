"""MPC counterpart of train.parallel_train.train_multiple_models for the evolution loop.

Returns [(xml_path, fitness), ...] in the input order. A design that raises gets fitness -inf and a
failure_<name>.json with an "error" key in output_dir, the same contract the loop already uses for
failed SAC runs. Per-design details go to mpc_<name>.json.
"""
from __future__ import annotations

import json
import math
import os
import time
import traceback
from pathlib import Path

from mpc.mppi import MPPIConfig, evaluate_design


def evaluate_multiple_designs(xml_paths, *, terrain="flat", seed=0, episodes=3, threads=8,
                              output_dir="runs", cfg: MPPIConfig | None = None, terrain_scale: float = 1.0):
    os.makedirs(output_dir, exist_ok=True)
    cfg = cfg or MPPIConfig(threads=threads, terrain_scale=terrain_scale)
    results = []
    for path in xml_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        t0 = time.time()
        try:
            r = evaluate_design(Path(path).read_text(), terrain, episodes, seed, cfg)
            fitness = r["fitness"]
            if not math.isfinite(fitness):
                raise ValueError(f"MPC returned nonfinite fitness: {fitness}")
            with open(os.path.join(output_dir, f"mpc_{name}.json"), "w") as f:
                json.dump({"fitness": fitness, "fitness_first200": r["fitness_first200"], "nu": r["nu"],
                           "seconds": time.time() - t0, "episodes": r["episodes"]}, f, indent=2)
        except Exception as e:
            fitness = float("-inf")
            with open(os.path.join(output_dir, f"failure_{name}.json"), "w") as f:
                json.dump({"error": f"MPC evaluation failed: {e}", "traceback": traceback.format_exc()}, f, indent=2)
        results.append((path, fitness))
    return results
