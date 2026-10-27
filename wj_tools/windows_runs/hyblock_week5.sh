#!/usr/bin/env bash
# 第 5–6 周：Hyblock（水力君）仿真模型的离线测试 + 模型验证 + 标定批量。由 D:\RM\hyblock_week5.bat 调用。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/robomorph"
cd /mnt/d/RM/robomorph || exit 1
OUT=/mnt/d/RM/hyblock_runs
mkdir -p "$OUT"
LOG="$OUT/run_log.txt"
{
  echo "=== start $(date '+%F %T')  cpus=$(nproc)"
  echo "=== offline tests"
  uv run --locked python -m unittest discover -s tests 2>&1 | grep -v -i "warn" 
  echo "=== validate (pipeline 3.5 items 1-4)"
  uv run --locked python -m hyblock.validate --out "$OUT/validate" --n 200
  echo "=== LLM + MPC, Hyblock domain: 1 generation x 4 designs (Claude Haiku 5.5, flat)"
  uv run --locked python -m robomorph.main --exp_name hyb_llmtest1 --domain hyblock --evaluator mpc \
      --model claude-haiku-5-5 --no_log -n 1 --population_size 4 --max_modules 6 \
      --results_path "$OUT/llmtest1/results.csv" 2>&1 | grep -v -i "warn"
  echo "=== calibrate (flat 3 episodes x 63 designs, then terrains for the walkers)"
  uv run --locked python -m hyblock.calibrate --out "$OUT/calib" --n 60 --threads 8
  echo "=== end $(date '+%F %T')"
} 2>&1 | tee -a "$LOG"
