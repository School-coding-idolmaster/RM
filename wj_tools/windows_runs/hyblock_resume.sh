#!/usr/bin/env bash
# 10-26 关机后续跑：只跑 hyblock_week5 剩下的地形标定（phase B），已完成的行自动跳过，最后写 summary。
# 由 D:\RM\hyblock_resume.bat 调用。参数与 hyblock_week5.sh 的标定一步相同。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/robomorph"
cd /mnt/d/RM/robomorph || exit 1
OUT=/mnt/d/RM/hyblock_runs
LOG="$OUT/run_log.txt"
{
  echo "=== resume after shutdown $(date '+%F %T'): calibrate phase B only"
  uv run --locked python -m hyblock.calibrate --out "$OUT/calib" --n 60 --threads 8 --phase B
  echo "=== end $(date '+%F %T')"
} 2>&1 | tee -a "$LOG"
