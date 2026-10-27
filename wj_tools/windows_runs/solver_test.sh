#!/usr/bin/env bash
set -uo pipefail
LOG=/mnt/d/RM/solver_test_log.txt
exec > >(tee -a "$LOG") 2>&1
echo "===== solver_test start $(date '+%F %T') ====="
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd /mnt/d/RM/robomorph || exit 1
if pgrep -f "sac_batch\.py|sac_long\.py|mpc\.eval_batch|hyblock\.calibrate|speed10\.sh|rank70\.sh" >/dev/null; then
  echo "错误：已有另一个 SAC / MPC 任务在运行，停止"; exit 1; fi
rm -f /mnt/d/RM/mpc_runs/solver_test/results.jsonl /mnt/d/RM/mpc_runs/solver_test/episodes.jsonl
uv run --locked python ../tools/solver_test.py 2>&1 | grep -v "^WARNING: Nan\|^$"
echo "===== solver_test end $(date '+%F %T') ====="
