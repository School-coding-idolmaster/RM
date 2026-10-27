#!/usr/bin/env bash
# 10-27: RM reference domain, MPC with 20 Newton iterations (was the XML's 4) + max_speed 10.
# Called by D:\RM\iter20.bat. Resumable. 0. offline tests -> 1. MPC on the 70 designs -> mpc_runs/run4_it20 (~1 h) -> 2. summary
set -uo pipefail
RM=/mnt/d/RM
mkdir -p "$RM/logs"
LOG="$RM/logs/iter20_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo
echo "===== iter20 start $(date '+%F %T') ====="
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
[ -d "$UV_PROJECT_ENVIRONMENT" ] || { echo "错误：找不到虚拟环境"; exit 1; }
cd "$RM/robomorph" || exit 1
if pgrep -f "sac_batch\.py|sac_long\.py|mpc\.eval_batch|hyblock\.calibrate|solver_test" >/dev/null; then
  echo "错误：已有另一个 SAC / MPC 任务在运行，停止"; exit 1; fi
echo "--- 0. 离线测试 $(date '+%T') ---"
uv run --locked python -m unittest discover -s tests >/tmp/iter20_tests.txt 2>&1 \
  || { grep -v -i "warn" /tmp/iter20_tests.txt | tail -40; echo "离线测试失败，停止"; exit 1; }
tail -3 /tmp/iter20_tests.txt
echo "--- 1. MPC（20 次迭代，max_speed 10）$(date '+%T') ---"
uv run --locked python -m mpc.eval_batch --xml_dir ../sac_runs/run1 --out ../mpc_runs/run4_it20 \
  --terrain flat --episodes 3 --seed 0 --threads 8 --steps 250 --samples 64 \
  --fall_penalty 100 --noise_warm 0.5 --noise_hist 0.1 --max_speed 10 --min_iterations 20 2>&1 \
  | grep -v "^WARNING: Nan\|^$" || echo "MPC 中途出错（重新双击会补跑缺的）"
echo "--- 2. 汇总 $(date '+%T') ---"
uv run --locked python ../tools/iter20_summary.py
echo "===== iter20 end $(date '+%F %T') ====="
