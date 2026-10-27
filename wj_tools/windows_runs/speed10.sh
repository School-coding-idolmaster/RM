#!/usr/bin/env bash
# 10-27: MPPI speed guard 20 -> 10 m/s. Called by D:\RM\speed10.bat. Resumable (double-click again).
#   0. offline tests (stop if any fail)
#   1. MPC rerun on the same 70 RM designs, max_speed 10, otherwise identical to run2_x2 -> mpc_runs/run3_v10 (~1 h)
#   2. SAC 1M on 7 designs fixed in advance: 20,21,57,66 (MPC-favoured) and 2,19,25 (SAC-favoured) -> sac_runs/long1M (~3.5 h)
#   3. summary -> mpc_runs/run3_v10/summary_speed10.txt
set -uo pipefail
RM=/mnt/d/RM
LOG="$RM/speed10_log.txt"
exec > >(tee -a "$LOG") 2>&1
echo
echo "===== speed10 start $(date '+%F %T') ====="
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
[ -d "$UV_PROJECT_ENVIRONMENT" ] || { echo "错误：找不到虚拟环境 $UV_PROJECT_ENVIRONMENT"; exit 1; }
command -v uv >/dev/null || { echo "错误：找不到 uv"; exit 1; }
cd "$RM/robomorph" || exit 1
if pgrep -f "sac_batch\.py|sac_long\.py|mpc\.eval_batch|hyblock\.calibrate|hyblock_week5|rank70\.sh" >/dev/null; then
  echo "错误：已有另一个 SAC / MPC 任务在运行，停止（关掉其他窗口；仍不行就在 cmd 里执行 wsl --shutdown 再双击）"
  exit 1
fi

echo "--- 0. 离线测试 $(date '+%T') ---"
uv run --locked python -m unittest discover -s tests >/tmp/speed10_tests.txt 2>&1 \
  || { grep -v -i "warn" /tmp/speed10_tests.txt | tail -40; echo "离线测试失败，停止"; exit 1; }
tail -3 /tmp/speed10_tests.txt

echo "--- 1. MPC 重跑（max_speed 10）$(date '+%T') ---"
uv run --locked python -m mpc.eval_batch --xml_dir ../sac_runs/run1 --out ../mpc_runs/run3_v10 \
  --terrain flat --episodes 3 --seed 0 --threads 8 --steps 250 --samples 64 \
  --fall_penalty 100 --noise_warm 0.5 --noise_hist 0.1 --max_speed 10 2>&1 | grep -v "^WARNING: Nan\|^$" \
  || echo "MPC 中途出错（继续；重新双击会补跑缺的）"

echo "--- 2. SAC 100 万步（7 个设计）$(date '+%T') ---"
uv run --locked python ../tools/sac_long.py --designs 20,21,57,66,2,19,25 || echo "SAC 中途出错（重新双击会补跑缺的）"

echo "--- 3. 汇总 $(date '+%T') ---"
uv run --locked python ../tools/speed10_summary.py
echo "===== speed10 end $(date '+%F %T') ====="
