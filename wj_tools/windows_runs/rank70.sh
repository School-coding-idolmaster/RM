#!/usr/bin/env bash
# RM 参照域：把 MPC–SAC 排序一致性检验从 20 个设计补到 70 个（2026-10-26 事先定好 n = 70）。
# 由 D:\RM\rank70.bat 调用。可中断：重新双击会跳过已完成的设计接着跑。
#   1. 等 hyblock_week5 跑完（避免抢 CPU、干扰 Hyblock 计时）
#   2. 用同一生成器、同一种子扩到 70 个设计，并确认前 20 个逐字节一致
#   3. MPC：噪声 ×2、3 回合（与 run2_x2 相同），结果追加到 mpc_runs/run2_x2，约 45 分钟
#   4. SAC：50 万步缩小配置（与 run1 相同），结果追加到 sac_runs/run1，约 13 小时
#   5. 汇总：mpc_runs/run2_x2/summary_70.txt、paired_70.csv
set -uo pipefail

RM=/mnt/d/RM
LOG="$RM/rank70_log.txt"
exec > >(tee -a "$LOG") 2>&1

echo
echo "===== rank70 start $(date '+%F %T') ====="

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
if [ ! -d "$UV_PROJECT_ENVIRONMENT" ]; then
  echo "错误：找不到虚拟环境 $UV_PROJECT_ENVIRONMENT（为防止 uv 在 D 盘另建 .venv，停止）"
  exit 1
fi
command -v uv >/dev/null || { echo "错误：找不到 uv"; exit 1; }
cd "$RM/robomorph" || exit 1

# 1. 等 week5
while pgrep -f "hyblock_week5|hyblock/calibrate|hyblock\.calibrate|hyblock/validate|hyblock\.validate" >/dev/null; do
  echo "$(date '+%T') hyblock_week5 还在运行，5 分钟后再看"
  sleep 300
done
if pgrep -f "sac_batch\.py|mpc\.eval_batch" >/dev/null; then
  echo "错误：已有另一个 SAC / MPC 批量任务在运行，停止（不要同时开两个窗口）"
  exit 1
fi

# MPC 设置必须与 run2_x2 一致（eval_batch 会重写 config.json，先备份并核对）
CFG="$RM/mpc_runs/run2_x2/config.json"
if [ -f "$CFG" ]; then
  [ -f "$RM/mpc_runs/run2_x2/config_run20.json" ] || cp "$CFG" "$RM/mpc_runs/run2_x2/config_run20.json"
  uv run --locked python - "$RM/mpc_runs/run2_x2/config_run20.json" <<'PY' || exit 1
import json, sys
a = json.load(open(sys.argv[1]))["args"]
want = {"terrain": "flat", "episodes": 3, "seed": 0, "steps": 250, "samples": 64, "mjx_solver": False,
        "fall_penalty": 100.0, "noise_warm": 0.5, "noise_hist": 0.1, "no_filter": False}
bad = {k: (a.get(k), v) for k, v in want.items() if a.get(k) != v}
if bad:
    print("错误：run2_x2 的设置与本脚本不一致（旧值, 本脚本）：", bad); sys.exit(1)
print("run2_x2 的 MPC 设置核对一致")
PY
else
  echo "错误：找不到 $CFG"; exit 1
fi

# 2. 扩到 70 个设计
echo "--- 2. 生成设计 $(date '+%T') ---"
uv run --locked python ../tools/extend_designs.py --n 70 --seed 0 --dir ../sac_runs/run1 || { echo "设计生成/核对失败，停止"; exit 1; }

# 3. MPC
echo "--- 3. MPC $(date '+%T') ---"
uv run --locked python -m mpc.eval_batch --xml_dir ../sac_runs/run1 --out ../mpc_runs/run2_x2 \
  --terrain flat --episodes 3 --seed 0 --threads 8 --steps 250 --samples 64 \
  --fall_penalty 100 --noise_warm 0.5 --noise_hist 0.1 || echo "MPC 批量中途出错（继续跑 SAC；重新双击会补跑缺的）"

# 4. SAC
echo "--- 4. SAC $(date '+%T') ---"
uv run --locked python ../tools/sac_batch.py --n 70 --seed 0 --steps 500000 --envs 64 --ep 200 \
  --updates 8 --sac_seed 0 --terrain flat --out ../sac_runs/run1 || echo "SAC 批量中途出错（重新双击会补跑缺的）"

# 5. 汇总
echo "--- 5. 汇总 $(date '+%T') ---"
uv run --locked python ../tools/rank70_summary.py

echo "===== rank70 end $(date '+%F %T') ====="
