#!/usr/bin/env bash
# 第 7 周 Mac 任务 2（约 4–5 小时）：打开自碰撞（10-28 决定）之后的重新验证、筛选误杀检查、重新标定。
#   0 若 week7_mac1 还在跑：等它的平地阶段（calib2/flat.csv 123 行）跑完再停掉它——它的地形阶段是\"无自碰撞\"的旧物理，
#     不再需要；平地阶段留作参照（与 Windows 10-26 标定同一批设计，可做 Hyblock 域跨机器比较）
#   1 离线测试（全部）；不通过就停
#   2 模型验证 200 个随机设计 → ~/RM/mac_runs/validate_sc
#   3 示例设计打分（新物理）→ ~/RM/mac_runs/example_sc
#   4 ① 筛选误杀检查：200 个随机设计（上限 6）+ 3 个手工设计，全部跑 MPC（每设计 2 回合），记录筛选结论
#     → ~/RM/mac_runs/screen_audit
#   5 重新标定（新物理）：随机 120 + 手工 3，平地 3 回合；能走的在沟坎 / 横梁 × {1.0, 1.25, 1.5, 2, 3} 各 1 回合
#     → ~/RM/mac_runs/calib_sc
# 启动：Windows 双击 run_mac.bat → 输入 week7_mac2。可中断后重新启动（已完成的部分会跳过）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
mkdir -p "$RUNS"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
py() { uv run --locked python "$@" 2>&1 | grep -v -i warn; }

# 0 ------------------------------------------------------------------------------------------
if pgrep -f "wj_tools/week7_mac1.sh" >/dev/null; then
  echo "=== 0 week7_mac1 还在运行：等它的平地阶段结束 $(date '+%T')"
  while pgrep -f "wj_tools/week7_mac1.sh" >/dev/null; do
    n=0; [ -f "$RUNS/calib2/flat.csv" ] && n=$(($(wc -l < "$RUNS/calib2/flat.csv") - 1))
    [ "$n" -ge 123 ] && break
    sleep 60
  done
  if pgrep -f "wj_tools/week7_mac1.sh" >/dev/null; then
    echo "    平地阶段已完成，停止 week7_mac1 $(date '+%T')"
    pkill -f "wj_tools/week7_mac1.sh"; pkill -f "hyblock.calibrate"; pkill -f "hyblock.score"; sleep 5
  fi
  [ -f "$RUNS/calib2/flat.csv" ] && py -m hyblock.calibrate --out "$RUNS/calib2" --summary_only > /dev/null
  echo "stopped by week7_mac2: flat phase only (old physics, no self-collision)" > "$RUNS/calib2/STOPPED"
fi

# 1 ------------------------------------------------------------------------------------------
echo "=== 1 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests_week7.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests_week7.txt" | tail -4
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/tests_week7.txt）"; exit 1; }

# 2 ------------------------------------------------------------------------------------------
out="$RUNS/validate_sc"
if [ ! -f "$out/DONE" ]; then
  echo "=== 2 模型验证 $(date '+%T')"
  py wj_tools/run_meta.py start "$out" --threads 1 --note "hyblock.validate with self-collision"
  py -m hyblock.validate --out "$out" --n 200
  [ -f "$out/report.txt" ] && py wj_tools/run_meta.py done "$out"
fi

# 3 ------------------------------------------------------------------------------------------
out="$RUNS/example_sc"
if [ ! -f "$out/DONE" ]; then
  echo "=== 3 示例设计打分 $(date '+%T')"
  py wj_tools/run_meta.py start "$out" --threads 12 --note "example design, self-collision on (flat, 3 episodes)"
  py -m hyblock.score --from_prompt --out "$out" --episodes 3 --seed 0 --threads 12
  [ -f "$out/result.json" ] && py wj_tools/run_meta.py done "$out"
fi

# 4 ------------------------------------------------------------------------------------------
if [ ! -f "$RUNS/screen_audit/DONE" ]; then
  echo "=== 4 筛选误杀检查 $(date '+%T')"
  py -m hyblock.batch --random 200 --seed 1 --hand --max_modules 6 --screen record --episodes 2 \
     --threads 12 --out "$RUNS/screen_audit"
fi

# 5 ------------------------------------------------------------------------------------------
out="$RUNS/calib_sc"
if [ ! -f "$out/DONE" ]; then
  echo "=== 5 重新标定 $(date '+%T')"
  py wj_tools/run_meta.py start "$out" --threads 12 --note "calibration with self-collision: n=120, scales 1.0-3"
  py -m hyblock.calibrate --out "$out" --n 120 --seed 0 --mpc_seed 0 --episodes 3 --threads 12 \
     --scales 1.0,1.25,1.5,2,3
  [ -f "$out/summary.txt" ] && grep -q "地形" "$out/summary.txt" && py wj_tools/run_meta.py done "$out"
fi
echo "=== end $(date '+%F %T')"
