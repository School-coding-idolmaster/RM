#!/usr/bin/env bash
# 第 7 周 Mac 任务 1（约 3 小时）：
#   1/2 示例设计打分（系统提示里的示例，平地 3 回合）→ ~/RM/mac_runs/example/result.json
#   2/2 地形缩放补测：随机设计 120 个 + 手工 3 个，平地 3 回合（阶段 A）；平地能走的设计在
#       沟坎 / 横梁 × 缩放 {1.0, 1.25, 1.5, 2, 3} 上各 1 回合（阶段 B）→ ~/RM/mac_runs/calib2/
#       五个比例全部在 Mac 上重测（同一组对比只在一台机器上跑）；前 60 个随机设计与 Windows 10-26 标定相同，
#       顺便得到 Hyblock 域的跨机器参考比较。
# 启动：Windows 双击 run_mac.bat → 输入 week7_mac1。可中断后重新启动（已完成的设计会跳过）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
mkdir -p "$RUNS"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
py() { uv run --locked python "$@" 2>&1 | grep -v -i warn; }

out="$RUNS/example"
if [ ! -f "$out/DONE" ]; then
  echo "=== 1/2 示例设计打分 $(date '+%T')"
  py wj_tools/run_meta.py start "$out" --threads 12 --note "score example design in hyblock/prompts/system.txt (flat, 3 episodes)"
  py -m hyblock.score --from_prompt --out "$out" --episodes 3 --seed 0 --threads 12
  [ -f "$out/result.json" ] && py wj_tools/run_meta.py done "$out"
fi

out="$RUNS/calib2"
if [ ! -f "$out/DONE" ]; then
  echo "=== 2/2 地形缩放补测 $(date '+%T')"
  py wj_tools/run_meta.py start "$out" --threads 12 --note "week7 terrain-scale retest: n=120, scales 1.0-3"
  py -m hyblock.calibrate --out "$out" --n 120 --seed 0 --mpc_seed 0 --episodes 3 --threads 12 \
     --scales 1.0,1.25,1.5,2,3
  [ -f "$out/summary.txt" ] && grep -q "地形" "$out/summary.txt" && py wj_tools/run_meta.py done "$out"
fi
echo "=== end $(date '+%F %T')"
