#!/usr/bin/env bash
# 地形试点（约 1.5–2 小时，不调用 API）：普通 GA，开发种子 0，沟坎（ridged）与横梁（beams）各一个运行，s = 1.0。
# 目的：看得分范围、能走比例、每次评估用时，选定地形；也作为第 12 周"LLM 用在地形上"探索的 GA 基准（第11周记录 第 9 节）。
# 结果在 mac_runs/terrain_pilot/。formal_flat 会等本任务跑完再开始，可以两个都启动。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
OUT="$(cd .. && pwd)/mac_runs/terrain_pilot"
mkdir -p "$OUT"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
while pgrep -f "wj_tools/week11_mac" >/dev/null || pgrep -f "hyblock.search" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 300
done
for t in ridged beams; do
  out="$OUT/ga_${t}_s0"
  [ -f "$out/DONE" ] && continue
  echo "=== GA ${t} 种子 0 $(date '+%T')"
  uv run --locked python -m hyblock.search --method ga --terrain "$t" --terrain_scale 1.0 --seed 0 --budget 200 --batch 16 \
      --max_modules 8 --threads 12 --out "$out" 2>&1 | grep -v -i warn
done
uv run --locked python -m hyblock.compare_search "$OUT" --out "$OUT/compare" > /dev/null 2>&1 \
  && echo "写好了 mac_runs/terrain_pilot/compare/report.md" || echo "对比报告没写成"
echo "=== end $(date '+%F %T')"
