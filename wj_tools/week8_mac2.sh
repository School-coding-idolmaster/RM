#!/usr/bin/env bash
# 第 8 周 Mac 任务 2（约 15–20 小时）：三个基线，按 10-28 晚的三个决定重跑——
#   模块数上限 6 → 8；提示词 v2（加\"零件不能互相穿透\"与模块计数说明）；重复设计占预算（沿用旧分数）。
#   随机搜索 / GA / RM 式 LLM × 种子 0、1、2，每个运行预算 200 次评估，每代 16 个候选，平地 s = 1.0。
#   开头先停掉还在跑的 week8_mac1（上限 6、提示词 v1 的试跑，结果留在 base_flat/ 作参照）。
#   结果：~/RM/mac_runs/base_flat8/<方法>_s<种子>/
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week8_mac2 → wait_mac.bat。可中断后重新启动（从断点继续）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat8"
mkdir -p "$RUNS"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
# 10-29：第一次启动在 01:08 无报错中断，记录开机时间与最近的睡眠 / 关机事件，便于判断原因
echo "--- uptime: $(uptime)"
last reboot 2>/dev/null | head -3 | sed 's/^/--- /'
pmset -g log 2>/dev/null | grep -E " (Sleep|Wake|Shutdown|Restart) " | grep "$(date -v-1d '+%Y-%m-%d')\|$(date '+%Y-%m-%d')" | tail -8 | sed 's/^/--- /'

if pgrep -f "wj_tools/week8_mac1.sh" >/dev/null; then
  echo "=== 停止 week8_mac1 $(date '+%T')"
  pkill -f "wj_tools/week8_mac1.sh"; sleep 1; pkill -f "hyblock.search"; sleep 5
  echo "stopped by week8_mac2 (cap 6 / prompt v1 / duplicates free: pilot only)" > "$(cd .. && pwd)/mac_runs/base_flat/STOPPED"
fi

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/base_flat8/tests.txt）"; exit 1; }

have_key=0
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null && have_key=1
[ $have_key -eq 1 ] || echo "!!! .env 里没有 ANTHROPIC_API_KEY：跳过 llm（Windows 双击 copy_env_to_mac.bat 后重新启动本任务即可补跑）"

for s in 0 1 2; do
  for m in llm random ga; do
    out="$RUNS/${m}_s${s}"
    [ -f "$out/DONE" ] && continue
    if [ "$m" = llm ] && [ $have_key -eq 0 ]; then continue; fi
    echo "=== ${m} 种子 ${s} $(date '+%T')"
    uv run --locked python -m hyblock.search --method "$m" --seed "$s" --budget 200 --batch 16 --max_modules 8 \
        --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
  done
done
echo "=== end $(date '+%F %T')"
