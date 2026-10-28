#!/usr/bin/env bash
# 第 8 周 Mac 任务 1（约 8–10 小时）：三个基线的试跑（Hyblock 域，平地 s = 1.0，新物理，① 筛选开）。
#   随机搜索 / 进化算法（GA）/ RM 式 LLM（Claude Haiku 5.5）× 种子 0、1、2，每个运行预算 200 次 MPC，每代 16 个候选。
#   顺序按种子交错（llm s0 → random s0 → ga s0 → llm s1 → ...），中途停下也有每种方法的结果。
#   结果：~/RM/mac_runs/base_flat/<方法>_s<种子>/（evals.csv、summary.txt、curve.csv；llm 另有 llm_cache/）
# 前提：Mac 的 .env 有 ANTHROPIC_API_KEY（Windows 双击 copy_env_to_mac.bat）；没有的话跳过 llm、只跑另外两个。
# 启动：Windows 双击 run_mac.bat → 输入 week8_mac1；然后 wait_mac.bat。可中断后重新启动（从断点继续）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat"
mkdir -p "$RUNS"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/base_flat/tests.txt）"; exit 1; }

have_key=0
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null && have_key=1
[ $have_key -eq 1 ] || echo "!!! .env 里没有 ANTHROPIC_API_KEY：跳过 llm（Windows 双击 copy_env_to_mac.bat 后重新启动本任务即可补跑）"

for s in 0 1 2; do
  for m in llm random ga; do
    out="$RUNS/${m}_s${s}"
    [ -f "$out/DONE" ] && continue
    if [ "$m" = llm ] && [ $have_key -eq 0 ]; then continue; fi
    echo "=== ${m} 种子 ${s} $(date '+%T')"
    uv run --locked python -m hyblock.search --method "$m" --seed "$s" --budget 200 --batch 16 \
        --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
  done
done
echo "=== end $(date '+%F %T')"
