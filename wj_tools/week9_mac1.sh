#!/usr/bin/env bash
# 第 9 周 Mac 任务 1（约 8–11 小时）：第四阶段（中）前半（pipeline v4.6 的 2 × 2 设计，先做\"LLM 变异\"一项）。
#   ① GA + LLM 变异算子（search.py --method ga --mutation llm）× 种子 0、1、2 → base_flat8/ga_llmmut_s<种子>/
#      与 week8_mac2 的普通 GA（ga_s<种子>）配对：同一第 0 代、预算 200、每代 16、上限 8、① 筛选 v1、MPC 2 回合；
#      只把\"随机变异\"换成\"LLM 变异\"（看亲本、得分、好设计、失败例子、相似的已评估设计；无效则带原因重问 1 次，
#      仍无效退回随机变异）。系统提示 = 提示词 v2 的零件 / 规则部分 + prompts/mutate.txt
#   ② RM 式最小忠实版（--method llm --no_scores：示例不写得分，其余与公开代码版 llm 相同）× 种子 0、1、2
#      → base_flat8/llm_noscore_s<种子>/。第 0 代提示词与 llm_s<种子> 相同：复制它的 LLM 缓存，第 0 代重放同样的回复（配对）
#   开头：离线测试 → LLM 变异冒烟检查（ga_s0 的前 4 代当历史，对第 4 代问一次，约 10–15 次调用、约 $0.03；
#   第 1 次就通过的比例 < 0.3 时停止，避免整夜跑成\"退回随机变异\"的 GA）。最后写对比报告 base_flat8/compare_week9/。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week9_mac1 → wait_mac.bat。可中断后重新启动（从断点继续）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat8"
SMOKE="$(cd .. && pwd)/mac_runs/week9_smoke"
mkdir -p "$RUNS" "$SMOKE"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"
last reboot 2>/dev/null | head -3 | sed 's/^/--- /'
pmset -g log 2>/dev/null | grep -E " (Sleep|Wake|Shutdown|Restart) " | grep "$(date -v-1d '+%Y-%m-%d')\|$(date '+%Y-%m-%d')" | tail -8 | sed 's/^/--- /'

# 同一时间只让一个任务吃满 CPU（工作流规则 5）
while pgrep -f "hyblock.search" >/dev/null; do
  echo "--- 还有别的 hyblock.search 在跑，等待 $(date '+%T')"; sleep 600
done

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests_week9.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests_week9.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/base_flat8/tests_week9.txt）"; exit 1; }

grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止（Windows 双击 copy_env_to_mac.bat 后重新启动本任务）"; exit 1; }

if [ ! -f "$SMOKE/OK" ]; then
  echo "=== LLM 变异冒烟检查 $(date '+%T')"
  uv run --locked python -m hyblock.llm_mutation --history "$RUNS/ga_s0/evals.csv" --gen 4 --out "$SMOKE" \
      > "$SMOKE/smoke.txt" 2>&1
  rc=$?
  grep -v -i warn "$SMOKE/smoke.txt"
  [ $rc -eq 0 ] || { echo "!!! 冒烟检查没通过，停止（详见 mac_runs/week9_smoke/smoke.txt，交给 Claude）"; exit 1; }
  date '+%F %T' > "$SMOKE/OK"
fi

for s in 0 1 2; do
  out="$RUNS/ga_llmmut_s${s}"
  [ -f "$out/DONE" ] && continue
  echo "=== GA + LLM 变异 种子 ${s} $(date '+%T')"
  uv run --locked python -m hyblock.search --method ga --mutation llm --seed "$s" --budget 200 --batch 16 \
      --max_modules 8 --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
done

for s in 0 1 2; do
  out="$RUNS/llm_noscore_s${s}"
  [ -f "$out/DONE" ] && continue
  echo "=== RM 式（示例不写得分）种子 ${s} $(date '+%T')"
  if [ ! -d "$out/llm_cache" ] && [ -d "$RUNS/llm_s${s}/llm_cache" ]; then
    mkdir -p "$out" && cp -R "$RUNS/llm_s${s}/llm_cache" "$out/llm_cache"
  fi
  uv run --locked python -m hyblock.search --method llm --no_scores --seed "$s" --budget 200 --batch 16 \
      --max_modules 8 --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
done

echo "=== 对比报告 $(date '+%T')"
uv run --locked python -m hyblock.compare_search "$RUNS" --out "$RUNS/compare_week9" > /dev/null 2>&1 \
  && echo "写好了 mac_runs/base_flat8/compare_week9/report.md" || echo "对比报告没写成（不影响结果，Windows 上重做即可）"
echo "=== end $(date '+%F %T')"
