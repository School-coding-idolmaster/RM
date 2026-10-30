#!/usr/bin/env bash
# 第 10 周 Mac 任务 1（约 1 小时，只调用 LLM API，几乎不占 CPU；API 约 $2）：② LLM 挑选的离线回放。
#   用 base_flat8 的 ga_s0–2、ga_llmtyped_s0–2（每代 16 个候选都有 MPC 得分）逐代重放挑选（共 72 道题）：
#   挑选器只看本代以前的记录、给本代候选排序，与真实得分比较（Spearman、前一半召回、gain）。
#   LLM 的设置：经验池 8 / 16 / 24 × 全部排序 / 只挑前一半 × 1 / 3 票；对照 random、parent、regress、knn。
#   开头：等别的 hyblock.search 结束 → 离线测试 → 冒烟检查（第一道题，ok 比例 < 0.8 就停止）。
#   输出：mac_runs/select_replay/report.md（+ problems.csv、calls.csv、llm_cache/）。结果决定 week10_mac2 的 ② 设置。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week10_mac1 → wait_mac.bat。可中断后重新启动（LLM 回复有缓存）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat8"
OUT="$(cd .. && pwd)/mac_runs/select_replay"
SMOKE="$(cd .. && pwd)/mac_runs/select_replay_smoke"
mkdir -p "$OUT" "$SMOKE"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"

while pgrep -f "hyblock.search" >/dev/null; do
  echo "--- 别的 hyblock.search 还在跑，等待 $(date '+%T')"; sleep 600
done

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$OUT/tests_week10.txt" 2>&1
rc=$?
grep -v -i warn "$OUT/tests_week10.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/select_replay/tests_week10.txt）"; exit 1; }

grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止（Windows 双击 copy_env_to_mac.bat 后重新启动本任务）"; exit 1; }

RUNLIST="$RUNS/ga_s0 $RUNS/ga_s1 $RUNS/ga_s2 $RUNS/ga_llmtyped_s0 $RUNS/ga_llmtyped_s1 $RUNS/ga_llmtyped_s2"
if [ ! -f "$SMOKE/OK" ]; then
  echo "=== 冒烟检查 $(date '+%T')"
  uv run --locked python -m hyblock.select_replay --smoke --runs $RUNLIST --out "$SMOKE" > "$SMOKE/smoke.txt" 2>&1
  rc=$?
  grep -v -i warn "$SMOKE/smoke.txt" | tail -12
  [ $rc -eq 0 ] || { echo "!!! 冒烟检查没通过，停止（详见 mac_runs/select_replay_smoke/smoke.txt，交给 Claude）"; exit 1; }
  date '+%F %T' > "$SMOKE/OK"
fi

echo "=== 离线回放 $(date '+%T')"
uv run --locked python -m hyblock.select_replay --runs $RUNLIST --out "$OUT" --workers 8 2>&1 | grep -v -i warn | grep -E "道题|组（" 
[ -f "$OUT/report.md" ] && echo "写好了 mac_runs/select_replay/report.md" || { echo "!!! 回放没有写出 report.md"; exit 1; }
date '+%F %T' > "$OUT/DONE"
echo "=== end $(date '+%F %T')"
