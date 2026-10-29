#!/usr/bin/env bash
# 第 9 周 Mac 任务 2（约 4–5 小时）：按种类引导的 LLM 变异（WJ 10-30 选的方案 A）× 种子 0、1、2
#   → base_flat8/ga_llmtyped_s<种子>/。与 ga_s<种子>（普通 GA）、ga_llmmut_s<种子>（第 9 周自由形式版）配对：
#   同一第 0 代、预算 200、每代 16、上限 8、① 筛选 v1、MPC 2 回合。
#   与自由形式版的区别：修改种类按随机变异同样的 9 种均匀抽，LLM 只决定改哪里、怎么改（种类不符按无效处理、带原因重问，
#   仍不行退回同一种类的随机变异）；系统提示 = v2 的零件 / 规则部分 + prompts/mutate_typed.txt（写明初始角只决定起始姿态）。
#   原因：自由形式版 64% 的修改是初始关节角，搜索在第 33–143 次评估后停滞（第四阶段（中）记录 第 8 节）。
#   开头：等 week9_mac1 跑完 → 离线测试 → 冒烟检查（ga_s0 前 4 代当历史问一代；第 1 次通过率 < 0.3 就停止）。
#   最后写对比报告 base_flat8/compare_week9b/。
# 启动：week9_mac1 跑完（wait_mac 弹窗）后，Windows 双击 push_tools.bat → run_mac.bat 输入 week9_mac2 → wait_mac.bat。
#   可中断后重新启动（从断点继续）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat8"
SMOKE="$(cd .. && pwd)/mac_runs/week9_smoke_typed"
mkdir -p "$RUNS" "$SMOKE"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"
last reboot 2>/dev/null | head -3 | sed 's/^/--- /'

# 同一时间只让一个任务吃满 CPU（工作流规则 5）
while pgrep -f "wj_tools/week9_mac1.sh" >/dev/null || pgrep -f "hyblock.search" >/dev/null; do
  echo "--- week9_mac1 / 别的 hyblock.search 还在跑，等待 $(date '+%T')"; sleep 600
done

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests_week9b.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests_week9b.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/base_flat8/tests_week9b.txt）"; exit 1; }

grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止（Windows 双击 copy_env_to_mac.bat 后重新启动本任务）"; exit 1; }

if [ ! -f "$SMOKE/OK" ]; then
  echo "=== 按种类引导的 LLM 变异：冒烟检查 $(date '+%T')"
  uv run --locked python -m hyblock.llm_mutation --mutation llm_typed --history "$RUNS/ga_s0/evals.csv" --gen 4 \
      --out "$SMOKE" > "$SMOKE/smoke.txt" 2>&1
  rc=$?
  grep -v -i warn "$SMOKE/smoke.txt"
  [ $rc -eq 0 ] || { echo "!!! 冒烟检查没通过，停止（详见 mac_runs/week9_smoke_typed/smoke.txt，交给 Claude）"; exit 1; }
  date '+%F %T' > "$SMOKE/OK"
fi

for s in 0 1 2; do
  out="$RUNS/ga_llmtyped_s${s}"
  [ -f "$out/DONE" ] && continue
  echo "=== GA + 按种类引导的 LLM 变异 种子 ${s} $(date '+%T')"
  uv run --locked python -m hyblock.search --method ga --mutation llm_typed --seed "$s" --budget 200 --batch 16 \
      --max_modules 8 --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
done

echo "=== 对比报告 $(date '+%T')"
uv run --locked python -m hyblock.compare_search "$RUNS" --out "$RUNS/compare_week9b" > /dev/null 2>&1 \
  && echo "写好了 mac_runs/base_flat8/compare_week9b/report.md" || echo "对比报告没写成（不影响结果，Windows 上重做即可）"
echo "=== end $(date '+%F %T')"
