#!/usr/bin/env bash
# 第 10 周 Mac 任务 2（约 7 小时）：② 挑选的在线试跑 × 种子 0、1、2 → base_flat8/ga_sel_<挑选器>_s<种子>/。
#   GA（随机变异）+ ② 挑选：每代提出 32 个（2 倍），① 筛选后送 16 个做 MPC（随机名额 2，其余按挑选器排序取前 14）；
#   没挑中的记为 unselected、不占预算；第 0 代不挑选（与 ga_s<种子> 同一批，逐种子配对）。预算 200、上限 8、MPC 2 回合不变。
#   挑选器：llm（经验池 24、全部排序、3 票；按 week10_mac1 的离线回放定）与 knn（回放中最强的非 LLM 对照）。
#   对照：ga_s<种子>（第 8 周）。随机挑选在分布上等同普通 GA（从 32 个里随机挑 16 个 = 直接生成 16 个），试跑不重复跑。
#   最后写对比报告 base_flat8/compare_week10/。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week10_mac2
#   → wait_mac.bat。可中断后重新启动（从断点继续）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat8"
mkdir -p "$RUNS"
# ② LLM 设置（10-30 按 week10_mac1 的离线回放定；见 claude/第四阶段（下）记录.md 第 2 节）：
#   3 票在所有设置里都明显好于 1 票；经验池 24 + 全部排序的 Spearman 最高（0.37），与 knn 的配对 gain 差 +13；
#   全部排序便于在线用随机名额测排序质量。每个种子 LLM 挑选约 36 次调用、约 $0.1
SEL_POOL=24
SEL_MODE=rank
SEL_VOTES=3
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- ② LLM 设置：经验池 ${SEL_POOL}，${SEL_MODE}，${SEL_VOTES} 票"
echo "--- uptime: $(uptime)"

while pgrep -f "wj_tools/week10_mac1.sh" >/dev/null || pgrep -f "hyblock.search" >/dev/null || pgrep -f "hyblock.select_replay" >/dev/null; do
  echo "--- week10_mac1 / 别的搜索还在跑，等待 $(date '+%T')"; sleep 600
done

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests_week10.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests_week10.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/base_flat8/tests_week10.txt）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止（Windows 双击 copy_env_to_mac.bat 后重新启动本任务）"; exit 1; }

for sel in llm knn; do
  for s in 0 1 2; do
    out="$RUNS/ga_sel_${sel}_s${s}"
    [ -f "$out/DONE" ] && continue
    echo "=== GA + ② ${sel} 种子 ${s} $(date '+%T')"
    uv run --locked python -m hyblock.search --method ga --select "$sel" --oversample 2 --rand_slots 2 \
        --sel_pool "$SEL_POOL" --sel_mode "$SEL_MODE" --sel_votes "$SEL_VOTES" --seed "$s" --budget 200 --batch 16 \
        --max_modules 8 --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
  done
done

echo "=== 对比报告 $(date '+%T')"
uv run --locked python -m hyblock.compare_search "$RUNS" --out "$RUNS/compare_week10" > /dev/null 2>&1 \
  && echo "写好了 mac_runs/base_flat8/compare_week10/report.md" || echo "对比报告没写成（不影响结果，Windows 上重做即可）"
echo "=== end $(date '+%F %T')"
