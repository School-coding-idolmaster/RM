#!/usr/bin/env bash
# 第 10 周 Mac 任务 3（约 5–7 小时）：WJ 10-31 选的 A + D。
#   D 诊断（约 30 分钟）：ga_sel_llm_s0–2、ga_sel_knn_s0–2 各均匀抽 10 个被刷掉（unselected）的候选，用同一评估器补跑 MPC，
#     与同代被挑中的比较（有没有"错过的突破"）→ mac_runs/base_flat8/diag_unselected/report.md。原运行不改。
#   A 按修改种类分层挑选：GA + ② LLM（经验池 24、排序、3 票，与 week10_mac2 相同）+ --sel_strata family × 种子 0–2
#     → base_flat8/ga_sel_llmstrat_s<种子>/。只改"取名额"一步：各类修改（交叉 / 删减 / 增加 / 微调）按候选数比例分名额、类内按 LLM 排序。
#     原因：week10_mac2 中 ② 偏爱增加 / 微调、压低交叉 / 删减，最好得分 LLM 451、knn 500 < GA 717（第四阶段（下）记录 第 5 节）。
#   最后写对比报告 base_flat8/compare_week10b/。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week10_mac3 → wait_mac.bat。可中断后重新启动（从断点继续）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat8"
DIAG="$RUNS/diag_unselected"
mkdir -p "$RUNS" "$DIAG"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"

while pgrep -f "wj_tools/week10_mac2.sh" >/dev/null || pgrep -f "hyblock.search" >/dev/null || pgrep -f "hyblock.select_replay" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests_week10c.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests_week10c.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/base_flat8/tests_week10c.txt）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止（Windows 双击 copy_env_to_mac.bat 后重新启动本任务）"; exit 1; }

if [ ! -f "$DIAG/DONE" ]; then
  echo "=== D 诊断：补跑被刷掉的候选 $(date '+%T')"
  uv run --locked python -m hyblock.eval_unselected --per_run 10 --threads 12 --out "$DIAG" \
      --runs "$RUNS/ga_sel_llm_s0" "$RUNS/ga_sel_llm_s1" "$RUNS/ga_sel_llm_s2" \
             "$RUNS/ga_sel_knn_s0" "$RUNS/ga_sel_knn_s1" "$RUNS/ga_sel_knn_s2" 2>&1 | grep -v -i warn | grep -E "自检|<<<|^\|" 
  [ -f "$DIAG/report.md" ] && date '+%F %T' > "$DIAG/DONE" || echo "!!! 诊断没写出 report.md（不影响 A，继续）"
fi

for s in 0 1 2; do
  out="$RUNS/ga_sel_llmstrat_s${s}"
  [ -f "$out/DONE" ] && continue
  echo "=== GA + ② LLM 分层 种子 ${s} $(date '+%T')"
  uv run --locked python -m hyblock.search --method ga --select llm --sel_strata family --oversample 2 --rand_slots 2 \
      --sel_pool 24 --sel_mode rank --sel_votes 3 --seed "$s" --budget 200 --batch 16 \
      --max_modules 8 --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
done

echo "=== 对比报告 $(date '+%T')"
uv run --locked python -m hyblock.compare_search "$RUNS" --out "$RUNS/compare_week10b" > /dev/null 2>&1 \
  && echo "写好了 mac_runs/base_flat8/compare_week10b/report.md" || echo "对比报告没写成（不影响结果，Windows 上重做即可）"
echo "=== end $(date '+%F %T')"
