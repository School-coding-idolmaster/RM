#!/usr/bin/env bash
# 第 10 周 Mac 任务 4（约 3–4 小时）：WJ 10-31 选的 E（预热）。
#   GA + ② LLM 分层（与 week10_mac3 相同：经验池 24、排序、3 票、--sel_strata family），但第 5 代起才挑选（--sel_from 5）；
#   第 0–4 代与普通 GA 完全相同 → 直接复制 ga_s<种子> 的前 5 代（--init_from，开头核对第 0 代；同一台机器上与重跑逐位相同）。
#   → base_flat8/ga_sel_llmstrat5_s<种子>/。原因：② 让亲本池在前几代就收窄（第 2 代相似度 0.23–0.28 vs GA 0.13），
#   且回放里 LLM 在第 1–4 代不占优（第四阶段（下）记录 第 9.4–9.5 节）。
#   注意：GA 第 4 代结束时种子 0 已 881、种子 2 已 606（最终值），这两个种子的最好得分天然接近 GA；主要看第 5–12 代与种子 1。
#   最后写对比报告 base_flat8/compare_week10c/。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week10_mac4 → wait_mac.bat。可中断后重新启动（从断点继续）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat8"
mkdir -p "$RUNS"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"

while pgrep -f "wj_tools/week10_mac3.sh" >/dev/null || pgrep -f "hyblock.search" >/dev/null || pgrep -f "hyblock.eval_unselected" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests_week10d.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests_week10d.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/base_flat8/tests_week10d.txt）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止（Windows 双击 copy_env_to_mac.bat 后重新启动本任务）"; exit 1; }

for s in 0 1 2; do
  out="$RUNS/ga_sel_llmstrat5_s${s}"
  [ -f "$out/DONE" ] && continue
  [ -f "$RUNS/ga_s${s}/DONE" ] || { echo "!!! 缺少 ga_s${s}（普通 GA），停止"; exit 1; }
  echo "=== GA + ② LLM 分层，第 5 代起 种子 ${s} $(date '+%T')"
  uv run --locked python -m hyblock.search --method ga --select llm --sel_strata family --sel_from 5 \
      --init_from "$RUNS/ga_s${s}/evals.csv" --oversample 2 --rand_slots 2 \
      --sel_pool 24 --sel_mode rank --sel_votes 3 --seed "$s" --budget 200 --batch 16 \
      --max_modules 8 --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
done

echo "=== 对比报告 $(date '+%T')"
uv run --locked python -m hyblock.compare_search "$RUNS" --out "$RUNS/compare_week10c" > /dev/null 2>&1 \
  && echo "写好了 mac_runs/base_flat8/compare_week10c/report.md" || echo "对比报告没写成（不影响结果，Windows 上重做即可）"
echo "=== end $(date '+%F %T')"
