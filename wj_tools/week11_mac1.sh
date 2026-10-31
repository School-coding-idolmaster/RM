#!/usr/bin/env bash
# 第 11 周 Mac 任务 1（约 3 小时；API 约 $7–15）：今晚的两个低成本检查（第11周记录 第 1–2 节，判断标准已事先写好）。
#   a. Sonnet 离线回放：与第 10 周 Haiku 回放同样的 72 道题、同样的提示词（经验池 24、排序、3 票），只换模型 claude-sonnet-5-5
#      → mac_runs/select_replay_sonnet/（report.md、paired.txt = 与 Haiku 逐题配对）。只调用 API，几乎不占 CPU，放在前面。
#      冒烟检查没通过时跳过 a，继续 b。
#   b. E + 温和过滤（Ev）：当前 ② 最好配置（预热 4 代 + 分层、经验池 24、排序、3 票）只把 --oversample 2 改成 1.25
#      （每代提出 20 个：随机 2 个 + 排序前 14 个 ≈ 刷掉预测最差的 4 个），种子 0–2 → base_flat8/ga_sel_llmstrat5v_s<种子>/
#      最后写对比报告 base_flat8/compare_week11a/。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week11_mac1 → wait_mac.bat。可中断后重新启动（LLM 回复有缓存，搜索从断点继续）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
MR="$(cd .. && pwd)/mac_runs"
RUNS="$MR/base_flat8"
HAIKU="$MR/select_replay"
OUT="$MR/select_replay_sonnet"
SMOKE="$MR/select_replay_sonnet_smoke"
MODEL="claude-sonnet-5-5"
mkdir -p "$OUT" "$SMOKE"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"

while pgrep -f "wj_tools/week10_mac" >/dev/null || pgrep -f "hyblock.search" >/dev/null || pgrep -f "hyblock.eval_unselected" >/dev/null || pgrep -f "hyblock.select_replay" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests_week11a.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests_week11a.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/base_flat8/tests_week11a.txt）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止（Windows 双击 copy_env_to_mac.bat 后重新启动本任务）"; exit 1; }

RUNLIST="$RUNS/ga_s0 $RUNS/ga_s1 $RUNS/ga_s2 $RUNS/ga_llmtyped_s0 $RUNS/ga_llmtyped_s1 $RUNS/ga_llmtyped_s2"

# ---------------- a. Sonnet 离线回放
if [ -f "$OUT/DONE" ]; then
  echo "=== a 已完成，跳过"
else
  sonnet_ok=1
  echo "=== a0 配对脚本自检（Haiku 对 Haiku，只读缓存；差值应全为 0，rho 应为 0.37）$(date '+%T')"
  uv run --locked python -m hyblock.replay_paired --runs $RUNLIST --base "claude-haiku-5-5=$HAIKU" \
      --new "claude-haiku-5-5=$HAIKU" --out "$OUT/selfcheck_paired.txt" > /dev/null 2>&1 \
    && grep -E "缓存缺失|^\| rho（3 票）|^\| claude-haiku-5-5_v3 " "$OUT/selfcheck_paired.txt" | head -4 \
    || echo "!!! 配对脚本自检没跑通（不影响回放本身；回来后 Claude 在 Windows 上查）"
  if [ ! -f "$SMOKE/OK" ]; then
    echo "=== a1 Sonnet 冒烟检查 $(date '+%T')"
    uv run --locked python -m hyblock.select_replay --smoke --model "$MODEL" --pools 24 --modes rank --votes 3 \
        --runs $RUNLIST --out "$SMOKE" > "$SMOKE/smoke.txt" 2>&1
    rc=$?
    grep -v -i warn "$SMOKE/smoke.txt" | grep -E "冒烟|p24 rank|!!!" | tail -4
    if [ $rc -eq 0 ]; then date '+%F %T' > "$SMOKE/OK"; else sonnet_ok=0
      echo "!!! Sonnet 冒烟检查没通过，跳过 a、继续 b（详见 mac_runs/select_replay_sonnet_smoke/smoke.txt，交给 Claude）"; fi
  fi
  if [ $sonnet_ok -eq 1 ]; then
    echo "=== a2 Sonnet 离线回放（216 次调用）$(date '+%T')"
    uv run --locked python -m hyblock.select_replay --model "$MODEL" --pools 24 --modes rank --votes 3 \
        --runs $RUNLIST --out "$OUT" --workers 12 2>&1 | grep -v -i warn | grep -E "道题|组（|p24 rank"
    if [ -f "$OUT/report.md" ]; then
      echo "=== a3 与 Haiku 逐题配对 $(date '+%T')"
      uv run --locked python -m hyblock.replay_paired --runs $RUNLIST --base "claude-haiku-5-5=$HAIKU" \
          --new "$MODEL=$OUT" --out "$OUT/paired.txt" > /dev/null 2>&1 \
        && grep -E "缓存缺失|^\| rho（" "$OUT/paired.txt" || echo "!!! 配对比较没写成（回来后在 Windows 上重做）"
      date '+%F %T' > "$OUT/DONE"
    else
      echo "!!! Sonnet 回放没有写出 report.md，继续 b"
    fi
  fi
fi

# ---------------- b. E + 温和过滤
for s in 0 1 2; do
  out="$RUNS/ga_sel_llmstrat5v_s${s}"
  [ -f "$out/DONE" ] && continue
  [ -f "$RUNS/ga_s${s}/DONE" ] || { echo "!!! 缺少 ga_s${s}（普通 GA），停止"; exit 1; }
  echo "=== b GA + ② LLM 分层，第 5 代起，温和过滤（×1.25）种子 ${s} $(date '+%T')"
  uv run --locked python -m hyblock.search --method ga --select llm --sel_strata family --sel_from 5 \
      --init_from "$RUNS/ga_s${s}/evals.csv" --oversample 1.25 --rand_slots 2 \
      --sel_pool 24 --sel_mode rank --sel_votes 3 --seed "$s" --budget 200 --batch 16 \
      --max_modules 8 --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
done

echo "=== 对比报告 $(date '+%T')"
uv run --locked python -m hyblock.compare_search "$RUNS" --out "$RUNS/compare_week11a" > /dev/null 2>&1 \
  && echo "写好了 mac_runs/base_flat8/compare_week11a/report.md" || echo "对比报告没写成（不影响结果，Windows 上重做即可）"
echo "=== end $(date '+%F %T')"
