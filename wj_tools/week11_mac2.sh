#!/usr/bin/env bash
# 第 11 周 Mac 任务 2（约 5 小时；API 约 $1）：2 × 2 初步实验缺的两组（第11周记录 第 5–7 节）。
#   ② 已冻结为 Ev = 预热 4 代 + 按修改种类分层 + 温和过滤（--oversample 1.25），LLM 用 Haiku（10-31 WJ 同意）。
#   1. GA + 按种类引导的 LLM 变异 + ② Ev（2 × 2 第 4 组）× 种子 0–2 → base_flat8/ga_llmtyped_sel_llmstrat5v_s<种子>/
#      第 0–4 代复制 ga_llmtyped_s<种子>：evals.csv（--init_from）+ llm_mut.csv 的第 0–4 代（LLM 变异的提示词要用到之前被拒的设计）
#      + llm_cache（第 5 代前 16 个变异与 ga_llmtyped 相同 → 与「GA + LLM 变异」只差 ②）
#   2. GA + ② knn Ev（判断 ② 的效果是不是 LLM 特有）× 种子 0–2 → base_flat8/ga_sel_knnstrat5v_s<种子>/（不调用 API）
#   最后写对比报告 base_flat8/compare_week11b/。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week11_mac2 → wait_mac.bat。可中断后重新启动（从断点继续）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat8"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"

while pgrep -f "wj_tools/week11_mac1" >/dev/null || pgrep -f "hyblock.search" >/dev/null || pgrep -f "hyblock.select_replay" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RUNS/tests_week11b.txt" 2>&1
rc=$?
grep -v -i warn "$RUNS/tests_week11b.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/base_flat8/tests_week11b.txt）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止（Windows 双击 copy_env_to_mac.bat 后重新启动本任务）"; exit 1; }

SEL="--sel_strata family --sel_from 5 --oversample 1.25 --rand_slots 2 --budget 200 --batch 16 --max_modules 8 --threads 12"

# ---------------- 1. GA + LLM 变异 + ② Ev
for s in 0 1 2; do
  out="$RUNS/ga_llmtyped_sel_llmstrat5v_s${s}"
  src="$RUNS/ga_llmtyped_s${s}"
  [ -f "$out/DONE" ] && continue
  [ -f "$src/DONE" ] || { echo "!!! 缺少 ga_llmtyped_s${s}，停止"; exit 1; }
  if [ ! -f "$out/evals.csv" ]; then
    mkdir -p "$out"
    [ -d "$out/llm_cache" ] || cp -R "$src/llm_cache" "$out/llm_cache"
    uv run --locked python - "$src/llm_mut.csv" "$out/llm_mut.csv" <<'PY' || { echo "!!! 复制 llm_mut.csv 失败，停止"; exit 1; }
import csv, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)
    rows = [x for x in r if int(x["gen"]) < 5]
    fields = r.fieldnames
with open(dst, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)
print(f"llm_mut.csv：复制第 0–4 代 {len(rows)} 行")
PY
  fi
  echo "=== GA + LLM 变异 + ② LLM Ev 种子 ${s} $(date '+%T')"
  uv run --locked python -m hyblock.search --method ga --mutation llm_typed --select llm $SEL \
      --init_from "$src/evals.csv" --sel_pool 24 --sel_mode rank --sel_votes 3 --seed "$s" \
      --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
done

# ---------------- 2. GA + ② knn Ev
for s in 0 1 2; do
  out="$RUNS/ga_sel_knnstrat5v_s${s}"
  [ -f "$out/DONE" ] && continue
  [ -f "$RUNS/ga_s${s}/DONE" ] || { echo "!!! 缺少 ga_s${s}，停止"; exit 1; }
  echo "=== GA + ② knn Ev 种子 ${s} $(date '+%T')"
  uv run --locked python -m hyblock.search --method ga --select knn $SEL \
      --init_from "$RUNS/ga_s${s}/evals.csv" --seed "$s" --out "$out" 2>&1 | grep -v -i warn
done

echo "=== 对比报告 $(date '+%T')"
uv run --locked python -m hyblock.compare_search "$RUNS" --out "$RUNS/compare_week11b" > /dev/null 2>&1 \
  && echo "写好了 mac_runs/base_flat8/compare_week11b/report.md" || echo "对比报告没写成（不影响结果，Windows 上重做即可）"
echo "=== end $(date '+%F %T')"
