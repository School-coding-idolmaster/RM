#!/usr/bin/env bash
# 正式实验（平地）：2 × 2 + 对照，新种子 100–109（第11周记录 第 10 节，10-31 WJ 同意；种子 0–2 只用于开发，这里不用）。
# 共约 2.5 天，API 约 $15。随时可中断后重新启动（每个运行有 DONE 就跳过，没跑完的从断点继续）。
#   A1 GA                       A2 GA + ② LLM Ev（复制 A1 第 0–4 代）   A3 GA + ② knn Ev（复制 A1）
#   A6 GA + ② LLM Ev，Sonnet（复制 A1）
#   A4 GA + LLM 变异            A5 GA + LLM 变异 + ② LLM Ev（复制 A4 的 evals / llm_mut 第 0–4 代 / llm_cache）
#   外部参照（种子 100–104）：随机搜索、RM 式 LLM（公开代码版：写得分、top 3）
# 顺序：先按种子跑 A1→A2→A3→A6（共享前 5 代的配对），再按种子跑 A4→A5，最后外部参照。结果在 mac_runs/formal_flat/。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 formal_flat → wait_mac.bat。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
OUT="$(cd .. && pwd)/mac_runs/formal_flat"
mkdir -p "$OUT"
SEEDS="100 101 102 103 104 105 106 107 108 109"
REF_SEEDS="100 101 102 103 104"
COMMON="--budget 200 --batch 16 --max_modules 8 --threads 12 --llm_workers 8"
EV="--sel_strata family --sel_from 5 --oversample 1.25 --rand_slots 2 --sel_pool 24 --sel_mode rank --sel_votes 3"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"

while pgrep -f "wj_tools/week11_mac" >/dev/null || pgrep -f "wj_tools/terrain_pilot" >/dev/null || pgrep -f "hyblock.search" >/dev/null || pgrep -f "hyblock.select_replay" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done

echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$OUT/tests_formal_flat.txt" 2>&1
rc=$?
grep -v -i warn "$OUT/tests_formal_flat.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（详见 mac_runs/formal_flat/tests_formal_flat.txt）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止（Windows 双击 copy_env_to_mac.bat 后重新启动本任务）"; exit 1; }

run() {   # run <名称> <种子> <参数…>
  local name="$1" s="$2"; shift 2
  local out="$OUT/${name}_s${s}"
  [ -f "$out/DONE" ] && return 0
  echo "=== ${name} 种子 ${s} $(date '+%T')"
  uv run --locked python -m hyblock.search --seed "$s" $COMMON "$@" --out "$out" 2>&1 | grep -v -i warn
  [ -f "$out/DONE" ] || echo "!!! ${name} 种子 ${s} 没有 DONE（继续下一个；重新启动本任务会从断点继续）"
}

copy_llm_mut() {   # copy_llm_mut <源目录> <目标目录>：复制 llm_mut.csv 的第 0–4 代与 llm_cache（只在目标还没开始时）
  local src="$1" out="$2"
  [ -f "$out/evals.csv" ] && return 0
  mkdir -p "$out"
  [ -d "$out/llm_cache" ] || cp -R "$src/llm_cache" "$out/llm_cache"
  uv run --locked python - "$src/llm_mut.csv" "$out/llm_mut.csv" <<'PY'
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
}

# ---------------- 随机变异：A1 → A2 / A3 / A6
for s in $SEEDS; do
  run ga "$s" --method ga
  [ -f "$OUT/ga_s${s}/DONE" ] || continue
  run ga_sel_llmEv "$s" --method ga --select llm $EV --init_from "$OUT/ga_s${s}/evals.csv"
  run ga_sel_knnEv "$s" --method ga --select knn $EV --init_from "$OUT/ga_s${s}/evals.csv"
  run ga_sel_llmEv_sonnet "$s" --method ga --select llm $EV --model claude-sonnet-5-5 --init_from "$OUT/ga_s${s}/evals.csv"
done

# ---------------- LLM 变异：A4 → A5
for s in $SEEDS; do
  run ga_llmtyped "$s" --method ga --mutation llm_typed
  [ -f "$OUT/ga_llmtyped_s${s}/DONE" ] || continue
  if [ ! -f "$OUT/ga_llmtyped_sel_llmEv_s${s}/DONE" ]; then
    copy_llm_mut "$OUT/ga_llmtyped_s${s}" "$OUT/ga_llmtyped_sel_llmEv_s${s}" || { echo "!!! 复制 llm_mut.csv 失败，跳过种子 ${s}"; continue; }
  fi
  run ga_llmtyped_sel_llmEv "$s" --method ga --mutation llm_typed --select llm $EV --init_from "$OUT/ga_llmtyped_s${s}/evals.csv"
done

# ---------------- 外部参照
for s in $REF_SEEDS; do
  run random "$s" --method random
  run llm "$s" --method llm
done

echo "=== 对比报告 $(date '+%T')"
uv run --locked python -m hyblock.compare_search "$OUT" --out "$OUT/compare" > /dev/null 2>&1 \
  && echo "写好了 mac_runs/formal_flat/compare/report.md" || echo "对比报告没写成（不影响结果，Windows 上重做即可）"
echo "=== end $(date '+%F %T')"
