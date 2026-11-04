#!/usr/bin/env bash
# 第 13 周（claude/正式实验计划.md 变更记录第 10、12 条，11-04 登记，跑之前写定）：
#   第一部分  平地复现 H3：新种子 200–209，A4（GA + LLM 变异）与 A5（本方案，复制 A4 第 0–4 代）→ mac_runs/replicate_flat/
#   第二部分  横梁正式实验：种子 100–109，B1–B6 → mac_runs/formal_beams/
#     B1 普通 GA                         B2 GA + ② LLM Ev（复制 B1 第 0–4 代）
#     B3 平地迁移（--init transfer）       B4 LLM 改写后迁移（--init llm_transfer）
#        迁移来源 = mac_runs/formal_flat/ga_s<种子>/evals.csv（同种子平地 A1）
#     B5 GA + 地形版 LLM 变异              B6 B5 + ② LLM Ev（复制 B5 第 0–4 代：evals、llm_mut、llm_cache）
# 横梁的 LLM 提示词由 hyblock/terrain_text.py 加地形段（平地版逐字节不变）。约 30 + 100 小时，API 约 $27。
# 等 formal_flat 跑完再开始；随时可中断，重新启动从断点继续。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week13_mac1 → wait_mac.bat
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
REP="$RUNS/replicate_flat"
BEAMS="$RUNS/formal_beams"
mkdir -p "$REP" "$BEAMS"
EV="--sel_strata family --sel_from 5 --oversample 1.25 --rand_slots 2 --sel_pool 24 --sel_mode rank --sel_votes 3"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"
while pgrep -f "wj_tools/formal_flat" >/dev/null || pgrep -f "hyblock.search" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done
for s in 100 101 102 103 104 105 106 107 108 109; do
  [ -f "$RUNS/formal_flat/ga_s${s}/DONE" ] || { echo "!!! 缺少迁移来源 formal_flat/ga_s${s}，停止"; exit 1; }
done
echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$REP/tests_week13.txt" 2>&1
rc=$?
grep -v -i warn "$REP/tests_week13.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（mac_runs/replicate_flat/tests_week13.txt）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY（copy_env_to_mac.bat）"; exit 1; }

run() {   # run <输出根目录> <名称> <种子> <参数…>
  local root="$1" name="$2" s="$3"; shift 3
  local out="$root/${name}_s${s}"
  [ -f "$out/DONE" ] && return 0
  echo "=== $(basename "$root")/${name} 种子 ${s} $(date '+%F %T')"
  uv run --locked python -m hyblock.search --seed "$s" --budget 200 --batch 16 --max_modules 8 --threads 12 --llm_workers 8 \
      "$@" --out "$out" 2>&1 | grep -v -i warn
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

# ---------------- 第一部分：平地复现 H3（种子 200–209）
for s in 200 201 202 203 204 205 206 207 208 209; do
  run "$REP" ga_llmtyped "$s" --method ga --mutation llm_typed
  [ -f "$REP/ga_llmtyped_s${s}/DONE" ] || continue
  if [ ! -f "$REP/ga_llmtyped_sel_llmEv_s${s}/DONE" ]; then
    copy_llm_mut "$REP/ga_llmtyped_s${s}" "$REP/ga_llmtyped_sel_llmEv_s${s}" || { echo "!!! 复制 llm_mut.csv 失败，跳过种子 ${s}"; continue; }
  fi
  run "$REP" ga_llmtyped_sel_llmEv "$s" --method ga --mutation llm_typed --select llm $EV --init_from "$REP/ga_llmtyped_s${s}/evals.csv"
done

# ---------------- 第二部分：横梁正式实验（种子 100–109）
B="--method ga --terrain beams --terrain_scale 1.0"
for s in 100 101 102 103 104 105 106 107 108 109; do
  src="$RUNS/formal_flat/ga_s${s}/evals.csv"
  run "$BEAMS" ga "$s" $B
  [ -f "$BEAMS/ga_s${s}/DONE" ] && run "$BEAMS" ga_sel_llmEv "$s" $B --select llm $EV --init_from "$BEAMS/ga_s${s}/evals.csv"
  run "$BEAMS" ga_init_transfer "$s" $B --init transfer --init_source "$src"
  run "$BEAMS" ga_init_llmtransfer "$s" $B --init llm_transfer --init_source "$src"
  run "$BEAMS" ga_llmtyped "$s" $B --mutation llm_typed
  [ -f "$BEAMS/ga_llmtyped_s${s}/DONE" ] || continue
  if [ ! -f "$BEAMS/ga_llmtyped_sel_llmEv_s${s}/DONE" ]; then
    copy_llm_mut "$BEAMS/ga_llmtyped_s${s}" "$BEAMS/ga_llmtyped_sel_llmEv_s${s}" || { echo "!!! 复制 llm_mut.csv 失败，跳过种子 ${s}"; continue; }
  fi
  run "$BEAMS" ga_llmtyped_sel_llmEv "$s" $B --mutation llm_typed --select llm $EV --init_from "$BEAMS/ga_llmtyped_s${s}/evals.csv"
done

for root in "$REP" "$BEAMS"; do
  uv run --locked python -m hyblock.compare_search "$root" --out "$root/compare" > /dev/null 2>&1 \
    && echo "写好了 $(basename "$root")/compare/report.md" || echo "$(basename "$root") 对比报告没写成"
done
echo "=== end $(date '+%F %T')"
