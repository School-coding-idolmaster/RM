#!/usr/bin/env bash
# 第 15 周（claude/正式实验计划.md 变更记录第 18–20 条，11-11 登记，跑之前写定）：
#   追加 C  最优设计重评：formal_flat、replicate_flat、formal_beams、sensitivity/{p14,m15,v370} 每个运行的最终最好设计，
#           新回合种子重跑 10 回合 → mac_runs/reeval_best/（hyblock/reeval_best.py；约 5–10 小时）
#   追加 D  v370 + 提示词写 370 deg/s：平地，--max_speed_deg 370 --prompt_speed_deg 370，种子 300–302，
#           A4′ GA + LLM 变异 / A5′ A4′ + ②（复制 A4′ 第 0–4 代）→ mac_runs/sensitivity/v370_prompt370/（约 10–12 小时）
#           A1 不重跑：GA 不用 LLM，沿用 sensitivity/v370/ga_s<种子>
#   追加 A  H6 横梁确认：种子 400–423，B1 普通 GA / B5 GA + 地形版 LLM 变异 / B6 B5 + ②（复制 B5 第 0–4 代）
#           → mac_runs/confirm_beams/（约 110–125 小时）。按种子交错跑，中途停下也能拿到完整的种子
#   其余设置与 formal_flat / formal_beams 完全相同。全部约 130–145 小时，API 约 $33。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week15_mac1 → wait_mac.bat。随时可中断，重新启动从断点继续。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
SEN="$RUNS/sensitivity"
VP="$SEN/v370_prompt370"
CONF="$RUNS/confirm_beams"
RE="$RUNS/reeval_best"
mkdir -p "$VP" "$CONF" "$RE"
EV="--sel_strata family --sel_from 5 --oversample 1.25 --rand_slots 2 --sel_pool 24 --sel_mode rank --sel_votes 3"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"
while pgrep -f "wj_tools/week14_mac1" >/dev/null || pgrep -f "hyblock.search" >/dev/null || pgrep -f "hyblock.reeval_best" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done
for d in formal_flat replicate_flat formal_beams sensitivity/p14 sensitivity/m15 sensitivity/v370; do
  [ -d "$RUNS/$d" ] || { echo "!!! 缺少 mac_runs/${d}，停止"; exit 1; }
done
for s in 300 301 302; do
  [ -f "$SEN/v370/ga_s${s}/DONE" ] || { echo "!!! 缺少 sensitivity/v370/ga_s${s}（追加 D 的 A1），停止"; exit 1; }
done
echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$RE/tests_week15.txt" 2>&1
rc=$?
grep -v -i warn "$RE/tests_week15.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（mac_runs/reeval_best/tests_week15.txt）"; exit 1; }
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

pair_llm() {   # pair_llm <根目录> <种子> <参数…>：GA + LLM 变异，然后 + ②（复制第 0–4 代）
  local root="$1" s="$2"; shift 2
  run "$root" ga_llmtyped "$s" "$@" --mutation llm_typed
  [ -f "$root/ga_llmtyped_s${s}/DONE" ] || return 0
  if [ ! -f "$root/ga_llmtyped_sel_llmEv_s${s}/DONE" ]; then
    copy_llm_mut "$root/ga_llmtyped_s${s}" "$root/ga_llmtyped_sel_llmEv_s${s}" || { echo "!!! 复制 llm_mut.csv 失败，跳过种子 ${s}"; return 0; }
  fi
  run "$root" ga_llmtyped_sel_llmEv "$s" "$@" --mutation llm_typed --select llm $EV --init_from "$root/ga_llmtyped_s${s}/evals.csv"
}

# ---------------- 追加 C：最优设计重评
if [ ! -f "$RE/DONE" ]; then
  echo "=== 最优设计重评 $(date '+%F %T')"
  uv run --locked python -m hyblock.reeval_best run --threads 12 --out "$RE" --roots "$RUNS/formal_flat" "$RUNS/replicate_flat" \
      "$RUNS/formal_beams" "$SEN/p14" "$SEN/m15" "$SEN/v370" 2>&1 | grep -v -i warn
  [ -f "$RE/DONE" ] || echo "!!! 重评没有 DONE（继续；重新启动本任务会从断点继续）"
fi

# ---------------- 追加 D：v370 + 提示词写 370 deg/s（种子 300–302）
for s in 300 301 302; do
  pair_llm "$VP" "$s" --method ga --max_speed_deg 370 --prompt_speed_deg 370
done

# ---------------- 追加 A：H6 横梁确认（种子 400–423）
B="--method ga --terrain beams --terrain_scale 1.0"
for s in $(seq 400 423); do
  run "$CONF" ga "$s" $B
  pair_llm "$CONF" "$s" $B
done

uv run --locked python -m hyblock.reeval_best report --out "$RE" > /dev/null 2>&1 && echo "写好了 reeval_best/report.md"
for root in "$VP" "$CONF"; do
  uv run --locked python -m hyblock.compare_search "$root" --out "$root/compare" > /dev/null 2>&1 \
    && echo "写好了 $(basename "$root")/compare/report.md" || echo "$(basename "$root") 对比报告没写成"
done
echo "=== end $(date '+%F %T')"
