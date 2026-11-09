#!/usr/bin/env bash
# 第 14 周敏感性分析（claude/正式实验计划.md 变更记录第 16 条，11-09 登记，跑之前写定）：
#   平地，新种子 300–302；三种物理条件（每次只改一项，其余为默认）：
#     p14  压力 14 MPa（关节力矩上限 ×1.75）     --pressure 14
#     m15  质量 ×1.5（模块与分支件；被动脚不变）  --mass_scale 1.5
#     v370 关节速度上限 370 deg/s（默认 1185）     --max_speed_deg 370
#   每种条件三组：A1 普通 GA（ga）/ A4 GA + LLM 变异（ga_llmtyped）/ A5 A4 + ② LLM Ev（复制 A4 第 0–4 代）
#   其余设置与 formal_flat 完全相同（预算 200、每代 16、模块上限 8、2 回合、提示词冻结——提示词里写的仍是默认参数）
# 结果 mac_runs/sensitivity/<条件>/<组>_s<种子>/；按种子交错跑，中途停下也能拿到完整的种子。约 30–40 小时，API 约 $10。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week14_mac1 → wait_mac.bat
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
SEN="$RUNS/sensitivity"
mkdir -p "$SEN"
EV="--sel_strata family --sel_from 5 --oversample 1.25 --rand_slots 2 --sel_pool 24 --sel_mode rank --sel_votes 3"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"
while pgrep -f "wj_tools/week13_mac1" >/dev/null || pgrep -f "hyblock.search" >/dev/null; do
  echo "--- 别的任务还在跑，等待 $(date '+%T')"; sleep 600
done
echo "=== 离线测试 $(date '+%T')"
uv run --locked python -m unittest discover -s tests > "$SEN/tests_week14.txt" 2>&1
rc=$?
grep -v -i warn "$SEN/tests_week14.txt" | tail -3
[ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（mac_runs/sensitivity/tests_week14.txt）"; exit 1; }
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

cond_args() {   # cond_args <条件名>
  case "$1" in
    p14) echo "--pressure 14" ;;
    m15) echo "--mass_scale 1.5" ;;
    v370) echo "--max_speed_deg 370" ;;
  esac
}

for s in 300 301 302; do
  for c in p14 m15 v370; do
    root="$SEN/$c"; mkdir -p "$root"
    P="--method ga $(cond_args "$c")"
    run "$root" ga "$s" $P
    run "$root" ga_llmtyped "$s" $P --mutation llm_typed
    [ -f "$root/ga_llmtyped_s${s}/DONE" ] || continue
    if [ ! -f "$root/ga_llmtyped_sel_llmEv_s${s}/DONE" ]; then
      copy_llm_mut "$root/ga_llmtyped_s${s}" "$root/ga_llmtyped_sel_llmEv_s${s}" || { echo "!!! 复制 llm_mut.csv 失败，跳过 ${c} 种子 ${s}"; continue; }
    fi
    run "$root" ga_llmtyped_sel_llmEv "$s" $P --mutation llm_typed --select llm $EV --init_from "$root/ga_llmtyped_s${s}/evals.csv"
  done
done

for c in p14 m15 v370; do
  uv run --locked python -m hyblock.compare_search "$SEN/$c" --out "$SEN/$c/compare" > /dev/null 2>&1 \
    && echo "写好了 sensitivity/$c/compare/report.md" || echo "sensitivity/$c 对比报告没写成"
done
echo "=== end $(date '+%F %T')"
