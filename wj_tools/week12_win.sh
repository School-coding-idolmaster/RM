#!/usr/bin/env bash
# 第 12 周阶段 2（第12周记录 第 2d 节）在 Windows（WSL，12100F）上跑：横梁、开发种子 0–2，
#   G0 普通 GA / I2 平地原样迁移（--init transfer）/ I3 LLM 迁移（--init llm_transfer）各 3 个种子，共 9 个运行。
# 同一组比较全部在这台机器上（G0 种子 0 也在这里重跑，不用 Mac 的 terrain_pilot）。迁移来源 = Mac 的平地 GA
# from_mac/mac_runs/base_flat8/ga_s<种子>/evals.csv。按种子交错跑，中途停下也能拿到完整的种子。
# 由 D:\RM\week12_win.bat 调用；暂停 = D:\RM\stop_win.bat（或在窗口里 Ctrl+C）；再双击 week12_win.bat 从断点继续。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/robomorph"
cd /mnt/d/RM/robomorph || exit 1
OUT=/mnt/d/RM/win_runs/beams_dev
SRC=/mnt/d/RM/from_mac/mac_runs/base_flat8
mkdir -p "$OUT" /mnt/d/RM/logs
COMMON="--method ga --terrain beams --terrain_scale 1.0 --budget 200 --batch 16 --max_modules 8 --threads 8 --llm_workers 8"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)（工作区含未提交的第 12 周改动）"
if [ "$(pgrep -fc 'hyblock.search')" -gt 0 ]; then echo "!!! 已经有 hyblock.search 在跑（重复双击？），退出"; exit 1; fi
if [ ! -f "$OUT/tests_ok" ]; then
  echo "=== 离线测试（第一次运行时，约几分钟）$(date '+%T')"
  uv run --locked python -m unittest discover -s tests > "$OUT/tests_week12_win.txt" 2>&1
  rc=$?
  grep -v -i warn "$OUT/tests_week12_win.txt" | tail -3
  [ $rc -eq 0 ] || { echo "离线测试没有全部通过，停止（win_runs/beams_dev/tests_week12_win.txt，交给 Claude）"; exit 1; }
  touch "$OUT/tests_ok"
fi
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止"; exit 1; }

run() {   # run <名称> <种子> <参数…>
  local name="$1" s="$2"; shift 2
  local out="$OUT/${name}_s${s}"
  [ -f "$out/DONE" ] && return 0
  echo "=== ${name} 种子 ${s} $(date '+%F %T')"
  uv run --locked python -m hyblock.search --seed "$s" $COMMON "$@" --out "$out" 2>&1 | grep -v -i warn
  [ -f "$out/DONE" ] || { echo "!!! ${name} 种子 ${s} 没有 DONE（被停下或出错；再双击 week12_win.bat 从断点继续）"; exit 1; }
}

for s in 0 1 2; do
  run ga "$s"
  run ga_init_transfer "$s" --init transfer --init_source "$SRC/ga_s${s}/evals.csv"
  run ga_init_llmtransfer "$s" --init llm_transfer --init_source "$SRC/ga_s${s}/evals.csv"
done
uv run --locked python -m hyblock.compare_search "$OUT" --out "$OUT/compare" > /dev/null 2>&1 \
  && echo "写好了 win_runs/beams_dev/compare/report.md" || echo "对比报告没写成"
echo "=== end $(date '+%F %T')"
