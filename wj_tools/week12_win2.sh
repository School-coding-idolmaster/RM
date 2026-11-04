#!/usr/bin/env bash
# 第 12 周阶段 2 追加（第12周记录 第 2e 节，11-02 06:5x 登记，阶段 2 出结果之前）：横梁上的 2 × 2 = 第 0 代 {随机, 平地迁移}
# × 变异 {随机, 地形版 LLM 变异（--mutation llm_typed，提示词含横梁描述）}。已有 G0（随机 / 随机）、I2（迁移 / 随机）
# 在 week12_win.sh 里；这里补 I5（随机 / LLM）与 I4（迁移 / LLM），各种子 0–2，共 6 个运行，同一台 Windows。
# 等 week12_win.sh 的 9 个运行全部 DONE 再开始（按文件判断，暂停时不会抢跑）。
# 由 D:\RM\week12_win2.bat 调用；暂停 = D:\RM\stop_win.bat；再双击 week12_win2.bat 从断点继续。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/robomorph"
cd /mnt/d/RM/robomorph || exit 1
OUT=/mnt/d/RM/win_runs/beams_dev
SRC=/mnt/d/RM/from_mac/mac_runs/base_flat8
COMMON="--method ga --terrain beams --terrain_scale 1.0 --budget 200 --batch 16 --max_modules 8 --threads 8 --llm_workers 8"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)（工作区含未提交的第 12 周改动）"
need() { for s in 0 1 2; do for n in ga ga_init_transfer ga_init_llmtransfer; do [ -f "$OUT/${n}_s${s}/DONE" ] || return 1; done; done; }
until need; do echo "--- 等 week12_win 的 9 个运行跑完 $(date '+%T')"; sleep 600; done
[ -f "$OUT/tests_ok" ] || { echo "!!! 离线测试还没通过（先跑 week12_win.bat）"; exit 1; }
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止"; exit 1; }

run() {
  local name="$1" s="$2"; shift 2
  local out="$OUT/${name}_s${s}"
  [ -f "$out/DONE" ] && return 0
  echo "=== ${name} 种子 ${s} $(date '+%F %T')"
  uv run --locked python -m hyblock.search --seed "$s" $COMMON "$@" --out "$out" 2>&1 | grep -v -i warn
  [ -f "$out/DONE" ] || { echo "!!! ${name} 种子 ${s} 没有 DONE（被停下或出错；再双击 week12_win2.bat 从断点继续）"; exit 1; }
}
for s in 0 1 2; do
  run ga_init_transfer_llmtyped "$s" --init transfer --init_source "$SRC/ga_s${s}/evals.csv" --mutation llm_typed
  run ga_llmtyped "$s" --mutation llm_typed
done
uv run --locked python -m hyblock.compare_search "$OUT" --out "$OUT/compare" > /dev/null 2>&1 \
  && echo "写好了 win_runs/beams_dev/compare/report.md" || echo "对比报告没写成"
echo "=== end $(date '+%F %T')"
