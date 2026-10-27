#!/usr/bin/env bash
# Mac mini 第一天：并行测速 → 跨机器一致性检验（参照域 20 个设计 × 种子 0 / 1 × 3 回合）。
# 用法（终端）：cd ~/RM/robomorph && git pull && bash wj_tools/mac_day1.sh
# 结果：~/RM/mac_runs/{bench,xcheck_s0,xcheck_s1}；日志 ~/RM/logs/mac_day1.txt。可中断后重新运行（已完成的设计会跳过）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs"
mkdir -p "$RUNS" ../logs
exec > >(tee -a ../logs/mac_day1.txt) 2>&1
echo "=== start $(date '+%F %T')  $(git describe --tags --always)  dirty=$(git status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
run() { caffeinate -i uv run --locked "$@"; }

if [ ! -f "$RUNS/bench/DONE" ]; then
  echo "=== 1/3 并行测速（约 10 分钟）"
  run python wj_tools/bench_parallel.py --out "$RUNS/bench" 2>&1 | grep -v -i warn
fi
for s in 0 1; do
  out="$RUNS/xcheck_s$s"
  [ -f "$out/DONE" ] && continue
  echo "=== $((s + 2))/3 一致性检验 种子 $s（20 个设计 × 3 回合，与 Windows run4_it20 同一设置）"
  run python wj_tools/run_meta.py start "$out" --threads 12 --note "xcheck seed $s (run4_it20 settings)"
  run python -m mpc.eval_batch --xml_dir wj_tools/xcheck/rm20 --out "$out" --episodes 3 --seed $s \
      --threads 12 --steps 250 --samples 64 --fall_penalty 100 --noise_warm 0.5 --noise_hist 0.1 \
      --max_speed 10 --min_iterations 20 2>&1 | grep -v -i warn
  n=$(($(wc -l < "$out/results.csv") - 1))
  [ "$n" -ge 20 ] && run python wj_tools/run_meta.py done "$out"
done
echo "=== end $(date '+%F %T')"
