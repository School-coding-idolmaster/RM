#!/usr/bin/env bash
# !!! 10-29 起不再计划运行（pipeline v4.6）：top_k 消融并入 LLM 变异 / ② 的经验池消融。保留作记录。
# 第 8 周 Mac 任务 3（约 4–5 小时）：消融——RM 式 LLM 只把 \"最好 3 个\" 换成 \"最好 16 个\" 当示例（--top_k 16），
#   其余与 week8_mac2 的 llm 完全相同（上限 8、提示词 v2、重复占预算、预算 200、每代 16、种子 0–2）。
#   目的：单独测\"只多给示例（更大的经验）\"能把 RM 式改善多少（10-29 发现：top3 很快变成同一设计的变体，
#   重复里大半是重新提出已评估过的设计）。
#   结果：~/RM/mac_runs/base_flat8/llm_top16_s<种子>/（与 week8_mac2 放在一起，compare_search 一起比较）
# 若 week8_mac2 还在跑，先等它跑完（同一时间只让一个任务吃满 CPU）。
# 启动：Windows 双击 push_tools.bat → run_mac.bat 输入 week8_mac3 → wait_mac.bat。可中断后重新启动。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
RUNS="$(cd .. && pwd)/mac_runs/base_flat8"
mkdir -p "$RUNS"
echo "=== start $(date '+%F %T')  $(git log --oneline -1)"
echo "--- uptime: $(uptime)"
while pgrep -f "wj_tools/week8_mac2.sh" >/dev/null; do
  echo "--- week8_mac2 还在跑，等待 $(date '+%T')"; sleep 600
done
grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null || { echo "!!! .env 里没有 ANTHROPIC_API_KEY，停止"; exit 1; }
for s in 0 1 2; do
  out="$RUNS/llm_top16_s${s}"
  [ -f "$out/DONE" ] && continue
  echo "=== llm top_k 16 种子 ${s} $(date '+%T')"
  # 第 0 代（还没有示例）的提示词与 top_k 3 相同：复制 llm_s<种子> 的 LLM 缓存，第 0 代直接重放同样的回复，
  # 两者从同一批设计出发（配对比较，减少噪声）；之后提示词不同，缓存不命中，正常调用 API
  if [ ! -d "$out/llm_cache" ] && [ -d "$RUNS/llm_s${s}/llm_cache" ]; then
    mkdir -p "$out" && cp -R "$RUNS/llm_s${s}/llm_cache" "$out/llm_cache"
  fi
  uv run --locked python -m hyblock.search --method llm --top_k 16 --seed "$s" --budget 200 --batch 16 \
      --max_modules 8 --threads 12 --llm_workers 8 --out "$out" 2>&1 | grep -v -i warn
done
echo "=== end $(date '+%F %T')"
