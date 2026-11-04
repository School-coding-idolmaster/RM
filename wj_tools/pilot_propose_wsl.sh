#!/usr/bin/env bash
# 第 12 周阶段 1：在 Windows（WSL）上让 LLM 提出横梁第 0 代（I1 LLM 初始化、I3 LLM 迁移；Haiku，约 20 次调用、< $0.1）。
# 只调用 API、编译和 ① 筛选，不跑 MPC，也不碰 Mac。Claude 的虚拟机连不上 API，所以这一步由 WJ 双击
# D:\RM\pilot_propose.bat。结果：from_mac/mac_runs/terrain_pilot/analysis_w12/pilot/（props.csv、llm_cache、I*/init.csv）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/robomorph"
cd /mnt/d/RM/robomorph || exit 1
PYTHONPATH=/mnt/d/RM/robomorph uv run --locked python /mnt/d/RM/from_mac/mac_runs/terrain_pilot/analysis_w12/pilot_init.py propose 2>&1 \
  | grep -v -i warn | tee /mnt/d/RM/from_mac/mac_runs/terrain_pilot/analysis_w12/pilot/propose_log.txt
