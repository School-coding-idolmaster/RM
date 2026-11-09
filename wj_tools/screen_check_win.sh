#!/usr/bin/env bash
# ① 筛选的误杀率检验（第13周记录 第 7 节）：在 Windows（WSL）上把被 ① 筛掉的约 80 个设计送 MPC（6 线程，约 1–1.5 小时）。
# 由 D:\RM\screen_check_win.bat 调用；暂停 = D:\RM\stop_win.bat；再双击 screen_check_win.bat 从断点继续。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/robomorph"
export PYTHONPATH=/mnt/d/RM/robomorph
cd /mnt/d/RM/robomorph || exit 1
mkdir -p /mnt/d/RM/logs
echo "=== start $(date '+%F %T')"
if pgrep -f "hyblock[.]search" >/dev/null || pgrep -f "xcheck_formal[.]py" >/dev/null || [ "$(pgrep -fc 'screen_check[.]py')" -gt 0 ]; then
  echo "!!! 已经有计算在跑（重复双击？），退出"; exit 1
fi
uv run --locked python wj_tools/screen_check.py --threads 6 2>&1 | grep -v -i warn
echo "=== end $(date '+%F %T')"
