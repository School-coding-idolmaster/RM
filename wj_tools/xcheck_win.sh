#!/usr/bin/env bash
# 跨机器抽查（第13周记录 第 0 节）：在 Windows（WSL）上重评 formal_flat 的 5%（676 个设计，约 8–10 小时）。
# 由 D:\RM\xcheck_win.bat 调用；暂停 = D:\RM\stop_win.bat；再双击 xcheck_win.bat 从断点继续。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/robomorph"
export PYTHONPATH=/mnt/d/RM/robomorph
cd /mnt/d/RM/robomorph || exit 1
mkdir -p /mnt/d/RM/logs
echo "=== start $(date '+%F %T')"
if pgrep -f "hyblock[.]search" >/dev/null || [ "$(pgrep -fc 'xcheck_formal[.]py')" -gt 0 ]; then
  echo "!!! 已经有计算在跑（重复双击？），退出"; exit 1
fi
uv run --locked python wj_tools/xcheck_formal.py 2>&1 | grep -v -i warn
echo "=== end $(date '+%F %T')"
