#!/usr/bin/env bash
# 在 Mac 上：拉最新代码，后台启动 wj_tools/<任务>.sh（防睡眠），日志写 ~/RM/logs/<任务>.out。
# 由 Windows 的 D:\RM\run_mac.bat 通过 ssh 调用：bash ~/RM/robomorph/wj_tools/mac_launch.sh <任务>
set -u
task="${1:-}"
cd "$HOME/RM/robomorph" || exit 1
git pull -q || { echo "git pull 失败"; exit 1; }
echo "代码：$(git log --oneline -1)"
[ -n "$task" ] && [ -f "wj_tools/${task}.sh" ] || { echo "找不到 wj_tools/${task}.sh"; ls wj_tools/*.sh; exit 1; }
if pgrep -f "wj_tools/${task}.sh" >/dev/null; then echo "${task} 已经在运行"; exit 0; fi
mkdir -p "$HOME/RM/logs"
nohup caffeinate -i bash "wj_tools/${task}.sh" > "$HOME/RM/logs/${task}.out" 2>&1 < /dev/null &
echo "已启动：${task}（PID $!），日志 ~/RM/logs/${task}.out"
