#!/usr/bin/env bash
# 在 Mac 上：正在运行的任务 + 最新日志的最后几行。由 D:\RM\mac_status.bat 通过 ssh 调用。
echo "--- 正在运行："
pgrep -fl 'wj_tools/.*\.sh' || echo "（没有）"
f=$(ls -t "$HOME"/RM/logs/*.out "$HOME"/RM/logs/*.txt 2>/dev/null | head -1)
[ -n "$f" ] && { echo "--- $f"; tail -8 "$f"; }
echo "--- 已完成的结果（有 DONE）："
ls -d "$HOME"/RM/mac_runs/*/DONE 2>/dev/null | sed 's#/DONE##; s#.*/##' | tr '\n' ' '; echo
