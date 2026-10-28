#!/usr/bin/env bash
# WSL：每 5 分钟问一次 Mac 还有没有 wj_tools 任务在跑；全部跑完后把结果 rsync 到 D:\RM\from_mac 并退出（返回 0）。
# 连不上 Mac 时不算跑完，继续等。由 D:\RM\wait_mac.bat 调用。
interval="${1:-300}"
echo "等待 Mac mini 上的任务跑完（每 $((interval / 60)) 分钟检查一次；关掉这个窗口就停止等待，不影响 Mac）"
while true; do
  running=$(ssh -o ConnectTimeout=15 -o BatchMode=yes macmini "pgrep -fl 'wj_tools/[^ ]*[.]sh'" 2>/dev/null)
  rc=$?
  now=$(date '+%H:%M')
  if [ $rc -eq 255 ]; then
    echo "$now 连不上 Mac（休眠、断网或 IP 变了？），5 分钟后再试"
  elif [ -n "$running" ]; then
    last=$(ssh -o ConnectTimeout=15 macmini 'f=$(ls -t ~/RM/logs/*.out 2>/dev/null | head -1); [ -n "$f" ] && tail -1 "$f"' 2>/dev/null)
    echo "$now 还在跑：$(echo "$running" | sed 's#.*wj_tools/##' | tr '\n' ' ')| ${last:0:110}"
  else
    echo "$now Mac 上没有任务在跑了，拉回结果……"
    mkdir -p /mnt/d/RM/from_mac
    rsync -a macmini:RM/mac_runs macmini:RM/logs /mnt/d/RM/from_mac/ && echo "已拉回到 D:\\RM\\from_mac"
    ssh macmini 'f=$(ls -t ~/RM/logs/*.out 2>/dev/null | head -1); echo "--- $f"; tail -5 "$f"'
    exit 0
  fi
  sleep "$interval"
done
