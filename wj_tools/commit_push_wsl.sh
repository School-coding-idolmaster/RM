#!/usr/bin/env bash
# WSL：提交仓库里的全部改动并推送到 GitHub（wj 远端）。用法：commit_push_wsl.sh "提交说明"。由 D:\RM\push_tools.bat 调用。
set -u
cd /mnt/d/RM/robomorph || exit 1
git checkout -q wj-main || exit 1
git add -A
git status --short
if git diff --cached --quiet; then echo "没有要提交的改动"; else
MSG="${1:-WJ: update $(date '+%F %H:%M')}"
git commit -q -F - <<EOF2
$MSG

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RX18SzeAwMGxthrFetEU8H
EOF2
fi
git push wj wj-main && git log --oneline -1
