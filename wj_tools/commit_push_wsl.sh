#!/usr/bin/env bash
# WSL：提交 wj_tools 等改动并推送到 GitHub（wj 远端）。由 D:\RM\push_tools.bat 调用。
set -u
cd /mnt/d/RM/robomorph || exit 1
git checkout -q wj-main || exit 1
git add -A
git status --short
if git diff --cached --quiet; then echo "没有要提交的改动"; else
git commit -q -F - <<'MSG'
Mac day 1: parallel benchmark, cross-machine check, design copies

- wj_tools/bench_parallel.py: processes x threads throughput on 3 Hyblock designs
- wj_tools/mac_day1.sh: benchmark + cross-machine check (run4_it20 settings, seeds 0/1)
- wj_tools/xcheck/rm20/: byte-identical copies of sac_runs/run1 design_00..19.xml

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RX18SzeAwMGxthrFetEU8H
MSG
fi
git push wj wj-main && git log --oneline -1
