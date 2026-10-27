#!/usr/bin/env bash
# D1 定版（WSL 端）：离线测试 → 生成 WSL 环境清单 → 提交到 wj-main 并打标签 mac-base。不推送。
# 由 D:\RM\mac_prep.bat 调用。日志：D:\RM\logs\mac_prep_log.txt
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/robomorph"
cd /mnt/d/RM/robomorph || exit 1
LOG=/mnt/d/RM/logs/mac_prep_log.txt
exec > >(tee -a "$LOG") 2>&1
echo "=== start $(date '+%F %T')"

if [ -z "$(git config user.name)" ] || [ -z "$(git config user.email)" ]; then
  echo "git 还没设置作者。先在 Ubuntu 窗口里执行（换成你自己的名字和邮箱）："
  echo '  git config --global user.name  "你的名字"'
  echo '  git config --global user.email "你的邮箱"'
  echo "然后再双击 mac_prep.bat"; exit 1
fi

echo "=== 1/4 离线测试（原 50 项 + 确定性 3 项）"
uv run --locked python -m unittest discover -s tests > /mnt/d/RM/logs/tests_wsl.txt 2>&1
rc=$?
grep -v -i warn /mnt/d/RM/logs/tests_wsl.txt | tail -6
[ $rc -eq 0 ] || { echo "离线测试失败，停止（详见 D:\\RM\\logs\\tests_wsl.txt）"; exit 1; }

echo "=== 2/4 WSL 环境清单"
uv run --locked python wj_tools/env_report.py --out wj_tools/env/wsl.json || exit 1

echo "=== 3/4 提交"
git rev-parse --verify -q wj-main >/dev/null || git checkout -b wj-main
git checkout -q wj-main
git add --renormalize . >/dev/null 2>&1
git add -A
git status --short
git commit -q -F - <<'MSG'
WJ: CPU MPC evaluator, Claude backend, Hyblock domain, Mac tooling (as of 2026-10-27)

- mpc/: CPU MPPI evaluator (RoboGrammar port, RM V3 settings), batch eval, diagnostics
- robomorph/: --evaluator {sac,mpc}, --domain {rm,hyblock}, native Claude backend (llm.py)
- hyblock/: Suirikun module + hub + passive foot model, grammar, MJCF compiler, generator
- tests/: MPC, LLM, Hyblock, determinism (thread count does not change MPPI results)
- wj_tools/: env_report / compare_env / run_meta / setup_mac.sh; windows_runs/ = scripts
  behind the 10-19..10-27 Windows results (they hard-code /mnt/d/RM)
- wj_tools/env/wsl.json: reference environment for the Mac mini

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RX18SzeAwMGxthrFetEU8H
MSG
git tag -f mac-base >/dev/null

echo "=== 4/4 结果"
git log --oneline -2
git describe --tags
git status --short | head
echo "=== end $(date '+%F %T')"
