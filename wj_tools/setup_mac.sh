#!/usr/bin/env bash
# Mac mini（Apple Silicon）环境配置：装与 WSL 完全相同的锁定依赖，逐个比对版本，跑离线测试。
# 用法（终端）：bash ~/RM/robomorph/wj_tools/setup_mac.sh
# 日志：~/RM/logs/setup_mac_<时间>.txt。任何一步失败都会停下，把日志发给 Claude。
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RM_ROOT="$(cd "$REPO/.." && pwd)"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
REF="$REPO/wj_tools/env/wsl.json"
mkdir -p "$RM_ROOT/logs"
LOG="$RM_ROOT/logs/setup_mac_$(date +%Y%m%d_%H%M).txt"
exec > >(tee -a "$LOG") 2>&1
say() { printf '\n==> %s\n' "$*"; }

say "0/6 机器与位置"
if [ "$(uname -s)" != Darwin ] || [ "$(uname -m)" != arm64 ]; then
  echo "这个脚本只用于 Apple Silicon 的 macOS"; exit 1
fi
sw_vers
sysctl -n machdep.cpu.brand_string
sysctl hw.ncpu hw.memsize; sysctl hw 2>/dev/null | grep -E 'perflevel[0-9]\.(name|logicalcpu):' || true
case "$REPO" in
  "$HOME/Desktop"*|"$HOME/Documents"*) echo "仓库在\"桌面\"或\"文稿\"里，可能被 iCloud 同步。请放到 ~/RM/robomorph 再运行"; exit 1;;
esac
echo "仓库：$REPO   commit $(git -C "$REPO" rev-parse --short HEAD)   $(git -C "$REPO" describe --tags --always)"

say "1/6 命令行工具（git、clang；pytinyrenderer 需要从源码编译）"
if ! xcode-select -p >/dev/null 2>&1; then
  echo "缺少命令行工具：先运行  xcode-select --install  ，装完再运行本脚本"; exit 1
fi
git --version; clang --version | head -1

say "2/6 uv"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
if ! grep -q 'UV_PROJECT_ENVIRONMENT' "$HOME/.zshrc" 2>/dev/null; then
  { echo ""; echo "# RM 研究（setup_mac.sh 写入）";
    echo "export UV_PROJECT_ENVIRONMENT=\"$UV_PROJECT_ENVIRONMENT\"";
    echo 'export PATH="$HOME/.local/bin:$PATH"'; } >> "$HOME/.zshrc"
  echo "已写入 ~/.zshrc（新开的终端生效）"
fi

say "3/6 Python（与 WSL 同一个补丁版本）"
[ -f "$REF" ] || { echo "缺少 $REF（WSL 端定版时生成并提交）"; exit 1; }
PYVER=$(sed -n 's/^ *"python": *"\([0-9.]*\)".*/\1/p' "$REF" | head -1)
echo "WSL 的 Python：$PYVER"
uv python install "$PYVER"

say "4/6 按 uv.lock 安装依赖（CPU）"
cd "$REPO"
uv sync --locked --python "$PYVER"

say "5/6 与 WSL 逐个比对版本"
uv run --locked python wj_tools/env_report.py --out wj_tools/env/mac.json
if ! uv run --locked python wj_tools/compare_env.py "$REF" wj_tools/env/mac.json; then
  echo "版本不一致：停在这里，不要自己升级或改锁文件。把日志 $LOG 发给 Claude"; exit 1
fi

say "6/6 离线测试（含线程数不影响结果的确定性测试）"
set +e
uv run --locked python -m unittest discover -s tests > "$RM_ROOT/logs/tests_mac.txt" 2>&1
rc=$?
set -e
grep -v -i warn "$RM_ROOT/logs/tests_mac.txt" | tail -6
if [ $rc -ne 0 ]; then echo "离线测试失败（详见 $RM_ROOT/logs/tests_mac.txt）"; exit 1; fi

if [ ! -f .env ]; then
  { echo "LLM_PROVIDER=anthropic"; echo "LLM_EFFORT=high"; echo "ANTHROPIC_API_KEY="; } > .env
  echo "已生成 .env（key 是空的：用文本编辑器填上，不要贴到对话里）"
fi
say "完成。日志：$LOG"
