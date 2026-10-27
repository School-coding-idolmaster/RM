#!/usr/bin/env bash
# 在 WSL2 (Ubuntu 22.04) 里配置 RoboMorph 公开版（V3）的 CPU 环境。
# 代码放在 D:\RM\robomorph（WSL 里是 /mnt/d/RM/robomorph），虚拟环境放在 WSL 自己的磁盘上（快、无符号链接问题）。
# 用法：bash /mnt/d/RM/tools/setup_wsl.sh
set -euo pipefail

RM_DIR="${RM_DIR:-/mnt/d/RM}"
REPO="$RM_DIR/robomorph"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"

say() { printf '\n==> %s\n' "$*"; }

[ -d "$REPO" ] || { echo "找不到 $REPO ，请先确认 D:\\RM\\robomorph 存在"; exit 1; }

say "1/5 系统工具（git、curl、编译工具）"
need=()
for c in git curl; do command -v "$c" >/dev/null || need+=("$c"); done
command -v gcc >/dev/null || need+=(build-essential)
if [ ${#need[@]} -gt 0 ]; then
  echo "缺少：${need[*]}，将用 sudo apt 安装（会要求输入 WSL 的密码）"
  sudo apt-get update && sudo apt-get install -y ca-certificates "${need[@]}"
else
  echo "已齐全"
fi

say "2/5 uv（Python 包管理器）"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
grep -q 'UV_PROJECT_ENVIRONMENT' "$HOME/.bashrc" 2>/dev/null || {
  echo "export UV_PROJECT_ENVIRONMENT=\"$UV_PROJECT_ENVIRONMENT\"" >> "$HOME/.bashrc"
  echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
}

say "3/5 安装依赖（锁定版本，Python 3.11，CPU）"
cd "$REPO"
git config core.fileMode false 2>/dev/null || true
uv sync --locked --python 3.11
uv run --locked robomorph --help >/dev/null && echo "robomorph 命令可用"

say "4/5 离线测试（23 项，不调用 LLM、不需要 GPU）"
uv run --locked python -m unittest discover -s tests 2>&1 | tail -4

say "5/5 冒烟测试与 .env"
uv run --locked python "$RM_DIR/tools/smoke_test.py" 2>&1 | grep -v -i warn
if [ ! -f .env ]; then
  cp .env.example .env
  cat >> .env <<'ENVEOF'

# 接 DeepSeek 时取消下面两行的注释（key 只写在这里，不要发到对话里）：
# OPENAI_BASE_URL=https://api.deepseek.com
# 并把 OPENAI_API_KEY= 填成 DeepSeek 的 key，运行时加 --model deepseek-chat
ENVEOF
  echo "已生成 $REPO/.env （未填 key）"
fi
say "完成。以后在 $REPO 里用：uv run --locked robomorph --help"
