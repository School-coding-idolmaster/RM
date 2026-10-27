#!/usr/bin/env bash
# 工作流自检：在 Mac 上写一个带 run_meta.json 和 DONE 的结果目录 ~/RM/mac_runs/hello（约 10 秒）。
set -u
export PATH="$HOME/.local/bin:$PATH"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$HOME/.venvs/robomorph}"
cd "$(dirname "$0")/.." || exit 1
out="$(cd .. && pwd)/mac_runs/hello"
uv run --locked python wj_tools/run_meta.py start "$out" --note "workflow self-test"
uv run --locked python -c "import mujoco, numpy; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__)" | tee "$out/hello.txt"
sleep 5
date '+%F %T' >> "$out/hello.txt"
uv run --locked python wj_tools/run_meta.py done "$out"
echo "hello 完成"
