"""环境指纹：Python、系统、CPU、全部已装包的版本、uv.lock 的 sha256、git 状态。

用法（仓库根目录下）：
    uv run --locked python wj_tools/env_report.py --out wj_tools/env/wsl.json
双机方案（claude/双机运行方案.md）用它比对 WSL 与 Mac 的依赖版本（compare_env.py），run_meta.py 也调用它。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _run(*cmd) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def cpu_name() -> str:
    if platform.system() == "Darwin":
        return _run("sysctl", "-n", "machdep.cpu.brand_string") or "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def git_state() -> dict:
    commit = _run("git", "-C", str(REPO), "rev-parse", "HEAD")
    status = _run("git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no")
    return {"git_commit": commit, "git_dirty": None if status is None else bool(status)}


def packages() -> dict:
    out = {}
    for d in metadata.distributions():
        name = (d.metadata["Name"] or "").strip().lower().replace("_", "-")
        if name:
            out[name] = d.version
    return dict(sorted(out.items()))


def report() -> dict:
    lock = REPO / "uv.lock"
    uv = shutil.which("uv") or os.path.expanduser("~/.local/bin/uv")
    rep = {
        "host": platform.node(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_impl": platform.python_implementation(),
        "uv": _run(uv, "--version") if uv else None,
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.exists() else None,
    }
    rep.update(git_state())
    rep["packages"] = packages()
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", help="写入的 JSON 路径（不给则打印）")
    a = p.parse_args()
    rep = report()
    text = json.dumps(rep, indent=1, ensure_ascii=False)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"写入 {a.out}：{rep['system']} {rep['machine']}，Python {rep['python']}，{len(rep['packages'])} 个包")
    else:
        print(text)


if __name__ == "__main__":
    main()
