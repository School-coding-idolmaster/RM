"""每个运行目录的 run_meta.json 与 DONE 标记（双机方案第 3 节规则 4、6）。

Python 里：
    from wj_tools.run_meta import start, done
    start(out_dir, cmd=..., threads=8, note="...")   # 运行开始时
    done(out_dir, n_designs=70)                      # 正常结束时
续跑时再次 start：旧记录的要点移进 previous_starts。
命令行：
    python wj_tools/run_meta.py start OUT_DIR --threads 8 --note "xcheck seed 0" --cmd "..."
    python wj_tools/run_meta.py done OUT_DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from env_report import report  # noqa: E402

KEY_PACKAGES = ("mujoco", "mujoco-mjx", "jax", "jaxlib", "brax", "numpy", "scipy", "httpx", "openai")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def start(out_dir, cmd: str | None = None, threads: int | None = None, note: str | None = None, **extra) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rep = report()
    meta = {k: rep[k] for k in ("host", "system", "release", "machine", "cpu", "cpu_count", "python", "uv",
                                "uv_lock_sha256", "git_commit", "git_dirty")}
    meta["packages"] = {k: rep["packages"].get(k) for k in KEY_PACKAGES}
    meta.update({
        "started_at": _now(),
        "cmd": cmd if cmd is not None else " ".join(sys.argv),
        "threads": threads,
        "note": note,
        "llm_provider": os.environ.get("LLM_PROVIDER"),
        "llm_effort": os.environ.get("LLM_EFFORT"),
    })
    meta.update(extra)
    prev = out / "run_meta.json"
    if prev.exists():                        # 续跑：保留之前每次启动的记录
        try:
            old = json.loads(prev.read_text(encoding="utf-8"))
            meta["previous_starts"] = old.pop("previous_starts", []) + [
                {k: old.get(k) for k in ("started_at", "host", "git_commit", "git_dirty", "cmd")}]
        except (OSError, ValueError):
            pass
    (out / "run_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    done_flag = out / "DONE"
    if done_flag.exists():   # 重新开始（或续跑）的运行在结束前不算完成
        done_flag.rename(out / f"DONE.prev_{time.strftime('%Y%m%d_%H%M%S')}")
    return meta


def done(out_dir, **summary) -> None:
    out = Path(out_dir)
    info = {"finished_at": _now(), **summary}
    (out / "DONE").write_text(json.dumps(info, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "done"])
    p.add_argument("out_dir")
    p.add_argument("--threads", type=int)
    p.add_argument("--note")
    p.add_argument("--cmd")
    a = p.parse_args()
    if a.action == "start":
        m = start(a.out_dir, cmd=a.cmd, threads=a.threads, note=a.note)
        print(f"run_meta.json → {a.out_dir}（{m['host']}，commit {str(m['git_commit'])[:8]}，dirty={m['git_dirty']}）")
    else:
        done(a.out_dir)
        print(f"DONE → {a.out_dir}")


if __name__ == "__main__":
    main()
