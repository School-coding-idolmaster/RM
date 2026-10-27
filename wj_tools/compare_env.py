"""比对两份 env_report.py 的输出：Python 版本、uv.lock、每个包的版本必须完全一致。

用法：python wj_tools/compare_env.py wj_tools/env/wsl.json wj_tools/env/mac.json [--allow-missing pytinyrenderer]
一致时退出码 0，否则打印差异并退出 1。主机名、CPU、git 状态不参与比对。
"""
from __future__ import annotations

import argparse
import json
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ref")
    p.add_argument("new")
    p.add_argument("--allow-missing", nargs="*", default=[],
                   help="允许只在一边存在的包（例如 Mac 上跳过编译的 pytinyrenderer）")
    a = p.parse_args()
    ref, new = (json.load(open(f, encoding="utf-8")) for f in (a.ref, a.new))
    bad = []
    for key in ("python", "python_impl", "uv_lock_sha256"):
        if ref.get(key) != new.get(key):
            bad.append(f"{key}: {ref.get(key)} != {new.get(key)}")
    rp, np_ = ref["packages"], new["packages"]
    allowed = {x.lower() for x in a.allow_missing}
    for name in sorted(set(rp) | set(np_)):
        va, vb = rp.get(name), np_.get(name)
        if va == vb:
            continue
        if (va is None or vb is None) and name in allowed:
            print(f"  （允许）{name}: {va} / {vb}")
            continue
        bad.append(f"{name}: {va} != {vb}")
    print(f"参照 {ref.get('system')} {ref.get('machine')} ({ref.get('host')}) vs "
          f"新环境 {new.get('system')} {new.get('machine')} ({new.get('host')})；"
          f"共 {len(set(rp) | set(np_))} 个包")
    if bad:
        print("不一致：")
        for b in bad:
            print("  " + b)
        sys.exit(1)
    print("一致：Python、uv.lock 与全部包版本相同")


if __name__ == "__main__":
    main()
