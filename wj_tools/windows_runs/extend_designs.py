"""把 RM 参照域的随机设计从 20 个扩到 N 个（默认 70），并确认前面已有的设计能逐字节重现。

用同一个生成器、同一个设计种子（sac_batch.py 的 make_designs，seed 0）重新生成 N 个设计：
  - 已有的 design_XX.xml 必须与重新生成的完全一致，否则直接退出（返回 2），不写任何文件；
  - 一致时，只写入缺少的 design_XX.xml（不覆盖已有文件）。

用法（在 robomorph 目录下）：
    uv run --locked python ../tools/extend_designs.py --n 70 --seed 0 --dir ../sac_runs/run1
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sac_batch import make_designs  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=70)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dir", default="../sac_runs/run1")
    a = p.parse_args()

    d = Path(a.dir).resolve()
    existing = sorted(d.glob("design_*.xml"))
    print(f"目录 {d}：已有 {len(existing)} 个 design_XX.xml", flush=True)
    designs = make_designs(a.n, a.seed)

    mismatch = []
    for x in designs:
        f = d / f"design_{x['idx']:02d}.xml"
        if f.exists() and f.read_text() != x["xml"]:
            mismatch.append(f.name)
    if mismatch:
        print(f"错误：{len(mismatch)} 个已有设计与重新生成的不一致：{mismatch[:10]}", flush=True)
        print("说明生成器不是确定性的（或代码变过），新设计与旧设计不是同一序列。未写入任何文件。", flush=True)
        sys.exit(2)

    n_new = 0
    for x in designs:
        f = d / f"design_{x['idx']:02d}.xml"
        if not f.exists():
            f.write_text(x["xml"])
            n_new += 1
    print(f"已有设计逐字节一致；新写入 {n_new} 个 XML，共 {a.n} 个", flush=True)
    print("nu 分布（新设计）：", [x["nu"] for x in designs if x["idx"] >= a.n - n_new], flush=True)


if __name__ == "__main__":
    main()
