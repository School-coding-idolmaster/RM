"""测一下 RM 公开版的 SAC 训练在本机 CPU 上能不能跑、要多久、占多少内存。

公开版把并行环境数写死为 4096（train/train_brax.py 的 NUM_ENVS，按 GPU 设计）。
这里在运行时把它改小，只用于"摸底计时"，不改仓库文件。
用法（在 robomorph 目录下）：
    uv run --locked python ../tools/sac_timing.py --envs 64 --ep 200 --steps 20000
    uv run --locked python ../tools/sac_timing.py --envs 256 --ep 1000 --steps 100000
输出：总耗时、峰值内存、训练得分。先小后大，别一上来就开大配置。
"""
import argparse, random, resource, sys, tempfile, time
from pathlib import Path

import mujoco
from compiler.compile import compile_robot, make_robot
from compiler.robot_generator import convert_to_components, random_design_robot

p = argparse.ArgumentParser()
p.add_argument("--envs", type=int, default=64)
p.add_argument("--ep", type=int, default=200, help="每回合步数（公开版默认 1000）")
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--terrain", default="flat")
p.add_argument("--seed", type=int, default=3, help="随机设计的种子")
a = p.parse_args()

random.seed(a.seed)
while True:
    _, s = random_design_robot()
    try:
        xml = make_robot(compile_robot(convert_to_components(s)))
        mujoco.MjModel.from_xml_string(xml)
        break
    except Exception:
        continue

work = Path(tempfile.mkdtemp(prefix="sac_timing_"))
xml_path = work / "design.xml"
xml_path.write_text(xml)

import train.train_brax as tb
tb.NUM_ENVS, tb.EP_LENGTH = a.envs, a.ep
sys.argv = ["train_brax", "--xml", str(xml_path), "--terrain", a.terrain, "--timesteps", str(a.steps),
            "--no_log", "--name", "timing", "--output_dir", str(work / "out"),
            "--min_replay_size", "1000", "--max_replay_size", "100000", "--grad_updates_per_step", "8"]
print(f"配置：envs={a.envs} ep={a.ep} steps={a.steps} terrain={a.terrain}；输出目录 {work}", flush=True)
t = time.time()
tb.main()
el = time.time() - t
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"总耗时 {el:.1f} s，峰值内存 {peak:.0f} MB")
