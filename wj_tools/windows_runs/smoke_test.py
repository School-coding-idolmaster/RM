"""冒烟测试（CPU）：随机设计 -> 编译 -> MuJoCo 加载与仿真 -> rollout 计时与线程并行。

用法（在 robomorph 目录下）：
    uv run --locked python ../tools/smoke_test.py
第一次在本机跑，输出里的"步/秒"就是第二阶段估算 MPC 预算的第一个真实数字。
"""
import collections, os, random, time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np
from mujoco import rollout

from compiler.compile import compile_robot, make_robot
from compiler.robot_generator import convert_to_components, random_design_robot

print(f"mujoco {mujoco.__version__}, CPU 逻辑核数 {os.cpu_count()}")

# 1) 随机设计能否编译/加载/仿真
random.seed(1)
err, models = collections.Counter(), []
N_TRY = 300
for _ in range(N_TRY):
    _, s = random_design_robot()
    try:
        xml = make_robot(compile_robot(convert_to_components(s)))
        models.append(mujoco.MjModel.from_xml_string(xml))
    except Exception as e:
        err[str(e)[:40]] += 1
print(f"[1] 随机设计 {N_TRY} 个：编译成功 {len(models)}，失败 {sum(err.values())} {dict(err)}")

m = models[0]
d = mujoco.MjData(m)
for _ in range(100):
    mujoco.mj_step(m, d)
assert np.isfinite(d.qpos).all(), "仿真状态出现非有限数"
print(f"[2] 示例设计 nq={m.nq} nv={m.nv} nu={m.nu}，100 步仿真状态有限")

# 3) rollout：64 条轨迹 x 16 步
N, H = 64, 16
def make_inputs():
    dd = mujoco.MjData(m)
    init = np.tile(np.concatenate([[dd.time], dd.qpos, dd.qvel]), (N, 1))
    ctrl = np.random.uniform(-1, 1, (N, H, m.nu))
    return dd, init, ctrl
def run(args):
    dd, init, ctrl, reps = args
    for _ in range(reps):
        st, _s = rollout.rollout(m, dd, init, ctrl)
    return st
dd, init, ctrl = make_inputs()
run((dd, init, ctrl, 3))  # 预热
reps = 30
t = time.perf_counter(); st = run((dd, init, ctrl, reps)); one = (time.perf_counter() - t) / reps
sps1 = N * H / one
print(f"[3] 单线程：{N} 条轨迹 x {H} 步 = {one*1000:.1f} ms/次，{sps1:,.0f} 步/秒，状态有限={bool(np.isfinite(st).all())}")
for nt in sorted({2, 4, min(8, os.cpu_count() or 1)}):
    if nt > (os.cpu_count() or 1):
        continue
    ins = [make_inputs() for _ in range(nt)]
    t = time.perf_counter()
    with ThreadPoolExecutor(nt) as ex:
        list(ex.map(run, [(a, b, c, reps) for (a, b, c) in ins]))
    el = time.perf_counter() - t
    print(f"[3] {nt} 线程并行：吞吐 {nt*reps*N*H/el:,.0f} 步/秒，相对单线程 {nt*reps*one/el:.2f}x")
print("冒烟测试完成")
