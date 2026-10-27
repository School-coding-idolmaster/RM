"""批量补跑 SAC：生成一批固定的随机设计，在 CPU 上依次训练并记录得分（可中断、可续跑）。

用途：为"MPC 与 SAC 的排序一致性"检验准备 SAC 得分。同一个 --seed 生成的设计序列是确定的，
所以先 --n 3 试跑，之后改成 --n 20 会复用前 3 个的结果，只补跑新增的。
续跑按（设计编号、SAC 种子、步数、环境数、回合长度、梯度更新数）判断是否已完成：换种子或换配置会重新训练。
配置是缩小版（默认 64 环境、回合 200 步、每步 8 次梯度更新），不是 RM 论文的 4096 环境配置，
得分只能在同一套设置内比较相对排序，不能和 RM 论文的数值直接比。

用法（在 robomorph 目录下，窗口保持打开）：
    uv run --locked python ../tools/sac_batch.py --n 3 --steps 500000 2>&1 | tee ../sac_runs/pilot.log
    uv run --locked python ../tools/sac_batch.py --n 20 --steps 500000 2>&1 | tee ../sac_runs/full.log
先检查不训练：加 --check
"""
import argparse, csv, json, random, shutil, subprocess, sys, time
from pathlib import Path


def make_designs(n, seed):
    import mujoco
    from compiler.compile import compile_robot, make_robot
    from compiler.robot_generator import convert_to_components, random_design_robot
    random.seed(seed)
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        _, structure = random_design_robot()
        comp = convert_to_components(structure)
        try:
            xml = make_robot(compile_robot(comp))
            m = mujoco.MjModel.from_xml_string(xml)
        except Exception:
            continue
        out.append({"idx": len(out), "structure": structure, "comp": comp, "xml": xml, "nu": int(m.nu)})
    return out


def worker(a):
    import train.train_brax as tb
    tb.NUM_ENVS, tb.EP_LENGTH = a.envs, a.ep
    cfg = tb.ExperimentConfig(xml=a.xml, name=a.name, render=False, log=False, algorithm="sac",
                              timesteps=a.steps, num_evals=a.num_evals, terrain=a.terrain,
                              seed=a.sac_seed, output_dir=a.run_dir)
    hp = tb.Hyperparameters(lr=1e-3, batch_size=256, discounting=0.99, grad_updates_per_step=a.updates,
                            min_replay_size=1000, max_replay_size=100000, num_minibatches=0, entropy_cost=0)
    if a.check:
        tb._validate_config(cfg, hp)
        print("配置校验通过（未训练）")
        return
    reward = tb.train_policy(cfg, hp)
    print(f"RESULT {reward}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=3, help="设计个数")
    p.add_argument("--seed", type=int, default=0, help="随机设计的种子（决定设计序列）")
    p.add_argument("--steps", type=int, default=500_000)
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--ep", type=int, default=200)
    p.add_argument("--updates", type=int, default=8, help="每个演员步的梯度更新次数")
    p.add_argument("--num_evals", type=int, default=5, help="训练中评估几次（用来看学习曲线）")
    p.add_argument("--terrain", default="flat")
    p.add_argument("--sac_seed", type=int, default=0)
    p.add_argument("--out", default="../sac_runs/run1")
    p.add_argument("--check", action="store_true", help="只生成设计并校验配置，不训练")
    # 内部：子进程模式
    p.add_argument("--worker", action="store_true")
    p.add_argument("--xml"); p.add_argument("--name"); p.add_argument("--run_dir")
    a = p.parse_args()
    if a.worker:
        return worker(a)

    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    designs = make_designs(a.n, a.seed)
    csv_path = out / "results.csv"
    done = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                if r["status"] == "ok":
                    done[(int(r["idx"]), int(r["sac_seed"]), int(r["steps"]), int(r["envs"]),
                          int(r["ep"]), int(r["updates"]))] = r
    new_file = not csv_path.exists()
    with open(out / "designs.jsonl", "w") as f:
        for d in designs:
            f.write(json.dumps({k: d[k] for k in ("idx", "structure", "comp", "nu")}, ensure_ascii=False) + "\n")
    for d in designs:
        (out / f"design_{d['idx']:02d}.xml").write_text(d["xml"])
    cfg_key = (a.sac_seed, a.steps, a.envs, a.ep, a.updates)
    n_done = sum(1 for k in done if k[1:] == cfg_key and k[0] < a.n)
    print(f"设计 {len(designs)} 个（seed={a.seed}），该配置下已完成 {n_done} 个；输出目录 {out}", flush=True)
    print("nu 分布：", [d["nu"] for d in designs], flush=True)

    with open(csv_path, "a", newline="") as cf:
        w = csv.writer(cf)
        if new_file:
            w.writerow(["idx", "nu", "reward", "seconds", "status", "steps", "envs", "ep", "updates", "sac_seed"])
        for d in designs:
            i = d["idx"]
            if (i,) + cfg_key in done:
                print(f"[{i}] 已完成，跳过（reward={done[(i,) + cfg_key]['reward']}）", flush=True)
                continue
            name = f"design_{i:02d}"
            run_dir = out / "runs"
            stale = run_dir / f"{name}_s_{a.sac_seed}"
            if stale.exists():
                shutil.rmtree(stale)
            cmd = [sys.executable, __file__, "--worker", "--xml", str(out / f"{name}.xml"), "--name", name,
                   "--run_dir", str(run_dir), "--steps", str(a.steps), "--envs", str(a.envs), "--ep", str(a.ep),
                   "--updates", str(a.updates), "--num_evals", str(a.num_evals), "--terrain", a.terrain,
                   "--sac_seed", str(a.sac_seed)]
            if a.check:
                cmd.append("--check")
            print(f"\n[{i}] 开始训练 nu={d['nu']} steps={a.steps} ...", flush=True)
            t = time.time(); reward = None
            with open(out / f"{name}.log", "w") as lf:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    lf.write(line)
                    if line.startswith(("Step:", "RESULT", "Training finished", "配置校验", "Traceback")) or "Error" in line:
                        print(f"  {line.rstrip()}", flush=True)
                    if line.startswith("RESULT"):
                        reward = float(line.split()[1])
                proc.wait()
            sec = time.time() - t
            ok = proc.returncode == 0 and (reward is not None or a.check)
            if not a.check:
                w.writerow([i, d["nu"], reward, f"{sec:.0f}", "ok" if ok else "fail", a.steps, a.envs, a.ep,
                            a.updates, a.sac_seed]); cf.flush()
            print(f"[{i}] {'完成' if ok else '失败（见 ' + name + '.log）'}：reward={reward}，用时 {sec/60:.1f} 分钟", flush=True)
    print("\n全部完成。结果在", csv_path)


if __name__ == "__main__":
    main()
