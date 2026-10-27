"""CPU MPPI evaluator for RoboMorph designs (MuJoCo 3.2.6, no JAX/Brax).

Algorithm follows RoboGrammar core/src/optim.cpp (MPPIOptimizer + DefaultInputSampler);
parameters follow the RoboMorph V3 settings given by Kevin Qiu (2026-10-21):
64 samples per control step (512 on the first update), half noise std 0.25 / half 0.05,
horizon 16 control steps, 4 env steps per control step, kappa 1.0, ~250 control steps
(= 1000 env steps) per episode, no value function.

Environment semantics copied from RoboMorph train/envs/flat_env.py so that MPC scores are on
the same scale as the SAC fitness (eval/episode_reward_forward):
  * env step = n_frames (5) physics steps of opt.timestep (0.01 s from the compiled XML) -> dt 0.05 s
  * action in [-1, 1] mapped linearly onto actuator ctrlrange (RoboMorph uses ctrlrange -1..1, gear 50)
  * reward per env step = -(torso x velocity)        (robots walk in -x)
  * episode ends when torso z < 0.05 or > 5.0       (terminate_when_unhealthy=True)
  * reset noise: qpos += U(-0.1, 0.1), qvel = 0.1 * N(0, 1)
  * solver: the XML's own options (Newton, 4 iterations) by default. RoboMorph's MJX backend
    overrides to 1 iteration / 4 line-search iterations; on CPU MuJoCo that setting blows up
    under random 50 N*m torques, and MuJoCo then auto-resets the data, which MPPI exploits as a
    huge "forward" jump. `mjx_solver=True` reproduces the MJX override for comparison.
  * physics glitches (MuJoCo reset, non-finite state, |v_x| > max_speed) end the rollout with
    zero reward from that step on; in the executed episode they end the episode (flag `unstable`).
"""
from __future__ import annotations

import math
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout


# ----------------------------------------------------------------------------------------------
# Terrain (identical geometry to robomorph train/envs/*.py)
# ----------------------------------------------------------------------------------------------

def apply_terrain(xml_string: str, terrain: str, scale: float = 1.0) -> str:
    """Return the robot XML with the RoboMorph terrain added.

    `scale` multiplies all terrain lengths (Hyblock domain; 1.0 = RoboMorph). If the floor geom has a
    `priority` attribute (Hyblock XMLs), the obstacles get the same priority so that the terrain, not
    the robot, sets the contact friction."""
    if terrain == "flat":
        return xml_string
    root = ET.fromstring(xml_string)
    worldbody = root.find(".//worldbody")
    floors = root.findall(".//geom[@type='plane']")
    priority = floors[0].get("priority") if len(floors) == 1 else None
    if terrain in ("ridged", "beams"):
        spread, radius = 2.0 * scale, 0.2 * scale
        height = 0.5 * scale if terrain == "beams" else 0.0
        for i in range(15):
            g = ET.SubElement(worldbody, "geom", name=f"{'beam' if terrain == 'beams' else 'ridge'}_{i}",
                              pos=f"-{i * spread + 1.0 * scale} 0 {height}", size=f"{radius} 10.0",
                              quat="0.7071067811865476 0.7071067811865476 0.0 0.0", type="cylinder",
                              contype="0", conaffinity="1", rgba="0.7 0.5 0.3 1.0")
            if priority is not None:
                g.set("priority", priority)
    elif terrain == "frozen_lake":
        floor = root.findall(".//geom[@type='plane']")
        if len(floor) != 1:
            raise ValueError(f"Expected exactly 1 ground plane in XML, found {len(floor)}")
        floor[0].set("friction", "0.05")
    else:
        raise ValueError(f"Unknown terrain: {terrain}")
    return ET.tostring(root, encoding="unicode")


def load_model(xml_string: str, terrain: str = "flat", mjx_solver: bool = False,
               terrain_scale: float = 1.0, min_iterations: int | None = None) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_string(apply_terrain(xml_string, terrain, terrain_scale))
    if min_iterations and not mjx_solver:
        model.opt.iterations = max(model.opt.iterations, int(min_iterations))
    if mjx_solver:
        model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        model.opt.iterations = 1
        model.opt.ls_iterations = 4
        model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_EULERDAMP
    return model


# ----------------------------------------------------------------------------------------------
# Low-pass FIR for the sampling noise (RoboGrammar: 5th-order FIR, 15 Hz control, passband < 2 Hz)
# ----------------------------------------------------------------------------------------------

def _custom_numeric(model: mujoco.MjModel, name: str):
    """Value of a <custom><numeric name=...> entry, or None."""
    i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    if i < 0:
        return None
    adr, size = model.numeric_adr[i], model.numeric_size[i]
    return np.array(model.numeric_data[adr:adr + size])


def fir_lowpass(numtaps: int = 6, cutoff_hz: float = 2.0, fs_hz: float = 15.0) -> np.ndarray:
    """Windowed-sinc (Hamming) low-pass, unit DC gain. Normalised cutoff copied from RoboGrammar."""
    n = np.arange(numtaps) - (numtaps - 1) / 2.0
    fc = cutoff_hz / fs_hz
    h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(numtaps)
    return h / h.sum()


@dataclass
class MPPIConfig:
    num_samples: int = 64
    warmup_samples: int = 512
    horizon: int = 16            # control steps
    interval: int = 4            # env steps per control step
    n_frames: int = 5            # physics steps per env step (RoboMorph FlatEnv)
    episode_control_steps: int = 250
    kappa: float = 1.0
    discount: float = 0.99       # per control step inside the horizon
    noise_std_history: float = 0.1   # 2x Kevin's 0.05: the FIR (gain 0.54) brings the effective std back to ~0.05
    noise_std_warm: float = 0.5      # 2x Kevin's 0.25, same reason (chosen 10-26, see 第二阶段记录)
    healthy_z: tuple = (0.05, 5.0)
    reset_noise: float = 0.1
    threads: int = 8
    mjx_solver: bool = False
    fall_penalty: float = 100.0  # planner-only: subtracted when a sample terminates/glitches (score unaffected)
    max_speed: float = 10.0       # m/s; faster torso motion is treated as a physics glitch
                                  # (20 -> 10 on 10-27: rank70 found 6/210 RM episodes riding glitches at
                                  #  18-20 m/s; all other episodes peak <= 5.6 m/s)
    filter_noise: bool = True
    min_solver_iterations: int = 20  # 10-27: RM XMLs ship 4 Newton iterations; with 4 the planner rode
                                     # contact glitches (18-20 m/s); at 20 they vanish (tools/solver_test.py).
                                     # Hyblock XMLs already use 20, so they are unchanged.
    terrain_scale: float = 1.0    # Hyblock domain: terrain lengths x scale (RoboMorph = 1.0)


@dataclass
class EpisodeResult:
    score: float                         # sum of forward reward over the episode (SAC fitness scale)
    score_first200: float                # same, first 200 env steps (SAC batch used ep=200)
    distance: float                      # torso displacement in -x [m]
    env_steps: int
    terminated: bool
    unstable: bool
    seconds: float
    physics_steps: int
    rewards: list = field(default_factory=list, repr=False)
    x_start: float = 0.0                 # torso x at reset (Hyblock terrain pass rates)


class MPPIEvaluator:
    def __init__(self, model: mujoco.MjModel, cfg: MPPIConfig = MPPIConfig(), seed: int = 0):
        self.m = model
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.nu = model.nu
        if self.nu == 0:
            raise ValueError("design has no actuators")
        if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE or model.jnt_bodyid[0] != 1:
            raise ValueError("expected the torso (body 1) to carry the free joint")
        self.ctrl_lo = model.actuator_ctrlrange[:, 0].copy()
        self.ctrl_hi = model.actuator_ctrlrange[:, 1].copy()
        # Hyblock XMLs (hyblock/mjcf.py) declare themselves through <custom> numerics; RoboMorph XMLs don't.
        box = _custom_numeric(model, "hyblock_torso_box")
        self.hyblock = box is not None
        if self.hyblock:
            x0, x1, y0, y1, z0, z1 = box
            self.torso_corners = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
            self.torso_touch = float(_custom_numeric(model, "hyblock_torso_touch")[0])
            self.init_clearance = float(_custom_numeric(model, "hyblock_init_clearance")[0])
            nf = _custom_numeric(model, "hyblock_n_frames")
            self.n_frames = int(nf[0]) if nf is not None else cfg.n_frames
        else:
            self.n_frames = cfg.n_frames
        self.dt_env = model.opt.timestep * self.n_frames
        self.steps_per_ctrl = cfg.interval * self.n_frames
        self.fir = fir_lowpass() if cfg.filter_noise else np.array([1.0])
        self.state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self.datas = [mujoco.MjData(model) for _ in range(max(1, cfg.threads))]
        self.pool = ThreadPoolExecutor(max_workers=cfg.threads) if cfg.threads > 1 else None
        self.physics_steps = 0

    # ---- helpers ------------------------------------------------------------------------------
    def scale(self, a: np.ndarray) -> np.ndarray:
        if self.hyblock:
            # zero-centred: a > 0 -> fraction of the extension limit, a < 0 -> of the retraction limit
            return np.where(a >= 0, a * self.ctrl_hi, -a * self.ctrl_lo)
        return (a + 1.0) * (self.ctrl_hi - self.ctrl_lo) * 0.5 + self.ctrl_lo

    def unhealthy(self, qpos_free: np.ndarray) -> np.ndarray:
        """Termination test on the free-joint pose (..., 7) = [x, y, z, qw, qx, qy, qz].

        RoboMorph: torso z outside healthy_z. Hyblock: the torso (root module A) touches the ground
        (its lowest box corner below torso_touch) or is upside down (its local +y axis points down)."""
        z = qpos_free[..., 2]
        if not self.hyblock:
            return (z < self.cfg.healthy_z[0]) | (z > self.cfg.healthy_z[1])
        w, x, y, zq = (qpos_free[..., k] for k in range(3, 7))
        r3 = np.stack([2 * (x * zq - w * y), 2 * (y * zq + w * x), 1 - 2 * (x * x + y * y)], axis=-1)
        corner_z = z[..., None] + r3 @ self.torso_corners.T
        return (corner_z.min(axis=-1) < self.torso_touch) | (r3[..., 1] < 0) | (z > self.cfg.healthy_z[1])

    def _filtered_noise(self, n: int, std: float) -> np.ndarray:
        """Gaussian noise (n, H, nu), low-pass filtered along time (causal FIR, warm-started)."""
        H, k = self.cfg.horizon, len(self.fir)
        raw = self.rng.standard_normal((n, H + k - 1, self.nu)) * std
        if k == 1:
            return raw
        out = np.zeros((n, H, self.nu))
        for i, c in enumerate(self.fir[::-1]):
            out += c * raw[:, i:i + H, :]
        return out

    def _history_base(self, history: np.ndarray) -> np.ndarray:
        """RoboGrammar history sampler: repeat the last p executed actions with period p."""
        H = self.cfg.horizon
        if len(history) == 0:
            return np.zeros((H, self.nu))
        p = int(self.rng.integers(H // 2 + 1, H + 1))
        p = min(p, len(history))
        tail = history[-p:]
        return np.stack([tail[j % p] for j in range(H)])

    def _sample(self, plan: np.ndarray, history: np.ndarray, n: int) -> np.ndarray:
        cfg = self.cfg
        n_hist = n // 2
        n_warm = n - n_hist
        hist_bases = np.stack([self._history_base(history) for _ in range(n_hist)]) if n_hist else np.zeros((0, cfg.horizon, self.nu))
        # linear transition from the current plan into the periodic history (RoboGrammar blends)
        w = np.linspace(0.0, 1.0, cfg.horizon)[None, :, None]
        hist_bases = (1 - w) * plan[None] + w * hist_bases
        u_hist = hist_bases + self._filtered_noise(n_hist, cfg.noise_std_history)
        u_warm = plan[None] + self._filtered_noise(n_warm, cfg.noise_std_warm)
        u = np.empty((n, cfg.horizon, self.nu))
        u[0::2][:n_hist] = u_hist
        u[1::2][:n_warm] = u_warm
        return np.clip(u, -1.0, 1.0)

    def _rollout_chunk(self, data, state0, ctrl):
        nroll, nstep = ctrl.shape[0], ctrl.shape[1]
        init = np.repeat(state0[None], nroll, axis=0)
        st, _ = mj_rollout.rollout(self.m, data, init, ctrl)
        return st

    def _returns(self, state0: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Discounted planning return of each action sequence u (n, H, nu)."""
        cfg = self.cfg
        n, H = u.shape[0], cfg.horizon
        ctrl = np.repeat(self.scale(u), self.steps_per_ctrl, axis=1)  # (n, H*steps, nu)
        if self.pool is None:
            states = self._rollout_chunk(self.datas[0], state0, ctrl)
        else:
            chunks = np.array_split(np.arange(n), len(self.datas))
            futs = [self.pool.submit(self._rollout_chunk, d, state0, ctrl[idx])
                    for d, idx in zip(self.datas, chunks) if len(idx)]
            states = np.concatenate([f.result() for f in futs], axis=0)
        self.physics_steps += ctrl.shape[0] * ctrl.shape[1]
        # full-physics state = [time, qpos, qvel, act, ...]; torso free joint is qpos[0:3]
        nf = self.n_frames
        t = states[:, nf - 1::nf, 0]
        x = states[:, nf - 1::nf, 1]          # torso x after each env step
        pose = states[:, nf - 1::nf, 1:8]     # torso free joint [x y z qw qx qy qz]
        x_prev = np.concatenate([np.full((n, 1), state0[1]), x[:, :-1]], axis=1)
        t_prev = np.concatenate([np.full((n, 1), state0[0]), t[:, :-1]], axis=1)
        with np.errstate(invalid="ignore"):
            r = -(x - x_prev) / self.dt_env                            # (n, H*interval)
            glitch = ~np.isfinite(r) | (t <= t_prev) | (np.abs(r) > cfg.max_speed)
            unhealthy = self.unhealthy(pose)
        r = np.where(glitch, 0.0, r)
        alive = np.cumprod(~glitch, axis=1)                             # a glitch zeroes its own step
        healthy_before = np.concatenate([np.ones((n, 1), bool), ~unhealthy[:, :-1]], axis=1)
        alive = alive * np.cumprod(healthy_before, axis=1)              # the step that falls still counts
        r = (r * alive).reshape(n, H, cfg.interval).sum(axis=2)
        disc = cfg.discount ** np.arange(H)
        ret = (r * disc).sum(axis=1)
        # RoboMorph has no alive bonus, so a short-horizon planner happily lunges and falls (the SAC
        # critic sees the lost future reward; MPPI does not). Penalise ending the episode instead.
        ended = glitch | unhealthy
        any_end = ended.any(axis=1)
        first = np.where(any_end, ended.argmax(axis=1), 0) // cfg.interval
        ret = ret - any_end * cfg.fall_penalty * disc[first]
        return ret

    def update(self, data: mujoco.MjData, plan: np.ndarray, history: np.ndarray, n: int) -> np.ndarray:
        state0 = np.empty(self.state_size)
        mujoco.mj_getState(self.m, data, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        u = self._sample(plan, history, n)
        R = self._returns(state0, u)
        w = np.exp(self.cfg.kappa * (R - R.max()))
        return np.tensordot(w / w.sum(), u, axes=1)

    # ---- episode ------------------------------------------------------------------------------
    def reset(self, seed: int) -> mujoco.MjData:
        rng = np.random.default_rng(seed)
        d = mujoco.MjData(self.m)
        if self.hyblock:
            # joints only (a ±0.1 m noise on the height or the quaternion would push a 6 kg module
            # into the floor); then re-lift so that the lowest point is init_clearance above the ground
            d.qpos[:] = self.m.qpos0
            d.qpos[7:] += rng.uniform(-self.cfg.reset_noise, self.cfg.reset_noise, self.m.nq - 7)
            lo, hi = self.m.jnt_range[1:, 0], self.m.jnt_range[1:, 1]
            d.qpos[7:] = np.clip(d.qpos[7:], lo, hi)
            mujoco.mj_forward(self.m, d)
            from hyblock.mjcf import lowest_point
            d.qpos[2] += self.init_clearance - lowest_point(self.m, d)
        else:
            d.qpos[:] = self.m.qpos0 + rng.uniform(-self.cfg.reset_noise, self.cfg.reset_noise, self.m.nq)
        d.qvel[:] = self.cfg.reset_noise * rng.standard_normal(self.m.nv)
        mujoco.mj_forward(self.m, d)
        return d

    def run_episode(self, seed: int = 0, verbose: bool = False) -> EpisodeResult:
        cfg = self.cfg
        t0 = time.time()
        self.physics_steps = 0
        d = self.reset(seed)
        plan = np.zeros((cfg.horizon, self.nu))
        history = np.zeros((0, self.nu))
        rewards = []
        terminated = unstable = False
        x_start = d.qpos[0]
        badqacc = int(mujoco.mjtWarning.mjWARN_BADQACC)
        for k in range(cfg.episode_control_steps):
            n = cfg.warmup_samples if k == 0 else cfg.num_samples
            plan = self.update(d, plan, history, n)
            a = plan[0].copy()
            d.ctrl[:] = self.scale(a)
            for _ in range(cfg.interval):
                x0, nwarn = d.qpos[0], d.warning[badqacc].number
                mujoco.mj_step(self.m, d, nstep=self.n_frames)
                r = -(d.qpos[0] - x0) / self.dt_env
                if d.warning[badqacc].number > nwarn or not np.isfinite(r) or abs(r) > cfg.max_speed:
                    unstable = terminated = True
                    break
                rewards.append(r)
                if self.unhealthy(d.qpos[:7]):
                    terminated = True
                    break
            self.physics_steps += self.n_frames * cfg.interval
            history = np.vstack([history, a[None]])[-cfg.horizon:]
            plan = np.vstack([plan[1:], np.zeros((1, self.nu))])
            if terminated:
                break
            if verbose and (k + 1) % 50 == 0:
                print(f"  ctrl step {k + 1}: score so far {sum(rewards):.2f}, x={d.qpos[0]:.3f}, "
                      f"{time.time() - t0:.1f}s", flush=True)
        return EpisodeResult(score=float(np.sum(rewards)), score_first200=float(np.sum(rewards[:200])),
                             distance=float(np.sum(rewards) * self.dt_env), env_steps=len(rewards),
                             terminated=terminated, unstable=unstable, seconds=time.time() - t0,
                             physics_steps=self.physics_steps, rewards=[float(r) for r in rewards],
                             x_start=float(x_start))

    def close(self):
        if self.pool is not None:
            self.pool.shutdown()


def evaluate_design(xml_string: str, terrain: str = "flat", episodes: int = 1, seed: int = 0,
                    cfg: MPPIConfig = MPPIConfig(), verbose: bool = False) -> dict:
    """Score one design: mean MPC episode score (same scale as RoboMorph SAC fitness)."""
    model = load_model(xml_string, terrain, cfg.mjx_solver, cfg.terrain_scale, cfg.min_solver_iterations)
    ev = MPPIEvaluator(model, cfg, seed=seed)
    try:
        eps = [ev.run_episode(seed=seed * 1000 + i, verbose=verbose) for i in range(episodes)]
    finally:
        ev.close()
    return {
        "fitness": float(np.mean([e.score for e in eps])),
        "fitness_first200": float(np.mean([e.score_first200 for e in eps])),
        "episodes": [{k: v for k, v in asdict(e).items() if k != "rewards"} for e in eps],
        "rewards": [e.rewards for e in eps],
        "nu": int(model.nu),
        "max_distance": float(max(_max_progress(e.rewards, ev.dt_env) for e in eps)),
    }


def _max_progress(rewards, dt_env) -> float:
    """Furthest torso position reached in -x during an episode [m] (for obstacle pass rates)."""
    if not rewards:
        return 0.0
    return float(np.max(np.cumsum(rewards)) * dt_env)
