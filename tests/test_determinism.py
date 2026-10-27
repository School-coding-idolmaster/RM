"""确定性（双机方案第 1、2 节的前提）：同一台机器、同一种子下，MPPI 的回合结果与线程数无关、重复运行逐位相同。

跨机器（WSL x86 ↔ Mac arm64）不要求逐位相同，由一致性检验（第 4 节）判断。
"""
import unittest
from pathlib import Path

import numpy as np

from mpc.mppi import MPPIConfig, evaluate_design

ROOT = Path(__file__).resolve().parents[1]
ANT = ROOT / "mpc" / "ant_test.xml"
HYB = "h0=root; m1=h0.zp@90 mirror; foot=m1.P2"
STEPS = 25          # 控制步（完整回合是 250），足以覆盖热启动采样和接触


def _rewards(xml: str, threads: int, seed: int = 0) -> np.ndarray:
    cfg = MPPIConfig(threads=threads, episode_control_steps=STEPS)
    r = evaluate_design(xml, "flat", 1, seed, cfg)
    return np.asarray(r["rewards"][0], dtype=float)


class TestDeterminism(unittest.TestCase):
    def _check(self, xml: str):
        ref = _rewards(xml, 1)
        self.assertGreater(len(ref), 0)
        for t in (6, 12, 12):            # 12 跑两次 = 重复运行
            with self.subTest(threads=t):
                out = _rewards(xml, t)
                self.assertTrue(np.array_equal(out, ref),
                                f"threads={t} 与 threads=1 不同（长度 {len(out)} vs {len(ref)}）")

    def test_reference_domain_thread_invariant(self):
        self._check(ANT.read_text())

    def test_hyblock_domain_thread_invariant(self):
        from hyblock.mjcf import compile_design
        self._check(compile_design(HYB, max_modules=6))

    def test_seed_matters(self):
        xml = ANT.read_text()
        self.assertFalse(np.array_equal(_rewards(xml, 6, 0), _rewards(xml, 6, 1)))


if __name__ == "__main__":
    unittest.main()
