"""敏感性分析的物理参数选项（第 14 周，claude/正式实验计划.md 变更记录第 16 条）：默认值与之前逐字节相同；改了的参数确实进了 MJCF。"""
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from hyblock import search
from hyblock.mjcf import compile_design
from hyblock.model import load_spec

EXAMPLE = "h0=root; m1=h0.zp@90 q60/120 mirror; h2=h0.xm@0; m3=h2.zp@0 q90/150 mirror; foot=m1.P2; foot=m3.P2"
FAST = ["--steps", "2", "--threads", "1", "--episodes", "1"]


def args(*extra):
    return search.build_parser().parse_args(["--method", "ga", "--out", "x", *extra])


def forceranges(xml):
    return [tuple(float(v) for v in m.split()) for m in re.findall(r'forcerange="([^"]+)"', xml)]


def masses(xml):
    return [float(m) for m in re.findall(r'mass="([^"]+)"', xml)]


class SensitivityArgsTests(unittest.TestCase):
    def test_defaults_unchanged(self):
        self.assertEqual(search.physics_kw(args()), {})
        self.assertEqual(compile_design(EXAMPLE, max_modules=8, **search.physics_kw(args())), compile_design(EXAMPLE, max_modules=8))

    def test_pressure_scales_force_limits(self):
        base = forceranges(compile_design(EXAMPLE, max_modules=8))
        hi = forceranges(compile_design(EXAMPLE, max_modules=8, **search.physics_kw(args("--pressure", "14"))))
        self.assertTrue(base and len(base) == len(hi))
        for (a0, a1), (b0, b1) in zip(base, hi):
            self.assertAlmostEqual(b0 / a0, 14 / 8, places=6)
            self.assertAlmostEqual(b1 / a1, 14 / 8, places=6)

    def test_mass_scale(self):
        base = masses(compile_design(EXAMPLE, max_modules=8))
        hi = masses(compile_design(EXAMPLE, max_modules=8, **search.physics_kw(args("--mass_scale", "1.5"))))
        self.assertTrue(base and len(base) == len(hi))
        foot = load_spec().foot_mass                      # 被动脚不随 mass_scale 变（mjcf.py 原有行为，评估器不改）
        for x, y in zip(base, hi):
            if abs(x - foot) < 1e-9:
                self.assertAlmostEqual(y, foot, places=9)
            else:
                self.assertAlmostEqual(y / x, 1.5, places=6)

    def test_speed_limit_changes_model(self):
        kw = search.physics_kw(args("--max_speed_deg", "370"))
        self.assertEqual(kw, {"max_speed_deg_s": 370.0})
        self.assertNotEqual(compile_design(EXAMPLE, max_modules=8, **kw), compile_design(EXAMPLE, max_modules=8))

    def test_search_records_physics(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            out = tmp / "p14"
            search.main(["--method", "ga", "--budget", "3", "--batch", "3", "--pressure", "14", "--out", str(out)] + FAST)
            self.assertIn("pressure_MPa = 14.0", (out / "summary.txt").read_text(encoding="utf-8"))
            meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["args"]["pressure"], 14.0)
            self.assertTrue((out / "DONE").exists())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
