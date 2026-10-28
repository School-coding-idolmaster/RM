"""① 筛选与自碰撞（第 7 周，10-28）。"""
import random
import unittest

import mujoco

from hyblock.design import DesignError, to_text
from hyblock.generator import mutate, random_design
from hyblock.mjcf import check_self_collision, compile_design, self_penetration
from hyblock.screen import ScreenConfig, postcheck, screen

QUAD = ("h0=root; m1=h0.zp@90 q90/150 mirror; m2=h0.xm@0 q90; h3=m2.P2@0; m4=h3.zp@90 q90/150 mirror; "
        "foot=m1.P2; foot=m4.P2")
FOLDED = "m0=root; m1=m0.P2@0 q150; m2=m1.P2@0 q150; m3=m2.P2@0 q150"     # chain folded into itself


class SelfCollisionTests(unittest.TestCase):
    def test_folded_chain_is_invalid(self):
        with self.assertRaises(DesignError) as cm:
            compile_design(FOLDED)
        self.assertEqual(cm.exception.kind, "self_collision")
        with self.assertRaises(DesignError):
            check_self_collision(FOLDED)

    def test_straight_chain_and_quad_are_valid(self):
        for text in ("m0=root; m1=m0.P2@0; m2=m1.P2@0", QUAD):
            self.assertLessEqual(check_self_collision(text), 0.005)
            m = mujoco.MjModel.from_xml_string(compile_design(text))
            self.assertLessEqual(self_penetration(m)[0], 0.005)

    def test_robot_parts_collide_in_simulation(self):
        m = mujoco.MjModel.from_xml_string(compile_design(QUAD))
        robot = m.geom_bodyid > 0
        self.assertTrue((m.geom_conaffinity[robot] == 1).all())
        m0 = mujoco.MjModel.from_xml_string(compile_design(QUAD, self_collision=False))
        self.assertTrue((m0.geom_conaffinity[m0.geom_bodyid > 0] == 0).all())

    def test_penetration_check_leaves_model_unchanged(self):
        m = mujoco.MjModel.from_xml_string(compile_design(QUAD))
        before = (m.geom_contype.copy(), m.geom_conaffinity.copy(), m.body_contype.copy())
        self_penetration(m)
        for x, y in zip(before, (m.geom_contype, m.geom_conaffinity, m.body_contype)):
            self.assertTrue((x == y).all())

    def test_generator_and_mutation_avoid_self_collision(self):
        rng = random.Random(7)
        for _ in range(15):
            d = random_design(rng, max_modules=6)
            compile_design(d, max_modules=6)              # raises if self-colliding
            d2, _ = mutate(d, rng, max_modules=6)
            compile_design(d2, max_modules=6)


class ScreenTests(unittest.TestCase):
    def test_quad_passes(self):
        r = screen(QUAD)
        self.assertTrue(r.passed, r.detail)
        self.assertGreater(r.metrics["torso_gap_max_m"], 0.005)

    def test_torso_lowest_is_screened(self):
        # a single module lying on its block: the torso is the lowest part in every reset
        r = screen("h0=root; m1=h0.yp@0")
        self.assertFalse(r.passed)
        self.assertEqual(r.reason, "torso_ground")

    def test_invalid_design_keeps_kind(self):
        r = screen(FOLDED)
        self.assertEqual(r.reason, "self_collision")
        r = screen("m0=root; m1=m0.P2@45")
        self.assertEqual(r.reason, "roll")

    def test_postcheck_flags_fast_episodes(self):
        res = {"fitness": 100.0, "rewards": [[1.0, 2.0], [0.5, 7.5]]}
        pc = postcheck(res, ScreenConfig(score_bound=50.0))
        self.assertEqual(pc["fast_episodes"], 1)
        self.assertTrue(pc["over_bound"] and pc["suspect"])
        self.assertFalse(postcheck({"fitness": 1.0, "rewards": [[1.0]]})["suspect"])


if __name__ == "__main__":
    unittest.main()
