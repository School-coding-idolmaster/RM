"""Offline tests for the Hyblock domain (hyblock/, v5: 水力君 two-joint module + passive hub + passive foot).

Covers the static model (torque limits, dU/dθ, hand calculations), the design text format and mirror
expansion, the MJCF generator (gravity term vs. static model = pipeline 3.5 item 1, masses, actuators,
terrain friction/scale) and the Hyblock hooks in mpc/mppi.py.
"""
import math
import random
import unittest

import mujoco
import numpy as np

from hyblock.design import DesignError, build_robot, expand, parse, to_text
from hyblock.generator import mutate, random_design
from hyblock.mjcf import GROUND_X, compile_design, joint_qpos_index, lowest_point
from hyblock.model import G, Node, Robot, load_spec

QUAD = ("h0=root; m1=h0.zp@90 q90/150 mirror; m2=h0.xm@0 q90; h3=m2.P2@0; m4=h3.zp@90 q90/150 mirror; "
        "foot=m1.P2; foot=m4.P2")


class StaticModelTests(unittest.TestCase):
    def test_torque_limits_and_masses(self):
        for rmin in (-30.0, 30.0):
            sp = load_spec(range_min_deg=rmin)
            A_ext = math.pi * 0.025 ** 2 / 4
            A_ret = A_ext - math.pi * 0.0125 ** 2 / 4
            lp, ln = sp.tau_limits(8e6)
            self.assertAlmostEqual(lp, 8e6 * A_ext * 0.0423, places=6)
            self.assertAlmostEqual(ln, 8e6 * A_ret * 0.0423, places=6)
            self.assertLess(abs(lp - 166.0), 1.0)
            self.assertLess(abs(lp * 21 / 8 - 436.0), 3.0)
            self.assertAlmostEqual(math.degrees(sp.range_max - sp.range_min), 180.0 if rmin < 0 else 120.0)
        sp = load_spec()
        self.assertAlmostEqual(sum(b.mass for b in sp.parts["module"].bodies.values()), 8.0)
        self.assertEqual(len(sp.parts["module"].joints), 2)
        self.assertEqual(sorted(sp.parts["module"].ports), ["P1", "P2"])
        self.assertEqual(len(sp.parts["hub"].ports), 6)

    def test_single_module_hand_calc(self):
        """Root module lying flat. θ = 0: plates horizontal, COM and foot 0.042 m from the hinge."""
        sp = load_spec()
        r = Robot(sp, [Node("module", -1, q0=(0.0, 0.0))], feet=[(0, "P1")])
        tau = r.static_torques([0.0, 0.0])
        plate = sp.parts["module"].bodies["P1"]
        self.assertAlmostEqual(abs(tau[0]), (plate.mass + sp.foot_mass) * 0.042 * G, places=9)
        self.assertAlmostEqual(abs(tau[1]), plate.mass * 0.042 * G, places=9)
        tau90 = r.static_torques([math.pi / 2] * 2)      # plates vertical: COM 0.0125 m outboard
        self.assertAlmostEqual(abs(tau90[1]), plate.mass * 0.0125 * G, places=9)

    def test_torque_is_gradient_of_potential(self):
        rng = random.Random(1)
        worst, tested = 0.0, 0
        for k in range(60):
            robot, _ = build_robot(random_design(rng, max_modules=8), max_modules=8,
                                   mass_scale=rng.choice([0.75, 1.0, 1.5]))
            q = np.random.default_rng(k).uniform(-0.4, 2.5, robot.n_joints)
            tau = robot.static_torques(q)
            for i in range(robot.n_joints):
                qp, qm = q.copy(), q.copy()
                qp[i] += 1e-6
                qm[i] -= 1e-6
                worst = max(worst, abs(tau[i] - (robot.potential_energy(qp) - robot.potential_energy(qm)) / 2e-6))
            tested += robot.n_joints
        self.assertGreater(tested, 100)
        self.assertLess(worst, 1e-5)

    def test_validation_rules(self):
        sp = load_spec()

        def m(parent, port="P2", **k):
            return Node("module", parent, port, q0=(1.0, 1.0), **k)

        def raises(nodes, **kw):
            with self.assertRaises(ValueError):
                Robot(sp, nodes, **kw)
        raises([m(-1, ""), m(0), m(0)])                                   # same port twice
        raises([Node("hub", -1), m(0, "xm"), m(1, "P1")])                 # child's own P1 is taken
        raises([m(-1, ""), m(0, roll_deg=45)])
        raises([m(-1, "")] + [m(i) for i in range(8)], max_modules=8)     # 9 modules
        raises([Node("hub", -1)] + [Node("hub", 0, p) for p in ("xp", "zp", "zm", "yp", "ym")], max_hubs=4)
        raises([m(-1, ""), m(0)], feet=[(0, "P2")])                       # foot on a used port
        raises([m(-1, ""), m(0)], feet=[(1, "P1")])                       # foot on the child's attach port
        raises([Node("module", -1, q0=(1.0,))])                           # wrong number of joint angles
        Robot(sp, [m(-1, ""), m(0, "P1"), m(0, "P2")], feet=[(1, "P2")])  # root module: both plates usable


class DesignTextTests(unittest.TestCase):
    def test_round_trip_and_mirror(self):
        text = "h0=root; m1=h0.zp@90 q60/120 mirror; m2=m1.P2@0 x2 q120; h3=m2.P2@0; foot=h3.zp"
        d = parse(text)
        self.assertEqual(to_text(d), text)
        self.assertEqual(to_text(parse("DESIGN: " + text.replace("zp", "ZP"))), text)
        ex = expand(d)
        self.assertEqual(ex.labels, ["h0", "m1", "m2", "h3", "m1'", "m2'", "h3'"])
        self.assertEqual(ex.nodes[4].parent_port, "zm")
        self.assertEqual(ex.nodes[4].roll_deg, 270)                       # mirrored roll
        self.assertEqual(ex.module_count(), 2 * (1 + 2))
        self.assertEqual(ex.hub_count(), 3)
        self.assertEqual(ex.feet, [(3, "zp"), (6, "zm")])                 # mirrored foot port

    def test_mirror_is_a_reflection(self):
        """With equal angles on twin joints the robot is mirror-symmetric about the torso's mid-plane."""
        text = ("h0=root; m1=h0.zp@90 q60/120 mirror; m2=m1.P2@180 q30/150; h3=m2.P2@270; "
                "m4=h3.zm@90 mirror; foot=m4.P2; m5=h0.xp@0 q100/40; foot=m5.P2")
        robot, ex = build_robot(text, max_modules=20, ground_x=GROUND_X)
        m = mujoco.MjModel.from_xml_string(compile_design(text, max_modules=20))
        d = mujoco.MjData(m)
        rng = np.random.default_rng(1)
        base, q = {}, []
        for i, js in robot.joint_index:
            q.append(base.setdefault((ex.labels[i].replace("'", ""), js.name), rng.uniform(-0.4, 2.5)))
        d.qpos[joint_qpos_index(m, robot)] = q
        mujoco.mj_forward(m, d)
        normal = d.xmat[1].reshape(3, 3)[:, 2]                            # torso z axis = mid-plane normal
        c = d.xpos[1]
        pts = d.geom_xpos[m.geom_bodyid > 0]
        refl = pts - 2 * ((pts - c) @ normal)[:, None] * normal[None, :]
        dist = np.min(np.linalg.norm(pts[:, None, :] - refl[None, :, :], axis=-1), axis=1)
        self.assertLess(dist.max(), 1e-9)

    def test_invalid_designs_have_kinds(self):
        cases = {
            "m0=root; m1=m0.P2@45": "roll",
            "m0=root; m1=m0.P2@0; m2=m0.P2@0": "port_conflict",
            "h0=root; m1=h0.xp@0 mirror": "mirror",
            "m0=root; m1=m0.P2@0 mirror": "mirror",
            "m0=root; m1=m9.P2@0": "unknown_parent",
            "m0=root; m1=m0.P2@0 q170": "joint_range",
            "m0=root; foot=m0.P2; m1=m0.P2@0": "foot",
            "m0=root; m1=m0.P2@0; foot=m1.P1": "foot",
            "m0=root; m1=m0.zp@0": "unknown_port",
            "h0=root; h1=h0.zp@0 x2": "parallel",
            "hello": "syntax",
            "m0=root; m1=m0.P2@0 q10/20/30": "syntax",
        }
        for text, kind in cases.items():
            with self.assertRaises(DesignError, msg=text) as cm:
                build_robot(text)
            self.assertEqual(cm.exception.kind, kind, text)
        with self.assertRaises(DesignError) as cm:
            build_robot("m0=root; " + "; ".join(f"m{i + 1}=m{i}.P2@0" for i in range(8)), max_modules=8)
        self.assertEqual(cm.exception.kind, "module_cap")
        with self.assertRaises(DesignError) as cm:
            build_robot("h0=root; " + "; ".join(f"h{i + 1}=h0.{p}@0" for i, p in enumerate(("xp", "xm", "zp", "zm", "yp"))))
        self.assertEqual(cm.exception.kind, "hub_cap")

    def test_generator_and_mutation_give_valid_designs(self):
        rng = random.Random(0)
        for _ in range(40):
            d = random_design(rng, max_modules=6)
            _, ex = build_robot(d, max_modules=6)
            self.assertLessEqual(ex.module_count(), 6)
            self.assertEqual(to_text(parse(to_text(d))), to_text(d))
            d2, _ = mutate(d, rng, max_modules=6)
            build_robot(d2, max_modules=6)


class MJCFTests(unittest.TestCase):
    def test_gravity_term_matches_static_model(self):
        rng = random.Random(3)
        worst = 0.0
        for k in range(20):
            d = random_design(rng, max_modules=8)
            m = mujoco.MjModel.from_xml_string(compile_design(d, fixed_base=True, max_modules=8))
            data = mujoco.MjData(m)
            robot, _ = build_robot(d, ground_x=GROUND_X, max_modules=8)
            idx = joint_qpos_index(m, robot)
            for t in range(3):
                q = np.random.default_rng(10 * k + t).uniform(-0.5, 2.6, len(idx))
                data.qpos[idx] = q
                mujoco.mj_forward(m, data)
                worst = max(worst, np.abs(data.qfrc_bias[idx] - robot.static_torques(q)).max())
        self.assertLess(worst, 1e-7)

    def test_mass_actuators_and_custom_fields(self):
        text = "m0=root q60/120; m1=m0.P2@0 x2; foot=m1.P2"
        m = mujoco.MjModel.from_xml_string(compile_design(text))
        robot, _ = build_robot(text)
        self.assertAlmostEqual(m.body_mass.sum(), robot.total_mass(), places=9)
        self.assertEqual(m.nu, 4)
        lim = np.array(sorted(map(tuple, m.actuator_ctrlrange)))
        np.testing.assert_allclose(lim, [[-124.6 * 1.6, 166.1 * 1.6]] * 2 + [[-124.6, 166.1]] * 2, atol=0.2)
        np.testing.assert_allclose(m.actuator_forcerange, m.actuator_ctrlrange)
        self.assertTrue(np.all(m.actuator_dyntype == mujoco.mjtDyn.mjDYN_FILTEREXACT))
        self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_NUMERIC, "hyblock_torso_box"), 0)
        self.assertEqual(m.jnt_type[0], mujoco.mjtJoint.mjJNT_FREE)
        self.assertEqual(m.jnt_bodyid[0], 1)
        self.assertEqual(m.opt.iterations, 20)
        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        self.assertAlmostEqual(lowest_point(m, data), 0.02, places=9)
        np.testing.assert_allclose(np.degrees(data.qpos[joint_qpos_index(m, robot)]), [60, 120, 90, 90])

    def test_floor_priority_sets_ice_friction(self):
        from mpc.mppi import load_model
        m = load_model(compile_design(QUAD), "frozen_lake")
        d = mujoco.MjData(m)
        for _ in range(300):
            mujoco.mj_step(m, d)
        self.assertGreater(d.ncon, 0)
        self.assertTrue(np.allclose(d.contact.friction[:d.ncon, 0], 0.05))

    def test_terrain_scale(self):
        from mpc.mppi import load_model
        m = load_model(compile_design(QUAD), "beams", terrain_scale=2.0)
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "beam_0")
        np.testing.assert_allclose(m.geom_pos[g], [-2.0, 0.0, 1.0])
        self.assertAlmostEqual(m.geom_size[g][0], 0.4)
        self.assertEqual(m.geom_priority[g], 1)


class HyblockMPPITests(unittest.TestCase):
    def test_zero_centred_action_map_and_termination(self):
        from mpc.mppi import MPPIConfig, MPPIEvaluator, load_model
        ev = MPPIEvaluator(load_model(compile_design(QUAD)), MPPIConfig(threads=1))
        self.assertTrue(ev.hyblock)
        nu = ev.nu
        np.testing.assert_allclose(ev.scale(np.zeros(nu)), 0.0)
        np.testing.assert_allclose(ev.scale(np.ones(nu)), ev.ctrl_hi)
        np.testing.assert_allclose(ev.scale(-np.ones(nu)), ev.ctrl_lo)
        np.testing.assert_allclose(ev.scale(np.full(nu, -0.5)), 0.5 * ev.ctrl_lo)
        quat = ev.m.body_quat[1].copy()
        upright = np.concatenate([[0, 0, 0.5], quat])
        self.assertFalse(ev.unhealthy(upright))
        low = upright.copy()
        low[2] = 0.06                                     # hub half size 0.06: bottom face on the ground
        self.assertTrue(ev.unhealthy(low))
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, np.array([0, 1.0, 0, 0]), quat)   # 180° about world x
        self.assertTrue(ev.unhealthy(np.concatenate([[0, 0, 0.5], q])))
        ev.close()

    def test_short_episode(self):
        from mpc.mppi import MPPIConfig, evaluate_design
        r = evaluate_design(compile_design(QUAD), "flat", 1, 0,
                            MPPIConfig(threads=2, episode_control_steps=5, warmup_samples=64))
        self.assertTrue(math.isfinite(r["fitness"]))
        self.assertEqual(r["episodes"][0]["env_steps"], 20)
        self.assertFalse(r["episodes"][0]["unstable"])
        self.assertIn("max_distance", r)


if __name__ == "__main__":
    unittest.main()


class HyblockLoopTests(unittest.TestCase):
    RESPONSE = ("STRUCTURAL RULES:\nStep 1: r0) h0=root\nStep 2: r3) h0=root; m1=h0.zp@90\n"
                "Step 3: r4) h0=root; m1=h0.zp@90 mirror\nStep 4: 2x r6) h0=root; m1=h0.zp@90 mirror; foot=m1.P2\n\n"
                "**DESIGN:** `h0=root; m1=h0.zp@90 q60/120 mirror; foot=m1.P2`\n")

    def test_extraction(self):
        from hyblock import llm_io
        self.assertEqual(llm_io.extract_design(self.RESPONSE), "h0=root; m1=h0.zp@90 q60/120 mirror; foot=m1.P2")
        self.assertEqual(llm_io.extract_rules(self.RESPONSE), "r0 r3 r4 r6 r6")
        self.assertIn("At most 6 modules", llm_io.system_prompt(6))
        self.assertGreater(llm_io.diversity(llm_io.extract_design(self.RESPONSE)), 0)

    def test_parser_and_loop_dispatch(self):
        import tempfile
        from pathlib import Path
        from unittest.mock import Mock, patch
        import mpc.evaluate
        from robomorph.evolution import loop
        from robomorph.main import build_parser
        a = build_parser().parse_args(["--exp_name", "x", "--evaluator", "mpc", "--domain", "hyblock",
                                       "--max_modules", "4", "--terrain_scale", "2"])
        self.assertEqual((a.domain, a.max_modules, a.terrain_scale), ("hyblock", 4, 2.0))
        self.assertEqual(build_parser().parse_args(["--exp_name", "x"]).domain, "rm")
        self.assertIsNone(a.mpc_episodes)                       # resolved in main() by domain
        from robomorph.main import default_mpc_episodes
        self.assertEqual((default_mpc_episodes("hyblock"), default_mpc_episodes("rm")), (2, 3))
        bad = self.RESPONSE.replace("q60/120", "q60/170")
        with tempfile.TemporaryDirectory() as tmp:
            population = loop.create_population(2, "Design a robot.")
            with patch.object(loop, "generate_grammar", side_effect=[self.RESPONSE, bad]), \
                    patch.object(loop, "train_multiple_models") as train, \
                    patch.object(mpc.evaluate, "evaluate_multiple_designs",
                                 side_effect=lambda xmls, **kw: [(x, 5.0) for x in xmls]) as ev:
                loop._evaluate_fitness("t", 0, population, Mock(), "flat", "unused", str(Path(tmp) / "r.csv"),
                                       evaluator="mpc", domain="hyblock", max_modules=6, terrain_scale=2.0)
            train.assert_not_called()
            self.assertEqual(ev.call_args.kwargs["terrain_scale"], 2.0)
            self.assertEqual(len(ev.call_args.args[0]), 1)
            u0, u1 = population.units
            self.assertEqual(u0.fitness, 5.0)
            self.assertEqual(u0.structure, "r0 r3 r4 r6 r6")
            self.assertIn("joint_range", u1.error)
            prompt = (Path(tmp) / "examples_t" / "grammar_prompt.txt").read_text()
            self.assertIn("Hyblock", prompt)
