"""Offline tests for the CPU MPPI evaluator (mpc/) and its hook in the evolution loop."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from robomorph.evolution import loop
from robomorph.main import build_parser

ANT = Path(__file__).resolve().parents[1] / "mpc" / "ant_test.xml"
RESPONSE = ('Step 1: r0) H-U-(C-M-J-L-E)-T\n'
            '[null]-[body=15cm]-([connector]-[mount]-[knee=30deg]-[limb=10cm]-[null])-[null]')


class MPCTests(unittest.TestCase):
    def test_parser_accepts_mpc_evaluator(self):
        args = build_parser().parse_args(["--exp_name", "x", "--evaluator", "mpc", "--mpc_episodes", "2"])
        self.assertEqual((args.evaluator, args.mpc_episodes, args.mpc_threads), ("mpc", 2, 8))
        self.assertEqual(build_parser().parse_args(["--exp_name", "x"]).evaluator, "sac")

    def test_loop_dispatches_to_mpc_without_sac(self):
        import mpc.evaluate
        with tempfile.TemporaryDirectory() as temporary:
            population = loop.create_population(1, "Design a robot")
            with patch.object(loop, "generate_grammar", return_value=RESPONSE), \
                    patch.object(loop, "train_multiple_models") as train, \
                    patch.object(mpc.evaluate, "evaluate_multiple_designs",
                                 side_effect=lambda xmls, **kw: [(xmls[0], 7.0)]) as ev:
                loop._evaluate_fitness("test", 0, population, Mock(), "flat", "unused",
                                       str(Path(temporary) / "results.csv"), evaluator="mpc")
            train.assert_not_called()
            ev.assert_called_once()
            self.assertEqual(population.elites[0].fitness, 7.0)

    def test_unknown_evaluator_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            population = loop.create_population(1, "Design a robot")
            with patch.object(loop, "generate_grammar", return_value=RESPONSE):
                with self.assertRaisesRegex(ValueError, "Unknown evaluator"):
                    loop._evaluate_fitness("test", 0, population, Mock(), "flat", "unused",
                                           str(Path(temporary) / "results.csv"), evaluator="nope")

    def test_failed_design_gets_minus_inf_and_failure_file(self):
        from mpc.evaluate import evaluate_multiple_designs
        with tempfile.TemporaryDirectory() as temporary:
            bad = Path(temporary) / "bad.xml"
            bad.write_text("<mujoco><worldbody/></mujoco>")   # no actuators -> ValueError
            [(path, fitness)] = evaluate_multiple_designs([str(bad)], output_dir=temporary, threads=1)
            self.assertEqual(fitness, float("-inf"))
            self.assertIn("MPC evaluation failed", json.loads((Path(temporary) / "failure_bad.json").read_text())["error"])

    def test_short_mppi_episode_on_ant(self):
        from mpc.mppi import MPPIConfig, evaluate_design
        cfg = MPPIConfig(threads=2, episode_control_steps=5, warmup_samples=64)
        r = evaluate_design(ANT.read_text(), "flat", 1, 0, cfg)
        self.assertTrue(math.isfinite(r["fitness"]))
        self.assertEqual(r["episodes"][0]["env_steps"], 20)
        self.assertFalse(r["episodes"][0]["unstable"])


    def test_speed_guard_default_is_10(self):
        from mpc.mppi import MPPIConfig
        self.assertEqual(MPPIConfig().max_speed, 10.0)

    def test_speed_guard_ends_episode(self):
        from mpc.mppi import MPPIConfig, evaluate_design
        cfg = MPPIConfig(threads=2, episode_control_steps=5, warmup_samples=64, max_speed=1e-6)
        ep = evaluate_design(ANT.read_text(), "flat", 1, 0, cfg)["episodes"][0]
        self.assertTrue(ep["unstable"])
        self.assertLess(ep["env_steps"], 20)

    def test_solver_iterations_raised_to_20(self):
        from mpc.mppi import MPPIConfig, load_model
        self.assertEqual(MPPIConfig().min_solver_iterations, 20)
        self.assertGreaterEqual(load_model(ANT.read_text(), min_iterations=20).opt.iterations, 20)
        self.assertEqual(load_model(ANT.read_text(), mjx_solver=True, min_iterations=20).opt.iterations, 1)

if __name__ == "__main__":
    unittest.main()
