"""第 12 周：地形版提示词（平地逐字节不变）与第 0 代来源 --init（假 LLM）。"""
import csv
import hashlib
import random
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hyblock import init_pop, llm_mutation, search, selection
from hyblock.design import parse, to_text
from hyblock.generator import mutate, random_design
from hyblock.terrain_text import adapt

FAST = ["--steps", "2", "--threads", "1", "--episodes", "1"]


def rows(p):
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def flat_evals(path, n=20, seed=3):
    rng = random.Random(seed)
    fields = ["gen", "idx", "status", "design", "fitness"]
    seen, out = set(), []
    while len(out) < n:
        t = to_text(random_design(rng))
        if t in seen:
            continue
        seen.add(t)
        out.append({"gen": 0, "idx": len(out), "status": "mpc", "design": t, "fitness": 100 + len(out)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    return out


class TerrainTextTests(unittest.TestCase):
    def test_flat_unchanged(self):
        for typed in (False, True):
            t = llm_mutation.system_prompt(8, typed)
            self.assertEqual(t, llm_mutation.system_prompt(8, typed, "flat"))
            self.assertIn("flat ground", t)
            self.assertNotIn("### TERRAIN ###", t)
        self.assertEqual(selection.system_prompt(8), selection.system_prompt(8, "flat"))

    def test_flat_sha_matches_week11_runs(self):
        root = Path(__file__).resolve().parents[2]
        cand = [root / p / "base_flat8/ga_llmtyped_sel_llmstrat5v_s0/run_meta.json" for p in ("from_mac/mac_runs", "mac_runs")]
        meta = next((c for c in cand if c.exists()), cand[0])
        if not meta.exists():
            self.skipTest("week-11 run not here (Mac)")
        import json
        m = json.loads(meta.read_text(encoding="utf-8"))
        self.assertEqual(m["mutation_prompt_sha256"], llm_mutation.prompt_sha256(8, True))
        self.assertEqual(m["select_prompt_sha256"], selection.prompt_sha256(8))

    def test_beams_prompts(self):
        for t in (llm_mutation.system_prompt(8, True, "beams"), llm_mutation.system_prompt(8, False, "beams"),
                  selection.system_prompt(8, "beams"), init_pop.system_prompt("init", "beams", 8),
                  init_pop.system_prompt("transfer", "beams", 8)):
            self.assertIn("### TERRAIN ###", t)
            self.assertIn("0.3 m", t)
            self.assertNotIn("{name}", t)
            self.assertNotIn("{n}", t)
            self.assertEqual(t.count("### TERRAIN ###"), 1)
        self.assertNotIn("flat ground for", llm_mutation.system_prompt(8, True, "beams"))
        with self.assertRaises(ValueError):
            adapt("nothing to replace\n### TASK ###", "beams")
        with self.assertRaises(ValueError):
            adapt("walk fast on flat ground ### TASK ###", "ridged")


class InitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = flat_evals(self.tmp / "flat.csv")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_transfer_takes_top_distinct(self):
        out = self.tmp / "tr"
        search.main(["--method", "ga", "--init", "transfer", "--init_source", str(self.tmp / "flat.csv"),
                     "--terrain", "beams", "--budget", "4", "--batch", "4", "--out", str(out)] + FAST)
        r = [x for x in rows(out / "evals.csv") if x["gen"] == "0"]
        want = [to_text(parse(s["design"])) for s in sorted(self.src, key=lambda s: -s["fitness"])[:4]]
        self.assertEqual([x["design"] for x in r], want)
        self.assertTrue(all(x["source"] == "transfer" for x in r))
        self.assertTrue((out / "init.csv").exists())

    def test_llm_init_retry_fill_and_replay(self):
        rng = random.Random(7)
        good = [to_text(random_design(rng)) for _ in range(3)]
        calls = []

        def fake(system, user, model):
            calls.append(user)
            self.assertIn("### TERRAIN ###", system)
            if len(calls) == 1:    # 1 ok, 2 invalid, 3 duplicate of 1, 4 missing
                txt = f"IDEA 1: a\nDESIGN 1: {good[0]}\nDESIGN 2: m0=root; m1=m0.P9@0\nDESIGN 3: {good[0]}\n"
            else:                  # retry fixes 2 and 3, not 4
                self.assertIn("could not be used", user)
                txt = f"DESIGN 2: {good[1]}\nDESIGN 3: {good[2]}\nDESIGN 4: m0=root; m1=m0.P9@0\n"
            return txt, {"input_tokens": 10, "output_tokens": 5}

        out = self.tmp / "li"
        args = ["--method", "ga", "--init", "llm", "--terrain", "beams", "--budget", "4", "--batch", "4",
                "--llm_workers", "1"]
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x"}), \
                mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=fake):
            search.main(args + ["--out", str(out)] + FAST)
        self.assertEqual(len(calls), 2)
        r = [x for x in rows(out / "evals.csv") if x["gen"] == "0"]
        self.assertEqual([x["source"] for x in r], ["llm_init"] * 3 + ["llm_init_fail>random"])
        self.assertEqual([x["design"] for x in r[:3]], good)
        out2 = self.tmp / "li2"            # replay from the cache: same generation 0, no API
        with mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=AssertionError("no API")):
            search.main(args + ["--replay", "--llm_cache", str(out / "llm_cache"), "--out", str(out2)] + FAST)
        self.assertEqual([x["design"] for x in rows(out2 / "evals.csv") if x["gen"] == "0"], [x["design"] for x in r])

    def test_llm_transfer(self):
        top = [to_text(parse(s["design"])) for s in sorted(self.src, key=lambda s: -s["fitness"])[:4]]
        rng = random.Random(1)

        def fake(system, user, model):
            self.assertIn("level ground", user)
            k = [i for i, t in enumerate(top) if t in user][0]
            if "could not be used" in user:
                return "DESIGN: m0=root; m1=m0.P9@0", {}           # still invalid -> original
            if k == 0:
                return f"CHANGE: none\nDESIGN: {top[0]}", {}
            if k == 1:
                return "DESIGN: m0=root; m1=m0.P9@0", {}
            if k == 2:
                return f"DESIGN: {top[3]}", {}                      # duplicate of another source design
            while True:
                c, _ = mutate(parse(top[3]), rng, max_modules=8)
                t = to_text(c)
                if t not in top:
                    return f"CHANGE: x\nDESIGN: {t}", {}

        out = self.tmp / "lt"
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x"}), \
                mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=fake):
            search.main(["--method", "ga", "--init", "llm_transfer", "--init_source", str(self.tmp / "flat.csv"),
                         "--terrain", "beams", "--budget", "4", "--batch", "4", "--llm_workers", "1",
                         "--out", str(out)] + FAST)
        r = [x for x in rows(out / "evals.csv") if x["gen"] == "0"]
        self.assertEqual([x["source"] for x in r], ["llm_transfer_none", "llm_transfer_fail>transfer",
                                                    "llm_transfer_fail>transfer", "llm_transfer"])
        self.assertEqual([x["design"] for x in r[:3]], top[:3])
        self.assertNotEqual(r[3]["design"], top[3])

    def test_guards(self):
        with self.assertRaises(SystemExit):
            search.main(["--method", "ga", "--init", "transfer", "--out", str(self.tmp / "g")] + FAST)
        with self.assertRaises(SystemExit):
            search.main(["--method", "random", "--init", "llm", "--out", str(self.tmp / "g2")] + FAST)


class BeamsLLMSearchTests(unittest.TestCase):
    """横梁上 ② 挑选与 LLM 变异的系统提示都带地形段（第 13 周 B2 / B5 / B6 用到的路径，假 LLM）。"""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_select_and_mutation_get_terrain_prompt(self):
        from tests.test_selection import feet_ranking
        seen = {"sel": 0, "mut": 0}

        def fake(system, user, model, retries=5):
            self.assertIn("### TERRAIN ###", system)
            self.assertNotIn("flat ground for", system)
            if "surrogate model" in system:
                seen["sel"] += 1
                return "ANALYSIS: x\nRANKING: " + ", ".join(feet_ranking(user)), {"input_tokens": 1, "output_tokens": 1}
            seen["mut"] += 1
            return "ANALYSIS: x\nCHANGE: none\nDESIGN: m0=root; m1=m0.P9@0", {"input_tokens": 1, "output_tokens": 1}

        out = self.tmp / "b6"
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x"}), \
                mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=fake):
            search.main(["--method", "ga", "--terrain", "beams", "--mutation", "llm_typed", "--select", "llm",
                         "--budget", "8", "--batch", "4", "--rand_slots", "1", "--sel_pool", "4", "--llm_workers", "1",
                         "--out", str(out)] + FAST)
        self.assertGreater(seen["sel"], 0)
        self.assertGreater(seen["mut"], 0)
        import json
        m = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
        self.assertEqual(m["select_prompt_sha256"], selection.prompt_sha256(8, "beams"))
        self.assertEqual(m["mutation_prompt_sha256"], llm_mutation.prompt_sha256(8, True, "beams"))


if __name__ == "__main__":
    unittest.main()
