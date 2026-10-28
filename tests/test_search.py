"""Hyblock 搜索驱动（第 8 周）：随机搜索 / 进化算法 / RM 式 LLM（假 LLM）、预算、续跑、缓存重放、交叉。"""
import csv
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hyblock import search
from hyblock.design import to_text
from hyblock.generator import crossover, random_design
from hyblock.mjcf import compile_design

FAST = ["--steps", "2", "--threads", "1", "--episodes", "1"]
EXAMPLE = "h0=root; m1=h0.zp@90 q60/120 mirror; h2=h0.xm@0; m3=h2.zp@0 q90/150 mirror; foot=m1.P2; foot=m3.P2"


def rows(out):
    with open(Path(out) / "evals.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_crossover_children_are_valid(self):
        rng = random.Random(5)
        made = 0
        for _ in range(10):
            c = crossover(random_design(rng), random_design(rng), rng)
            if c is not None:
                compile_design(c)
                made += 1
        self.assertGreater(made, 5)

    def test_ga_budget_resume_and_determinism(self):
        out = self.tmp / "ga"
        search.main(["--method", "ga", "--budget", "6", "--batch", "4", "--out", str(out)] + FAST)
        first = rows(out)
        self.assertEqual(sum(r["status"] == "mpc" for r in first), 6)
        self.assertTrue((out / "DONE").exists())
        search.main(["--method", "ga", "--budget", "9", "--batch", "4", "--out", str(out)] + FAST)   # resume
        resumed = rows(out)
        self.assertEqual([r["design"] for r in resumed[:len(first)]], [r["design"] for r in first])
        fresh = self.tmp / "ga2"
        search.main(["--method", "ga", "--budget", "9", "--batch", "4", "--out", str(fresh)] + FAST)
        self.assertEqual([r["design"] for r in rows(fresh)], [r["design"] for r in resumed])
        self.assertIn("mutate", rows(fresh)[-1]["source"] + rows(fresh)[-2]["source"] + rows(fresh)[-3]["source"])

    def test_random_and_ga_share_generation_zero(self):
        a, b = self.tmp / "r", self.tmp / "g"
        search.main(["--method", "random", "--budget", "3", "--batch", "3", "--out", str(a)] + FAST)
        search.main(["--method", "ga", "--budget", "3", "--batch", "3", "--out", str(b)] + FAST)
        self.assertEqual([r["design"] for r in rows(a)], [r["design"] for r in rows(b)])

    def test_llm_statuses_cache_and_replay(self):
        replies = {0: f"STRUCTURAL RULES:\nStep 1: r0) h0=root\n\nDESIGN: {EXAMPLE}",
                   1: f"DESIGN: {EXAMPLE}",                                    # duplicate of 0
                   2: "DESIGN: m0=root; m1=m0.P2@0 q150; m2=m1.P2@0 q150; m3=m2.P2@0 q150",  # self_collision
                   3: "I cannot design that."}                                 # no DESIGN line
        calls = []

        def fake(system, user, model, retries=5):
            calls.append(user)
            idx = len(calls) - 1
            return replies.get(idx % 4, replies[0]), {"input_tokens": 10, "output_tokens": 20}

        out = self.tmp / "llm"
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x"}), \
                mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=fake):
            search.main(["--method", "llm", "--budget", "2", "--batch", "4", "--llm_workers", "1",
                         "--out", str(out)] + FAST)
        st = [r["status"] for r in rows(out)]
        self.assertEqual(st[:4], ["mpc", "duplicate", "invalid", "no_design"])
        self.assertEqual(rows(out)[2]["kind"], "self_collision")
        self.assertIn("Example 0:", calls[-1])                              # gen 1 prompt shows the best design
        self.assertTrue(any((out / "llm_cache").glob("*.json")))
        # replay into a fresh directory with the same cache: no API calls, same designs
        out2 = self.tmp / "llm2"
        with mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=AssertionError("no API")):
            search.main(["--method", "llm", "--budget", "2", "--batch", "4", "--llm_workers", "1", "--replay",
                         "--llm_cache", str(out / "llm_cache"), "--out", str(out2)] + FAST)
        self.assertEqual([r["design"] for r in rows(out2)], [r["design"] for r in rows(out)])


if __name__ == "__main__":
    unittest.main()
