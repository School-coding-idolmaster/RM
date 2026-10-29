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
            search.main(["--method", "llm", "--budget", "3", "--batch", "4", "--llm_workers", "1",
                         "--out", str(out)] + FAST)
        st = [r["status"] for r in rows(out)]
        self.assertEqual(st[:4], ["mpc", "duplicate", "invalid", "no_design"])
        self.assertEqual(st.count("mpc") + st.count("duplicate"), 3)          # duplicates use budget
        self.assertTrue((out / "DONE").exists())
        self.assertEqual(rows(out)[2]["kind"], "self_collision")
        self.assertIn("Example 0:", calls[-1])                              # gen 1 prompt shows the best design
        self.assertTrue(any((out / "llm_cache").glob("*.json")))
        # replay into a fresh directory with the same cache: no API calls, same designs
        out2 = self.tmp / "llm2"
        with mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=AssertionError("no API")):
            search.main(["--method", "llm", "--budget", "3", "--batch", "4", "--llm_workers", "1", "--replay",
                         "--llm_cache", str(out / "llm_cache"), "--out", str(out2)] + FAST)
        self.assertEqual([r["design"] for r in rows(out2)], [r["design"] for r in rows(out)])

    def test_llm_no_scores_prompt(self):
        """--no_scores（RM 论文 V3 的最小忠实版）：示例不写得分；第 0 代提示词与公开代码版相同（可共用缓存配对）。"""
        calls = []

        def fake(system, user, model, retries=5):
            calls.append(user)
            d = to_text(random_design(random.Random(len(calls))))
            return f"DESIGN: {d}", {"input_tokens": 10, "output_tokens": 20}

        out = self.tmp / "noscore"
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x"}), \
                mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=fake):
            search.main(["--method", "llm", "--no_scores", "--budget", "5", "--batch", "4", "--llm_workers", "1",
                         "--out", str(out)] + FAST)
        self.assertIn("Example 0:", calls[-1])
        self.assertIn("Here are some examples of designs. How can you design a better robot?", calls[-1])
        self.assertNotIn("itness", calls[-1])
        self.assertNotIn("Example", calls[0])                 # gen 0: same prompt as the public-code version
        a1 = search.build_parser().parse_args(["--method", "llm", "--out", str(out), "--problem", "Design a robot."])
        a2 = search.build_parser().parse_args(["--method", "llm", "--out", str(out), "--problem", "Design a robot.",
                                               "--no_scores"])
        l1, l2 = search.LLM(a1, out), search.LLM(a2, out)
        empty = search.Log(self.tmp / "empty_dir_does_not_exist")
        self.assertEqual(l1._key(l1.user_prompt(empty, 0), 0, 0, l1.system),
                         l2._key(l2.user_prompt(empty, 0), 0, 0, l2.system))

    def test_ga_llm_mutation(self):
        """GA + LLM 变异：通过 / 与输入相同 → 重问 / 自碰撞两次 → 退回随机变异 / 没有 DESIGN 行；缓存重放。"""
        import re
        from hyblock.design import parse
        from hyblock.generator import mutate
        collide = "m0=root; m1=m0.P2@0 q150; m2=m1.P2@0 q150; m3=m2.P2@0 q150"
        calls, firsts = [], []

        def fake(system, user, model, retries=5):
            self.assertIn("mutation operator", system)
            self.assertNotIn("STRUCTURAL RULES:", system)
            calls.append(user)
            inp = re.search(r"## Design to modify\nDESIGN: (.+)", user).group(1)
            good = to_text(mutate(parse(inp), random.Random(len(calls)))[0])
            if "previous answer was rejected" in user:
                reply = f"DESIGN: {collide}" if "overlap" in user else f"CHANGE: c\nDESIGN: {good}"
            else:
                firsts.append(user)
                k = len(firsts) % 4
                reply = {1: f"ANALYSIS: a\nCHANGE: none\nDESIGN: {inp}", 2: f"DESIGN: {collide}",
                         3: "I would rather not."}.get(k, f"ANALYSIS: a\nCHANGE: c\nDESIGN: {good}")
            return reply, {"input_tokens": 100, "output_tokens": 50}

        out = self.tmp / "gallm"
        args = ["--method", "ga", "--mutation", "llm", "--budget", "12", "--batch", "4", "--llm_workers", "1"]
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x"}), \
                mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=fake):
            search.main(args + ["--out", str(out)] + FAST)
        ev = rows(out)
        self.assertTrue((out / "DONE").exists())
        self.assertEqual(sum(r["status"] == "mpc" for r in ev), 12)
        self.assertFalse(any(r["status"] in ("invalid", "duplicate") for r in ev))   # checked before evaluation
        src = " ".join(r["source"] for r in ev)
        for tag in ("llm_mut", "llm_mut_retry", "llm_fail>mutate:"):
            self.assertIn(tag, src)
        self.assertEqual(len({r["design"] for r in ev}), len(ev))
        with open(out / "llm_mut.csv", newline="", encoding="utf-8") as f:
            att = list(csv.DictReader(f))
        outcomes = {r["outcome"] for r in att}
        self.assertTrue({"ok", "unchanged", "invalid", "no_design"} <= outcomes)
        bad = [r for r in att if r["outcome"] == "invalid"]
        self.assertTrue(all(r["kind"] == "self_collision" and " mm" in r["reason"] for r in bad))
        self.assertEqual(len(att), len(calls))
        self.assertIn("## Other evaluated designs most similar", firsts[-1])
        self.assertTrue(any("(rank " in u for u in firsts))               # an evaluated parent: score + rank
        self.assertTrue(any("made by crossover" in u for u in firsts))      # a crossover child: parents' scores
        self.assertTrue(any("## Rejected designs" in u and "overlap" in u for u in firsts))
        mut_rows = [r for r in ev if "llm" in r["source"]]
        self.assertTrue(all(int(r["llm_calls"]) >= 1 for r in mut_rows))
        summ = (out / "summary.txt").read_text(encoding="utf-8")
        self.assertIn("LLM 变异", summ)
        self.assertIn("子代优于亲本", summ)
        # replay into a fresh directory: no API calls, identical designs and attempt log
        out2 = self.tmp / "gallm2"
        with mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=AssertionError("no API")):
            search.main(args + ["--replay", "--llm_cache", str(out / "llm_cache"), "--out", str(out2)] + FAST)
        self.assertEqual([r["design"] for r in rows(out2)], [r["design"] for r in ev])
        with open(out2 / "llm_mut.csv", newline="", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), len(att))


if __name__ == "__main__":
    unittest.main()
