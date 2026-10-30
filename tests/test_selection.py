"""② 挑选（第 10 周）：回复解析、经验池、--select 的预算 / 续跑 / 缓存重放、与 LLM 变异叠加（假 LLM）。"""
import csv
import random
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from hyblock import search, selection
from hyblock.design import to_text

FAST = ["--steps", "2", "--threads", "1", "--episodes", "1"]
KEYS = ["gen", "idx", "source", "parents", "design", "status", "fitness", "sel_rank", "sel_slot", "sel_pred"]


def rows(out):
    with open(Path(out) / "evals.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def feet_ranking(user):
    """假 LLM 的\"预测\"：脚多的排前面（并列按标签号）。"""
    cands = re.findall(r"^(C\d+) \| .*? \| (\d+) modules, (\d+) joints, (\d+) feet \|", user, flags=re.M)
    cands.sort(key=lambda c: (-int(c[3]), int(c[0][1:])))
    return [c[0] for c in cands]


class SelectionUnitTests(unittest.TestCase):
    def test_parse_answer(self):
        got, oc, ana = selection.parse_answer("ANALYSIS: feet help\n**RANKING:** C3, C1, c2", 3, "rank", 0)
        self.assertEqual((got, oc, ana), ([2, 0, 1], "ok", "feet help"))
        got, oc, _ = selection.parse_answer("RANKING: C2, C2, C9", 3, "rank", 0)
        self.assertEqual((got, oc), ([1], "partial"))
        got, oc, _ = selection.parse_answer("SELECTED: C4, C1, C2", 4, "half", 2)
        self.assertEqual((got, oc), ([3, 0], "ok"))
        self.assertEqual(selection.parse_answer("no idea", 4, "rank", 0)[1], "fail")
        pos = selection.positions([1], [2, 0, 1], 3)          # label C2 = candidate 0 is best; rest tied
        self.assertEqual(list(pos), [0.0, 1.5, 1.5])

    def test_pool_rows(self):
        rng = random.Random(0)
        from hyblock.generator import random_design
        hist = [{"design": to_text(random_design(rng)), "fitness": str(k), "status": "mpc", "modules": 1,
                 "joints": 2, "feet": 0} for k in range(40)]
        cands = [{"design": hist[3]["design"]}, {"design": hist[7]["design"]}]
        scored = sorted(hist, key=lambda r: -float(r["fitness"]))
        pool = selection.pool_rows(scored, cands, 12)
        self.assertEqual(len(pool), 12)
        self.assertEqual(len({r["design"] for r in pool}), 12)
        self.assertEqual([float(r["fitness"]) for r in pool], sorted((float(r["fitness"]) for r in pool), reverse=True))
        self.assertTrue({str(k) for k in range(34, 40)} <= {r["fitness"] for r in pool})   # best half
        self.assertIn(hist[3]["design"], {r["design"] for r in pool})                       # most similar
        self.assertEqual(selection.pool_rows(scored[:5], cands, 12), scored[:5])

    def test_regress_and_knn_predict(self):
        rng = random.Random(1)
        from hyblock.generator import random_design
        from hyblock.design import summary
        hist = []
        for k in range(30):
            d = to_text(random_design(rng))
            hist.append({"design": d, "fitness": str(10.0 * summary(d)["feet"]), "status": "mpc"})
        cands = [{"design": to_text(random_design(rng)), "pfit": []} for _ in range(10)]
        truth = [summary(c["design"])["feet"] for c in cands]
        pred, _ = selection.RegressSel("k").predict(1, cands, hist)
        from scipy.stats import spearmanr
        self.assertGreater(spearmanr(pred, truth).statistic, 0.8)
        pk, _ = selection.KnnSel("k").predict(1, cands, hist)
        self.assertEqual(len(pk), 10)


class StrataTests(unittest.TestCase):
    def test_family_and_quotas(self):
        self.assertEqual([selection.family(x) for x in ["crossover+mutate:roll", "mutate:remove_foot", "llm_mut[add_hub]",
                                                         "llm_fail>mutate:q0", "random"]],
                         ["crossover", "remove", "add", "tweak", "other"])
        q = selection.quotas({"crossover": 9, "remove": 7, "add": 8, "tweak": 6}, 14)
        self.assertEqual(sum(q.values()), 14)
        self.assertEqual(q, {"crossover": 4, "remove": 3, "add": 4, "tweak": 3})

    def test_stratified_choose(self):
        """预测值偏爱 add 时：不分层几乎全挑 add；分层后各类按比例。"""
        from types import SimpleNamespace
        srcs = ["mutate:add_foot"] * 8 + ["crossover"] * 8 + ["mutate:remove_part"] * 8 + ["mutate:q0"] * 8
        cands = [{"idx": k, "design": "", "source": s, "pfit": []} for k, s in enumerate(srcs)]

        class Fake(selection.Base):
            def predict(self, gen, cands, hist):
                return np.array([10.0 if "add" in c["source"] else 0.0 for c in cands]) + np.arange(len(cands)) * 1e-3, []
        for strata, lo, hi in (("none", 8, 8), ("family", 3, 5)):
            a = SimpleNamespace(rand_slots=0, seed=0, sel_strata=strata)
            res, _ = selection.choose(a, 1, cands, [], Fake("k"), 16)
            n_add = sum(selection.family(cands[p]["source"]) == "add" for p in res["picked"])
            self.assertTrue(lo <= n_add <= hi, (strata, n_add))
            self.assertEqual(len(res["picked"]), 16)


class SelectSearchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_select_budget_resume(self):
        """--select regress：每代提出 2 倍、只评估 batch 个；unselected 不占预算；中途中断后续跑得到同样的结果。"""
        out = self.tmp / "reg"
        args = ["--method", "ga", "--select", "regress", "--budget", "10", "--batch", "4", "--rand_slots", "1",
                "--out", str(out)] + FAST
        search.main(args)
        first = rows(out)
        self.assertTrue((out / "DONE").exists())
        self.assertEqual(sum(r["status"] == "mpc" for r in first), 10)
        g1 = [r for r in first if r["gen"] == "1"]
        self.assertEqual(len(g1), 8)
        self.assertEqual(sum(r["status"] == "mpc" for r in g1), 4)
        self.assertEqual(sorted(r["sel_slot"] for r in g1 if r["status"] == "mpc"), ["rand", "top", "top", "top"])
        self.assertTrue(all(r["sel_rank"] for r in g1 if r["status"] in ("mpc", "unselected")))
        self.assertEqual([r["source"] for r in first if r["gen"] == "0"],
                         ["random"] * 4)                                     # gen 0: no selection
        with open(out / "evals.csv", encoding="utf-8") as f:                 # interrupt in the middle of gen 2
            lines = f.read().splitlines(True)
        with open(out / "evals.csv", "w", encoding="utf-8") as f:
            f.writelines(lines[:-3])
        (out / "DONE").unlink()
        search.main(args)
        again = rows(out)
        key = lambda r: (int(r["gen"]), int(r["idx"]))
        self.assertEqual([[r[k] for k in KEYS] for r in sorted(again, key=key)],
                         [[r[k] for k in KEYS] for r in sorted(first, key=key)])
        self.assertIn("② 挑选", (out / "summary.txt").read_text(encoding="utf-8"))

    def test_strata_run_and_eval_unselected(self):
        """--sel_strata family 端到端；诊断脚本：自检逐位一致、补跑被刷掉的候选并写报告。"""
        out = self.tmp / "strat"
        search.main(["--method", "ga", "--select", "knn", "--sel_strata", "family", "--budget", "8", "--batch", "4",
                     "--rand_slots", "1", "--out", str(out)] + FAST)
        ev = rows(out)
        self.assertEqual(sum(r["status"] == "mpc" for r in ev), 8)
        self.assertIn("按修改种类分层", (out / "summary.txt").read_text(encoding="utf-8"))
        from hyblock import eval_unselected
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            eval_unselected.main(["--runs", str(out), "--per_run", "2", "--out", str(self.tmp / "diag"), "--threads", "1"])
        self.assertIn("一致", buf.getvalue())
        self.assertNotIn("不一致", buf.getvalue())
        with open(self.tmp / "diag" / "unselected_evals.csv", newline="", encoding="utf-8") as f:
            self.assertEqual(len(list(csv.DictReader(f))), 2)
        self.assertTrue((self.tmp / "diag" / "report.md").exists())

    def test_select_llm_cache_replay(self):
        """--select llm：每代 3 票（顺序不同）；坏回复记为 fail；top 名额 = 合并名次最前的；--replay 不调用 API、结果相同。"""
        calls = []

        def fake(system, user, model, retries=5):
            self.assertIn("surrogate model", system)
            calls.append(user)
            if len(calls) == 2:
                return "I cannot rank these.", {"input_tokens": 10, "output_tokens": 5}
            return "ANALYSIS: feet\nRANKING: " + ", ".join(feet_ranking(user)), {"input_tokens": 100, "output_tokens": 20}

        out = self.tmp / "llm"
        args = ["--method", "ga", "--select", "llm", "--budget", "10", "--batch", "4", "--rand_slots", "1",
                "--sel_pool", "4", "--llm_workers", "1", "--out", str(out)] + FAST
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x"}), \
                mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=fake):
            search.main(args)
        ev = rows(out)
        with open(out / "select.csv", newline="", encoding="utf-8") as f:
            sc = list(csv.DictReader(f))
        self.assertEqual(len(sc), 3 * len({r["gen"] for r in ev if r["gen"] != "0"}))
        self.assertIn("fail", {c["outcome"] for c in sc})
        self.assertEqual(len({c["labels"] for c in sc if c["gen"] == "1"}), 3)        # different orders
        u = calls[0]
        self.assertIn("## Designs already simulated (4 of the 4", u)
        self.assertIn("## New candidates, not simulated yet (8)", u)
        self.assertRegex(u, r"from a design scored|crossover of designs scored")
        for g in {r["gen"] for r in ev if r["gen"] != "0"}:
            gr = [r for r in ev if r["gen"] == g and r["sel_rank"]]
            top = [r for r in gr if r["sel_slot"] == "top"]
            others = [r for r in gr if r["sel_slot"] != "top"]
            self.assertTrue(max(int(r["sel_rank"]) for r in top) <= min(int(r["sel_rank"]) for r in others)
                            or any(r["sel_slot"] == "rand" for r in gr))
            feet = {r["design"]: int(r["feet"]) for r in gr}
            best_pred = min(gr, key=lambda r: int(r["sel_rank"]))
            self.assertEqual(feet[best_pred["design"]], max(feet.values()))
        rep = self.tmp / "rep"
        shutil.copytree(out / "llm_cache", rep / "llm_cache")
        with mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=AssertionError("no API")):
            search.main(args[:-len(FAST) - 2] + ["--out", str(rep), "--replay"] + FAST)
        self.assertEqual([[r[k] for k in KEYS] for r in rows(rep)], [[r[k] for k in KEYS] for r in ev])

    def test_select_llm_with_llm_typed_mutation(self):
        """2 × 2 的第 4 组：按种类引导的 LLM 变异 + ② LLM 挑选（假 LLM 分别回答两种请求）。"""
        from hyblock.design import parse
        from hyblock.generator import MUTATIONS
        from hyblock.llm_mutation import TYPE_TEXT
        from hyblock.model import load_spec
        spec = load_spec()
        by_text = {v: k for k, v in TYPE_TEXT.items()}
        n = {"mut": 0, "sel": 0}

        def fake(system, user, model, retries=5):
            if "surrogate model" in system:
                n["sel"] += 1
                return "RANKING: " + ", ".join(feet_ranking(user)), {"input_tokens": 100, "output_tokens": 20}
            n["mut"] += 1
            inp = re.search(r"## Design to modify\nDESIGN: (.+)", user).group(1)
            req = by_text[re.search(r"## Required kind of change\nYou must (.+?)\. Choose where", user).group(1)]
            rng = random.Random(n["mut"])
            for _ in range(40):
                c = MUTATIONS[req](parse(inp), rng=rng, spec=spec, max_modules=8)
                if c is not None and to_text(c) != inp:
                    return f"CHANGE: {req}\nDESIGN: {to_text(c)}", {"input_tokens": 100, "output_tokens": 50}
            return "no idea", {"input_tokens": 100, "output_tokens": 50}

        out = self.tmp / "both"
        with mock.patch.dict("os.environ", {"ANTHROPIC_API_KEY": "x"}), \
                mock.patch("robomorph.evolution.llm.anthropic_chat_full", side_effect=fake):
            search.main(["--method", "ga", "--mutation", "llm_typed", "--select", "llm", "--budget", "8",
                         "--batch", "4", "--rand_slots", "1", "--sel_votes", "2", "--llm_workers", "1",
                         "--out", str(out)] + FAST)
        ev = rows(out)
        self.assertTrue((out / "DONE").exists())
        self.assertGreater(n["mut"], 4)
        self.assertEqual(n["sel"], 2)                                         # gen 1 only (budget 8)
        self.assertEqual(sum(r["status"] == "unselected" for r in ev), 4)
        self.assertTrue(any("llm_mut" in r["source"] for r in ev))


if __name__ == "__main__":
    unittest.main()
