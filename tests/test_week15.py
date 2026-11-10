"""第 15 周的追加（claude/正式实验计划.md 变更记录第 18–20 条）：
追加 D 的提示词速度上限选项（默认逐字节不变）；追加 C 的最优设计重评（挑对设计、物理参数、可续跑）。"""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from hyblock import llm_mutation, prompt_phys, reeval_best, search, selection

# 已跑运行的 run_meta.json 里记下的 sha256（formal_flat / formal_beams 的 ga_llmtyped_sel_llmEv_s100）
FLAT_MUT = "d9832f345c55ace928f73b4be76daba2aed989215207be716e62bb87c10c94fd"
FLAT_SEL = "e4ba33dd76fc1411291d60fd6ab2b63ded633640394cb65e27c1dbcc1081ae4e"
BEAMS_MUT = "db1b334fda8308a6cc3dba7848b38b4ae6225b0edeb608d73d102e6340531b12"
BEAMS_SEL = "a4ad4aa85134bbc77cfb1241b9221db713bbe244d8a86ab8165abf089bc50b3b"
EXAMPLE = "h0=root; m1=h0.zp@90 q60/120 mirror; h2=h0.xm@0; m3=h2.zp@0 q90/150 mirror; foot=m1.P2; foot=m3.P2"


class PromptSpeedTests(unittest.TestCase):
    def test_defaults_byte_identical(self):
        self.assertEqual(llm_mutation.prompt_sha256(8, True, "flat"), FLAT_MUT)
        self.assertEqual(selection.prompt_sha256(8, "flat"), FLAT_SEL)
        self.assertEqual(llm_mutation.prompt_sha256(8, True, "beams"), BEAMS_MUT)
        self.assertEqual(selection.prompt_sha256(8, "beams"), BEAMS_SEL)
        self.assertEqual(llm_mutation.prompt_sha256(8, True, "flat", None), FLAT_MUT)

    def test_speed_replaced_once(self):
        for text in (llm_mutation.system_prompt(8, True, "flat", 370), selection.system_prompt(8, "flat", 370),
                     llm_mutation.system_prompt(8, True, "beams", 370)):
            self.assertEqual(text.count("max speed about 370 deg/s"), 1)
            self.assertNotIn("1200 deg/s", text)
        a = llm_mutation.system_prompt(8, True, "flat")
        b = llm_mutation.system_prompt(8, True, "flat", 370)
        self.assertEqual(a.replace("about 1200 deg/s", "about 370 deg/s"), b)     # 只改这一处
        with self.assertRaises(ValueError):
            prompt_phys.adapt_speed("no speed here", 370)

    def test_search_arg(self):
        a = search.build_parser().parse_args(["--method", "ga", "--out", "x", "--prompt_speed_deg", "370"])
        self.assertEqual(a.prompt_speed_deg, 370.0)
        self.assertIsNone(search.build_parser().parse_args(["--method", "ga", "--out", "x"]).prompt_speed_deg)
        tmp = Path(tempfile.mkdtemp())
        try:
            with self.assertRaises(SystemExit):     # RM 式 / LLM 第 0 代的提示词没有改写，不允许
                search.main(["--method", "ga", "--init", "llm", "--prompt_speed_deg", "370", "--summary_only",
                             "--replay", "--out", str(tmp / "x")])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def fake_run(root: Path, name: str, rows, args):
    d = root / name
    d.mkdir(parents=True)
    with open(d / "evals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["gen", "idx", "status", "fitness", "design", "ep_scores"])
        w.writeheader()
        w.writerows(rows)
    (d / "run_meta.json").write_text(json.dumps({"args": args}), encoding="utf-8")
    (d / "DONE").write_text("{}\n", encoding="utf-8")


class ReevalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rows = [{"gen": 0, "idx": 0, "status": "mpc", "fitness": 10, "design": "h0=root; foot=h0.ym", "ep_scores": "[10, 10]"},
                {"gen": 1, "idx": 3, "status": "mpc", "fitness": 50, "design": EXAMPLE, "ep_scores": "[40, 60]"},
                {"gen": 2, "idx": 1, "status": "mpc", "fitness": 50, "design": "h0=root; foot=h0.xp", "ep_scores": "[50, 50]"},
                {"gen": 2, "idx": 2, "status": "invalid", "fitness": "", "design": "x", "ep_scores": ""}]
        self.root = self.tmp / "v370"
        fake_run(self.root, "ga_s300", rows, {"terrain": "flat", "steps": 2, "max_speed_deg": 370.0, "pressure": 8.0,
                                              "mass_scale": 1.0, "max_modules": 8})
        fake_run(self.root, "ga_llmtyped_s300", rows, {"terrain": "flat", "steps": 2, "max_speed_deg": 370.0})
        (self.root / "compare").mkdir()             # 没有 DONE 的目录跳过

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_targets(self):
        T = reeval_best.targets([self.root])
        self.assertEqual(len(T), 2)
        t = T[0]
        self.assertEqual((t["gen"], t["idx"], t["design"]), ("1", "3", EXAMPLE))        # 最高分并列取最早的
        self.assertEqual(json.loads(t["physics"]), {"max_speed_deg": 370.0})              # 默认值不传
        self.assertEqual(reeval_best.physics_kw(json.loads(t["physics"])), {"max_speed_deg_s": 370.0})

    def test_run_and_resume(self):
        out = self.tmp / "re"
        argv = ["run", "--roots", str(self.root), "--out", str(out), "--threads", "1"]
        reeval_best.main(argv)
        R = list(csv.DictReader(open(out / "reeval.csv", encoding="utf-8")))
        self.assertEqual(len(R), 2)
        self.assertEqual(len(json.loads(R[0]["re_eps"])), reeval_best.EPISODES)
        self.assertEqual(sorted(bool(r["reused"]) for r in R), [False, True])            # 同一设计只评一次
        self.assertTrue((out / "DONE").exists() and (out / "report.md").exists())
        reeval_best.main(argv)                                                            # 续跑：不重复写
        self.assertEqual(len(list(csv.DictReader(open(out / "reeval.csv", encoding="utf-8")))), 2)


if __name__ == "__main__":
    unittest.main()
