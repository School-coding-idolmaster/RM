"""Hyblock 域在进化循环里的 LLM 输入输出：系统提示、从回复中取设计、编译、多样性。"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from hyblock.design import DesignError, summary
from hyblock.mjcf import compile_design

PROMPT_DIR = Path(__file__).with_name("prompts")


def system_prompt(max_modules: int) -> str:
    return (PROMPT_DIR / "system.txt").read_text(encoding="utf-8").replace("{max_modules}", str(max_modules))


def extract_design(text: str) -> str | None:
    """最后一个 \"DESIGN:\" 行（容忍 markdown 的 ** 和反引号）；没有则取最后一个 Step 行的设计。"""
    if not text:
        return None
    hits = re.findall(r"^\W*DESIGN\W*:\s*(.+)$", text, flags=re.M | re.I)
    if hits:
        return hits[-1].strip().strip("`* ").strip() or None
    steps = re.findall(r"Step\s*\d+\s*:\s*(?:\d+x\s*)?r\d\)\s*(.+)$", text, flags=re.M | re.I)
    return steps[-1].strip() if steps else None


def extract_rules(text: str) -> str | None:
    """Step 行里用过的规则序列，例如 \"r0 r3 r4 r6 r6\"（\"3x r6\" 展开为三次）。"""
    rules = []
    for n, r in re.findall(r"Step\s*\d+\s*:\s*(?:(\d+)\s*x\s*)?(r\d)\)", text or "", flags=re.I):
        rules += [r.lower()] * (int(n) if n else 1)
    return " ".join(rules) if rules else None


def compile_text(design_text: str, max_modules: int) -> str:
    """设计文本 -> MJCF（无效设计抛 DesignError，带 kind）。"""
    return compile_design(design_text, max_modules=max_modules)


def diversity(design_text: str) -> float:
    """与 RM 的 design_score 同思路：零件数 + 用到的结构种类越多分越高（只作日志与多样性统计）。"""
    try:
        s = summary(design_text)
    except DesignError:
        return 0.0
    kinds = Counter({"hub_root": s["root"] == "hub", "mirror": s["mirrored"] > 0, "parallel": s["parallel"] > 0,
                     "feet": s["feet"] > 0, "hubs": s["hubs"] > 0})
    n = sum(1 for v in kinds.values() if v)
    return s["modules"] + s["hubs"] + s["feet"] + (n * 2 ** (n - 1) if n > 1 else n)
