"""LLM 提示词里的地形描述（第 12 周，横梁）。

平地（flat）：原样返回，提示词逐字节不变（prompt_sha256 与 LLM 缓存键不变）。
其他地形：把任务说明里写死的 "flat ground" 按固定规则替换，并在 "### TASK ###" 前加一段 ### TERRAIN ###。
地形段只写事实（几何尺寸、位置、能否绕开），不写从搜索结果得出的建议（第12周记录 第 2a 节）。
"""
from __future__ import annotations

TERRAIN = {
    "beams": (
        "The robot moves over flat ground (the same high-friction floor as on flat terrain) with horizontal "
        "cylindrical beams fixed across its path. Each beam has a radius of 0.2 m and its axis is 0.5 m above the "
        "ground, so the lower edge of a beam is 0.3 m and its top is 0.7 m above the ground. The first beam is 1 m "
        "ahead of the robot's starting position and the next beams follow every 2 m in the direction of travel. "
        "The beams span the whole width of the terrain, so the robot cannot go around them, and they do not move."
    ),
}
NAME = {"beams": "the beams terrain"}

# (原文, 替换) —— 各提示词里出现的写法；每个地形版提示词至少要替换掉一处，且替换后不能再有 "flat ground"
RULES = [
    ("robots that walk fast on flat ground", "robots that move forward fast on {name}"),
    ("simulated on flat ground for", "simulated on {name} for"),
    ("walk fast on flat ground", "move forward fast on {name}"),
]
TASK_MARK = "### TASK ###"


def adapt(text: str, terrain: str = "flat") -> str:
    if terrain == "flat":
        return text
    if terrain not in TERRAIN:
        raise ValueError(f"no terrain description for {terrain!r} (hyblock/terrain_text.py)")
    n = 0
    for old, new in RULES:
        n += text.count(old)
        text = text.replace(old, new.format(name=NAME[terrain]))
    if n == 0 or "flat ground" in text:
        raise ValueError(f"terrain_text.adapt: prompt does not match the replacement rules ({n} replaced)")
    if text.count(TASK_MARK) != 1:
        raise ValueError("terrain_text.adapt: expected exactly one '### TASK ###'")
    section = f"### TERRAIN ###\n{TERRAIN[terrain]}\n\n"
    return text.replace(TASK_MARK, section + TASK_MARK)
