"""LLM 提示词里写的物理参数（第 15 周，追加 D；claude/正式实验计划.md 变更记录第 20 条）。

冻结的 system.txt 写的是默认参数（关节速度上限 "max speed about 1200 deg/s"，对应 modules.yaml 的 1185 deg/s）。
敏感性分析 v370 下 LLM 的优势接近 0；追加 D 检验"提示词与仿真不符"是不是原因之一：
--prompt_speed_deg 370 时，只把这一句换成 "max speed about 370 deg/s"，其余逐字不变。
默认（None）：原样返回，提示词与 LLM 缓存键逐字节不变。
"""
from __future__ import annotations

SPEED_TEXT = "max speed about 1200 deg/s"


def adapt_speed(text: str, speed_deg: float | None = None) -> str:
    if speed_deg is None:
        return text
    n = text.count(SPEED_TEXT)
    if n != 1:
        raise ValueError(f"prompt_phys.adapt_speed: expected exactly one {SPEED_TEXT!r}, found {n}")
    return text.replace(SPEED_TEXT, f"max speed about {int(round(speed_deg))} deg/s")
