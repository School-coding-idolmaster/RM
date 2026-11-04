"""地形相关的离线几何检查（第 12 周，横梁）：不跑 MPC，只看初始姿态的高度与尺寸。

横梁（mpc/mppi.py apply_terrain，s = 1.0）：半径 0.2 m 的横杆，中心离地 0.5 m（下沿 0.3 m），第一根在前方 1 m，
之后每隔 2 m 一根。机器人要从下面钻过去。
高度 = MPC reset 的初始姿态（不加噪声；最低点离地 init_clearance）下所有零件的最高点 z。控制器可以中途改变姿态，
所以这只是\"初始是否低矮\"，不是能否钻过的判据。
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from hyblock.design import DesignError, build_robot
from hyblock.mjcf import compile_design

BEAM_CLEARANCE_M = 0.3          # 横梁下沿（s = 1.0）


def _extent_z(model, data, g) -> float:
    t = model.geom_type[g]
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return float(np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g])
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(model.geom_size[g][0])
    return float(model.geom_rbound[g])


def geometry(design: str, max_modules: int | None = 8) -> dict:
    """初始姿态的高度、长、宽（m）与零件数。无效设计返回 {'error': kind}。"""
    try:
        xml = compile_design(design, max_modules=max_modules)
    except DesignError as e:
        return {"error": e.kind}
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo = np.full(3, math.inf)
    hi = np.full(3, -math.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] == 0:
            continue
        c = d.geom_xpos[g]
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            ext = np.abs(d.geom_xmat[g].reshape(3, 3)) @ m.geom_size[g]
        else:
            ext = np.full(3, _extent_z(m, d, g))
        lo, hi = np.minimum(lo, c - ext), np.maximum(hi, c + ext)
    torso_z = float(d.qpos[2])
    return {"height_m": round(float(hi[2]), 4), "length_m": round(float(hi[0] - lo[0]), 4),
            "width_m": round(float(hi[1] - lo[1]), 4), "torso_z_m": round(torso_z, 4),
            "under_beam": bool(hi[2] < BEAM_CLEARANCE_M)}
