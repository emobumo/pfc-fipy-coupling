# -*- coding: utf-8 -*-
from scipy.interpolate import Rbf


def write_field_to_particles(x, y, values, extra_id, clamp_min=None):
    """
    将 FiPy 网格上的标量场插值到 PFC 颗粒，并写入指定 extra 槽位。

    仅在 PFC（itasca）运行环境中执行写回；若不在 PFC 环境（如纯 FiPy 测试），
    则跳过写回并返回 False，使核心计算可在无 PFC 时独立运行。
    """
    try:
        import itasca.ball as balls
    except Exception:
        return False

    func = Rbf(x, y, values, function="linear")

    for b in balls.list():
        bx = b.pos_x()
        by = b.pos_y()
        val = float(func(bx, by))
        if (clamp_min is not None) and (val < clamp_min):
            val = float(clamp_min)
        b.set_extra(extra_id, val)


def write_vector_field_to_particles(x, y, vx, vy, extra_id):
    """
    将 FiPy 网格上的二维矢量场插值到 PFC 颗粒，并写入指定 extra 槽位。

    仅在 PFC（itasca）运行环境中执行写回；若不在 PFC 环境，则跳过并返回 False。
    """
    try:
        import itasca.ball as balls
    except Exception:
        return False

    func_x = Rbf(x, y, vx, function="linear")
    func_y = Rbf(x, y, vy, function="linear")

    for b in balls.list():
        bx = b.pos_x()
        by = b.pos_y()
        val_x = float(func_x(bx, by))
        val_y = float(func_y(bx, by))
        b.set_extra(extra_id, (val_x, val_y))
