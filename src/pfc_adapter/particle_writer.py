import itasca.ball as balls
from scipy.interpolate import Rbf


def write_field_to_particles(x, y, values, extra_id):
    """
    将 FiPy 网格上的标量场插值到 PFC 颗粒，并写入指定 extra 槽位。
    """
    func = Rbf(x, y, values, function="linear")

    for b in balls.list():
        bx = b.pos_x()
        by = b.pos_y()
        val = float(func(bx, by))
        b.set_extra(extra_id, val)


def write_vector_field_to_particles(x, y, vx, vy, extra_id):
    """
    将 FiPy 网格上的二维矢量场插值到 PFC 颗粒，并写入指定 extra 槽位。
    """
    func_x = Rbf(x, y, vx, function="linear")
    func_y = Rbf(x, y, vy, function="linear")

    for b in balls.list():
        bx = b.pos_x()
        by = b.pos_y()
        val_x = float(func_x(bx, by))
        val_y = float(func_y(bx, by))
        b.set_extra(extra_id, (val_x, val_y))