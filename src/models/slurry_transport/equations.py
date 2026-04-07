# -*- coding: utf-8 -*-
from fipy import TransientTerm, DiffusionTerm


def update_effective_properties(state):
    """
    第一版占位函数：
    根据当前 filling / clogging 更新 mobility 和 storage。

    这里先不给复杂本构，只做一个最简单、可运行的逻辑：
    - clogging 越大，mobility 越小
    - storage 暂时保持常数
    """
    mobility = state["mobility"]
    storage = state["storage"]
    clogging = state["clogging"]

    mobility.setValue(1.0 / (1.0 + clogging.value))
    storage.setValue(1.0)


def apply_boundary_conditions(state):
    """
    第一版边界条件占位：
    暂时只模拟顶部局部入流/高压区的效果。
    后面再替换成真正的浆液倾倒边界。
    """
    pressure = state["pressure"]
    mesh = state["mesh"]
    fx, fy = mesh.faceCenters()

    # 顶部中间一小段作为“浆液输入区”
    inlet_faces = mesh.facesTop & (fx > 0.8) & (fx < 1.7)

    # 先用固定 pressure 边界做最简单近似
    pressure.constrain(1.0, inlet_faces)

    # 其余边界的最简处理
    pressure.grad.constrain(0.0, mesh.facesLeft)
    pressure.grad.constrain(0.0, mesh.facesRight)
    pressure.grad.constrain(0.0, mesh.facesBottom)


def solve_slurry_step(state, dt=0.01):
    """
    第一版浆液连续场求解器骨架。

    当前只是一个“压力样变量”的扩散-储集形式：
        storage * dp/dt = div(mobility * grad(p))

    这不是最终浆液模型，只是为了先把耦合框架跑通。
    """
    pressure = state["pressure"]
    mobility = state["mobility"]
    storage = state["storage"]
    filling = state["filling"]
    clogging = state["clogging"]

    update_effective_properties(state)
    apply_boundary_conditions(state)

    eq = TransientTerm(coeff=storage) == DiffusionTerm(coeff=mobility)
    eq.solve(var=pressure, dt=dt)

    # 下面是第一版占位更新逻辑，不代表最终物理
    filling.setValue(filling.value + 0.1 * pressure.value * dt)
    clogging.setValue(clogging.value + 0.01 * filling.value * dt)

    result = {
        "scalar_pressure": pressure.value,
        "scalar_filling": filling.value,
        "scalar_clogging": clogging.value,
        "vector_flux_x": (-mobility * pressure.grad()[0]).value,
        "vector_flux_y": (-mobility * pressure.grad()[1]).value,
    }

    return result
