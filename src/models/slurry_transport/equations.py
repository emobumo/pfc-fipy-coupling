# -*- coding: utf-8 -*-

import numpy as np
from fipy import TransientTerm, DiffusionTerm


def get_slurry_parameters(state):
    return state["slurry_parameters"]


def get_inlet_x_range(params):
    """
    Return inlet x-range for placeholder top localized pouring.
    Prefer center+width keys; fall back to legacy center+half-width keys.
    """
    if ("inlet_zone_center_x" in params) and ("inlet_zone_width_x" in params):
        half_width = 0.5 * params["inlet_zone_width_x"]
        return (
            params["inlet_zone_center_x"] - half_width,
            params["inlet_zone_center_x"] + half_width,
        )
    return (
        params["inlet_center_x"] - params["inlet_half_width_x"],
        params["inlet_center_x"] + params["inlet_half_width_x"],
    )


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
    params = get_slurry_parameters(state)

    # Placeholder assumption: clogging only reduces a reference mobility.
    mobility_value = params["reference_mobility"] / (
        1.0 + params["mobility_clogging_factor"] * clogging.value
    )
    mobility.setValue(mobility_value)
    mobility.setValue(np.maximum(mobility_value, params["min_mobility"]))

    # Placeholder assumption: storage stays constant for now.
    storage.setValue(params["reference_storage"])


def apply_boundary_conditions(state):
    """
    第一版边界条件占位：
    暂时只模拟顶部局部入流/高压区的效果。
    后面再替换成真正的浆液倾倒边界。
    """
    pressure = state["pressure"]
    mesh = state["mesh"]
    params = get_slurry_parameters(state)
    fx, fy = mesh.faceCenters()
    # Placeholder engineering boundary: top pouring represented by
    # a fixed-pressure inlet patch with tunable center and half-width.
    inlet_x_min, inlet_x_max = get_inlet_x_range(params)

    # 顶部中间一小段作为“浆液输入区”
    inlet_faces = mesh.facesTop & (fx > inlet_x_min) & (fx < inlet_x_max)

    # 先用固定 pressure 边界做最简单近似
    # Still a placeholder boundary condition, not a full inflow model.
    pressure.constrain(params["inlet_pressure_value"], inlet_faces)

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
    params = get_slurry_parameters(state)

    update_effective_properties(state)
    apply_boundary_conditions(state)

    # Placeholder transport form for the current engineering stub only.
    eq = TransientTerm(coeff=storage) == DiffusionTerm(coeff=mobility)
    eq.solve(var=pressure, dt=dt)

    # 下面是第一版占位更新逻辑，不代表最终物理
    # Placeholder post-update rules, not a final constitutive model.
    filling.setValue(filling.value + params["filling_rate"] * pressure.value * dt)
    clogging.setValue(
        clogging.value + params["clogging_rate"] * filling.value * dt
    )

    result = {
        "scalar_pressure": pressure.value,
        "scalar_filling": filling.value,
        "scalar_clogging": clogging.value,
        "vector_flux_x": (-mobility * pressure.grad()[0]).value,
        "vector_flux_y": (-mobility * pressure.grad()[1]).value,
    }

    return result
