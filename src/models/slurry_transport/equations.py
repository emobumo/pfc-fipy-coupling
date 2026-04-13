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


def get_inlet_geometry(params):
    """
    Return placeholder top-pouring geometry and pressures.
    Core = stronger pouring patch; spread = weaker surrounding patch.
    """
    center = params.get("inlet_zone_center_x", params.get("inlet_center_x", 0.0))
    core_width = params.get("inlet_core_width_x", params.get("inlet_zone_width_x", 0.0))
    spread_width = params.get("inlet_spread_width_x", params.get("inlet_zone_width_x", core_width))
    if core_width <= 0.0:
        core_width = 0.5 * spread_width if spread_width > 0.0 else 0.4
    if spread_width <= 0.0:
        spread_width = core_width
    if spread_width <= core_width:
        spread_width = core_width * 1.2 if core_width > 0.0 else 0.48
    min_core_fraction = params.get("inlet_core_min_fraction_of_spread", 0.6)
    if (min_core_fraction <= 0.0) or (min_core_fraction >= 1.0):
        min_core_fraction = 0.6
    core_width = max(core_width, min_core_fraction * spread_width)

    core_half = 0.5 * core_width
    spread_half = 0.5 * spread_width
    core_p = params.get("inlet_pressure_core_value", params.get("inlet_pressure_value", 1.0))
    spread_factor = params.get("inlet_pressure_spread_factor", 0.6)
    if spread_factor <= 0.0:
        spread_factor = 0.6
    if spread_factor >= 1.0:
        spread_factor = 0.8
    spread_p = core_p * spread_factor

    return {
        "core_x_min": center - core_half,
        "core_x_max": center + core_half,
        "spread_x_min": center - spread_half,
        "spread_x_max": center + spread_half,
        "core_pressure_value": core_p,
        "spread_pressure_value": spread_p,
    }


def get_inlet_loading_factor(state, params):
    start_factor = params.get("inlet_loading_start_factor", 1.0)
    ramp_steps = int(params.get("inlet_loading_ramp_steps", 1))
    step_idx_raw = state.get("flow_step_index", 0)
    try:
        step_idx = int(step_idx_raw)
    except (TypeError, ValueError):
        step_idx = 0

    if start_factor < 0.0:
        start_factor = 0.0
    if start_factor > 1.0:
        start_factor = 1.0
    if ramp_steps <= 1:
        return 1.0

    alpha = float(step_idx) / float(ramp_steps - 1)
    alpha = min(max(alpha, 0.0), 1.0)
    return start_factor + (1.0 - start_factor) * alpha


def update_effective_properties(state):
    """
    第一版占位函数：
    根据当前 filling / clogging 更新 mobility 和 storage。

    这里先不给复杂本构，只做一个最简单、可运行的逻辑：
    - clogging 越大，mobility 越小
    - storage 暂时保持常数
    """
    mobility = state["mobility"]
    intrinsic_mobility = state["intrinsic_mobility"]
    storage = state["storage"]
    clogging = state["clogging"]
    params = get_slurry_parameters(state)

    # Placeholder assumption: structural mobility is initialized once from
    # porosity/permeability; later steps only apply clogging attenuation.
    mobility_value = intrinsic_mobility.value / (
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
    geo = get_inlet_geometry(params)
    load_factor = get_inlet_loading_factor(state, params)

    # 顶部中间一小段作为“浆液输入区”
    spread_faces = mesh.facesTop & (fx > geo["spread_x_min"]) & (fx < geo["spread_x_max"])
    core_faces = mesh.facesTop & (fx > geo["core_x_min"]) & (fx < geo["core_x_max"])

    # 先用固定 pressure 边界做最简单近似
    # Still a placeholder boundary condition, not a full inflow model.
    pressure.constrain(geo["spread_pressure_value"] * load_factor, spread_faces)
    pressure.constrain(geo["core_pressure_value"] * load_factor, core_faces)

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
    filling.setValue(np.maximum(filling.value, 0.0))
    clogging.setValue(np.maximum(clogging.value, 0.0))
    step_idx_raw = state.get("flow_step_index", 0)
    try:
        step_idx = int(step_idx_raw)
    except (TypeError, ValueError):
        step_idx = 0
    state["flow_step_index"] = step_idx + 1

    result = {
        "scalar_pressure": pressure.value,
        "scalar_filling": filling.value,
        "scalar_clogging": clogging.value,
        "vector_flux_x": (-mobility * pressure.grad()[0]).value,
        "vector_flux_y": (-mobility * pressure.grad()[1]).value,
    }

    return result
