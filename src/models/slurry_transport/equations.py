# -*- coding: utf-8 -*-

import numpy as np
from fipy import DiffusionTerm, TransientTerm


def get_slurry_parameters(state):
    return state["slurry_parameters"]


def get_rheology_model(params):
    model = params.get("rheology_model", "linear")
    if model not in ("linear", "gated_bingham", "porous_bingham"):
        model = "linear"
    # Backward compatibility: older tests may only toggle enable_bingham_yield.
    if (model == "linear") and params.get("enable_bingham_yield", False):
        return "gated_bingham"
    return model


def get_inlet_x_range(params):
    """
    Return inlet x-range for placeholder localized borehole-injection region.
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
    Return placeholder borehole-injection geometry and pressures.
    Core = stronger borehole-influence patch; spread = weaker surrounding patch.
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


def get_initial_porosity_for_filling_limit(state):
    """
    Cache initial porosity once for filling upper-bound evaluation.
    """
    initial_porosity = state.get("initial_porosity_for_filling_limit")
    if initial_porosity is None:
        initial_porosity = np.array(state["porosity"].value, copy=True)
        state["initial_porosity_for_filling_limit"] = initial_porosity
    return initial_porosity


def _to_array(value_or_var):
    if hasattr(value_or_var, "value"):
        return np.asarray(value_or_var.value, dtype=float)
    return np.asarray(value_or_var, dtype=float)


def compute_pressure_gradient_components(state):
    pressure = state["pressure"]
    grad_x = _to_array(pressure.grad()[0])
    grad_y = _to_array(pressure.grad()[1])
    return grad_x, grad_y


def compute_pressure_gradient_magnitude(state):
    grad_x, grad_y = compute_pressure_gradient_components(state)
    return np.sqrt(np.maximum(grad_x * grad_x + grad_y * grad_y, 0.0))


def compute_yield_factor(state, grad_mag):
    params = get_slurry_parameters(state)
    rheology_model = get_rheology_model(params)
    if rheology_model == "linear":
        return np.ones_like(grad_mag)

    if rheology_model == "gated_bingham":
        eps = max(float(params.get("yield_eps", 1.0e-12)), 1.0e-20)
        smoothing = max(float(params.get("yield_gradient_smoothing", 0.05)), eps)
        grad_crit = float(params.get("yield_gradient_crit", 0.25))
        # Smooth monotonic threshold activation:
        # ~0 below threshold, smooth transition around threshold, ~1 above threshold.
        arg = np.clip((grad_mag - grad_crit) / smoothing, -60.0, 60.0)
        yield_factor = 0.5 * (1.0 + np.tanh(arg))
        return np.clip(yield_factor, 0.0, 1.0)

    # Porous-Bingham activation is computed from the physically corrected
    # effective gradient, not the raw pressure-gradient magnitude.
    return np.ones_like(grad_mag)


def compute_porous_bingham_fields(state):
    """
    First-stage porous-medium Bingham approximation.

    This uses a pore-scale threshold estimate for the pressure gradient:
    grad_p_crit = yield_stress / characteristic_pore_size
    It is an engineering closure, not a full constitutive derivation.
    """
    params = get_slurry_parameters(state)
    grad_x, grad_y = compute_pressure_gradient_components(state)
    rho = float(params.get("slurry_density", 1800.0))
    gravity_y = float(params.get("gravity_y", -9.81))
    reg_eps = max(float(params.get("regularization_eps", 1.0e-12)), 1.0e-20)
    pore_size = max(float(params.get("characteristic_pore_size", 1.0e-2)), reg_eps)
    yield_stress = float(params.get("yield_stress", 50.0))
    plastic_viscosity = max(float(params.get("plastic_viscosity", 1.0)), reg_eps)
    band_scale = max(float(params.get("yield_regularization_band", 0.10)), 0.0)

    effective_grad_x = grad_x
    effective_grad_y = grad_y - rho * gravity_y
    effective_grad_mag = np.sqrt(
        np.maximum(
            effective_grad_x * effective_grad_x + effective_grad_y * effective_grad_y,
            0.0,
        )
    )

    grad_p_crit_scalar = yield_stress / pore_size
    grad_p_crit = np.zeros_like(effective_grad_mag) + grad_p_crit_scalar
    band = max(band_scale * grad_p_crit_scalar, reg_eps)
    arg = np.clip((effective_grad_mag - grad_p_crit_scalar) / band, -60.0, 60.0)
    activation = 0.5 * (1.0 + np.tanh(arg))
    activation = np.clip(activation, 0.0, 1.0)

    activation_floor = float(params.get("mobility_activation_floor", 1.0e-3))
    activation_floor = min(max(activation_floor, reg_eps), 1.0)
    activation_eff = np.maximum(activation, activation_floor)

    mu_app = plastic_viscosity / activation_eff
    mu_app = np.maximum(mu_app, float(params.get("min_apparent_viscosity", 1.0e-3)))
    mu_app = np.minimum(mu_app, float(params.get("max_apparent_viscosity", 1.0e6)))

    return {
        "effective_grad_mag": effective_grad_mag,
        "grad_p_crit": grad_p_crit,
        "activation": activation,
        "apparent_viscosity": mu_app,
    }


def update_effective_mobility(state, grad_mag=None, yield_factor=None):
    params = get_slurry_parameters(state)
    rheology_model = get_rheology_model(params)
    mobility_structural = state["mobility_structural"]
    mobility_effective = state["mobility_effective"]
    permeability = _to_array(state["permeability"])
    storage = state["storage"]
    min_mobility = float(params["min_mobility"])

    if rheology_model == "porous_bingham":
        porous_fields = compute_porous_bingham_fields(state)
        mobility_value = permeability / porous_fields["apparent_viscosity"]
        mobility_value = np.maximum(mobility_value, min_mobility)
        mobility_effective.setValue(mobility_value)
        state["mobility"].setValue(mobility_value)
        state["yield_factor"].setValue(porous_fields["activation"])
        state["effective_grad_mag_last"] = porous_fields["effective_grad_mag"]
        state["grad_p_crit_last"] = porous_fields["grad_p_crit"]
        state["apparent_viscosity_last"] = porous_fields["apparent_viscosity"]
        storage.setValue(params["reference_storage"])
        return

    if rheology_model == "gated_bingham":
        if grad_mag is None:
            grad_mag = compute_pressure_gradient_magnitude(state)
        if yield_factor is None:
            yield_factor = compute_yield_factor(state, grad_mag)
        floor_factor = float(params.get("yield_floor_factor", 0.0))
        floor_factor = min(max(floor_factor, 0.0), 1.0)
        yield_scale = np.maximum(yield_factor, floor_factor)
        if params.get("enable_clogging_feedback", False):
            clogging = state["clogging"]
            clogging_value = np.clip(clogging.value, 0.0, 1.0)
            clogging_scale = np.power(
                1.0 - clogging_value,
                float(params.get("mobility_blockage_exponent", 2.0)),
            )
        else:
            clogging_scale = 1.0
    else:
        # Default linear baseline path: do not compute gradient/yield physics.
        # Keep effective mobility equal to structural mobility.
        yield_factor = np.ones_like(mobility_structural.value, dtype=float)
        mobility_value = np.array(mobility_structural.value, copy=True)
        mobility_effective.setValue(mobility_value)
        state["mobility"].setValue(mobility_value)
        state["yield_factor"].setValue(yield_factor)
        state["effective_grad_mag_last"] = np.zeros_like(mobility_value)
        state["grad_p_crit_last"] = np.zeros_like(mobility_value)
        state["apparent_viscosity_last"] = np.zeros_like(mobility_value)
        storage.setValue(params["reference_storage"])
        return

    mobility_value = mobility_structural.value * yield_scale * clogging_scale
    mobility_value = np.maximum(mobility_value, min_mobility)
    mobility_effective.setValue(mobility_value)
    # Legacy alias retained.
    state["mobility"].setValue(mobility_value)

    state["yield_factor"].setValue(yield_factor)
    state["effective_grad_mag_last"] = np.array(grad_mag, copy=True)
    state["grad_p_crit_last"] = np.zeros_like(grad_mag) + float(params.get("yield_gradient_crit", 0.25))
    state["apparent_viscosity_last"] = np.maximum(permeability, min_mobility) / np.maximum(mobility_value, min_mobility)
    storage.setValue(params["reference_storage"])


def update_effective_properties(state):
    # Legacy compatibility wrapper.
    update_effective_mobility(state)


def _compute_net_inflow_from_flux(state):
    pressure = state["pressure"]
    mobility_effective = state["mobility_effective"]
    grad_x = _to_array(pressure.grad()[0])
    grad_y = _to_array(pressure.grad()[1])
    qx = -_to_array(mobility_effective.value) * grad_x
    qy = -_to_array(mobility_effective.value) * grad_y

    # Preferred path: true net inflow from FiPy divergence if supported.
    try:
        # q = -mobility_effective * grad(p), so net inflow = -div(q)
        # = div(mobility_effective * grad(p)).
        net_inflow = _to_array((mobility_effective * pressure.grad()).divergence)
        if net_inflow.shape == pressure.value.shape:
            return net_inflow
    except Exception:
        pass

    # Robust legacy fallback: finite-difference net inflow proxy on the cell grid.
    try:
        x = _to_array(state["x"])
        y = _to_array(state["y"])
        n = x.size
        ux = np.unique(np.round(x, 12))
        uy = np.unique(np.round(y, 12))
        nx = int(ux.size)
        ny = int(uy.size)
        if (nx > 1) and (ny > 1) and (nx * ny == n):
            order = np.lexsort((x, y))
            qx_grid = qx[order].reshape((ny, nx))
            qy_grid = qy[order].reshape((ny, nx))
            dx = float(np.median(np.diff(np.sort(ux))))
            dy = float(np.median(np.diff(np.sort(uy))))
            if dx <= 0.0:
                dx = 1.0
            if dy <= 0.0:
                dy = 1.0
            div_q = np.gradient(qx_grid, dx, axis=1) + np.gradient(qy_grid, dy, axis=0)
            net_inflow = -div_q.reshape(-1)
            out = np.zeros_like(net_inflow)
            out[order] = net_inflow
            return out
    except Exception:
        pass

    # Final fallback: use local flux magnitude as positive inflow proxy.
    return np.sqrt(np.maximum(qx * qx + qy * qy, 0.0))


def update_filling_from_flux(state, dt):
    filling = state["filling"]
    clogging = state["clogging"]
    params = get_slurry_parameters(state)
    rheology_model = get_rheology_model(params)

    porosity = _to_array(state["porosity"])
    initial_porosity = get_initial_porosity_for_filling_limit(state)
    eps = max(float(params.get("yield_eps", 1.0e-12)), 1.0e-20)
    fill_accumulation = max(float(params.get("fill_accumulation_factor", 0.1)), 0.0)
    filling_max = max(float(params.get("filling_max", 1.0)), 0.0)
    filling_limit_fraction = float(params.get("filling_limit_fraction", 0.95))
    filling_limit_fraction = min(max(filling_limit_fraction, 0.0), 1.0)
    local_filling_cap = np.clip(
        filling_limit_fraction * initial_porosity,
        0.0,
        filling_max,
    )

    net_inflow = _compute_net_inflow_from_flux(state)
    positive_inflow = np.maximum(net_inflow, 0.0)
    denom = np.maximum(porosity + eps, eps)
    filling_increment = dt * fill_accumulation * positive_inflow / denom
    filling_value = np.maximum(filling.value + filling_increment, filling.value)
    filling_value = np.maximum(filling_value, 0.0)
    filling_value = np.minimum(filling_value, local_filling_cap)
    filling.setValue(filling_value)

    # Keep default baseline unclogged unless explicitly enabled.
    if (rheology_model != "porous_bingham") and params.get("enable_clogging_feedback", False):
        clogging_denom = np.maximum(local_filling_cap, eps)
        clogging_value = np.clip(filling_value / clogging_denom, 0.0, 1.0)
    else:
        clogging_value = np.zeros_like(filling_value)
    clogging.setValue(clogging_value)
    state["filling_cap_last"] = local_filling_cap
    return net_inflow


def apply_boundary_conditions(state):
    """
    Apply placeholder borehole-injection boundary conditions.
    """
    pressure = state["pressure"]
    mesh = state["mesh"]
    params = get_slurry_parameters(state)
    fx, fy = mesh.faceCenters()
    geo = get_inlet_geometry(params)
    load_factor = get_inlet_loading_factor(state, params)

    spread_faces = mesh.facesTop & (fx > geo["spread_x_min"]) & (fx < geo["spread_x_max"])
    core_faces = mesh.facesTop & (fx > geo["core_x_min"]) & (fx < geo["core_x_max"])

    pressure.constrain(geo["spread_pressure_value"] * load_factor, spread_faces)
    pressure.constrain(geo["core_pressure_value"] * load_factor, core_faces)

    pressure.grad.constrain(0.0, mesh.facesLeft)
    pressure.grad.constrain(0.0, mesh.facesRight)
    pressure.grad.constrain(0.0, mesh.facesBottom)


def _solve_pressure_once(state, dt):
    pressure = state["pressure"]
    storage = state["storage"]
    mobility_effective = state["mobility_effective"]
    eq = TransientTerm(coeff=storage) == DiffusionTerm(coeff=mobility_effective)
    eq.solve(var=pressure, dt=dt)


def solve_slurry_step(state, dt=0.01):
    """
    Solve one slurry-transport step.

    Default path stays compatible with the current linear baseline.
    Bingham-like yield control is activated only when enable_bingham_yield=True.
    """
    pressure = state["pressure"]
    params = get_slurry_parameters(state)
    rheology_model = get_rheology_model(params)

    apply_boundary_conditions(state)

    if rheology_model in ("gated_bingham", "porous_bingham"):
        max_iters = int(params.get("picard_max_iters", 4))
        if max_iters < 1:
            max_iters = 1
        tol = float(params.get("picard_tol", 1.0e-6))
        if tol <= 0.0:
            tol = 1.0e-6

        previous_pressure = np.array(pressure.value, copy=True)
        picard_residual = 0.0
        it = 0
        for it in range(max_iters):
            if rheology_model == "gated_bingham":
                grad_mag = compute_pressure_gradient_magnitude(state)
                yield_factor = compute_yield_factor(state, grad_mag)
                update_effective_mobility(state, grad_mag=grad_mag, yield_factor=yield_factor)
            else:
                update_effective_mobility(state)
            _solve_pressure_once(state, dt=dt)

            picard_residual = float(np.max(np.abs(pressure.value - previous_pressure)))
            previous_pressure = np.array(pressure.value, copy=True)
            if picard_residual <= tol:
                break
        state["picard_iterations_last"] = it + 1
        state["picard_residual_last"] = picard_residual
    else:
        # Linear baseline.
        update_effective_mobility(state)
        _solve_pressure_once(state, dt=dt)
        state["picard_iterations_last"] = 1
        state["picard_residual_last"] = 0.0

    if rheology_model == "gated_bingham":
        grad_mag = compute_pressure_gradient_magnitude(state)
        yield_factor = compute_yield_factor(state, grad_mag)
        update_effective_mobility(state, grad_mag=grad_mag, yield_factor=yield_factor)
    elif rheology_model == "porous_bingham":
        grad_mag = compute_pressure_gradient_magnitude(state)
        update_effective_mobility(state)
        yield_factor = _to_array(state["yield_factor"])
    else:
        grad_mag = np.zeros_like(pressure.value, dtype=float)
        yield_factor = np.ones_like(pressure.value, dtype=float)
        update_effective_mobility(state, grad_mag=grad_mag, yield_factor=yield_factor)
    net_inflow = update_filling_from_flux(state, dt)

    step_idx_raw = state.get("flow_step_index", 0)
    try:
        step_idx = int(step_idx_raw)
    except (TypeError, ValueError):
        step_idx = 0
    state["flow_step_index"] = step_idx + 1

    mobility_effective = state["mobility_effective"]
    result = {
        "scalar_pressure": pressure.value,
        "scalar_filling": state["filling"].value,
        "scalar_filling_cap": state.get("filling_cap_last", np.zeros_like(state["filling"].value)),
        "scalar_clogging": state["clogging"].value,
        "scalar_mobility_effective": mobility_effective.value,
        "scalar_mobility_structural": state["mobility_structural"].value,
        "scalar_yield_factor": state["yield_factor"].value,
        "scalar_grad_mag": grad_mag,
        "scalar_effective_grad_mag": state.get("effective_grad_mag_last", np.zeros_like(grad_mag)),
        "scalar_grad_p_crit": state.get("grad_p_crit_last", np.zeros_like(grad_mag)),
        "scalar_apparent_viscosity": state.get("apparent_viscosity_last", np.zeros_like(grad_mag)),
        "scalar_net_inflow": net_inflow,
        "vector_flux_x": _to_array(-mobility_effective * pressure.grad()[0]),
        "vector_flux_y": _to_array(-mobility_effective * pressure.grad()[1]),
    }
    return result
