# -*- coding: utf-8 -*-

import numpy as np
from fipy import (
    CellVariable,
    DiffusionTerm,
    TransientTerm,
    FaceVariable,
    ImplicitSourceTerm,
    Variable,
)


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


def get_inlet_geometry(params):
    """
    Return placeholder borehole-injection geometry and pressures.
    Core = stronger borehole-influence patch; spread = weaker surrounding patch.
    """
    center = params.get("inlet_zone_center_x", 0.0)
    core_width = params.get("inlet_core_width_x", 0.0)
    spread_width = params.get("inlet_spread_width_x", core_width)
    if core_width <= 0.0:
        core_width = 0.5 * spread_width if spread_width > 0.0 else 0.4
    if spread_width <= 0.0:
        spread_width = core_width
    # spread == core is a valid single-zone borehole; only widen a spread that
    # is narrower than the core (the core constraint overrides the overlap).
    if spread_width < core_width:
        spread_width = core_width
    min_core_fraction = params.get("inlet_core_min_fraction_of_spread", 0.6)
    if (min_core_fraction <= 0.0) or (min_core_fraction >= 1.0):
        min_core_fraction = 0.6
    core_width = max(core_width, min_core_fraction * spread_width)

    core_half = 0.5 * core_width
    spread_half = 0.5 * spread_width
    core_p = params.get("inlet_pressure_core_value", 1.0)
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


def get_porous_bingham_eps(params):
    default_eps = 1.0e-12
    return {
        "length_eps_m": max(float(params.get("length_eps_m", default_eps)), 1.0e-20),
        "gradient_eps_pa_per_m": max(float(params.get("gradient_eps_pa_per_m", default_eps)), 1.0e-20),
        "activation_eps": max(float(params.get("activation_eps", default_eps)), 1.0e-20),
    }


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


def get_porous_bingham_activation(params):
    """
    Sub-mode of porous_bingham:
      - "threshold": mobility = (k/mu_p) * max(0, 1 - lambda/(|grad Phi|+eps)),
        lambda = 2*tau0/r_eff per cell, exact zero below threshold (default);
      - "tanh": legacy smooth on/off activation via apparent viscosity (kept
        as a switchable comparison mode, not used by default).
    """
    model = params.get("porous_bingham_activation", "threshold")
    if model not in ("threshold", "tanh"):
        model = "threshold"
    return model


def get_yield_truncation_eps(state, params):
    """
    Regularization epsilon [Pa/m] added to |grad Phi| in the truncation factor.
    Explicit yield_truncation_eps_pa_per_m wins; otherwise auto-default to
    1e-6 of the characteristic driving gradient p0 / L_domain.
    """
    eps = float(params.get("yield_truncation_eps_pa_per_m", 0.0))
    if eps > 0.0:
        return eps
    p0 = abs(float(params.get("inlet_pressure_core_value", 1.0)))
    x = _to_array(state["x"])
    y = _to_array(state["y"])
    extent = max(
        float(np.max(x)) - float(np.min(x)),
        float(np.max(y)) - float(np.min(y)),
        1.0e-12,
    )
    return max(1.0e-6 * p0 / extent, 1.0e-20)


def compute_threshold_bingham_fields(state):
    """
    Porous-Bingham mobility with the start-up pressure-gradient truncation:

        mobility   = (k / mu_p) * max(0, 1 - lambda / (|grad Phi| + eps))
        lambda     = 2 * tau0 / r_eff          [Pa/m, per cell]
        r_eff      = sqrt(8 k / phi)           [m, capillary-bundle radius]
        |grad Phi| = |grad p - rho * g_vec|    (g_vec = (0, gravity_y))

    Below the threshold the mobility is exactly zero (no creep), so a true
    stagnation front exists at L_max = p0 / lambda in 1D.
    """
    params = get_slurry_parameters(state)
    grad_x, grad_y = compute_pressure_gradient_components(state)
    rho = float(params.get("slurry_density", 1800.0))
    gravity_y = float(params.get("gravity_y", -9.81))
    yield_stress = float(params.get("yield_stress", 50.0))
    plastic_viscosity = max(float(params.get("plastic_viscosity", 1.0)), 1.0e-20)

    # Driving force consistent with the PDE: grad(p) - rho*g_vec.
    effective_grad_x = grad_x
    effective_grad_y = grad_y - rho * gravity_y
    effective_grad_mag = np.sqrt(
        np.maximum(
            effective_grad_x * effective_grad_x + effective_grad_y * effective_grad_y,
            0.0,
        )
    )

    permeability = np.maximum(_to_array(state["permeability"]), 1.0e-30)
    porosity = np.clip(_to_array(state["porosity"]), 1.0e-6, 1.0 - 1.0e-6)
    r_eff = np.sqrt(8.0 * permeability / porosity)
    grad_p_crit = 2.0 * yield_stress / np.maximum(r_eff, 1.0e-20)

    eps = get_yield_truncation_eps(state, params)
    truncation = np.maximum(0.0, 1.0 - grad_p_crit / (effective_grad_mag + eps))
    mobility = (permeability / plastic_viscosity) * truncation

    # Diagnostic apparent viscosity (infinite below threshold; report clipped).
    max_mu = float(params.get("max_apparent_viscosity", 1.0e6))
    mu_app = np.minimum(
        plastic_viscosity / np.maximum(truncation, 1.0e-30), max_mu
    )

    return {
        "effective_grad_mag": effective_grad_mag,
        "grad_p_crit": grad_p_crit,
        "activation": truncation,
        "apparent_viscosity": mu_app,
        "mobility": mobility,
    }


def get_pressure_coeff_form(params):
    """
    Pressure-equation diffusion-coefficient discretization (threshold mode):
      - "face": explicitly built FaceVariable, M_f = harmonic(k)/mu_p *
        truncation(face gradient) * upwind k_r  (default; required for the
        saturation transport's discrete consistency)
      - "cell": legacy cell coefficient (FiPy arithmetic face averaging),
        kept as a comparison/fallback mode.
    """
    form = params.get("pressure_coeff_form", "face")
    if form not in ("face", "cell"):
        form = "face"
    return form


def _face_upwind_relperm(state, normal_force):
    """
    Upwind (donor-cell) relative permeability at faces.

    normal_force is the face driving force along the face normal,
    (grad(p) + b) . n_hat; the flux runs along -normal_force, and FiPy face
    normals point from faceCellIDs[0] to faceCellIDs[1], so the donor is
    cell0 when normal_force <= 0 (flux along +n) and cell1 otherwise.
    Exterior inflow faces use the boundary saturation: 1 on the open
    injection faces, 0 elsewhere (sealed walls cannot supply slurry).

    Until the saturation field is wired in (k_r == 1 everywhere), this
    returns ones and acts as the upwind hook only.
    """
    params = get_slurry_parameters(state)
    saturation = state.get("saturation")
    mesh = state["mesh"]
    num_faces = mesh.numberOfFaces
    if (saturation is None) or (not params.get("enable_saturation_transport", False)):
        return np.ones(num_faces, dtype=float)

    exponent = float(params.get("relperm_exponent", 3.0))
    s = np.clip(_to_array(saturation), 0.0, 1.0)
    kr_cell = np.power(s, exponent)

    # faceCellIDs[1] is masked on exterior faces; both np.where branches are
    # evaluated, so sanitize the indices (the masked entries are never the
    # selected donor: exterior outflow picks cell0, exterior inflow is
    # overwritten with the boundary value below).
    ids1_raw = mesh.faceCellIDs[1]
    if hasattr(ids1_raw, "filled"):
        ids1_raw = ids1_raw.filled(0)
    ids0 = np.clip(np.asarray(mesh.faceCellIDs[0]).astype(int), 0, kr_cell.size - 1)
    ids1 = np.clip(np.asarray(ids1_raw).astype(int), 0, kr_cell.size - 1)
    donor_is_cell0 = normal_force <= 0.0
    kr_face = np.where(donor_is_cell0, kr_cell[ids0], kr_cell[ids1])

    exterior = np.asarray(mesh.exteriorFaces.value, dtype=bool)
    open_faces = state.get("open_pressure_faces")
    if open_faces is not None:
        open_mask = np.asarray(open_faces.value, dtype=bool)
    else:
        open_mask = np.zeros(num_faces, dtype=bool)
    # Exterior faces: inflow (flux along the inward normal, normal_force > 0
    # at an exterior face whose normal points outward) draws from the
    # boundary saturation; outflow keeps the interior donor value.
    boundary_s = np.where(open_mask, 1.0, 0.0)
    boundary_kr = np.power(boundary_s, exponent)
    inflow_ext = exterior & (normal_force > 0.0)
    kr_face = np.where(inflow_ext, boundary_kr, kr_face)
    return kr_face


def update_threshold_face_mobility(state):
    """
    Build the face diffusion coefficient for the threshold Bingham mode:

        M_f = (harmonic(k)_f / mu_p) * trunc_f * k_r_up_f
        trunc_f = max(0, 1 - lambda_f / (|gradPhi|_f + eps))
        lambda_f = 2*tau0 / sqrt(8 k_f / phi_f)
        gradPhi_f = faceGrad(p) + b_f   (same sealed-wall gravity face as
                                         the solve and the explicit fluxes)

    Under-relaxed like the cell mobility; exactly zero below threshold.
    Stores/refreshes state["mobility_face"] (FaceVariable).
    """
    params = get_slurry_parameters(state)
    mesh = state["mesh"]
    pressure = state["pressure"]
    yield_stress = float(params.get("yield_stress", 50.0))
    plastic_viscosity = max(float(params.get("plastic_viscosity", 1.0)), 1.0e-20)

    k_face = np.maximum(_to_array(state["permeability"].harmonicFaceValue), 1.0e-30)
    phi_face = np.clip(
        _to_array(state["porosity"].arithmeticFaceValue), 1.0e-6, 1.0 - 1.0e-6
    )
    r_eff = np.sqrt(8.0 * k_face / phi_face)
    lam_face = 2.0 * yield_stress / np.maximum(r_eff, 1.0e-20)

    gravity_face = _build_gravity_face(state)
    grad_face = pressure.faceGrad() + gravity_face
    gx = _to_array(grad_face[0])
    gy = _to_array(grad_face[1])
    grad_mag = np.sqrt(np.maximum(gx * gx + gy * gy, 0.0))

    normals = np.asarray(mesh.faceNormals)
    normal_force = gx * normals[0] + gy * normals[1]
    kr_face = _face_upwind_relperm(state, normal_force)

    eps = get_yield_truncation_eps(state, params)
    truncation = np.maximum(0.0, 1.0 - lam_face / (grad_mag + eps))
    mobility_new = (k_face / plastic_viscosity) * truncation * kr_face

    mobility_face = state.get("mobility_face")
    if mobility_face is None:
        mobility_face = FaceVariable(
            mesh=mesh,
            value=_to_array(state["mobility_effective"].arithmeticFaceValue),
        )
        state["mobility_face"] = mobility_face

    omega = float(params.get("picard_relaxation", 0.5))
    if (omega <= 0.0) or (omega > 1.0):
        omega = 0.5
    mobility_old = _to_array(mobility_face)

    # Mobility-update policy (round-4 experiments, see saturation_design.md):
    #
    # WITHOUT filling cells (fully saturated / S=1 modes): asymmetric
    # under-relaxation with snap-to-zero below threshold. This holds the
    # Bingham stall hard (rounds 2-3 results) and there is no activation
    # seam to choke.
    #
    # WITH filling cells (fill transport): the snap's discontinuity
    # limit-cycles at the activation seam (a just-activated cell decouples
    # from the wet column, choking the front ~250x), so use SYMMETRIC
    # relaxation (continuous map) plus a per-face yield latch with
    # hysteresis: faces whose step-end converged gradient was <= lambda stay
    # at exactly zero until the iterate gradient exceeds (1+h)*lambda.
    # Without the latch, near-stall jitter (head refill from the pressure
    # boundary + tail drain into penalized cells) pumps slurry across
    # sub-threshold faces and the front creeps far past L_max. h only acts
    # at the front face there (position cost ~h*dx), default 0.05.
    if state.get("fill_penalty_active", False):
        mobility_value = (1.0 - omega) * mobility_old + omega * mobility_new
        if params.get("enable_yield_latch", True):
            latched = state.get("face_yield_latched")
            if latched is None:
                latched = np.zeros(mesh.numberOfFaces, dtype=bool)
            hband = 1.0 + max(
                float(params.get("yield_hysteresis_band", 0.05)), 0.0
            )
            latched = latched & np.logical_not(grad_mag > hband * lam_face)
            state["face_yield_latched"] = latched
            mobility_value = np.where(latched, 0.0, mobility_value)
    else:
        mobility_value = np.where(
            mobility_new > 0.0,
            (1.0 - omega) * mobility_old + omega * mobility_new,
            0.0,
        )

    mobility_face.setValue(mobility_value)
    return mobility_face


def compute_porous_bingham_fields(state):
    """
    First-stage porous-medium Bingham approximation.

    This uses a pore-scale threshold estimate for the pressure gradient:
    grad_p_crit = yield_stress / characteristic_pore_size
    It is an engineering closure, not a full constitutive derivation.
    """
    params = get_slurry_parameters(state)
    grad_x, grad_y = compute_pressure_gradient_components(state)
    eps = get_porous_bingham_eps(params)
    rho = float(params.get("slurry_density", 1800.0))
    gravity_y = float(params.get("gravity_y", -9.81))
    pore_size = max(float(params.get("characteristic_pore_size", 1.0e-2)), eps["length_eps_m"])
    yield_stress = float(params.get("yield_stress", 50.0))
    plastic_viscosity = max(float(params.get("plastic_viscosity", 1.0)), eps["activation_eps"])
    band_scale = max(float(params.get("yield_regularization_band", 0.10)), 0.0)

    # Driving force consistent with PDE: grad(p) - rho*g_vec, g_vec=(0, gravity_y).
    effective_grad_x = grad_x
    effective_grad_y = grad_y - rho * gravity_y
    effective_grad_mag = np.sqrt(
        np.maximum(
            effective_grad_x * effective_grad_x + effective_grad_y * effective_grad_y,
            0.0,
        )
    )

    # Critical pressure gradient [Pa/m] from a pore-scale yield approximation.
    grad_p_crit_scalar = yield_stress / pore_size
    grad_p_crit = np.zeros_like(effective_grad_mag) + grad_p_crit_scalar
    band = max(band_scale * grad_p_crit_scalar, eps["gradient_eps_pa_per_m"])
    arg = np.clip((effective_grad_mag - grad_p_crit_scalar) / band, -60.0, 60.0)
    activation = 0.5 * (1.0 + np.tanh(arg))
    activation = np.clip(activation, 0.0, 1.0)

    activation_floor = float(params.get("mobility_activation_floor", 1.0e-3))
    activation_floor = min(max(activation_floor, eps["activation_eps"]), 1.0)
    activation_eff = np.maximum(activation, activation_floor)

    # Apparent viscosity [Pa·s].
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
        if get_porous_bingham_activation(params) == "threshold":
            porous_fields = compute_threshold_bingham_fields(state)
            omega = float(params.get("picard_relaxation", 0.5))
            if (omega <= 0.0) or (omega > 1.0):
                omega = 0.5
            mobility_new = porous_fields["mobility"]
            mobility_old = _to_array(mobility_effective)
            # SYMMETRIC under-relaxation (also toward zero); see the face
            # update for why an asymmetric snap-to-zero limit-cycles. The
            # truncation is continuous, so the fixed point is exactly zero
            # below the start-up gradient (no steady creep, no floor).
            mobility_value = (1.0 - omega) * mobility_old + omega * mobility_new
            mobility_effective.setValue(mobility_value)
            state["mobility"].setValue(mobility_value)
            state["yield_factor"].setValue(porous_fields["activation"])
            state["effective_grad_mag_last"] = porous_fields["effective_grad_mag"]
            state["grad_p_crit_last"] = porous_fields["grad_p_crit"]
            state["apparent_viscosity_last"] = porous_fields["apparent_viscosity"]
            storage.setValue(params["reference_storage"])
            if get_pressure_coeff_form(params) == "face":
                update_threshold_face_mobility(state)
            else:
                state["mobility_face"] = None
            return

        # Legacy tanh activation path (porous_bingham_activation == "tanh").
        porous_fields = compute_porous_bingham_fields(state)
        # mobility_effective = permeability / apparent_viscosity
        # with units [m^2 / (Pa·s)].
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


def _build_gravity_face(state):
    """
    Rank-1 face body force b = -rho*g_vec with the gravity flux zeroed on
    sealed exterior faces (every boundary face except the open injection
    faces). Shared by the pressure solve and the net-inflow diagnostic so
    both see the same gravity flux.
    """
    mesh = state["mesh"]
    params = get_slurry_parameters(state)
    rho = float(params.get("slurry_density", 2000.0))
    gravity_y = float(params.get("gravity_y", -9.81))
    num_faces = mesh.numberOfFaces
    body_force = np.zeros((2, num_faces), dtype=float)
    body_force[1, :] = -rho * gravity_y
    exterior = np.asarray(mesh.exteriorFaces.value, dtype=bool)
    open_faces = state.get("open_pressure_faces")
    if open_faces is not None:
        open_mask = np.asarray(open_faces.value, dtype=bool)
    else:
        open_mask = np.zeros(num_faces, dtype=bool)
    sealed = exterior & np.logical_not(open_mask)
    body_force[1, sealed] = 0.0
    return FaceVariable(mesh=mesh, rank=1, value=body_force)


def _compute_net_inflow_from_flux(state):
    pressure = state["pressure"]
    mobility_effective = state["mobility_effective"]
    grad_x = _to_array(pressure.grad()[0])
    grad_y = _to_array(pressure.grad()[1])
    params = get_slurry_parameters(state)
    rho = float(params.get("slurry_density", 2000.0))
    gravity_y = float(params.get("gravity_y", -9.81))
    # Flux: q = -M_eff * (grad(p) - rho*g_vec), g_vec=(0, gravity_y), gravity_y<0.
    qx = -_to_array(mobility_effective.value) * grad_x
    qy = -_to_array(mobility_effective.value) * (grad_y - rho * gravity_y)

    # Preferred path: true net inflow from FiPy divergence if supported.
    try:
        # q = -M_eff*(grad(p) - rho*g_vec), so net inflow = -div(q)
        # = div(M_f*faceGrad(p)) + div(M_f * b), with the same face mobility
        # and sealed-wall gravity face as the pressure solve.
        gravity_face = _build_gravity_face(state)
        mobility_face = state.get("mobility_face")
        if mobility_face is not None:
            net_inflow = _to_array(
                (mobility_face * (pressure.faceGrad() + gravity_face)).divergence
            )
        else:
            net_inflow = _to_array(
                (mobility_effective * pressure.grad()).divergence
                + (mobility_effective.arithmeticFaceValue * gravity_face).divergence
            )
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

    Constraints are installed ONCE (FiPy constrain() is append-only, so
    re-constraining every step would accumulate duplicate constraints). The
    Dirichlet values are fipy Variables kept in state, so later calls only
    refresh the inlet loading factor; the inlet geometry is fixed after the
    first call.
    """
    pressure = state["pressure"]
    mesh = state["mesh"]
    params = get_slurry_parameters(state)
    load_factor = get_inlet_loading_factor(state, params)

    handles = state.get("boundary_condition_handles")
    if handles is not None:
        geo = handles["geometry"]
        handles["spread_value"].setValue(geo["spread_pressure_value"] * load_factor)
        handles["core_value"].setValue(geo["core_pressure_value"] * load_factor)
        return

    fx, fy = mesh.faceCenters()
    geo = get_inlet_geometry(params)

    spread_faces = mesh.facesTop & (fx > geo["spread_x_min"]) & (fx < geo["spread_x_max"])
    core_faces = mesh.facesTop & (fx > geo["core_x_min"]) & (fx < geo["core_x_max"])

    spread_value = Variable(value=geo["spread_pressure_value"] * load_factor)
    core_value = Variable(value=geo["core_pressure_value"] * load_factor)
    pressure.constrain(spread_value, spread_faces)
    pressure.constrain(core_value, core_faces)

    pressure.grad.constrain(0.0, mesh.facesLeft)
    pressure.grad.constrain(0.0, mesh.facesRight)
    pressure.grad.constrain(0.0, mesh.facesBottom)

    # Open (Dirichlet injection) faces; everything else on the boundary is a
    # sealed no-flow wall. Used to zero gravity flux through sealed walls.
    state["open_pressure_faces"] = spread_faces | core_faces
    state["boundary_condition_handles"] = {
        "geometry": geo,
        "spread_value": spread_value,
        "core_value": core_value,
    }


def get_saturation_transport_enabled(state):
    """
    True when the variable-saturation fill transport is active. Requires the
    threshold porous-Bingham mode with the face-built coefficient (the fill
    fluxes must be the matrix fluxes; see docs/saturation_design.md).
    """
    params = get_slurry_parameters(state)
    if not params.get("enable_saturation_transport", False):
        return False
    if get_rheology_model(params) != "porous_bingham":
        raise ValueError(
            "enable_saturation_transport requires rheology_model='porous_bingham'"
        )
    if get_porous_bingham_activation(params) != "threshold":
        raise ValueError(
            "enable_saturation_transport requires porous_bingham_activation='threshold'"
        )
    if get_pressure_coeff_form(params) != "face":
        raise ValueError(
            "enable_saturation_transport requires pressure_coeff_form='face'"
        )
    return True


def _update_fill_mask(state):
    """
    Freeze the active/filling split for this step: cells with
    S < saturation_active_threshold are 'filling' (pinned at air pressure ~0
    by the penalty source); the rest are active pressure unknowns.
    """
    params = get_slurry_parameters(state)
    threshold = float(params.get("saturation_active_threshold", 1.0 - 1.0e-3))
    s = np.clip(_to_array(state["saturation"]), 0.0, 1.0)
    empty = s < threshold
    state["fill_empty_mask"] = empty

    penalty_var = state.get("fill_penalty_var")
    if penalty_var is None:
        penalty_var = CellVariable(mesh=state["mesh"], value=0.0)
        state["fill_penalty_var"] = penalty_var
    penalty_var.setValue(np.where(empty, 1.0, 0.0))
    state["fill_penalty_active"] = bool(np.any(empty))
    return empty


def _fill_penalty_beta(state, dt):
    """Penalty magnitude [1/(Pa*s)]: must dominate both the diffusion row
    scale M/dx^2 and the storage row scale c/dt."""
    params = get_slurry_parameters(state)
    factor = max(float(params.get("fill_penalty_factor", 1.0e6)), 1.0)
    mobility_face = state.get("mobility_face")
    if mobility_face is not None:
        m_max = float(np.max(_to_array(mobility_face)))
    else:
        m_max = float(np.max(_to_array(state["mobility_effective"])))
    m_max = max(m_max, 1.0e-30)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    dx2 = max(float(np.min(vols)), 1.0e-30)
    c = float(params.get("reference_storage", 1.0e-7))
    return factor * (m_max / dx2 + c / max(float(dt), 1.0e-30))


def compute_total_face_flux(state):
    """
    Explicit total face flux q_f = -M_f*(faceGrad(p) + b), with the SAME face
    mobility and sealed-wall gravity face as the pressure matrix (discrete
    consistency). Returns (div_q [1/s], q_normal [m/s], face_areas [m]).
    """
    mesh = state["mesh"]
    pressure = state["pressure"]
    mobility_face = state["mobility_face"]
    gravity_face = _build_gravity_face(state)
    q_face = -(mobility_face * (pressure.faceGrad() + gravity_face))
    div_q = _to_array(q_face.divergence)
    qx = _to_array(q_face[0])
    qy = _to_array(q_face[1])
    normals = np.asarray(mesh.faceNormals)
    q_normal = qx * normals[0] + qy * normals[1]
    areas = np.asarray(mesh._faceAreas, dtype=float)
    return div_q, q_normal, areas


def _solve_pressure_once(state, dt):
    pressure = state["pressure"]
    storage = state["storage"]
    mobility_effective = state["mobility_effective"]

    # Gravity body force as a rank-1 face vector b = -rho*g_vec, g_vec=(0,gravity_y).
    # Sign matches Darcy flux q = -M*(grad(p) - rho*g_vec). For the TOTAL flux to
    # vanish at sealed no-flow walls, the gravity flux is zeroed on every exterior
    # face except the open injection faces; otherwise gravity leaks through sealed
    # walls and doubles the steady gradient.
    gravity_face = _build_gravity_face(state)

    # Face-built coefficient (threshold mode default) or legacy cell
    # coefficient; the gravity source must use the SAME face mobility as the
    # diffusion operator for the total flux to be consistent.
    mobility_face = state.get("mobility_face")
    if mobility_face is not None:
        diff_coeff = mobility_face
        face_mob = mobility_face
    else:
        diff_coeff = mobility_effective
        face_mob = mobility_effective.arithmeticFaceValue
    grav_source = (face_mob * gravity_face).divergence

    rhs = DiffusionTerm(coeff=diff_coeff) + grav_source
    # Fill closure: cells still filling (S < threshold) are pinned at air
    # pressure ~0 by a strong implicit sink; their inflow is credited to the
    # saturation update instead of the pressure storage.
    if state.get("fill_penalty_active", False):
        beta = _fill_penalty_beta(state, dt)
        rhs = rhs + ImplicitSourceTerm(coeff=-beta * state["fill_penalty_var"])

    eq = TransientTerm(coeff=storage) == rhs
    eq.solve(var=pressure, dt=dt)


def solve_pressure_step(state, dt=0.01):
    """
    Run the (possibly nonlinear) pressure solve for one step.

    Assumes boundary conditions are already installed on state["pressure"]
    (apply_boundary_conditions or test-specific constraints). Records
    picard_iterations_last / picard_residual_last / picard_history_last
    (list of max|dp| per Picard iteration) in state.

    Convergence: picard_tol is RELATIVE to the first-iteration residual of
    this step (default 1e-4), so the criterion is pressure-scale invariant
    (an absolute Pa tolerance was unreachable at engineering MPa scales).
    """
    pressure = state["pressure"]
    params = get_slurry_parameters(state)
    rheology_model = get_rheology_model(params)

    if rheology_model in ("gated_bingham", "porous_bingham"):
        max_iters = int(params.get("picard_max_iters", 20))
        if max_iters < 1:
            max_iters = 1
        tol_rel = float(params.get("picard_tol", 1.0e-4))
        if tol_rel <= 0.0:
            tol_rel = 1.0e-4

        # Transient handling of the Picard iteration:
        #   "chained" (legacy): each iteration treats the PREVIOUS ITERATE as
        #     the old state (FiPy reads the variable at assembly time), so a
        #     step advances up to max_iters*dt of pseudo-time -- fast steady
        #     marches, but c*dp/dt = -div q does not hold per step, and near
        #     the yield margin the head-refill/tail-drain ratchet lets a
        #     pinned-front column creep past the stagnation length.
        #   "backward_euler": every iteration re-solves the SAME step from
        #     the step's initial pressure; transported volume per step is
        #     c-storage-bounded, so the fill front overrun is structurally
        #     capped (~1 cell). Required by the fill-transport cases.
        be_mode = params.get("picard_transient_mode", "chained") == "backward_euler"
        pressure_start = np.array(pressure.value, copy=True)
        previous_pressure = np.array(pressure.value, copy=True)
        picard_residual = 0.0
        history = []
        initial_residual = None
        it = 0
        for it in range(max_iters):
            if rheology_model == "gated_bingham":
                grad_mag = compute_pressure_gradient_magnitude(state)
                yield_factor = compute_yield_factor(state, grad_mag)
                update_effective_mobility(state, grad_mag=grad_mag, yield_factor=yield_factor)
            else:
                update_effective_mobility(state)
            if be_mode:
                pressure.setValue(pressure_start)
            _solve_pressure_once(state, dt=dt)

            picard_residual = float(np.max(np.abs(pressure.value - previous_pressure)))
            previous_pressure = np.array(pressure.value, copy=True)
            history.append(picard_residual)
            if initial_residual is None:
                initial_residual = picard_residual
            if picard_residual <= tol_rel * initial_residual:
                break
        state["picard_iterations_last"] = it + 1
        state["picard_residual_last"] = picard_residual
        state["picard_history_last"] = history
        # Static-yield hysteresis: latch faces that ended the step at/below
        # the start-up gradient (threshold mode only; no-op otherwise).
        _update_yield_latch(state)
    else:
        # Linear baseline.
        update_effective_mobility(state)
        _solve_pressure_once(state, dt=dt)
        state["picard_iterations_last"] = 1
        state["picard_residual_last"] = 0.0
        state["picard_history_last"] = [0.0]


def _update_yield_latch(state):
    """
    Static-yield hysteresis bookkeeping for the FILL mode, evaluated once
    per step from the converged step-end state: LATCH faces whose driving
    gradient is at/below lambda. Unlatching happens per-iteration inside
    update_threshold_face_mobility when the iterate gradient exceeds
    (1+h)*lambda -- whole-step latching was tried and forces a cross-step
    period-2 at the activation seam (a cell's supply and drain faces latch
    alternately and the intra-step flux balance point is never reachable).
    """
    params = get_slurry_parameters(state)
    if not state.get("fill_penalty_active", False):
        return
    if not params.get("enable_yield_latch", True):
        return
    if get_rheology_model(params) != "porous_bingham":
        return
    if get_porous_bingham_activation(params) != "threshold":
        return
    if get_pressure_coeff_form(params) != "face":
        return
    mesh = state["mesh"]
    pressure = state["pressure"]
    yield_stress = float(params.get("yield_stress", 50.0))
    k_face = np.maximum(_to_array(state["permeability"].harmonicFaceValue), 1.0e-30)
    phi_face = np.clip(
        _to_array(state["porosity"].arithmeticFaceValue), 1.0e-6, 1.0 - 1.0e-6
    )
    lam_face = 2.0 * yield_stress / np.maximum(
        np.sqrt(8.0 * k_face / phi_face), 1.0e-20
    )
    gravity_face = _build_gravity_face(state)
    grad_face = pressure.faceGrad() + gravity_face
    gx = _to_array(grad_face[0])
    gy = _to_array(grad_face[1])
    grad_mag = np.sqrt(np.maximum(gx * gx + gy * gy, 0.0))
    latched = state.get("face_yield_latched")
    if latched is None:
        latched = np.zeros(mesh.numberOfFaces, dtype=bool)
    state["face_yield_latched"] = latched | (grad_mag <= lam_face)


def _ensure_saturation_ledger(state):
    """Snapshot the initial S and p fields the first time transport runs."""
    if state.get("saturation_initial_for_ledger") is None:
        state["saturation_initial_for_ledger"] = np.array(
            _to_array(state["saturation"]), copy=True
        )
        state["pressure_initial_for_ledger"] = np.array(
            _to_array(state["pressure"]), copy=True
        )
        state.setdefault("injected_volume_total", 0.0)
        state.setdefault("clipped_volume_total", 0.0)
        state.setdefault("compressed_volume_total", 0.0)


def _fill_cfl_dt(state, div_q):
    """
    Fill CFL: dt such that no filling cell gains more than
    dt_cfl * (1 - S) of saturation this step (structural no-overshoot,
    hence clipped_volume ~ 0). Returns a large number when nothing fills.
    """
    params = get_slurry_parameters(state)
    cfl = float(params.get("dt_cfl", 0.5))
    if (cfl <= 0.0) or (cfl > 1.0):
        cfl = 0.5
    empty = state.get("fill_empty_mask")
    if empty is None or not np.any(empty):
        return 1.0e30
    s = np.clip(_to_array(state["saturation"]), 0.0, 1.0)
    porosity = np.clip(_to_array(state["porosity"]), 1.0e-6, 1.0)
    inflow_rate = np.maximum(-np.asarray(div_q, dtype=float), 0.0)  # [1/s]
    cand = empty & (inflow_rate > 1.0e-30)
    if not np.any(cand):
        return 1.0e30
    gap = np.maximum(1.0 - s, 0.0)
    dt_per_cell = gap[cand] * porosity[cand] / inflow_rate[cand]
    return cfl * float(np.min(dt_per_cell))


def compute_adaptive_dt(state, dt_cap):
    """
    dt = min(dt_cap, dt_p, dt_S):
      dt_p = theta * c * L_c^2 / (2 M_wet)  (front time constant, round 2;
             toggleable via enable_front_dt_constraint)
      dt_S = fill CFL evaluated with the LAGGED fluxes of the previous step
             (the in-step re-solve in solve_transport_step is the hard guard).
    """
    params = get_slurry_parameters(state)
    dt = float(dt_cap)
    if params.get("enable_front_dt_constraint", True):
        theta = max(float(params.get("dt_front_theta", 0.01)), 1.0e-6)
        c = float(params.get("reference_storage", 1.0e-7))
        length = float(params.get("front_char_length", 0.0))
        if length <= 0.0:
            x = _to_array(state["x"])
            y = _to_array(state["y"])
            length = max(
                float(np.max(x)) - float(np.min(x)),
                float(np.max(y)) - float(np.min(y)),
                1.0e-12,
            )
        m_wet = max(float(np.max(_to_array(state["mobility_effective"]))), 1.0e-30)
        dt = min(dt, theta * c * length * length / (2.0 * m_wet))
    last_div_q = state.get("last_div_q")
    if last_div_q is not None:
        dt = min(dt, _fill_cfl_dt(state, last_div_q))
    return max(dt, 1.0e-12)


def update_saturation_from_flux(state, dt, div_q, q_normal, face_areas):
    """
    Explicit fill update + conservation ledger.

    Filling cells gain dt * (net inflow)/(n V); active cells keep their S
    (their net inflow is absorbed by the c*dp/dt storage, which the ledger
    books as V_comp). Clip residue (structurally ~0 thanks to the fill CFL)
    is accumulated into clipped_volume_total.
    """
    mesh = state["mesh"]
    saturation = state["saturation"]
    empty = state["fill_empty_mask"]
    porosity = np.clip(_to_array(state["porosity"]), 1.0e-6, 1.0)
    vols = np.asarray(mesh.cellVolumes, dtype=float)

    s_old = np.clip(_to_array(saturation), 0.0, 1.0)
    inflow_rate = -np.asarray(div_q, dtype=float)  # [1/s], >0 = net inflow
    raw = s_old + np.where(empty, dt * inflow_rate / porosity, 0.0)
    s_new = np.clip(raw, 0.0, 1.0)
    saturation.setValue(s_new)

    clipped_step = float(np.sum((raw - s_new) * porosity * vols))
    state["clipped_volume_total"] = (
        state.get("clipped_volume_total", 0.0) + clipped_step
    )

    # Active-cell absorption booked from the SAME explicit flux field (the
    # chained-iterate Picard does not satisfy c*dp/dt = -div q exactly, so
    # booking c*dp here would leak the Picard transient slip into the
    # ledger; with flux-field booking the balance is Gauss-exact).
    active = np.logical_not(empty)
    v_comp_step = float(np.sum(np.where(active, inflow_rate, 0.0) * vols)) * float(dt)
    state["compressed_volume_total"] = (
        state.get("compressed_volume_total", 0.0) + v_comp_step
    )

    # Net boundary inflow of this step (exterior normals point outward).
    exterior = np.asarray(mesh.exteriorFaces.value, dtype=bool)
    v_in_step = -float(
        np.sum(q_normal[exterior] * face_areas[exterior])
    ) * float(dt)
    state["injected_volume_total"] = (
        state.get("injected_volume_total", 0.0) + v_in_step
    )
    state["last_div_q"] = np.asarray(div_q, dtype=float)
    return s_new


def conservation_report(state):
    """
    Global ledger: V_in (net boundary inflow) vs
    dV_store (sum n*dS*V) + V_comp (active-cell absorption, booked from the
    same explicit flux field) + clipped_volume. All four terms come from one
    flux field, so the drift is Gauss-exact (float noise only) and catches
    any wiring inconsistency in the fill update.

    v_comp_pressure (sum c*dp*V) is reported as a separate physical
    cross-check; its gap to v_comp measures the Picard transient slip of the
    chained-iterate pressure solve (diagnostic, not asserted).
    """
    mesh = state["mesh"]
    params = get_slurry_parameters(state)
    vols = np.asarray(mesh.cellVolumes, dtype=float)
    porosity = np.clip(_to_array(state["porosity"]), 1.0e-6, 1.0)

    s = np.clip(_to_array(state["saturation"]), 0.0, 1.0)
    s0 = state.get("saturation_initial_for_ledger")
    if s0 is None:
        s0 = np.zeros_like(s)
    v_store = float(np.sum(porosity * (s - s0) * vols))

    p = _to_array(state["pressure"])
    p0 = state.get("pressure_initial_for_ledger")
    if p0 is None:
        p0 = np.zeros_like(p)
    c = float(params.get("reference_storage", 1.0e-7))
    v_comp_pressure = float(np.sum(c * (p - p0) * vols))

    v_in = float(state.get("injected_volume_total", 0.0))
    v_comp = float(state.get("compressed_volume_total", 0.0))
    v_clip = float(state.get("clipped_volume_total", 0.0))
    drift = v_in - v_store - v_comp - v_clip
    rel = abs(drift) / max(abs(v_in), 1.0e-30)
    return {
        "v_in": v_in,
        "v_store": v_store,
        "v_comp": v_comp,
        "v_comp_pressure": v_comp_pressure,
        "v_clip": v_clip,
        "drift": drift,
        "drift_rel": rel,
    }


def solve_transport_step(state, dt_cap=0.01):
    """
    One IMPES fill step (boundary conditions must already be installed):

      1. freeze the active/filling split from the current S,
      2. dt = min(dt_cap, dt_p, lagged fill CFL),
      3. implicit pressure solve (Picard); filling cells penalized to ~0,
      4. explicit total face flux; if it violates the fill CFL, restore the
         pressure and re-solve with the compliant dt (hard no-overshoot
         guard, keeps clipped_volume at zero),
      5. explicit S update + conservation ledger.

    Returns the dt actually used.
    """
    if not get_saturation_transport_enabled(state):
        solve_pressure_step(state, dt=dt_cap)
        return dt_cap

    pressure = state["pressure"]
    _ensure_saturation_ledger(state)
    _update_fill_mask(state)
    dt = compute_adaptive_dt(state, dt_cap)

    pressure_before = np.array(pressure.value, copy=True)
    div_q = None
    q_normal = None
    face_areas = None
    for _attempt in range(3):
        solve_pressure_step(state, dt=dt)
        div_q, q_normal, face_areas = compute_total_face_flux(state)
        dt_required = _fill_cfl_dt(state, div_q)
        if dt <= dt_required * (1.0 + 1.0e-12):
            break
        pressure.setValue(pressure_before)
        dt = max(dt_required * 0.9, 1.0e-12)

    update_saturation_from_flux(state, dt, div_q, q_normal, face_areas)
    state["dt_last"] = dt
    return dt


def solve_slurry_step(state, dt=0.01):
    """
    Solve one slurry-transport step.

    Default path stays compatible with the current linear baseline.
    Bingham-like yield control is activated only when enable_bingham_yield=True.
    With enable_saturation_transport the step runs the IMPES fill transport
    and dt acts as the adaptive-dt cap. dt is interpreted in seconds [s].
    """
    pressure = state["pressure"]
    params = get_slurry_parameters(state)
    rheology_model = get_rheology_model(params)
    saturation_on = get_saturation_transport_enabled(state)

    apply_boundary_conditions(state)
    if saturation_on:
        dt_used = solve_transport_step(state, dt_cap=dt)
    else:
        solve_pressure_step(state, dt=dt)
        dt_used = dt

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

    if saturation_on:
        # The saturation update inside solve_transport_step already consumed
        # the fluxes; the legacy filling indicator is not advanced.
        net_inflow = -np.asarray(state["last_div_q"], dtype=float)
    elif params.get("legacy_filling_mode", False):
        # Deprecated pre-saturation filling accumulation (0.1 factor), kept
        # only for PFC visualization continuity.
        net_inflow = update_filling_from_flux(state, dt)
    else:
        net_inflow = _compute_net_inflow_from_flux(state)

    step_idx_raw = state.get("flow_step_index", 0)
    try:
        step_idx = int(step_idx_raw)
    except (TypeError, ValueError):
        step_idx = 0
    state["flow_step_index"] = step_idx + 1

    mobility_effective = state["mobility_effective"]
    rho = float(params.get("slurry_density", 2000.0))
    gravity_y = float(params.get("gravity_y", -9.81))
    # Result-field unit semantics:
    # scalar_pressure [Pa]
    # scalar_grad_mag / scalar_effective_grad_mag / scalar_grad_p_crit [Pa/m]
    # scalar_apparent_viscosity [Pa·s]
    # vector_flux_x / vector_flux_y [m/s]  q = -M_eff*(grad(p)-rho*g_vec)
    if saturation_on:
        filling_out = state["saturation"].value
    else:
        filling_out = state["filling"].value
    result = {
        "scalar_pressure": pressure.value,
        "scalar_saturation": state["saturation"].value,
        "scalar_filling": filling_out,
        "dt_used": dt_used,
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
        "vector_flux_y": _to_array(-mobility_effective * (pressure.grad()[1] - rho * gravity_y)),
    }
    return result
