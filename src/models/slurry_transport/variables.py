# -*- coding: utf-8 -*-
from fipy import CellVariable


def build_placeholder_slurry_parameters():
    """
    Placeholder engineering parameters for the current slurry-transport stub.

    These values stay explicit so the coupling loop remains easy to test and
    easy to revise later without committing to a final constitutive model.
    """
    return {
        # Provisional engineering SI defaults.
        # reference_mobility [m^2 / (Pa·s)] at permeability_max.
        "reference_mobility": 2.0e-4,
        # reference_storage [Pa^-1].
        "reference_storage": 1.0e-7,
        # min_mobility [m^2 / (Pa·s)] lower numerical bound.
        "min_mobility": 1.0e-16,
        # Rheology switch:
        # - linear: backward-compatible baseline
        # - gated_bingham: legacy empirical gradient gate
        # - porous_bingham: first-stage physics-based porous Bingham mode
        "rheology_model": "linear",
        # First-version Bingham-like yield gate (disabled in default baseline).
        "enable_bingham_yield": False,
        "yield_gradient_crit": 0.25,
        "yield_gradient_smoothing": 0.05,
        # yield_eps is retained for legacy gated_bingham compatibility.
        "yield_eps": 1.0e-12,
        "yield_floor_factor": 0.05,
        # First-stage porous Bingham slurry parameters (neutral placeholder
        # defaults; the engineering case overrides these).
        # These are used only when rheology_model == "porous_bingham".
        # slurry_density [kg / m^3]
        "slurry_density": 1500.0,
        # plastic_viscosity [Pa·s]
        "plastic_viscosity": 0.05,
        # yield_stress [Pa]
        "yield_stress": 5.0,
        # gravity_y [m / s^2]
        "gravity_y": -9.81,
        # length_eps_m [m]
        "length_eps_m": 1.0e-9,
        # gradient_eps_pa_per_m [Pa / m]
        "gradient_eps_pa_per_m": 1.0e-6,
        # activation_eps [dimensionless]
        "activation_eps": 1.0e-12,
        # yield_regularization_band [dimensionless multiplier on grad_p_crit]
        "yield_regularization_band": 0.10,
        # mobility_activation_floor [dimensionless] (legacy tanh mode only).
        "mobility_activation_floor": 1.0e-3,
        # porous_bingham activation form:
        #   "threshold" -> mobility = (k/mu_p)*max(0, 1 - lambda/(|gradPhi|+eps)),
        #                  lambda = 2*tau0/sqrt(8k/phi) per cell, zero below
        #                  threshold (default)
        #   "tanh"      -> legacy smooth on/off gate via apparent viscosity
        "porous_bingham_activation": "threshold",
        # yield_truncation_eps_pa_per_m [Pa/m]: regularization added to
        # |grad Phi| in the truncation. 0 -> auto: 1e-6 * p0 / L_domain.
        "yield_truncation_eps_pa_per_m": 0.0,
        # Start-up-gradient truncation form (threshold mode):
        #   "papanastasiou" -> C1 exponential regularization T =
        #                      softplus(m*(1-lambda/|gradPhi|))/m, bounded
        #                      derivative -> Picard converges near the stall
        #                      (default; round 5)
        #   "hard"          -> max(0, 1-lambda/|gradPhi|) (rounds 2-3 ref)
        "yield_truncation_mode": "papanastasiou",
        # Papanastasiou sharpness m (dimensionless). Larger -> closer to the
        # hard kink (less sub-threshold creep, smaller long-time drift, but a
        # stiffer Picard map). 40 is the round-5 scan default.
        "yield_reg_m": 40.0,
        # Pressure-equation diffusion-coefficient discretization (threshold
        # mode): "face" = harmonic(k)*trunc(faceGrad)*upwind k_r FaceVariable
        # (default, required by saturation transport); "cell" = legacy cell
        # coefficient (FiPy arithmetic averaging), comparison/fallback only.
        "pressure_coeff_form": "face",
        # Saturation transport (design doc): k_r = S^a.
        "relperm_exponent": 3.0,
        # Saturation transport master switch (off in the neutral baseline so
        # the legacy linear tests are bit-identical; cases enable it).
        "enable_saturation_transport": False,
        # Fill closure (design doc "实施期修订"): cells with S below this
        # threshold are pinned at air pressure ~0 via a penalty source and
        # fill from the explicit face inflow; at/above it they join the
        # pressure domain.
        "saturation_active_threshold": 1.0 - 1.0e-3,
        # Penalty magnitude multiplier on max(M/dx^2, c/dt).
        "fill_penalty_factor": 1.0e6,
        # Adaptive dt: fill CFL number and the front-time-constant rule
        # dt_p = theta * c * L_c^2 / (2 M_wet) from round 2.
        "dt_cfl": 0.5,
        "dt_front_theta": 0.01,
        "enable_front_dt_constraint": True,
        # Characteristic front length for dt_p; 0 -> auto (domain max extent).
        "front_char_length": 0.0,
        # Deprecated filling-indicator path (pre-saturation accumulation with
        # the 0.1 factor); kept only for PFC visualization continuity.
        "legacy_filling_mode": False,
        # characteristic_pore_size [m]
        "characteristic_pore_size": 0.05,
        # max_apparent_viscosity [Pa·s]
        "max_apparent_viscosity": 1.0e6,
        # min_apparent_viscosity [Pa·s]
        "min_apparent_viscosity": 1.0e-3,
        # fill_accumulation_factor [dimensionless]
        "fill_accumulation_factor": 0.1,
        # filling_max [dimensionless]
        "filling_max": 1.0,
        "picard_max_iters": 20,
        # Picard tolerance, RELATIVE to the step's first-iteration residual.
        "picard_tol": 1.0e-4,
        # Picard transient handling: "chained" (legacy; previous iterate is
        # the old state -> fast pseudo-steady marches) or "backward_euler"
        # (each iteration re-solves the step from its initial pressure).
        "picard_transient_mode": "chained",
        # Under-relaxation factor for the threshold-mode mobility update.
        # Saturated (S=1 / no fill) modes converge at 0.5. The FILL path uses
        # picard_relaxation_fill: round-5 scans show the yield-margin Picard
        # map has a strong sign-flipping eigenvalue, so omega=0.5 oscillates
        # (does not contract) while omega=0.15 converges and cuts the
        # long-time front creep from ~+11% to ~+2.4% of L_max. This GLOBAL
        # under-relaxation was sufficient; per-face adaptive relaxation was
        # not needed.
        "picard_relaxation": 0.5,
        "picard_relaxation_fill": 0.15,
        # Yield latch (round-4 band-aid for the hard-truncation jitter). With
        # the round-5 fill relaxation it is nearly inert (omega=0.2 on/off
        # differ by 0.3% drift), so it is OFF by default; kept as an option.
        "enable_yield_latch": False,
        "yield_hysteresis_band": 0.05,
        # Gate for first-version baseline behavior.
        # Keep clogging feedback disabled unless explicitly enabled in a test/case.
        "enable_clogging_feedback": False,
        # One-time porosity -> permeability placeholder mapping parameters.
        "porosity_default": 0.35,
        "porosity_min": 0.05,
        "porosity_max": 0.95,
        "fallback_particle_area_ratio": 0.15,
        "porosity_clip_min": 0.05,
        "porosity_clip_max": 0.95,
        # permeability_* [m^2] provisional SI permeability bounds.
        # permeability_min/max are the OUTPUT range of the legacy power-law only.
        "permeability_min": 1.0e-7,
        "permeability_max": 1.0e-5,
        # Clip applied to the final permeability (both formulas). Kept wide so
        # Kozeny-Carman shows its true range; tighten once the realistic
        # waste-rock permeability magnitude is confirmed.
        "permeability_clip_min": 1.0e-10,
        "permeability_clip_max": 1.0e-1,
        "porosity_to_permeability_exponent": 2.0,
        # Active porosity->permeability law:
        #   "kozeny_carman"                -> k = d^2/C * phi^3/(1-phi)^2 (per-cell d)
        #   "power_normalized_linear_range"-> legacy placeholder power law
        "porosity_to_permeability_formula": "kozeny_carman",
        # Kozeny-Carman constant C (~180 for packed spheres).
        "kozeny_carman_constant": 180.0,
        # Fallback grain diameter [m] for cells with no balls / no PFC.
        "default_particle_diameter": 0.25,
        # Borehole-injection placeholder (neutral default; the engineering case
        # overrides center/width/pressure). Localized top influence region.
        # inlet_zone_center_x [m]
        "inlet_zone_center_x": 1.25,
        # inlet_core_width_x [m]
        "inlet_core_width_x": 0.70,
        # inlet_spread_width_x [m]
        "inlet_spread_width_x": 1.00,
        # inlet_core_min_fraction_of_spread [dimensionless]
        "inlet_core_min_fraction_of_spread": 0.60,
        # inlet_pressure_core_value [Pa]
        "inlet_pressure_core_value": 2.0e5,
        # inlet_pressure_spread_factor [dimensionless]
        "inlet_pressure_spread_factor": 0.50,
        # Constant-pressure borehole injection baseline:
        # keep pressure scale fixed at 1.0 unless a test explicitly changes it.
        "inlet_loading_start_factor": 1.0,
        "inlet_loading_ramp_steps": 1,
        # Placeholder filling/clogging/mobility feedback (Step 1):
        # filling and porosity are dimensionless.
        # filling is occupied pore-volume fraction, bounded by
        # filling_limit_fraction * initial porosity.
        # filling_limit_fraction [dimensionless]
        "filling_limit_fraction": 0.95,
        # Mobility attenuation: intrinsic_mobility * (1 - clogging)^n.
        "mobility_blockage_exponent": 2.0,
    }


def build_engineering_case_parameters():
    """
    Parameters for the model1 engineering case: top-borehole constant-pressure
    grouting of a cement slurry into a waste-rock pile. Starts from the neutral
    placeholder defaults and overrides the slurry rheology and injection.

    Slurry: 325 slag Portland cement, 55% mass concentration, W/C ~ 0.82,
    Bingham regime (density 1414 kg/m^3, yield stress 4.6 Pa, plastic
    viscosity 0.183 Pa.s). Injection: localized 1 m zone at the top center,
    constant 2 MPa (resolves on the 1.0 m mesh, where a top cell sits at x = 0).
    """
    params = build_placeholder_slurry_parameters()
    # NOTE reference_storage scaling rule (saturation design review): keep c
    # per-case and size it so c*p0 <= 1% * porosity. At p0 = 2 MPa and
    # n ~ 0.3 this requires c <= ~1.5e-9 Pa^-1; the inherited default 1e-7
    # makes c*p0 ~ 0.2 (same order as n) and is ONLY acceptable while the
    # legacy fully-saturated mode is in use. Revisit when this case switches
    # to enable_saturation_transport=True.
    params.update({
        "rheology_model": "porous_bingham",
        # Cement slurry (Bingham).
        "slurry_density": 1414.0,
        "yield_stress": 4.6,
        "plastic_viscosity": 0.183,
        "characteristic_pore_size": 0.10,
        # Borehole injection: 1 m zone at top center, constant 2 MPa.
        "inlet_zone_center_x": 0.0,
        "inlet_core_width_x": 1.0,
        "inlet_spread_width_x": 1.0,
        "inlet_pressure_core_value": 2.0e6,
        "inlet_pressure_spread_factor": 0.90,
    })
    return params


def initialize_slurry_variables(mesh, params=None):
    if params is None:
        params = build_placeholder_slurry_parameters()
    mobility_structural = CellVariable(name="mobility_structural", mesh=mesh, value=1.0)
    mobility_effective = CellVariable(name="mobility_effective", mesh=mesh, value=1.0)
    state = {
        "pressure": CellVariable(name="pressure", mesh=mesh, value=0.0),
        "mobility_structural": mobility_structural,
        "mobility_effective": mobility_effective,
        # Legacy aliases retained for compatibility.
        "intrinsic_mobility": mobility_structural,
        "mobility": mobility_effective,
        "permeability": CellVariable(name="permeability", mesh=mesh, value=1.0),
        "porosity": CellVariable(name="porosity", mesh=mesh, value=0.35),
        "storage": CellVariable(name="storage", mesh=mesh, value=1.0),
        "filling": CellVariable(name="filling", mesh=mesh, value=0.0),
        "clogging": CellVariable(name="clogging", mesh=mesh, value=0.0),
        "yield_factor": CellVariable(name="yield_factor", mesh=mesh, value=1.0),
        # Slurry saturation S (occupied pore fraction); the conserved field
        # of the fill closure. Initially dry; cases/tests may set 1.0 for the
        # fully-saturated degenerate mode.
        "saturation": CellVariable(name="saturation", mesh=mesh, value=0.0),
        "slurry_parameters": params,
        "structure_initialized_once": False,
        "flow_step_index": 0,
        "structure_init_report": {},
        # Conservation ledger (volumes per unit thickness, [m^3/m]).
        "injected_volume_total": 0.0,
        "clipped_volume_total": 0.0,
        "last_div_q": None,
    }
    return state
