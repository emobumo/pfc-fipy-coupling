from fipy import CellVariable


def build_placeholder_slurry_parameters():
    """
    Placeholder engineering parameters for the current slurry-transport stub.

    These values stay explicit so the coupling loop remains easy to test and
    easy to revise later without committing to a final constitutive model.
    """
    return {
        "reference_mobility": 1.0,
        "reference_storage": 1.0,
        "mobility_clogging_factor": 1.0,  # Legacy placeholder key (currently unused).
        "min_mobility": 1.0e-6,
        # First-version Bingham-like yield gate (disabled in default baseline).
        "enable_bingham_yield": False,
        "yield_gradient_crit": 0.25,
        "yield_gradient_smoothing": 0.05,
        "yield_eps": 1.0e-12,
        "yield_floor_factor": 0.05,
        "fill_accumulation_factor": 0.1,
        "filling_max": 1.0,
        "picard_max_iters": 4,
        "picard_tol": 1.0e-6,
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
        "permeability_min": 0.2,
        "permeability_max": 1.0,
        "permeability_clip_min": 0.2,
        "permeability_clip_max": 1.0,
        "porosity_to_permeability_exponent": 2.0,
        "porosity_to_permeability_formula": "power_normalized_linear_range",
        # Placeholder borehole-injection boundary
        # (localized top influence region as engineering equivalent).
        # Preferred geometry keys:
        #   core footprint + spread footprint on the top boundary.
        # This remains an engineering placeholder, not a final inflow model.
        "inlet_zone_center_x": 1.25,
        "inlet_core_width_x": 0.70,
        "inlet_spread_width_x": 1.00,
        "inlet_core_min_fraction_of_spread": 0.60,
        "inlet_pressure_core_value": 1.0,
        "inlet_pressure_spread_factor": 0.50,
        # Constant-pressure borehole injection baseline:
        # keep pressure scale fixed at 1.0 unless a test explicitly changes it.
        "inlet_loading_start_factor": 1.0,
        "inlet_loading_ramp_steps": 1,
        # Legacy-only aliases retained for compatibility and easy rollback.
        "inlet_zone_width_x": 0.90,
        "inlet_center_x": 1.25,
        "inlet_half_width_x": 0.45,
        "inlet_pressure_value": 1.0,
        # Placeholder filling/clogging/mobility feedback (Step 1):
        # filling is occupied pore-volume fraction, bounded by
        # filling_limit_fraction * initial porosity.
        "filling_limit_fraction": 0.95,
        # Mobility attenuation: intrinsic_mobility * (1 - clogging)^n.
        "mobility_blockage_exponent": 2.0,
        "filling_rate": 0.1,  # Legacy placeholder key (currently unused).
        "clogging_rate": 0.01,  # Legacy placeholder key (currently unused).
    }


def initialize_slurry_variables(mesh):
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
        "slurry_parameters": build_placeholder_slurry_parameters(),
        "structure_initialized_once": False,
        "flow_step_index": 0,
        "structure_init_report": {},
    }
    return state
