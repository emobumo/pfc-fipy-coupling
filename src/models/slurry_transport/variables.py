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
        "mobility_clogging_factor": 1.0,
        "min_mobility": 1.0e-6,
        # Placeholder top pouring boundary (localized on top surface).
        # Preferred geometry keys:
        #   core footprint + spread footprint on the top boundary.
        # This remains an engineering placeholder, not a final inflow model.
        "inlet_zone_center_x": 1.25,
        "inlet_core_width_x": 0.40,
        "inlet_spread_width_x": 1.00,
        "inlet_pressure_core_value": 1.0,
        "inlet_pressure_spread_factor": 0.50,
        # Legacy aliases retained for compatibility and easy rollback.
        "inlet_zone_width_x": 0.90,
        "inlet_center_x": 1.25,
        "inlet_half_width_x": 0.45,
        "inlet_pressure_value": 1.0,
        "filling_rate": 0.1,
        "clogging_rate": 0.01,
    }


def initialize_slurry_variables(mesh):
    state = {
        "pressure": CellVariable(name="pressure", mesh=mesh, value=0.0),
        "mobility": CellVariable(name="mobility", mesh=mesh, value=1.0),
        "storage": CellVariable(name="storage", mesh=mesh, value=1.0),
        "filling": CellVariable(name="filling", mesh=mesh, value=0.0),
        "clogging": CellVariable(name="clogging", mesh=mesh, value=0.0),
        "slurry_parameters": build_placeholder_slurry_parameters(),
    }
    return state
