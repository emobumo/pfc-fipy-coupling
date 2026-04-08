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
        "inlet_pressure": 1.0,
        "inlet_x_min": 0.8,
        "inlet_x_max": 1.7,
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
