from fipy import CellVariable


def initialize_slurry_variables(mesh):
    state = {
        "pressure": CellVariable(name="pressure", mesh=mesh, value=0.0),
        "mobility": CellVariable(name="mobility", mesh=mesh, value=1.0),
        "storage": CellVariable(name="storage", mesh=mesh, value=1.0),
        "filling": CellVariable(name="filling", mesh=mesh, value=0.0),
        "clogging": CellVariable(name="clogging", mesh=mesh, value=0.0),
    }
    return state