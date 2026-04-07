from src.fipy_adapter.mesh_init import build_mesh
from src.pfc_adapter.particle_writer import (
    write_field_to_particles,
    write_vector_field_to_particles,
)
from src.models.slurry_transport.variables import initialize_slurry_variables
from src.models.slurry_transport.equations import solve_slurry_step


def initialize_problem():
    mesh, x, y, fx, fy = build_mesh(nx=25, ny=15, dx=0.1, dy=0.1)
    state = {
        "mesh": mesh,
        "x": x,
        "y": y,
        "fx": fx,
        "fy": fy,
    }
    slurry_state = initialize_slurry_variables(state["mesh"])
    state.update(slurry_state)
    return state


def solve_one_step(state, dt=0.01):
    result = solve_slurry_step(state, dt=dt)
    return result


def map_back_to_particles(state, result):
    x = state["x"]
    y = state["y"]

    # extra(1): pressure-like field
    write_field_to_particles(
        x=x,
        y=y,
        values=result["scalar_pressure"],
        extra_id=1,
    )

    # extra(2): flux vector
    write_vector_field_to_particles(
        x=x,
        y=y,
        vx=result["vector_flux_x"],
        vy=result["vector_flux_y"],
        extra_id=2,
    )

    # extra(3): filling indicator
    write_field_to_particles(
        x=x,
        y=y,
        values=result["scalar_filling"],
        extra_id=3,
    )

    # extra(4): clogging indicator
    write_field_to_particles(
        x=x,
        y=y,
        values=result["scalar_clogging"],
        extra_id=4,
    )


def run_one_cycle(dt=0.01):
    state = initialize_problem()
    result = solve_one_step(state, dt=dt)
    map_back_to_particles(state, result)
    return state, result


if __name__ == "__main__":
    run_one_cycle()
