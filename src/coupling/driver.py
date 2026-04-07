from src.fipy_adapter.mesh_init import build_mesh
from src.pfc_adapter.particle_writer import (
    write_field_to_particles,
    write_vector_field_to_particles,
)
from src.models.slurry_transport.variables import initialize_slurry_variables


def initialize_problem():
    state = build_mesh(nx=25, ny=15, dx=0.1, dy=0.1)

    slurry_state = initialize_slurry_variables(state["mesh"])
    state.update(slurry_state)

    return state


def solve_one_step(state, dt=0.01):
    pressure = state["pressure"]

    # 这里只是占位，不代表最终浆液控制方程
    pressure.setValue(pressure.value + dt)

    grad_x = pressure.grad()[0].value
    grad_y = pressure.grad()[1].value

    result = {
        "scalar_pressure": pressure.value,
        "vector_flux_x": grad_x,
        "vector_flux_y": grad_y,
    }
    return result


def map_back_to_particles(state, result):
    x = state["x"]
    y = state["y"]

    write_field_to_particles(
        x=x,
        y=y,
        values=result["scalar_pressure"],
        extra_id=1,
    )

    write_vector_field_to_particles(
        x=x,
        y=y,
        vx=result["vector_flux_x"],
        vy=result["vector_flux_y"],
        extra_id=2,
    )


def run_one_cycle(dt=0.01):
    state = initialize_problem()
    result = solve_one_step(state, dt=dt)
    map_back_to_particles(state, result)
    return state, result


if __name__ == "__main__":
    run_one_cycle()