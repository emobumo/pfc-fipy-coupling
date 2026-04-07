from src.fipy_adapter.mesh_init import build_mesh
from src.models.slurry_transport.variables import initialize_slurry_variables
from src.models.slurry_transport.equations import solve_slurry_step


def main():
    print("Step 1: build mesh")
    mesh, x, y, fx, fy = build_mesh(nx=10, ny=6, dx=0.2, dy=0.2)
    state = {
        "mesh": mesh,
        "x": x,
        "y": y,
        "fx": fx,
        "fy": fy,
    }

    print("Step 2: initialize slurry variables")
    slurry_state = initialize_slurry_variables(state["mesh"])
    state.update(slurry_state)

    required_keys = ["mesh", "x", "y", "fx", "fy",
                     "pressure", "mobility", "storage", "filling", "clogging"]

    print("Step 3: check state keys")
    for key in required_keys:
        if key not in state:
            raise KeyError("Missing key in state: {0}".format(key))
    print("State keys OK")

    print("Step 4: solve one slurry step")
    result = solve_slurry_step(state, dt=0.01)

    required_result_keys = [
        "scalar_pressure",
        "scalar_filling",
        "scalar_clogging",
        "vector_flux_x",
        "vector_flux_y", 
    ]

    print("Step 5: check result keys")
    for key in required_result_keys:
        if key not in result:
            raise KeyError("Missing key in result: {0}".format(key))
    print("Result keys OK")

    print("Step 6: print simple diagnostics")
    print("pressure min/max:", result["scalar_pressure"].min(), result["scalar_pressure"].max())
    print("filling min/max:", result["scalar_filling"].min(), result["scalar_filling"].max())
    print("clogging min/max:", result["scalar_clogging"].min(), result["scalar_clogging"].max())

    print("Smoke test passed.")


if __name__ == "__main__":
    main()
