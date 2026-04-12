from __future__ import print_function

import itasca.ball as balls

from src.coupling import driver as drv


def _read_extra(ball, slot_id):
    if hasattr(ball, "extra"):
        return ball.extra(slot_id)
    if hasattr(ball, "get_extra"):
        return ball.get_extra(slot_id)
    if hasattr(ball, "extra_get"):
        return ball.extra_get(slot_id)
    raise AttributeError("Cannot read particle extra slot API on ball object")


def _get_vector2_components(value):
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return value[0], value[1]
    try:
        return value[0], value[1]
    except (TypeError, IndexError, KeyError):
        pass
    if hasattr(value, "x") and hasattr(value, "y"):
        xval = value.x() if callable(value.x) else value.x
        yval = value.y() if callable(value.y) else value.y
        return xval, yval
    return None


def _print_field_min_max(result):
    p = result["scalar_pressure"]
    f = result["scalar_filling"]
    c = result["scalar_clogging"]
    print("pressure min/max: {0:.6e} / {1:.6e}".format(float(p.min()), float(p.max())))
    print("filling  min/max: {0:.6e} / {1:.6e}".format(float(f.min()), float(f.max())))
    print("clogging min/max: {0:.6e} / {1:.6e}".format(float(c.min()), float(c.max())))


def _print_particle_extras(sample_count):
    plist = list(balls.list())
    if len(plist) == 0:
        print("No particles found; skip extra-value report.")
        return

    n = min(sample_count, len(plist))
    print("Representative particle extras (first {0}):".format(n))

    for idx, b in enumerate(plist[:n]):
        e1 = _read_extra(b, 1)
        e2 = _read_extra(b, 2)
        e3 = _read_extra(b, 3)
        e4 = _read_extra(b, 4)
        v2 = _get_vector2_components(e2)

        if v2 is None:
            print(
                "  p#{0}: e1={1}, e2={2}, e3={3}, e4={4}".format(
                    idx, e1, e2, e3, e4
                )
            )
        else:
            print(
                "  p#{0}: e1={1:.4e}, e2=({2:.4e},{3:.4e}), e3={4:.4e}, e4={5:.4e}".format(
                    idx,
                    float(e1),
                    float(v2[0]),
                    float(v2[1]),
                    float(e3),
                    float(e4),
                )
            )


def main(dt=0.01, sample_count=5):
    print("Run one verified coupling cycle ...")
    state, result = drv.run_one_cycle(dt=dt)

    params = state["slurry_parameters"]
    inlet_x_min = params["inlet_center_x"] - params["inlet_half_width_x"]
    inlet_x_max = params["inlet_center_x"] + params["inlet_half_width_x"]

    print("Inlet-zone x-range: [{0:.6f}, {1:.6f}]".format(inlet_x_min, inlet_x_max))
    _print_field_min_max(result)
    _print_particle_extras(sample_count)


if __name__ == "__main__":
    main()
