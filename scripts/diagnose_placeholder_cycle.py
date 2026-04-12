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


def _get_inlet_x_range(params):
    if ("inlet_zone_center_x" in params) and ("inlet_zone_width_x" in params):
        half_width = 0.5 * params["inlet_zone_width_x"]
        return (
            params["inlet_zone_center_x"] - half_width,
            params["inlet_zone_center_x"] + half_width,
        )
    return (
        params["inlet_center_x"] - params["inlet_half_width_x"],
        params["inlet_center_x"] + params["inlet_half_width_x"],
    )


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _report_group_stats(label, group):
    if len(group) == 0:
        print("{0}: no particles".format(label))
        return

    e1_vals = []
    e3_vals = []
    e4_vals = []
    for b in group:
        v1 = _safe_float(_read_extra(b, 1))
        v3 = _safe_float(_read_extra(b, 3))
        v4 = _safe_float(_read_extra(b, 4))
        if v1 is not None:
            e1_vals.append(v1)
        if v3 is not None:
            e3_vals.append(v3)
        if v4 is not None:
            e4_vals.append(v4)

    def _mean(vals):
        if len(vals) == 0:
            return None
        return sum(vals) / float(len(vals))

    m1 = _mean(e1_vals)
    m3 = _mean(e3_vals)
    m4 = _mean(e4_vals)
    print(
        "{0}: n={1}, mean(e1/e3/e4)=({2}, {3}, {4})".format(
            label,
            len(group),
            "NA" if m1 is None else "{0:.4e}".format(m1),
            "NA" if m3 is None else "{0:.4e}".format(m3),
            "NA" if m4 is None else "{0:.4e}".format(m4),
        )
    )


def _print_near_far_comparison(inlet_x_min, inlet_x_max, sample_count):
    plist = list(balls.list())
    if len(plist) == 0:
        print("Near/far comparison skipped: no particles found.")
        return

    near_particles = []
    far_particles = []
    for b in plist:
        bx = b.pos_x()
        if (bx >= inlet_x_min) and (bx <= inlet_x_max):
            near_particles.append(b)
        else:
            far_particles.append(b)

    print("Near-vs-far comparison by inlet x-zone:")
    _report_group_stats("near-inlet", near_particles)
    _report_group_stats("far-from-inlet", far_particles)

    print("Near-inlet sample extras:")
    _print_sample_from_group(near_particles, sample_count)
    print("Far-from-inlet sample extras:")
    _print_sample_from_group(far_particles, sample_count)


def _print_sample_from_group(group, sample_count):
    if len(group) == 0:
        print("  none")
        return
    n = min(sample_count, len(group))
    for idx, b in enumerate(group[:n]):
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
    inlet_x_min, inlet_x_max = _get_inlet_x_range(params)

    print("Inlet-zone x-range: [{0:.6f}, {1:.6f}]".format(inlet_x_min, inlet_x_max))
    _print_field_min_max(result)
    _print_particle_extras(sample_count)
    _print_near_far_comparison(inlet_x_min, inlet_x_max, sample_count)


if __name__ == "__main__":
    main()
