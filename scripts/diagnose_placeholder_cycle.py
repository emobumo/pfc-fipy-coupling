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


def _get_inlet_geometry(params):
    center = params.get("inlet_zone_center_x", params.get("inlet_center_x", 0.0))
    core_width = params.get("inlet_core_width_x", params.get("inlet_zone_width_x", 0.0))
    spread_width = params.get("inlet_spread_width_x", params.get("inlet_zone_width_x", core_width))
    if core_width <= 0.0:
        core_width = 0.5 * spread_width if spread_width > 0.0 else 0.4
    if spread_width <= 0.0:
        spread_width = core_width
    if spread_width <= core_width:
        spread_width = core_width * 1.2 if core_width > 0.0 else 0.48
    min_core_fraction = params.get("inlet_core_min_fraction_of_spread", 0.6)
    if (min_core_fraction <= 0.0) or (min_core_fraction >= 1.0):
        min_core_fraction = 0.6
    core_width = max(core_width, min_core_fraction * spread_width)

    core_half = 0.5 * core_width
    spread_half = 0.5 * spread_width
    core_p = params.get("inlet_pressure_core_value", params.get("inlet_pressure_value", 1.0))
    spread_factor = params.get("inlet_pressure_spread_factor", 0.6)
    if spread_factor <= 0.0:
        spread_factor = 0.6
    if spread_factor >= 1.0:
        spread_factor = 0.8
    spread_p = core_p * spread_factor

    return {
        "core_x_min": center - core_half,
        "core_x_max": center + core_half,
        "spread_x_min": center - spread_half,
        "spread_x_max": center + spread_half,
        "core_pressure_value": core_p,
        "spread_pressure_value": spread_p,
    }


def _get_inlet_loading_factor(state, params):
    start_factor = params.get("inlet_loading_start_factor", 1.0)
    ramp_steps = int(params.get("inlet_loading_ramp_steps", 1))
    step_idx_raw = state.get("flow_step_index", 0)
    try:
        step_idx = int(step_idx_raw)
    except (TypeError, ValueError):
        step_idx = 0
    if start_factor < 0.0:
        start_factor = 0.0
    if start_factor > 1.0:
        start_factor = 1.0
    if ramp_steps <= 1:
        return 1.0
    alpha = float(step_idx) / float(ramp_steps - 1)
    alpha = min(max(alpha, 0.0), 1.0)
    return start_factor + (1.0 - start_factor) * alpha


def _print_structure_init_report(state):
    report = state.get("structure_init_report", {})
    if len(report) == 0:
        print("Structure init report: unavailable")
        return
    print("Structure init report:")
    print(
        "  formula: {0}".format(
            report.get("porosity_to_permeability_formula", "NA")
        )
    )
    print(
        "  porosity range/clipping: [{0:.3f}, {1:.3f}] / [{2:.3f}, {3:.3f}]".format(
            report["porosity_range"][0],
            report["porosity_range"][1],
            report["porosity_clip_range"][0],
            report["porosity_clip_range"][1],
        )
    )
    print(
        "  permeability range/clipping: [{0:.3f}, {1:.3f}] / [{2:.3f}, {3:.3f}]".format(
            report["permeability_range"][0],
            report["permeability_range"][1],
            report["permeability_clip_range"][0],
            report["permeability_clip_range"][1],
        )
    )
    print(
        "  porosity min/max: {0:.6e} / {1:.6e}".format(
            report["porosity_min"], report["porosity_max"]
        )
    )
    print(
        "  permeability min/max: {0:.6e} / {1:.6e}".format(
            report["permeability_min"], report["permeability_max"]
        )
    )
    print(
        "  intrinsic_mobility min/max: {0:.6e} / {1:.6e}".format(
            report["intrinsic_mobility_min"], report["intrinsic_mobility_max"]
        )
    )
    print(
        "  mapping params: exponent={0:.3f}, reference_mobility={1:.3f}".format(
            report["porosity_to_permeability_exponent"],
            report["reference_mobility"],
        )
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


def _print_zone_comparison(geo, sample_count):
    plist = list(balls.list())
    if len(plist) == 0:
        print("Zone comparison skipped: no particles found.")
        return
    xvals = [float(b.pos_x()) for b in plist]
    print("Particle-cloud x-range: [{0:.6f}, {1:.6f}]".format(min(xvals), max(xvals)))

    core_particles = []
    spread_only_particles = []
    far_particles = []
    for b in plist:
        bx = b.pos_x()
        in_core = (bx >= geo["core_x_min"]) and (bx <= geo["core_x_max"])
        in_spread = (bx >= geo["spread_x_min"]) and (bx <= geo["spread_x_max"])
        if in_core:
            core_particles.append(b)
        elif in_spread:
            spread_only_particles.append(b)
        else:
            far_particles.append(b)

    print("Core/spread/far comparison by inlet x-zones:")
    _report_group_stats("core-zone", core_particles)
    _report_group_stats("spread-only-zone", spread_only_particles)
    _report_group_stats("far-from-inlet", far_particles)

    print("Core-zone sample extras:")
    _print_sample_from_group(core_particles, sample_count)
    print("Spread-only-zone sample extras:")
    _print_sample_from_group(spread_only_particles, sample_count)
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
    geo = _get_inlet_geometry(params)
    load_factor = _get_inlet_loading_factor(state, params)

    print(
        "Inlet core x-range: [{0:.6f}, {1:.6f}] (P={2:.6f})".format(
            geo["core_x_min"], geo["core_x_max"], geo["core_pressure_value"]
        )
    )
    print(
        "Inlet spread x-range: [{0:.6f}, {1:.6f}] (P={2:.6f})".format(
            geo["spread_x_min"], geo["spread_x_max"], geo["spread_pressure_value"]
        )
    )
    print("Inlet pressure scale (current step): {0:.6f}".format(load_factor))
    print(
        "Inlet widths (core/spread): {0:.6f} / {1:.6f}".format(
            geo["core_x_max"] - geo["core_x_min"],
            geo["spread_x_max"] - geo["spread_x_min"],
        )
    )
    _print_field_min_max(result)
    _print_structure_init_report(state)
    _print_particle_extras(sample_count)
    _print_zone_comparison(geo, sample_count)


if __name__ == "__main__":
    main()
