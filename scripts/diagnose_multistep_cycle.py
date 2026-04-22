from __future__ import print_function

import itasca.ball as balls

from src.coupling import driver as drv


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_extra(ball, slot_id):
    if hasattr(ball, "extra"):
        return ball.extra(slot_id)
    if hasattr(ball, "get_extra"):
        return ball.get_extra(slot_id)
    if hasattr(ball, "extra_get"):
        return ball.extra_get(slot_id)
    raise AttributeError("Cannot read particle extra slot API on ball object")


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
    return {
        "core_x_min": center - core_half,
        "core_x_max": center + core_half,
        "spread_x_min": center - spread_half,
        "spread_x_max": center + spread_half,
    }


def _calc_loading_factor(params, step_idx):
    start_factor = params.get("inlet_loading_start_factor", 1.0)
    ramp_steps = int(params.get("inlet_loading_ramp_steps", 1))

    if start_factor < 0.0:
        start_factor = 0.0
    if start_factor > 1.0:
        start_factor = 1.0
    if ramp_steps <= 1:
        return 1.0

    alpha = float(step_idx) / float(ramp_steps - 1)
    if alpha < 0.0:
        alpha = 0.0
    if alpha > 1.0:
        alpha = 1.0
    return start_factor + (1.0 - start_factor) * alpha


def _group_particle_sets(geo):
    plist = list(balls.list())
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
    return core_particles, spread_only_particles, far_particles


def _mean_extra(group, slot_id):
    vals = []
    for b in group:
        v = _safe_float(_read_extra(b, slot_id))
        if v is not None:
            vals.append(v)
    if len(vals) == 0:
        return None
    return sum(vals) / float(len(vals))


def _fmt_mean(value):
    if value is None:
        return "NA"
    return "{0:.4e}".format(value)


def _print_zone_means(step_idx, geo):
    core, spread_only, far = _group_particle_sets(geo)
    c3 = _mean_extra(core, 3)
    s3 = _mean_extra(spread_only, 3)
    f3 = _mean_extra(far, 3)
    c4 = _mean_extra(core, 4)
    s4 = _mean_extra(spread_only, 4)
    f4 = _mean_extra(far, 4)

    print(
        "[step {0}] zone means e3(core/spread/far)=({1}, {2}, {3})".format(
            step_idx, _fmt_mean(c3), _fmt_mean(s3), _fmt_mean(f3)
        )
    )
    print(
        "[step {0}] zone means e4(core/spread/far)=({1}, {2}, {3})".format(
            step_idx, _fmt_mean(c4), _fmt_mean(s4), _fmt_mean(f4)
        )
    )


def _print_mapping_setup(state):
    report = state.get("structure_init_report", {})
    if len(report) == 0:
        print("Mapping setup: unavailable")
        return
    print(
        "Mapping formula: {0}, exponent={1:.3f}, reference_mobility={2:.3f}".format(
            report.get("porosity_to_permeability_formula", "NA"),
            report["porosity_to_permeability_exponent"],
            report["reference_mobility"],
        )
    )
    print(
        "Porosity range/clipping: [{0:.3f}, {1:.3f}] / [{2:.3f}, {3:.3f}]".format(
            report["porosity_range"][0],
            report["porosity_range"][1],
            report["porosity_clip_range"][0],
            report["porosity_clip_range"][1],
        )
    )
    print(
        "Permeability range/clipping: [{0:.3f}, {1:.3f}] / [{2:.3f}, {3:.3f}]".format(
            report["permeability_range"][0],
            report["permeability_range"][1],
            report["permeability_clip_range"][0],
            report["permeability_clip_range"][1],
        )
    )


def main(num_steps=10, dt=0.01, zone_report_interval=5):
    print("Initialize once, then advance same state for {0} steps.".format(num_steps))
    state = drv.initialize_problem()
    params = state["slurry_parameters"]
    geo = _get_inlet_geometry(params)
    _print_mapping_setup(state)

    for i in range(num_steps):
        loading_factor = _calc_loading_factor(params, i)
        result = drv.solve_one_step(state, dt=dt)
        drv.map_back_to_particles(state, result)

        flow_step_index = state.get("flow_step_index")
        print(
            "[step {0}] flow_step_index={1}, inlet_pressure_scale={2:.6f}".format(
                i, flow_step_index, loading_factor
            )
        )
        print(
            "[step {0}] p(min/max)=({1:.6e}, {2:.6e})".format(
                i, float(result["scalar_pressure"].min()), float(result["scalar_pressure"].max())
            )
        )
        print(
            "[step {0}] f(min/max)=({1:.6e}, {2:.6e})".format(
                i, float(result["scalar_filling"].min()), float(result["scalar_filling"].max())
            )
        )
        print(
            "[step {0}] c(min/max)=({1:.6e}, {2:.6e})".format(
                i, float(result["scalar_clogging"].min()), float(result["scalar_clogging"].max())
            )
        )

        if zone_report_interval > 0 and ((i + 1) % int(zone_report_interval) == 0):
            _print_zone_means(i, geo)


if __name__ == "__main__":
    main()
