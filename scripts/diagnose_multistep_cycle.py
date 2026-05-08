from __future__ import print_function

from src.coupling import driver as drv
from diag_utils import (
    _safe_float,
    _read_extra,
    _get_inlet_geometry,
    _group_particle_sets,
    _mean_extra,
    _get_vector2_components,
)


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
