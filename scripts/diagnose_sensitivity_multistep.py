from __future__ import print_function

import copy
import sys

import itasca.ball as balls

from src.coupling import driver as drv
from src.coupling.porosity_to_permeability import initialize_structure_mobility_once


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


def _fmt(value):
    if value is None:
        return "NA"
    return "{0:.4e}".format(value)


def _apply_overrides(params, overrides):
    for key, val in overrides.items():
        params[key] = val


def _log(message):
    print(message)
    sys.stdout.flush()


def _run_case(case_name, group_name, overrides, num_steps, dt):
    state = drv.initialize_problem()
    params = state["slurry_parameters"]

    _log("[case start] [{0}] {1}".format(group_name, case_name))

    # Apply case parameters.
    _apply_overrides(params, overrides)

    # Mapping-group changes must be re-applied to one-time structure init.
    if group_name == "mapping":
        state["structure_initialized_once"] = False
        state["structure_init_report"] = {}
        initialize_structure_mobility_once(state)

    result = None
    for step_i in range(num_steps):
        _log("  step {0}/{1}".format(step_i + 1, num_steps))
        result = drv.solve_one_step(state, dt=dt)
        # Keep runtime cheaper: particle mapping is only needed for final zone stats.
        if step_i == (num_steps - 1):
            drv.map_back_to_particles(state, result)

    geo = _get_inlet_geometry(params)
    core, spread_only, far = _group_particle_sets(geo)

    summary = {
        "case": case_name,
        "group": group_name,
        "final_pressure_max": float(result["scalar_pressure"].max()),
        "final_filling_max": float(result["scalar_filling"].max()),
        "final_clogging_max": float(result["scalar_clogging"].max()),
        "final_mobility_min": float(state["mobility"].value.min()),
        "final_mobility_max": float(state["mobility"].value.max()),
        "core_e3": _mean_extra(core, 3),
        "spread_e3": _mean_extra(spread_only, 3),
        "far_e3": _mean_extra(far, 3),
        "core_e4": _mean_extra(core, 4),
        "spread_e4": _mean_extra(spread_only, 4),
        "far_e4": _mean_extra(far, 4),
    }
    _log("[case end]   [{0}] {1}".format(group_name, case_name))
    return summary


def _print_summary(summary):
    _log("")
    _log("[{0}] {1}".format(summary["group"], summary["case"]))
    _log(
        "  final max p/f/c = {0:.4e}, {1:.4e}, {2:.4e}".format(
            summary["final_pressure_max"],
            summary["final_filling_max"],
            summary["final_clogging_max"],
        )
    )
    _log(
        "  final mobility min/max = {0:.4e}, {1:.4e}".format(
            summary["final_mobility_min"],
            summary["final_mobility_max"],
        )
    )
    _log(
        "  mean e3 core/spread/far = {0}, {1}, {2}".format(
            _fmt(summary["core_e3"]),
            _fmt(summary["spread_e3"]),
            _fmt(summary["far_e3"]),
        )
    )
    _log(
        "  mean e4 core/spread/far = {0}, {1}, {2}".format(
            _fmt(summary["core_e4"]),
            _fmt(summary["spread_e4"]),
            _fmt(summary["far_e4"]),
        )
    )


def _build_cases(base_params):
    # Focused mobility-feedback sensitivity check only.
    return [
        {
            "name": "baseline",
            "group": "baseline",
            "overrides": {},
        },
        {
            "name": "flow_stronger_mobility_reduction",
            "group": "flow",
            "overrides": {
                "enable_clogging_feedback": True,
                "mobility_blockage_exponent": base_params.get("mobility_blockage_exponent", 2.0) * 2.5,
            },
        },
    ]


def main(num_steps=30, dt=0.01):
    base_state = drv.initialize_problem()
    base_params = copy.deepcopy(base_state["slurry_parameters"])
    cases = _build_cases(base_params)

    _log("[script start] sensitivity multistep diagnostics")
    _log("Run sensitivity over {0} cases, {1} steps each.".format(len(cases), num_steps))
    for case in cases:
        summary = _run_case(
            case_name=case["name"],
            group_name=case["group"],
            overrides=case["overrides"],
            num_steps=num_steps,
            dt=dt,
        )
        _print_summary(summary)
    _log("[script end] completed all cases")


if __name__ == "__main__":
    main()
