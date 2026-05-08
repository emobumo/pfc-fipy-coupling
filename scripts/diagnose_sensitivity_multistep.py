from __future__ import print_function

import copy
import sys

from src.coupling import driver as drv
from src.coupling.porosity_to_permeability import initialize_structure_mobility_once
from diag_utils import (
    _safe_float,
    _read_extra,
    _get_inlet_geometry,
    _group_particle_sets,
    _mean_extra,
    _get_vector2_components,
)


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
