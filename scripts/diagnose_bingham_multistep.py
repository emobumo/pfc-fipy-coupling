from __future__ import print_function

import copy

import numpy as np

from src.coupling import driver as drv
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


def _fmt_triplet(values):
    return "{0}, {1}, {2}".format(_fmt(values[0]), _fmt(values[1]), _fmt(values[2]))


def _apply_overrides(params, overrides):
    for key, val in overrides.items():
        params[key] = val


def _zone_masks_from_cells(state, geo):
    x = np.asarray(state["x"], dtype=float)
    core = (x >= geo["core_x_min"]) & (x <= geo["core_x_max"])
    spread = (x >= geo["spread_x_min"]) & (x <= geo["spread_x_max"]) & (~core)
    far = ~(core | spread)
    return core, spread, far


def _mean_masked(values, mask):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    if np.sum(mask) == 0:
        return None
    return float(np.mean(values[mask]))


def _zone_means(values, masks):
    return (
        _mean_masked(values, masks[0]),
        _mean_masked(values, masks[1]),
        _mean_masked(values, masks[2]),
    )


def _field_min_max(result, field_name):
    values = np.asarray(result[field_name], dtype=float)
    return float(values.min()), float(values.max())


def _run_case(case_name, overrides, num_steps, dt, active_threshold):
    state = drv.initialize_problem()
    params = state["slurry_parameters"]
    _apply_overrides(params, overrides)

    result = None
    for step_i in range(num_steps):
        result = drv.solve_one_step(state, dt=dt)
        if step_i == (num_steps - 1):
            drv.map_back_to_particles(state, result)

    geo = _get_inlet_geometry(params)
    core, spread_only, far = _group_particle_sets(geo)
    masks = _zone_masks_from_cells(state, geo)

    yield_factor = np.asarray(result["scalar_yield_factor"], dtype=float)
    yield_active_ratio = float(np.mean(yield_factor > active_threshold))
    eff_grad_min, eff_grad_max = _field_min_max(result, "scalar_effective_grad_mag")
    crit_min, crit_max = _field_min_max(result, "scalar_grad_p_crit")
    mu_min, mu_max = _field_min_max(result, "scalar_apparent_viscosity")
    clogging_min, clogging_max = _field_min_max(result, "scalar_clogging")

    return {
        "name": case_name,
        "rheology_model": params.get("rheology_model", "linear"),
        "yield_stress": float(params.get("yield_stress", 0.0)),
        "plastic_viscosity": float(params.get("plastic_viscosity", 0.0)),
        "characteristic_pore_size": float(params.get("characteristic_pore_size", 0.0)),
        "p_min": float(result["scalar_pressure"].min()),
        "p_max": float(result["scalar_pressure"].max()),
        "f_min": float(result["scalar_filling"].min()),
        "f_max": float(result["scalar_filling"].max()),
        "c_min": clogging_min,
        "c_max": clogging_max,
        "m_min": float(result["scalar_mobility_effective"].min()),
        "m_max": float(result["scalar_mobility_effective"].max()),
        "y_min": float(yield_factor.min()),
        "y_max": float(yield_factor.max()),
        "eff_grad_min": eff_grad_min,
        "eff_grad_max": eff_grad_max,
        "crit_min": crit_min,
        "crit_max": crit_max,
        "mu_min": mu_min,
        "mu_max": mu_max,
        "yield_active_ratio": yield_active_ratio,
        "zone_filling": _zone_means(result["scalar_filling"], masks),
        "zone_mobility": _zone_means(result["scalar_mobility_effective"], masks),
        "zone_yield": _zone_means(result["scalar_yield_factor"], masks),
        "zone_eff_grad": _zone_means(result["scalar_effective_grad_mag"], masks),
        "zone_grad_crit": _zone_means(result["scalar_grad_p_crit"], masks),
        "zone_mu_app": _zone_means(result["scalar_apparent_viscosity"], masks),
        "core_e3": _mean_extra(core, 3),
        "spread_e3": _mean_extra(spread_only, 3),
        "far_e3": _mean_extra(far, 3),
        "core_e4": _mean_extra(core, 4),
        "spread_e4": _mean_extra(spread_only, 4),
        "far_e4": _mean_extra(far, 4),
    }


def _print_summary(summary, active_threshold):
    print("")
    print("[{0}]".format(summary["name"]))
    print(
        "  rheology={0}, tau_y={1:.4e} Pa, mu_p={2:.4e} Pa*s, pore={3:.4e} m".format(
            summary["rheology_model"],
            summary["yield_stress"],
            summary["plastic_viscosity"],
            summary["characteristic_pore_size"],
        )
    )
    print("  final pressure min/max = {0:.4e}, {1:.4e}".format(summary["p_min"], summary["p_max"]))
    print("  final filling  min/max = {0:.4e}, {1:.4e}".format(summary["f_min"], summary["f_max"]))
    print("  final clogging min/max = {0:.4e}, {1:.4e}".format(summary["c_min"], summary["c_max"]))
    print("  final mobility min/max = {0:.4e}, {1:.4e}".format(summary["m_min"], summary["m_max"]))
    print("  final yield    min/max = {0:.4e}, {1:.4e}".format(summary["y_min"], summary["y_max"]))
    print(
        "  effective grad min/max = {0:.4e}, {1:.4e} Pa/m".format(
            summary["eff_grad_min"], summary["eff_grad_max"]
        )
    )
    print("  grad_p_crit   min/max = {0:.4e}, {1:.4e} Pa/m".format(summary["crit_min"], summary["crit_max"]))
    print("  mu_app        min/max = {0:.4e}, {1:.4e} Pa*s".format(summary["mu_min"], summary["mu_max"]))
    print(
        "  yield active ratio (>{0:.2f}) = {1:.4f}".format(
            active_threshold, summary["yield_active_ratio"]
        )
    )
    print("  cell filling core/spread/far = {0}".format(_fmt_triplet(summary["zone_filling"])))
    print("  cell mobility core/spread/far = {0}".format(_fmt_triplet(summary["zone_mobility"])))
    print("  cell yield    core/spread/far = {0}".format(_fmt_triplet(summary["zone_yield"])))
    print("  cell eff_grad core/spread/far = {0}".format(_fmt_triplet(summary["zone_eff_grad"])))
    print("  cell crit_grad core/spread/far = {0}".format(_fmt_triplet(summary["zone_grad_crit"])))
    print("  cell mu_app   core/spread/far = {0}".format(_fmt_triplet(summary["zone_mu_app"])))
    print(
        "  mean e3 core/spread/far = {0}, {1}, {2}".format(
            _fmt(summary["core_e3"]),
            _fmt(summary["spread_e3"]),
            _fmt(summary["far_e3"]),
        )
    )
    print(
        "  mean e4 core/spread/far = {0}, {1}, {2}".format(
            _fmt(summary["core_e4"]),
            _fmt(summary["spread_e4"]),
            _fmt(summary["far_e4"]),
        )
    )


def _build_cases(base_params):
    crit_base = float(base_params.get("yield_gradient_crit", 0.25))
    smooth_base = float(base_params.get("yield_gradient_smoothing", 0.05))
    tau_base = float(base_params.get("yield_stress", 50.0))
    mu_base = float(base_params.get("plastic_viscosity", 2.0))
    pore_base = float(base_params.get("characteristic_pore_size", 5.0e-3))
    return [
        {
            "name": "linear_baseline",
            "overrides": {
                "rheology_model": "linear",
                "enable_bingham_yield": False,
            },
        },
        {
            "name": "gated_bingham_mild_threshold",
            "overrides": {
                "rheology_model": "gated_bingham",
                "enable_bingham_yield": True,
                "yield_gradient_crit": max(1.0e-6, crit_base * 0.6),
                "yield_gradient_smoothing": smooth_base,
            },
        },
        {
            "name": "gated_bingham_stronger_threshold",
            "overrides": {
                "rheology_model": "gated_bingham",
                "enable_bingham_yield": True,
                "yield_gradient_crit": max(1.0e-6, crit_base * 1.8),
                "yield_gradient_smoothing": smooth_base,
            },
        },
        {
            "name": "porous_bingham_baseline",
            "overrides": {
                "rheology_model": "porous_bingham",
                "enable_bingham_yield": False,
            },
        },
        {
            "name": "porous_bingham_higher_yield",
            "overrides": {
                "rheology_model": "porous_bingham",
                "enable_bingham_yield": False,
                "yield_stress": tau_base * 2.0,
            },
        },
        {
            "name": "porous_bingham_higher_viscosity",
            "overrides": {
                "rheology_model": "porous_bingham",
                "enable_bingham_yield": False,
                "plastic_viscosity": mu_base * 2.0,
            },
        },
        {
            "name": "porous_bingham_larger_pore",
            "overrides": {
                "rheology_model": "porous_bingham",
                "enable_bingham_yield": False,
                "characteristic_pore_size": pore_base * 2.0,
            },
        },
    ]


def _print_delta(label, base, case):
    print(
        "  {0}: yield_active {1:.4f}->{2:.4f}, "
        "mobility_max {3:.4e}->{4:.4e}, mu_app_max {5:.4e}->{6:.4e}, "
        "filling_max {7:.4e}->{8:.4e}".format(
            label,
            base["yield_active_ratio"],
            case["yield_active_ratio"],
            base["m_max"],
            case["m_max"],
            base["mu_max"],
            case["mu_max"],
            base["f_max"],
            case["f_max"],
        )
    )


def _print_porous_bingham_comparison(summaries):
    lookup = {}
    for summary in summaries:
        lookup[summary["name"]] = summary

    base = lookup.get("porous_bingham_baseline")
    if base is None:
        return

    print("")
    print("[porous_bingham_case_comparison]")
    print("  baseline grad_p_crit min/max = {0:.4e}, {1:.4e} Pa/m".format(base["crit_min"], base["crit_max"]))
    for name in (
        "porous_bingham_higher_yield",
        "porous_bingham_higher_viscosity",
        "porous_bingham_larger_pore",
    ):
        case = lookup.get(name)
        if case is not None:
            _print_delta(name, base, case)


def main(num_steps=20, dt=0.01, active_threshold=0.3):
    base_state = drv.initialize_problem()
    base_params = copy.deepcopy(base_state["slurry_parameters"])
    cases = _build_cases(base_params)

    print("Run Bingham multistep diagnostics over {0} cases.".format(len(cases)))
    print("Steps per case: {0}, dt={1}".format(num_steps, dt))
    summaries = []
    for case in cases:
        summary = _run_case(
            case_name=case["name"],
            overrides=case["overrides"],
            num_steps=num_steps,
            dt=dt,
            active_threshold=active_threshold,
        )
        summaries.append(summary)
        _print_summary(summary, active_threshold)
    _print_porous_bingham_comparison(summaries)


if __name__ == "__main__":
    main()
