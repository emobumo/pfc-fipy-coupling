from __future__ import print_function

import copy

import numpy as np
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

    yield_factor = np.asarray(result["scalar_yield_factor"], dtype=float)
    yield_active_ratio = float(np.mean(yield_factor > active_threshold))

    return {
        "name": case_name,
        "p_min": float(result["scalar_pressure"].min()),
        "p_max": float(result["scalar_pressure"].max()),
        "f_min": float(result["scalar_filling"].min()),
        "f_max": float(result["scalar_filling"].max()),
        "m_min": float(result["scalar_mobility_effective"].min()),
        "m_max": float(result["scalar_mobility_effective"].max()),
        "y_min": float(yield_factor.min()),
        "y_max": float(yield_factor.max()),
        "yield_active_ratio": yield_active_ratio,
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
    print("  final pressure min/max = {0:.4e}, {1:.4e}".format(summary["p_min"], summary["p_max"]))
    print("  final filling  min/max = {0:.4e}, {1:.4e}".format(summary["f_min"], summary["f_max"]))
    print("  final mobility min/max = {0:.4e}, {1:.4e}".format(summary["m_min"], summary["m_max"]))
    print("  final yield    min/max = {0:.4e}, {1:.4e}".format(summary["y_min"], summary["y_max"]))
    print(
        "  yield active ratio (>{0:.2f}) = {1:.4f}".format(
            active_threshold, summary["yield_active_ratio"]
        )
    )
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
    return [
        {
            "name": "linear_baseline",
            "overrides": {
                "enable_bingham_yield": False,
            },
        },
        {
            "name": "bingham_mild_threshold",
            "overrides": {
                "enable_bingham_yield": True,
                "yield_gradient_crit": max(1.0e-6, crit_base * 0.6),
                "yield_gradient_smoothing": smooth_base,
            },
        },
        {
            "name": "bingham_stronger_threshold",
            "overrides": {
                "enable_bingham_yield": True,
                "yield_gradient_crit": max(1.0e-6, crit_base * 1.8),
                "yield_gradient_smoothing": smooth_base,
            },
        },
    ]


def main(num_steps=20, dt=0.01, active_threshold=0.3):
    base_state = drv.initialize_problem()
    base_params = copy.deepcopy(base_state["slurry_parameters"])
    cases = _build_cases(base_params)

    print("Run Bingham multistep diagnostics over {0} cases.".format(len(cases)))
    print("Steps per case: {0}, dt={1}".format(num_steps, dt))
    for case in cases:
        summary = _run_case(
            case_name=case["name"],
            overrides=case["overrides"],
            num_steps=num_steps,
            dt=dt,
            active_threshold=active_threshold,
        )
        _print_summary(summary, active_threshold)


if __name__ == "__main__":
    main()
