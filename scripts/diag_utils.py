# Shared helper functions for diagnostic scripts under scripts/.

import itasca.ball as balls


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
