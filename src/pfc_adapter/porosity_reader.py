import math

import numpy as np


def _build_cell_index_map(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx = len(x_unique)
    ny = len(y_unique)

    if nx > 1:
        dx = float(np.min(np.diff(x_unique)))
    else:
        dx = 1.0
    if ny > 1:
        dy = float(np.min(np.diff(y_unique)))
    else:
        dy = 1.0

    x0 = float(np.min(x_unique)) - 0.5 * dx
    y0 = float(np.min(y_unique)) - 0.5 * dy

    # Use floor here to match the ball-binning loop in read_cell_porosity_once.
    # round() would disagree (notably round(i+0.5) == i+1 on Python 2.7), which
    # shifts every cell key by one and leaves the top row / right column empty.
    cell_index_map = {}
    for i in range(n):
        ix = int(math.floor((x[i] - x0) / dx))
        iy = int(math.floor((y[i] - y0) / dy))
        cell_index_map[(ix, iy)] = i

    return cell_index_map, x0, y0, dx, dy, nx, ny


def _read_ball_radius(ball):
    for name in ("radius", "rad", "r"):
        if hasattr(ball, name):
            attr = getattr(ball, name)
            try:
                return float(attr() if callable(attr) else attr)
            except (TypeError, ValueError):
                return None
    return None


def read_cell_structure_once(
    x,
    y,
    default_porosity=0.35,
    min_porosity=0.05,
    max_porosity=0.95,
    fallback_particle_area_ratio=0.15,
    default_diameter=0.25,
):
    """
    One-time per-cell structure read from PFC particles mapped to FiPy cells:

      - porosity      = 1 - (sum of ball disk areas) / cell area   [2D areal]
      - cell_diameter = 2 * mean radius of the balls in the cell    [m]

    The per-cell diameter feeds a Kozeny-Carman permeability that uses the local
    grain size. If PFC data is unavailable, return uniform default fields.

    Returns (porosity, cell_diameter).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    porosity = np.ones(n, dtype=float) * float(default_porosity)
    cell_diameter = np.ones(n, dtype=float) * float(default_diameter)

    try:
        import itasca.ball as balls
    except Exception:
        return porosity, cell_diameter

    plist = list(balls.list())
    if len(plist) == 0:
        return porosity, cell_diameter

    cell_index_map, x0, y0, dx, dy, nx, ny = _build_cell_index_map(x, y)
    cell_area = max(dx * dy, 1.0e-12)
    occupancy = np.zeros(n, dtype=float)
    radius_sum = np.zeros(n, dtype=float)
    radius_count = np.zeros(n, dtype=float)

    for b in plist:
        bx = float(b.pos_x())
        by = float(b.pos_y())
        ix = int(math.floor((bx - x0) / dx))
        iy = int(math.floor((by - y0) / dy))
        if (ix < 0) or (ix > nx + 1) or (iy < 0) or (iy > ny + 1):
            continue
        idx = cell_index_map.get((ix, iy))
        if idx is None:
            continue

        br = _read_ball_radius(b)
        if (br is not None) and (br > 0.0):
            occupancy[idx] += math.pi * br * br
            radius_sum[idx] += br
            radius_count[idx] += 1.0
        else:
            occupancy[idx] += fallback_particle_area_ratio * cell_area

    porosity = 1.0 - (occupancy / cell_area)
    porosity = np.clip(porosity, float(min_porosity), float(max_porosity))

    # Per-cell characteristic diameter from the balls present; cells without a
    # readable-radius ball keep the default diameter.
    has_balls = radius_count > 0.0
    mean_radius = np.where(
        has_balls,
        radius_sum / np.maximum(radius_count, 1.0),
        0.5 * float(default_diameter),
    )
    cell_diameter = 2.0 * mean_radius
    return porosity, cell_diameter


def read_cell_porosity_once(
    x,
    y,
    default_porosity=0.35,
    min_porosity=0.05,
    max_porosity=0.95,
    fallback_particle_area_ratio=0.15,
):
    """Backward-compatible wrapper returning only the porosity field."""
    porosity, _ = read_cell_structure_once(
        x,
        y,
        default_porosity=default_porosity,
        min_porosity=min_porosity,
        max_porosity=max_porosity,
        fallback_particle_area_ratio=fallback_particle_area_ratio,
    )
    return porosity
