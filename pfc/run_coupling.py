# -*- coding: utf-8 -*-
"""
PFC-side entry point.

Run this from inside PFC AFTER restoring a particle model, e.g. in a .p2dat:

    restore 'D:\\PFC item\\pfc_fipy_model1\\sample.p2sav'
    call run_coupling.py

It runs the FiPy coupling against the live PFC particles. This first version
only performs the one-time structure transfer (PFC porosity -> FiPy
permeability/mobility) and reports the result, so you can confirm the porosity
is actually being read from YOUR model (varying values) rather than falling back
to the uniform default (min == max == default).

The flow solve / write-back is intentionally NOT run yet: the borehole injection
boundary still uses placeholder values that must be set to the real borehole
location and pressure first.
"""
import os
import sys

import numpy as np

# The repo root must be importable inside PFC's bundled Python 2.7.
# Override with the PFC_FIPY_REPO environment variable if the repo moves.
REPO_ROOT = os.environ.get("PFC_FIPY_REPO", r"D:\work\pfc-fipy-coupling")
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# PFC's embedded Python persists across `call`s and caches imported modules, so
# edits to src/ would NOT take effect on a re-run. Drop cached project modules
# first to force a fresh import of the current code.
for _m in list(sys.modules):
    if (_m == "src") or _m.startswith("src."):
        del sys.modules[_m]

from src.coupling.driver import initialize_problem
from src.pfc_adapter.porosity_reader import _read_ball_radius

# Domain MUST match the PFC model extent (x: -7.5..7.5 m, y: 0..40 m).
DOMAIN = (-7.5, 7.5, 0.0, 40.0)
# Cell size tuned so each cell averages over many balls (~33 at 1.5 m for this
# model: 8986 balls, mean radius 0.127 m). Too small -> porosity pins at the
# clip rails; too large -> coarse flow resolution. Re-tune per model.
CELL_SIZE = 1.5

state = initialize_problem(domain=DOMAIN, cell_size=CELL_SIZE)
report = state["structure_init_report"]
params = state["slurry_parameters"]

phi = np.asarray(state["porosity"].value, dtype=float)
num_cells = phi.size
clip_min = float(params.get("porosity_clip_min", 0.05))
clip_max = float(params.get("porosity_clip_max", 0.95))
n_at_min = int(np.sum(phi <= clip_min + 1.0e-9))
n_at_max = int(np.sum(phi >= clip_max - 1.0e-9))

print("=== structure init: porosity read from PFC balls ===")
print("mesh cells          : %d  (cell size %.3f m)" % (num_cells, CELL_SIZE))
print("porosity min/max    : %.4f / %.4f" % (report["porosity_min"], report["porosity_max"]))
print("porosity mean/median: %.4f / %.4f" % (float(np.mean(phi)), float(np.median(phi))))
print("cells at min clip   : %d (%.1f%%)" % (n_at_min, 100.0 * n_at_min / num_cells))
print("cells at max clip   : %d (%.1f%%)" % (n_at_max, 100.0 * n_at_max / num_cells))
print("permeability min/max: %.3e / %.3e" % (report["permeability_min"], report["permeability_max"]))
print("k formula           : %s" % report.get("porosity_to_permeability_formula", "?"))
if "cell_diameter_min" in report:
    print("cell diameter m/M/avg: %.4f / %.4f / %.4f m"
          % (report["cell_diameter_min"], report["cell_diameter_max"], report["cell_diameter_mean"]))

# --- Ball / resolution diagnostics. Porosity pinned at the clip rails usually
# means cells are too small (0-1 balls each) rather than a representative
# average. These numbers help pick a cell size that spans many balls.
try:
    import itasca.ball as _balls
    plist = list(_balls.list())
    n_balls = len(plist)
    radii = []
    xs = []
    ys = []
    for b in plist:
        xs.append(float(b.pos_x()))
        ys.append(float(b.pos_y()))
        r = _read_ball_radius(b)
        if (r is not None) and (r > 0.0):
            radii.append(r)
    print("")
    print("=== ball / resolution diagnostics ===")
    print("ball count          : %d" % n_balls)
    if xs:
        # Actual filled extent of the assembly (vs the stated DOMAIN). The pile
        # top is where the injection boundary should sit, not the domain top.
        print("ball x extent       : %.3f .. %.3f m  (domain x: %.1f .. %.1f)"
              % (min(xs), max(xs), DOMAIN[0], DOMAIN[1]))
        print("ball y extent       : %.3f .. %.3f m  (domain y: %.1f .. %.1f)"
              % (min(ys), max(ys), DOMAIN[2], DOMAIN[3]))
    print("radius readable      : %s (%d of %d)" % (len(radii) > 0, len(radii), n_balls))
    if radii:
        r_arr = np.asarray(radii, dtype=float)
        r_mean = float(np.mean(r_arr))
        ball_area = np.pi * r_mean * r_mean
        cell_area = CELL_SIZE * CELL_SIZE
        print("radius min/max/mean : %.4f / %.4f / %.4f m" % (float(r_arr.min()), float(r_arr.max()), r_mean))
        print("avg balls per cell  : %.2f" % (float(n_balls) / num_cells))
        print("cell_area/ball_area : %.1f  (want >> 1 for a meaningful average)" % (cell_area / max(ball_area, 1e-30)))
except Exception as exc:
    print("ball diagnostics unavailable:", exc)

# Coarse porosity map to reveal the pile geometry (top row = pile top).
# ' ' empty/outside pile, '.' sparse, '+'/'o' mid, '#' dense packing.
try:
    nx_map = int(round((DOMAIN[1] - DOMAIN[0]) / CELL_SIZE))
    ny_map = phi.size // nx_map
    grid = phi.reshape((ny_map, nx_map))

    def _ch(v):
        if v >= 0.90:
            return " "
        if v >= 0.50:
            return "."
        if v >= 0.35:
            return "+"
        if v >= 0.25:
            return "o"
        return "#"

    print("")
    print("=== porosity map (top = pile top; ' '=empty  #=dense) ===")
    for j in range(ny_map - 1, -1, -1):
        print("  |" + "".join(_ch(grid[j, i]) for i in range(nx_map)) + "|")
except Exception as exc:
    print("porosity map unavailable:", exc)

print("")
if abs(report["porosity_max"] - report["porosity_min"]) < 1.0e-9:
    print("RESULT: porosity uniform -> balls NOT read (check restore + DOMAIN).")
else:
    print("RESULT: porosity read from the PFC model.")
    if (n_at_min + n_at_max) > 0.3 * num_cells:
        print("  NOTE: many cells pinned at the clip rails -> cell size likely too")
        print("  small relative to ball size; consider increasing CELL_SIZE.")
