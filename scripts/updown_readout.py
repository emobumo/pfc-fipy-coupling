# -*- coding: utf-8 -*-
"""
Read-out error of the up/down spread ratio R_ud (post-processing, no solver).

    powershell -File scripts\\run_local.ps1 scripts\\updown_readout.py

The official R_ud (cases/inclined_hole_gradient.py front_metrics for the
solver, scripts/gravity_control.py up_down for the reach) takes CELL CENTRES:
r_up / r_down are the largest distances of filled (S >= 0.5, hole cells
excluded) or reachable (p_req <= p0) cell centres from the hole axis, along
its normal. Here the same field is read a second way -- bilinear
interpolation between cell centres on a grid of spacing dx/10, keeping the
points with p_req <= p0 or S >= 0.5 -- and the two ratios are compared. The
difference bounds the read-out error of the official numbers; the sub-grid
numbers support that bound and do not replace the anchors (the solver's S
front is one or two cells wide, so its interpolated contour is itself an
approximation; p_req is smooth).

Fields: line source, uniform 0.18 and gradient 0.12 -> 0.30, with and without
gravity, 1.25 and 2.5 m; solver states from outputs/rerun_rate/A/final_*.npz
(no 2.5 m no-gravity solver run exists), reach recomputed from the same field.

Output: outputs/updown_readout/updown_readout.txt
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain

G = imp.load_source("inclined_hole_gradient", os.path.join(REPO, "cases", "inclined_hole_gradient.py"))
RR = os.path.join(REPO, "outputs", "rerun_rate", "A")
OUT = os.path.join(REPO, "outputs", "updown_readout")
SUB = 10


def normal():
    return G.hole_geometry()[2]


def extents_cells(mask, mx, my):
    n = normal()
    perp = (mx[mask] - G.MOUTH[0]) * n[0] + (my[mask] - G.MOUTH[1]) * n[1]
    return float(-perp.min()), float(perp.max())


def extents_subgrid(field, mx, my, inside):
    n = normal()
    xs, ys = np.unique(mx), np.unique(my)
    g = field.reshape(len(ys), len(xs))
    h = (xs[1] - xs[0]) / SUB
    fx = np.arange(xs[0], xs[-1] + 1e-9, h)
    fy = np.arange(ys[0], ys[-1] + 1e-9, h)
    X, Y = np.meshgrid(fx, fy)
    X, Y = X.ravel(), Y.ravel()
    i = np.clip(np.searchsorted(xs, X) - 1, 0, len(xs) - 2)
    j = np.clip(np.searchsorted(ys, Y) - 1, 0, len(ys) - 2)
    tx = (X - xs[i]) / (xs[i + 1] - xs[i])
    ty = (Y - ys[j]) / (ys[j + 1] - ys[j])
    v = ((1 - tx) * (1 - ty) * g[j, i] + tx * (1 - ty) * g[j, i + 1]
         + (1 - tx) * ty * g[j + 1, i] + tx * ty * g[j + 1, i + 1])
    m = inside(v)
    perp = (X[m] - G.MOUTH[0]) * n[0] + (Y[m] - G.MOUTH[1]) * n[1]
    return float(-perp.min()), float(perp.max())


def rows():
    out = []
    for cell in (1.25, 2.5):
        G.CELL = cell
        try:
            for kind in ("uniform", "gradient"):
                for grav in (True, False):
                    st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
                    hc = np.asarray(hc, dtype=int)
                    cost = np.asarray(reachable_domain(st, hc, p0=G.P0, use_gravity=grav)["cost"], dtype=float)
                    c = extents_cells(cost <= G.P0, mx, my)
                    s = extents_subgrid(np.minimum(cost, 10.0 * G.P0), mx, my, lambda v: v <= G.P0)
                    out.append(("reach", kind, grav, cell, c, s))
                    p = os.path.join(RR, "final_%s_%s%s.npz" % (kind, "1.25" if cell == 1.25 else "2.5",
                                                                "" if grav else "_nograv"))
                    if not os.path.exists(p):
                        continue
                    f = np.load(p)
                    assert np.allclose(f["x"], mx) and np.allclose(f["y"], my)
                    sat = np.asarray(f["s"], dtype=float)
                    filled = sat >= 0.5
                    filled[hc] = False
                    c = extents_cells(filled, mx, my)
                    sat_sub = sat.copy()
                    sat_sub[hc] = 0.0              # hole cells excluded, as in front_metrics
                    s = extents_subgrid(sat_sub, mx, my, lambda v: v >= 0.5)
                    out.append(("solver", kind, grav, cell, c, s))
        finally:
            G.CELL = 2.5
    return out


def main():
    res = rows()
    lines = ["R_ud read-out: cell centres (official) vs bilinear sub-grid (dx/%d); r_up / r_down in m" % SUB, "",
             "%-6s %-8s %-6s %5s | %7s %7s %7s | %7s %7s %7s | %8s" % (
                 "side", "field", "grav", "cell", "up", "down", "R_ud", "up", "down", "R_ud", "sub-cell")]
    worst = {}
    for side, kind, grav, cell, c, s in res:
        rc, rs = c[0] / c[1], s[0] / s[1]
        lines.append("%-6s %-8s %-6s %5.2f | %7.3f %7.3f %7.4f | %7.3f %7.3f %7.4f | %+8.4f" % (
            side, kind, "on" if grav else "off", cell, c[0], c[1], rc, s[0], s[1], rs, rs - rc))
        key = (side, cell)
        worst[key] = max(worst.get(key, 0.0), abs(rs - rc))
    lines += ["", "largest |sub-grid - cell-centre| per side and grid:"]
    for side in ("reach", "solver"):
        for cell in (1.25, 2.5):
            if (side, cell) in worst:
                lines.append("  %-6s %4.2f m: %.4f" % (side, cell, worst[(side, cell)]))
    lines += ["", "table 4.4-2 (1.25 m), solver / reach: cell centres -> sub-grid"]
    by = dict(((side, kind, grav, cell), (c[0] / c[1], s[0] / s[1])) for side, kind, grav, cell, c, s in res)
    for kind, grav, name in (("uniform", True, "uniform"), ("gradient", True, "gradient"),
                             ("uniform", False, "uniform, no gravity"), ("gradient", False, "gradient, no gravity")):
        sv, rv = by[("solver", kind, grav, 1.25)], by[("reach", kind, grav, 1.25)]
        lines.append("  %-22s %.3f / %.3f -> %.3f / %.3f" % (name, sv[0], rv[0], sv[1], rv[1]))
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "updown_readout.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
