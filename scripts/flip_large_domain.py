# -*- coding: utf-8 -*-
"""
Flip criterion in a large synthetic box (reach only, no solver): does F* ~ 1
hold at tau0 = 7.5 Pa, where L_max (82 m at the hole's mid-height porosity)
outgrows the 60 x 60 m case box?

    powershell -File scripts\run_local.ps1 scripts\flip_large_domain.py

Same F family as scripts/flip_criterion.py (V3): n(y) = n_mid + n'(y - y_mid),
y_mid = 25.875 m (the hole's mid-height, hole unchanged), n_mid = 0.19763,
n' = F x 2 rho g n_mid (1 - n_mid) / p0 (rho 1830). Only the box changes:
M = 165 m beyond the hole's bounding box on every side (about two L_max at
7.5 Pa), aligned to the 1.25 m grid. The first try, M = 82.5 m (about one
L_max, 180 x 176 m), clipped every tau0 = 7.5 reach: gravity stretches the
downward reach to about L_max / (1 - Pi_g cos 35) = 108 m already at F = 0,
and more for F < 0. The family is used ONLY here; this box is not the case
pile.

Both tau0 = 30 (v0.6) and 7.5 are run in this box (A, rho at v0.6). Factor =
up/down / the geometric baseline (uniform n_mid, gravity off, same box);
F* = where the factor crosses 1, interpolated. Straight-ray estimate as in
flip_criterion (rays stop at the box). An F is EXCLUDED when the porosity over
the vertical span the grout actually reaches leaves 0.05 - 0.60 (the gradient
case's PHI_CLIP); a reach that touches the box is flagged.

Writes outputs/flip_large_domain/flip_large_domain.txt.
"""
from __future__ import print_function

import hashlib
import imp
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FC = imp.load_source("flip_criterion", os.path.join(REPO, "scripts", "flip_criterion.py"))
C, G = FC.C, FC.G
# Reach via the gravity-potential reweighting (same shortest paths as reachable_domain's
# Bellman-Ford, checked to 3e-13; docs/results_log.md). Needed here: far above the reach
# the F family exceeds n 0.455 at tau0 = 7.5 (Pi_g >= 1), and Bellman-Ford on ~75 000
# cells does not finish in reasonable time.
reachable_domain = imp.load_source(
    "tau0_low_sensitivity", os.path.join(REPO, "scripts", "tau0_low_sensitivity.py")).reach_potential
OUT = os.path.join(REPO, "outputs", "flip_large_domain")
CELL = 1.25
MARGIN = 165.0
F_LIST = (-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
TAU0S = (30.0, 7.5)
RANGE = (0.05, 0.60)
DIGEST = hashlib.sha1()


def big_box():
    end, _, _ = G.hole_geometry()
    x0, x1 = min(G.MOUTH[0], end[0]) - MARGIN, max(G.MOUTH[0], end[0]) + MARGIN
    y0, y1 = min(G.MOUTH[1], end[1]) - MARGIN, max(G.MOUTH[1], end[1]) + MARGIN
    f = lambda v: math.floor(v / CELL) * CELL
    c = lambda v: math.ceil(v / CELL) * CELL
    return f(x0), c(x1), f(y0), c(y1)


def reach(phi, gravity, mx, my):
    st, cx, cy, _, _, _, hc, _ = C.build_from_phi(phi)
    r = reachable_domain(st, hc, p0=G.P0, use_gravity=gravity)
    DIGEST.update(np.asarray(r["cost"], dtype=float).tobytes())
    hm = np.zeros(cx.size, dtype=bool)
    hm[hc] = True
    m = r["reachable"] & np.logical_not(hm)
    _, _, nrm = G.hole_geometry()
    perp = (cx[m] - G.MOUTH[0]) * nrm[0] + (cy[m] - G.MOUTH[1]) * nrm[1]
    box = bool(cx[m].min() < G.X_MIN + CELL or cx[m].max() > G.X_MAX - CELL
               or cy[m].min() < G.Y_MIN + CELL or cy[m].max() > G.Y_MAX - CELL)
    return {"ratio": float(-perp.min() / perp.max()), "up": float(-perp.min()), "down": float(perp.max()),
            "box": box, "neg": r["negative_edges"], "ymin": float(cy[m].min()), "ymax": float(cy[m].max()),
            "cells": int(m.sum())}


def crossing(xs, ys):
    for i in range(len(xs) - 1):
        if (ys[i] - 1.0) * (ys[i + 1] - 1.0) <= 0 and ys[i + 1] != ys[i]:
            return xs[i] + (1.0 - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
    return float("nan")


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    end, _, _ = G.hole_geometry()
    y_mid = 0.5 * (G.MOUTH[1] + end[1])
    n_mid = float(FC.gradient_n(y_mid))
    unit = 2.0 * 1830.0 * FC.G_ACC * n_mid * (1 - n_mid) / G.P0
    saved = (G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL)
    lines = []
    try:
        G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX = big_box()
        G.CELL = CELL
        _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, CELL)
        mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        lines += ["flip criterion in a large synthetic box: x %.2f..%.2f, y %.2f..%.2f m (%.2f x %.2f m, %d cells of %.2f m); "
                  "hole unchanged (mouth (%.1f, %.1f), 17 m, 35 deg)" % (
                      G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.X_MAX - G.X_MIN, G.Y_MAX - G.Y_MIN, mx.size, CELL,
                      G.MOUTH[0], G.MOUTH[1]),
                  "F family n(y) = n_mid + n'(y - y_mid), y_mid = %.3f m, n_mid = %.5f, n' = F x %.4e /m (rho 1830); "
                  "A, rho at v0.6; excluded = n over the reached vertical span outside %.2f-%.2f; box = reach touches the box"
                  % (y_mid, n_mid, unit, RANGE[0], RANGE[1]), ""]
        summary = []
        for tau0 in TAU0S:
            FC.set_params(tau0, 1.0, 1830.0)
            try:
                lmax = G.P0 / FC.lam(n_mid)
                base = reach(np.full(mx.size, n_mid), False, mx, my)
                lines += ["tau0 = %.1f Pa: L_max(n_mid) = %.2f m; geometric baseline (uniform n_mid, gravity off) "
                          "up/down %.4f (up %.2f / down %.2f m, box %s)" % (
                              tau0, lmax, base["ratio"], base["up"], base["down"], base["box"]),
                          "  %6s | %7s %7s | %8s %8s %7s | %8s %8s | %-5s %4s %s" % (
                              "F", "n(ymin)", "n(ymax)", "up [m]", "down [m]", "2D", "factor", "rays", "box", "neg",
                              "status")]
                rows = []
                for f1 in F_LIST:
                    slope = f1 * unit
                    nf = lambda yy, sl=slope: n_mid + sl * (yy - y_mid)
                    r = reach(nf(my), True, mx, my)
                    lo, hi = nf(r["ymin"]), nf(r["ymax"])
                    nlo, nhi = min(lo, hi), max(lo, hi)
                    excl = bool(nlo < RANGE[0] or nhi > RANGE[1])
                    rr = FC.ray_ratio(nf, True)
                    fac = r["ratio"] / base["ratio"]
                    status = "EXCLUDED (n out of range)" if excl else ("box-clipped" if r["box"] else "ok")
                    rows.append((f1, fac, rr["ratio"], excl or r["box"]))
                    lines.append("  %6.2f | %7.4f %7.4f | %8.2f %8.2f %7.4f | %8.4f %8.4f | %-5s %4d %s" % (
                        f1, nf(r["ymin"]), nf(r["ymax"]), r["up"], r["down"], r["ratio"], fac, rr["ratio"],
                        r["box"], r["neg"], status))
                    print(lines[-1])
                    sys.stdout.flush()
                good = [rw for rw in rows if not rw[3]]
                f2 = crossing([rw[0] for rw in good], [rw[1] for rw in good])
                fr = crossing([rw[0] for rw in good], [rw[2] for rw in good])
                summary.append((tau0, lmax, f2, fr, len(good)))
                lines.append("  F*: 2D %.3f, rays %.3f (from the %d usable F values)" % (f2, fr, len(good)))
                lines.append("")
            finally:
                FC.restore()
        lines.append("summary (large box): " + "; ".join(
            "tau0 %.1f: L_max %.1f m, F*_2D %.3f, F*_rays %.3f" % s[:4] for s in summary))
        lines.append("reference (60 x 60 m case box, flip_criterion.txt group 1): F*_2D 0.958, F*_rays 1.020")
        lines.append("sha1 of all cost arrays: %s" % DIGEST.hexdigest())
    finally:
        G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL = saved
    text = "\n".join(lines) + "\n"
    open(os.path.join(OUT, "flip_large_domain.txt"), "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
