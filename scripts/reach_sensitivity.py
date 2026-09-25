# -*- coding: utf-8 -*-
"""
Case 5 (parameter sensitivity), solver-free: how the headline up/down ratio
moves with tau0, A and rho over the brief's sensitivity bands.

    powershell -File scripts\run_local.ps1 scripts\reach_sensitivity.py [--cell 2.5]

The reach is set by lambda = 2 tau0 / sqrt(8 k / n) (with k = A n^3/(1-n)^2,
so lambda is proportional to tau0 / sqrt(A)) and by the climb rho g dy. A
ratio of lengths can therefore depend on the three parameters only through
the gravity number

    Pi_g = rho g / lambda   ~   s = (rho/1830) * sqrt(A/1.25e-7) / (tau0/30)

(s = 1 is v0.6), plus wherever the reach hits the 60 x 60 m box. This prints
the ratio for every combination so both claims can be checked: rows with
equal s should agree unless the box clips them, and the gradient field
should stay above the uniform one across the whole band.
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
OUT = os.path.join(REPO, "outputs", "reach_sensitivity")

TAU0S = (15.0, 30.0, 60.0)            # brief: 15~60 Pa
A_FACTORS = (0.3, 1.0, 3.0)           # brief: x0.3~x3
RHOS = (1450.0, 1830.0)               # brief's 1450 vs the adopted 1830


def main(argv):
    cell = 1.25
    if "--cell" in argv:
        cell = float(argv[argv.index("--cell") + 1])
    base = (G.TAU0, G.A_CAL, G.RHO, G.CELL)
    G.CELL = cell
    end, d, nrm = G.hole_geometry()
    rows = []
    try:
        for tau0 in TAU0S:
            for af in A_FACTORS:
                for rho in RHOS:
                    G.TAU0, G.A_CAL, G.RHO = tau0, base[1] * af, rho
                    s = (rho / 1830.0) * np.sqrt(af) / (tau0 / 30.0)
                    row = {"tau0": tau0, "A_factor": af, "rho": rho, "s": float(s)}
                    for kind in ("uniform", "gradient"):
                        st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
                        hm = np.zeros(mx.size, dtype=bool)
                        hm[hc] = True
                        r = reachable_domain(st, hc, p0=G.P0, use_gravity=True)
                        m = r["reachable"] & np.logical_not(hm)
                        perp = (mx[m] - G.MOUTH[0]) * nrm[0] + (my[m] - G.MOUTH[1]) * nrm[1]
                        edge = ((mx[m].max() > G.X_MAX - cell) or (my[m].max() > G.Y_MAX - cell)
                                or (my[m].min() < G.Y_MIN + cell))
                        row[kind] = {"up": float(-perp.min()), "down": float(perp.max()),
                                     "ratio": float(-perp.min() / perp.max()), "cells": int(m.sum()),
                                     "clipped": bool(edge)}
                    rows.append(row)
                    print("tau0 %4.0f  A x%.1f  rho %4.0f  s=%.3f | uniform %.3f%s | gradient %.3f%s" % (
                        tau0, af, rho, s, row["uniform"]["ratio"], " (box)" if row["uniform"]["clipped"] else "",
                        row["gradient"]["ratio"], " (box)" if row["gradient"]["clipped"] else ""))
                    sys.stdout.flush()
    finally:
        G.TAU0, G.A_CAL, G.RHO, G.CELL = base

    rows.sort(key=lambda r: r["s"])
    lines = ["reach up/down vs parameters (%.2f m cells, 16-neighbour, gravity on); s = (rho/1830)"
             "*sqrt(A/A0)/(tau0/30), v0.6 is s = 1; (box) = reach touches the domain edge" % cell, ""]
    lines.append("%6s %6s %6s %6s | %8s %8s | %s" % ("s", "tau0", "A x", "rho", "uniform", "gradient", "gradient > uniform?"))
    for r in rows:
        lines.append("%6.3f %6.0f %6.1f %6.0f | %7.3f%s %7.3f%s | %s" % (
            r["s"], r["tau0"], r["A_factor"], r["rho"],
            r["uniform"]["ratio"], "*" if r["uniform"]["clipped"] else " ",
            r["gradient"]["ratio"], "*" if r["gradient"]["clipped"] else " ",
            "yes" if r["gradient"]["ratio"] > r["uniform"]["ratio"] else "NO"))
    lines.append("")
    lines.append("* = reach touches the 60 x 60 m box: the ratio is clipped, not a model result")
    text = "\n".join(lines)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "reach_sensitivity_%.2f.txt" % cell), "w").write(text + "\n")
    print("")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
