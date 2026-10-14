# -*- coding: utf-8 -*-
"""
Resolution check of the tau0 / A / rho sensitivity rows (reach only):

    powershell -File scripts\\run_local.ps1 scripts\\sensitivity_0625.py

For the 10 rows of reach_sensitivity_1.25.txt that do not touch the box,
L_max / dx on 1.25 m (uniform 0.18, and the gradient field's hole mid-height
porosity 0.19763). Rows below the threshold 7 are recomputed on 0.625 m
(and on 1.25 m as the reference): uniform with gravity, gradient with
gravity, and the geometric baseline (uniform, no gravity), read at cell
centres the same way as scripts/reach_sensitivity.py (largest distance of
reachable cell centres normal to the hole axis, hole cells excluded).
Factor = R_ud / baseline. Also down - up of the uniform field with and
without gravity, against one cell.

Output: outputs/sensitivity_0625/sensitivity_0625.txt (+ md5 of the p_req fields)
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

from src.analysis.fill_diagnostics import reachable_domain

G = imp.load_source("inclined_hole_gradient", os.path.join(REPO, "cases", "inclined_hole_gradient.py"))
OUT = os.path.join(REPO, "outputs", "sensitivity_0625")
ROWS = [(60.0, 0.3, 1450.0), (60.0, 0.3, 1830.0), (60.0, 1.0, 1450.0), (30.0, 0.3, 1450.0), (60.0, 1.0, 1830.0),
        (30.0, 0.3, 1830.0), (60.0, 3.0, 1450.0), (30.0, 1.0, 1450.0), (60.0, 3.0, 1830.0), (30.0, 1.0, 1830.0)]
N_MID = 0.19763
THRESHOLD = 7.0
DIGEST = hashlib.md5()


def lam(n, tau0, af):
    return 2.0 * tau0 / math.sqrt(8.0 * G.A_CAL * af) * (1.0 - n) / n


def one(kind, grav, cell):
    G.CELL = cell
    st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
    hm = np.zeros(mx.size, dtype=bool)
    hm[hc] = True
    r = reachable_domain(st, hc, p0=G.P0, use_gravity=grav)
    DIGEST.update(np.asarray(r["cost"], dtype=float).tobytes())
    m = r["reachable"] & np.logical_not(hm)
    end, d, nrm = G.hole_geometry()
    perp = (mx[m] - G.MOUTH[0]) * nrm[0] + (my[m] - G.MOUTH[1]) * nrm[1]
    edge = bool((mx[m].max() > G.X_MAX - cell) or (my[m].max() > G.Y_MAX - cell) or (my[m].min() < G.Y_MIN + cell))
    return float(-perp.min()), float(perp.max()), edge


def main():
    base = (G.TAU0, G.A_CAL, G.RHO, G.CELL)
    lines = ["sensitivity rows: L_max / dx on 1.25 m (threshold %.0f); rows below it recomputed on 0.625 m (reach, 16-neighbour)" % THRESHOLD,
             "", "%5s %4s %5s %6s | %8s %8s | %9s %9s | %s" % ("tau0", "A x", "rho", "s", "Pi_g.18", "Pi_g mid",
                                                          "Lmax.18/dx", "Lmax mid/dx", "below")]
    low = []
    for tau0, af, rho in ROWS:
        s = (rho / 1830.0) * math.sqrt(af) / (tau0 / 30.0)
        l18, lmid = lam(0.18, tau0, af), lam(N_MID, tau0, af)
        r18, rmid = G.P0 / l18 / 1.25, G.P0 / lmid / 1.25
        below = r18 < THRESHOLD or rmid < THRESHOLD
        if below:
            low.append((tau0, af, rho, s))
        lines.append("%5.0f %4.1f %5.0f %6.3f | %8.4f %8.4f | %9.2f %9.2f | %s" % (
            tau0, af, rho, s, rho * 9.81 / l18, rho * 9.81 / lmid, r18, rmid, "yes" if below else ""))
    lines.append("")
    try:
        for tau0, af, rho, s in low:
            G.TAU0, G.A_CAL, G.RHO = tau0, base[1] * af, rho
            lines.append("tau0 %.0f  A x%.1f  rho %.0f  (s %.3f)" % (tau0, af, rho, s))
            for cell in (1.25, 0.625):
                bu, bd, be = one("uniform", False, cell)
                uu, ud, ue = one("uniform", True, cell)
                gu, gd, ge = one("gradient", True, cell)
                b, u, g = bu / bd, uu / ud, gu / gd
                lines.append("  %5.3f m | baseline up %7.3f down %7.3f R %.4f%s | uniform up %7.3f down %7.3f R %.4f%s | "
                             "gradient up %7.3f down %7.3f R %.4f%s" % (
                                 cell, bu, bd, b, " box" if be else "", uu, ud, u, " box" if ue else "",
                                 gu, gd, g, " box" if ge else ""))
                lines.append("          factor uniform %.4f  gradient %.4f | uniform down-up %.3f m (no gravity %.3f m), "
                             "gravity part %.3f m vs one cell %.3f m" % (u / b, g / b, ud - uu, bd - bu,
                                                                          (ud - uu) - (bd - bu), cell))
            lines.append("")
    finally:
        G.TAU0, G.A_CAL, G.RHO, G.CELL = base
    lines.append("md5 of the p_req fields: %s" % DIGEST.hexdigest())
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "sensitivity_0625.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
