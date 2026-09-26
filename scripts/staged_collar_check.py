# -*- coding: utf-8 -*-
"""
T0 for the 4.6 supplement: where is the staged case's "collar" band really?

    powershell -File scripts\run_local.ps1 scripts\staged_collar_check.py

cases/inclined_hole_staged.make_field("collar") hard-codes band_x = -13.75,
the collar column of the OLD 30 x 30 m domain. This prints where that band
sits relative to the hole on the v0.6 domain, and checks that the new
"collar_v06" field is cell for cell the field the channel case ran as
band_collar_x-28.75 (compared with the porosity column that case wrote).
Solver-free.
"""
from __future__ import print_function

import imp
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
S = imp.load_source("inclined_hole_staged", os.path.join(REPO, "cases", "inclined_hole_staged.py"))
C, G = S.C, S.G


def main():
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx = int(round((G.X_MAX - G.X_MIN) / G.CELL))
    ny = int(round((G.Y_MAX - G.Y_MIN) / G.CELL))
    hole = np.asarray(G.hole_cells(mx, my, nx, ny), dtype=int)
    end, d, nrm = G.hole_geometry()
    lines = ["hole: mouth (%.2f, %.2f) -> toe (%.2f, %.2f); hole cells x in [%.2f, %.2f]"
             % (G.MOUTH[0], G.MOUTH[1], end[0], end[1], mx[hole].min(), mx[hole].max())]
    for kind in ("collar", "collar_v06"):
        phi = S.make_field(kind, mx, my)
        band = phi > 0.3
        on_hole = int(np.sum(band[hole]))
        dx_toe = mx[band].min() - end[0]
        lines.append("%-10s band x in [%.2f, %.2f] (centre %.2f); hole cells inside the band: %d; "
                     "band's near edge minus toe x: %+.2f m"
                     % (kind, mx[band].min() - G.CELL / 2, mx[band].max() + G.CELL / 2,
                        mx[band].mean(), on_hole, dx_toe))
    ref = os.path.join(REPO, "outputs", "inclined_hole_channel", "band_collar_x-28.75", "cells.csv")
    if os.path.exists(ref):
        a = np.loadtxt(ref, delimiter=",", skiprows=1)
        phi_v06 = S.make_field("collar_v06", mx, my)
        # match rows by coordinates, not order
        key = dict(((round(u, 6), round(v, 6)), i) for i, (u, v) in enumerate(zip(mx, my)))
        idx = np.array([key[(round(u, 6), round(v, 6))] for u, v in zip(a[:, 0], a[:, 1])])
        same = bool(np.array_equal(phi_v06[idx], a[:, 2]))
        lines.append("collar_v06 vs band_collar_x-28.75 porosity (%d cells): identical = %s"
                     % (len(idx), same))
        old = S.make_field("collar", mx, my)
        lines.append("old 'collar' vs band_collar_x-28.75: identical = %s"
                     % bool(np.array_equal(old[idx], a[:, 2])))
    text = "\n".join(lines)
    out = os.path.join(REPO, "outputs", "stage_supplement")
    if not os.path.isdir(out):
        os.makedirs(out)
    open(os.path.join(out, "t0_collar_check.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
