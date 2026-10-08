# -*- coding: utf-8 -*-
"""
How far outside the reach diagnostic are the overshoot cells of the rate-rule
reruns? For every run with overshoot > 0, the path cost of each overshoot
cell / p0, against the 16-neighbour metric's bound (the diagnostic only
under-estimates reach, by at most 2.75 % in radius, i.e. cost / p0 <= ~1.0275
for a cell the exact metric would call reachable). Post-processing (reach
only, no solver).

    powershell -File scripts\run_local.ps1 scripts\overshoot_cost.py

Output: outputs/rerun_rate/overshoot_cost.txt
"""
from __future__ import print_function

import imp
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain, OVERSHOOT

S = imp.load_source("inclined_hole_staged", os.path.join(REPO, "cases", "inclined_hole_staged.py"))
C, G = S.C, S.G
R = os.path.join(REPO, "outputs", "rerun_rate")


def main():
    lines = ["overshoot cells: path cost / p0 from the full hole (16-neighbour reach, gravity on)", ""]
    worst = 0.0
    for lay in ("B", "C"):
        for r in json.load(open(os.path.join(R, lay, "results.json"))):
            if r.get("overshoot", 0) <= 0:
                continue
            a = np.loadtxt(os.path.join(R, lay, r["label"], "cells.csv"), delimiter=",", skiprows=1)
            cell = r.get("cell", 2.5)
            G.CELL, G.HOLE_LEN = cell, r.get("H", 17.0)
            try:
                _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, cell)
                mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
                key = dict(((round(u, 6), round(v, 6)), i) for i, (u, v) in enumerate(zip(mx, my)))
                idx = np.array([key[(round(u, 6), round(v, 6))] for u, v in zip(a[:, 0], a[:, 1])])
                phi = np.zeros(mx.size)
                phi[idx] = a[:, 2]
                cat = np.zeros(mx.size, dtype=int)
                cat[idx] = a[:, -1].astype(int)
                nx = int(round((G.X_MAX - G.X_MIN) / cell))
                ny = int(round((G.Y_MAX - G.Y_MIN) / cell))
                hole = np.asarray(G.hole_cells(mx, my, nx, ny), dtype=int)
                st = C.build_from_phi(phi)[0]
                cost = np.asarray(reachable_domain(st, hole, p0=G.P0, use_gravity=True)["cost"], dtype=float) / G.P0
            finally:
                G.CELL, G.HOLE_LEN = 2.5, 17.0
            ov = cost[cat == OVERSHOOT]
            worst = max(worst, float(ov.max()))
            lines.append("%-42s %d cells  cost/p0 %s" % (r["label"], ov.size, ", ".join("%.4f" % v for v in sorted(ov))))
    lines += ["", "largest overshoot cost / p0: %.4f (metric bound ~1.0275 in radius)" % worst]
    text = "\n".join(lines)
    open(os.path.join(R, "overshoot_cost.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
