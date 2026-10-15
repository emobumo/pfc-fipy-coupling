# -*- coding: utf-8 -*-
"""
Dense inclusions and bypass voids at the stop, by the 4.4.1 rule
(post-processing, no solver):

    powershell -File scripts\\run_local.ps1 scripts\\enclosed_census.py

Rule (4.4.1): filled = S >= 0.5 or source cell; the unfilled cells form
components through shared faces (4-adjacency); a component with a cell on
the outer boundary is open; cells of a closed component are a bypass void if
reachable, a dense inclusion if not. No solid cells outside the boulder model
(every cell is pore). Counted over the whole domain and over the design
domain (2.5 m: the 175-cell target without the 10 source cells = 165 cells).

Cases: the uniform baseline (2.5 m), the gradient field (2.5 m, and 1.25 m /
uniform 1.25 m over the whole domain only -- the design domain is a 2.5 m
cell set), the 5 channel fields and the 3 random seeds (2.5 m, layer B).
Reach: from cells.csv where stored, else recomputed (reachable_domain,
gravity on) on the case's own field.

Output: outputs/enclosed_census/enclosed_census.txt
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

CC = imp.load_source("connectivity_check", os.path.join(REPO, "scripts", "connectivity_check.py"))
G = imp.load_source("inclined_hole_gradient", os.path.join(REPO, "cases", "inclined_hole_gradient.py"))
RR = os.path.join(REPO, "outputs", "rerun_rate")
OUT = os.path.join(REPO, "outputs", "enclosed_census")
CSV_CASES = [("uniform baseline 2.5", "A/base_uniform_0.18_2.5"),
             ("channel collar n 0.45", "B/B_band_collar_phi0.45"),
             ("channel mid-hole n 0.45", "B/B_band_midhole_phi0.45"),
             ("channel past toe n 0.30", "B/B_band_toe_phi0.30"),
             ("channel past toe n 0.45", "B/B_band_toe_phi0.45"),
             ("channel past toe n 0.60", "B/B_band_toe_phi0.60"),
             ("random seed 1", "B/B_random_seed1"),
             ("random seed 2", "B/B_random_seed2"),
             ("random seed 3", "B/B_random_seed3")]


def census(x, y, filled, reach, design):
    lab, n, border = CC.components(x, y, np.logical_not(filled), 4)
    closed = (lab > 0) & np.logical_not(np.in1d(lab, list(border)))
    by, dense = closed & reach, closed & np.logical_not(reach)
    row = [int(len(set(lab[closed]))), int(by.sum()), int(dense.sum())]
    if design is not None:
        row += [int((by & design).sum()), int((dense & design).sum()), int(design.sum())]
    return row


def main():
    G.CELL = 2.5
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, 2.5)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    hole = np.asarray(G.hole_cells(mx, my, 24, 24), dtype=int)
    target = np.load(os.path.join(RR, "A", "target_mask.npy"))
    design = target.copy()
    design[hole] = False
    lines = ["Enclosed unfilled cells at the stop, 4.4.1 rule (4-adjacency, boundary-touching components open, no solids)", "",
             "%-26s %5s | %6s %7s %6s | %10s %11s %7s" % ("case", "grid", "closed", "bypass", "dense",
                                                          "bypass(D)", "dense(D)", "D cells")]
    rows = []
    for name, sub in CSV_CASES:
        a = np.loadtxt(os.path.join(RR, sub, "cells.csv"), delimiter=",", skiprows=1)
        assert np.allclose(a[:, 0], mx) and np.allclose(a[:, 1], my)
        filled = a[:, 3] >= CC.S_C
        filled[hole] = True
        rows.append((name, 2.5, census(mx, my, filled, a[:, 5] > 0.5, design)))
    for kind, cell in (("gradient", 2.5), ("gradient", 1.25), ("uniform", 1.25)):
        G.CELL = cell
        try:
            st, cx, cy, nx, ny, phi, hc, k = G.build_case(kind)
            f = np.load(os.path.join(RR, "A", "final_%s_%g.npz" % (kind, cell)))
            assert np.allclose(f["x"], cx) and np.allclose(f["y"], cy) and np.allclose(f["phi"], phi)
            reach = np.asarray(reachable_domain(st, hc, p0=G.P0, use_gravity=True)["reachable"], dtype=bool)
            filled = np.asarray(f["s"], dtype=float) >= CC.S_C
            filled[np.asarray(hc, dtype=int)] = True
            rows.append(("%s field" % kind, cell, census(cx, cy, filled, reach, design if cell == 2.5 else None)))
        finally:
            G.CELL = 2.5
    for name, cell, r in rows:
        tail = ("%10d %11d %7d" % tuple(r[3:])) if len(r) > 3 else "%10s %11s %7s" % ("-", "-", "-")
        lines.append("%-26s %5.2f | %6d %7d %6d | %s" % ((name, cell) + tuple(r[:3]) + (tail,)))
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "enclosed_census.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
