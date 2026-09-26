# -*- coding: utf-8 -*-
"""
4.5 supplement S2: the entry cost of a channel -- does the hole pass through it?

    powershell -File scripts\run_local.ps1 scripts\channel_entry_cost.py [--cell 1.25]

For the channel comparison's position sweep (phi 0.45 at the collar x=-28.75,
mid-hole x=-23.75, past the toe x=-11.25) and porosity sweep (past the toe,
phi 0.30/0.45/0.60), the entry cost is the least path cost (16-neighbour
reach, gravity on) over the band cells: 0 where the hole itself runs through
the band. Also: hole cells inside the band, reachable band cells, and whether
the reachable part of the band runs to the top / bottom boundary. The quota
time and target fill are read back from the channel case's own summaries.
The collar band touches the no-flow left boundary, so it has fill on one side
only -- noted, not corrected. Solver-free.
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

C = imp.load_source("inclined_hole_channel", os.path.join(REPO, "cases", "inclined_hole_channel.py"))
G = C.G
OUT = os.path.join(REPO, "outputs", "channel_entry_cost")

FIELDS = [  # (label in the channel case, band x, band phi, note)
    ("band_collar_x-28.75", -28.75, 0.45, "collar (one side on the no-flow boundary)"),
    ("band_midhole_x-23.75", -23.75, 0.45, "mid-hole"),
    ("band_toe_x-11.25", -11.25, 0.45, "past the toe"),
    ("band_phi0.30", -11.25, 0.30, "past the toe"),
    ("band_phi0.60", -11.25, 0.60, "past the toe"),
]


def summary(label):
    p = os.path.join(REPO, "outputs", "inclined_hole_channel", label, "summary.txt")
    d = {}
    if os.path.exists(p):
        for line in open(p):
            if " = " in line:
                k, v = line.split(" = ", 1)
                d[k.strip()] = v.strip()
    return d


def main(argv):
    cell = float(argv[argv.index("--cell") + 1]) if "--cell" in argv else G.CELL
    base_cell = G.CELL
    G.CELL = cell
    try:
        _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, cell)
        mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        nx = int(round((G.X_MAX - G.X_MIN) / cell))
        ny = int(round((G.Y_MAX - G.Y_MIN) / cell))
        hole = np.asarray(G.hole_cells(mx, my, nx, ny), dtype=int)
        hm = np.zeros(mx.size, dtype=bool)
        hm[hole] = True
        lines = ["S2 channel entry cost (%.2f m cells, 16-neighbour, gravity on, p0 = %.0f MPa)" % (cell, G.P0 / 1e6), "",
                 "%-22s %-40s %12s %7s %9s %10s %6s %6s %9s %7s" % (
                     "field", "position", "entry [Pa]", "/p0", "hole in", "band reach", "top", "bottom",
                     "quota t", "tgt%")]
        for label, bx, pb, note in FIELDS:
            phi = C.field("band", mx, my, phi_band=pb, band_x=bx)
            band = np.abs(phi - pb) < 1e-12
            st = C.build_from_phi(phi)[0]
            r = reachable_domain(st, hole, p0=G.P0, use_gravity=True)
            cost = np.asarray(r["cost"], dtype=float)
            reach = r["reachable"]
            entry = float(np.min(cost[band]))
            bre = band & reach
            sm = summary(label) if cell == base_cell else {}
            tgt = sm.get("target_filled_fraction")
            lines.append("%-22s %-40s %12.4g %7.3f %9d %10d %6s %6s %9s %7s" % (
                label, note, entry, entry / G.P0, int(np.sum(band & hm)), int(bre.sum()),
                bool(np.any(bre & (my > G.Y_MAX - cell))), bool(np.any(bre & (my < G.Y_MIN + cell))),
                ("%.0f s" % float(sm["t"])) if "t" in sm else "-",
                ("%.0f" % (100 * float(tgt))) if tgt else "-"))
    finally:
        G.CELL = base_cell
    text = "\n".join(lines)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "entry_cost_%.2f.txt" % cell), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
