# -*- coding: utf-8 -*-
"""
4.5 supplement S1: is random seed 2's stall real?

    powershell -File scripts\run_local.ps1 scripts\seed2_recheck.py [--pass2]

seed 2 stopped on "stall" (STALL_PATIENCE = 60 steps without a new cell
crossing S = 0.5) with 121 reachable cells unfilled, Q tail/peak 0.080 above
the 0.05 stall line and V_in 121.8 below the quota 166.15. Here the stall
window is widened AT RUN TIME for this script only (the module default is not
touched -- it is a stop criterion and would move the baseline quota and the
design target): seed 2 at patience 240 and 600, and the uniform baseline at
240 as a diagnostic (reported only; the quota and target are not changed).
Everything else is the channel case's own run(): field, quota 166.15, 2.5 m,
march. The march is wrapped only to record the filled-cell count per step.
Outputs: outputs/seed2_recheck/.
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

C = imp.load_source("inclined_hole_channel", os.path.join(REPO, "cases", "inclined_hole_channel.py"))
G = C.G
OUT = os.path.join(REPO, "outputs", "seed2_recheck")
QUOTA = 166.15
COUNTS = []
_march = C.march


def march_counting(state, hcells, vols, label, v_quota=None, keeper=None):
    """C.march unchanged, plus the filled-cell count after every step (read
    from the saturation the march leaves behind, via a step hook)."""
    del COUNTS[:]
    hmask = np.zeros(vols.size, dtype=bool)
    hmask[hcells] = True
    solve = C.solve_transport_step

    def hooked(st, dt_cap):
        dt = solve(st, dt_cap=dt_cap)
        s = np.asarray(st["saturation"].value, dtype=float)
        # the march sets the source cells to 1 after this call; they are
        # excluded from the count anyway
        COUNTS.append(int(np.sum((s >= 0.5) & np.logical_not(hmask))))
        return dt
    C.solve_transport_step = hooked
    try:
        return _march(state, hcells, vols, label, v_quota=v_quota, keeper=keeper)
    finally:
        C.solve_transport_step = solve


def target_mask(mx, my):
    base = np.loadtxt(os.path.join(REPO, "outputs", "inclined_hole_channel", "base_uniform_0.18", "cells.csv"),
                      delimiter=",", skiprows=1)
    key = dict(((round(u, 6), round(v, 6)), i) for i, (u, v) in enumerate(zip(mx, my)))
    t = np.zeros(mx.size, dtype=bool)
    for row in base:
        if row[3] >= 0.5:
            t[key[(round(row[0], 6), round(row[1], 6))]] = True
    return t


def one(label, phi, patience, quota, target):
    G.STALL_PATIENCE = patience
    try:
        row = C.run(label, phi, v_quota=quota, target_mask=target)
    finally:
        G.STALL_PATIENCE = 60
    row["patience"] = patience
    row["counts"] = list(COUNTS)
    return row


def main(argv):
    sfx = "_pass2" if "--pass2" in argv else ""
    C.OUT_ROOT = os.path.join(OUT, "runs" + sfx)
    C.march = march_counting
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    target = target_mask(mx, my)
    rows = []
    for patience in (240, 600):
        rows.append(one("random_seed2_p%d%s" % (patience, sfx), C.field("random", mx, my, seed=2),
                        patience, QUOTA, target))
    if not sfx:
        rows.append(one("base_uniform_0.18_p240", C.field("base", mx, my), 240, None, None))
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    with open(os.path.join(OUT, "results%s.json" % sfx), "w") as fh:
        # run() rows carry arrays (filled_mask); keep the scalars and lists
        json.dump([dict((k, v) for k, v in r.items() if not isinstance(v, np.ndarray)) for r in rows],
                  fh, indent=1, default=float)
    for r in rows:
        print("%-26s patience %3d: %-5s t=%.0f s V_in=%.2f filled=%d Qtail/peak=%.3f tgt=%s dsf=%s shortfall=%d"
              % (r["label"], r["patience"], r["reason"], r["t"], r["v_in"], r["filled_cells"], r["q_decay"],
                 ("%.0f%%" % (100 * r["target_filled_fraction"])) if "target_filled_fraction" in r else "-",
                 ("%.0f%%" % (100 * r["design_shortfall_fraction"])) if "design_shortfall_fraction" in r else "-",
                 r["front_shortfall"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
