# -*- coding: utf-8 -*-
"""
Single convex block, same local model as boulder variant A (solver run):

    powershell -File scripts\\run_local.ps1 scripts\\boulder_convex.py

cases/boulder_bypass.py variant A with the bridge (lid 4 x 1 m on two 1 x 2 m
legs, pocket 2 x 2 m under the lid) replaced by ONE solid 4 x 3 m block
covering the bridge's whole outline (x -2..2, y 6..9) -- convex, no rim
(matrix 0.18), same source, same 0.5 m grid, same rate stop rule (n_ref 0.18).
The 16 cells right below the block (x -1..1, y 4..6, the size of A's pocket)
are booked as "pocket" so the numbers line up with A.

Every 25 steps (as in boulder_bypass) the unfilled pore space is classified
with 4- and 8-adjacency (scripts/connectivity_check.py: S >= 0.5 or source
is filled, porosity <= 0.01 is solid, a component touching the outer
boundary is open): number of enclosed components, their cells, and how many
of them lie in the pocket. The run is checkpointed; rerun the command after
a crash.

Output: outputs/boulder_convex/results.jsonl, final_convex.npz
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

B = imp.load_source("boulder_bypass", os.path.join(REPO, "cases", "boulder_bypass.py"))
CC = imp.load_source("connectivity_check", os.path.join(REPO, "scripts", "connectivity_check.py"))

B.OUT = os.path.join(REPO, "outputs", "boulder_convex")
B.LID = (-2.0, 2.0, 6.0, 9.0)        # the bridge's outline, all solid
B.LEGS = []
B.POCKET = (-1.0, 1.0, 4.0, 6.0)     # 16 cells right below the block

_record = B.pocket_record


def _conn_record(state, src, reach, pocket, vols, t, v_in, step):
    rec = _record(state, src, reach, pocket, vols, t, v_in, step)
    x = np.asarray(state["x"], dtype=float)
    y = np.asarray(state["y"], dtype=float)
    phi = np.asarray(state["porosity"].value, dtype=float)
    filled = np.asarray(state["saturation"].value, dtype=float) >= CC.S_C
    filled[src] = True
    for conn in (4, 8):
        st = CC.enclosed_stats(x, y, filled, reach, phi, conn, True)
        by, dense = CC.unified_categories(x, y, filled, reach, phi, conn)
        rec["enc%d" % conn] = st["enclosed"]
        rec["enc%d_cells" % conn] = st["cells"]
        rec["enc%d_pocket_cells" % conn] = int(np.sum(pocket & (by | dense)))
    return rec


B.pocket_record = _conn_record


def main():
    if not os.path.isdir(B.OUT):
        os.makedirs(B.OUT)
    r = B.run_one("convex", "single convex block 4 x 3 m, no rim",
                  dict(phi_rim=B.PHI_MATRIX, phi_pocket=B.PHI_MATRIX))
    ser = r["series"]
    for conn in (4, 8):
        hit = [p for p in ser if p["enc%d" % conn] > 0]
        r["first_enclosed_%d" % conn] = hit[0] if hit else None
        r["max_enclosed_cells_%d" % conn] = max(p["enc%d_cells" % conn] for p in ser)
    open(os.path.join(B.OUT, "results.jsonl"), "w").write(json.dumps(r, default=float) + "\n")
    print("convex: %s at t=%.2f s, V_in %.4f / reachable %.4f; enclosed components ever (4 / 8): %s / %s; "
          "max enclosed cells %d / %d" % (r["reason"], r["t"], r["v_in"], r["cap"],
                                         r["first_enclosed_4"] is not None, r["first_enclosed_8"] is not None,
                                         r["max_enclosed_cells_4"], r["max_enclosed_cells_8"]))


if __name__ == "__main__":
    main()
