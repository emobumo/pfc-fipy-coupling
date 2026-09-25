# -*- coding: utf-8 -*-
"""
4.6 pilot: does a later pass inject only when the drilling increment exceeds
the stall length, Delta / L_max >= 1?

    powershell -File scripts\run_local.ps1 scripts\stage_ratio_pilot.py [--phi 0.10,0.14]

Every staged run so far sits at Delta < L_max (v0.6: L_max(0.18) = 18.3 m on
a 17 m hole), so it can only show that staging fails. This sweeps the ratio
with realistic ground: uniform fields phi = 0.10 / 0.14 / 0.18 (L_max 9.3 /
13.6 / 18.3 m) and four drilling plans on the same 17 m hole,

    single 17 | 5 -> 17 (Delta 12) | 7 -> 17 (Delta 10) | 7 -> 10 -> 17 (Delta 3, 7)

so the later-pass ratios span ~0.16 .. 1.30.

Stop rule, by definition (the brief's process, 3.2): every pass is injected
until it takes no more at constant pressure ("注至恒压注不进"), then sets
(stage_update closure) before the next pass is drilled. No volume cap. This
reuses cases/inclined_hole_staged.run_sequence with budget=None, unchanged.

Reads nothing; writes outputs/stage_ratio/<phi>/<plan>/ and
outputs/stage_ratio/summary.txt. Crash-safe through the staged case's
sequence checkpoints.
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

S = imp.load_source("inclined_hole_staged", os.path.join(REPO, "cases", "inclined_hole_staged.py"))
G = S.G
S.OUT_ROOT = os.path.join(REPO, "outputs", "stage_ratio")

PHIS = (0.10, 0.14, 0.18)
PLANS = (("single", (17.0,)), ("5_17", (5.0, 17.0)), ("7_17", (7.0, 17.0)),
         ("7_10_17", (7.0, 10.0, 17.0)))


def l_max(phi):
    return 83.33 * phi / (1.0 - phi)


def main(argv):
    phis = PHIS
    if "--phi" in argv:
        phis = tuple(float(v) for v in argv[argv.index("--phi") + 1].split(","))
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    nx = int(round((G.X_MAX - G.X_MIN) / G.CELL))
    ny = int(round((G.Y_MAX - G.Y_MIN) / G.CELL))
    full = np.asarray(G.hole_cells(mx, my, nx, ny), dtype=int)

    lines = []
    hdr = "%-20s %6s | %-40s | %7s %6s %6s %6s %7s" % (
        "run", "L_max", "later passes: Delta/L_max -> V (buried/src)", "V_tot", "fill", "reach", "shadow", "reach%")
    lines += [hdr, "-" * len(hdr)]
    for phi_v in phis:
        phi = np.full(mx.size, phi_v)
        st0 = S.C.build_from_phi(phi)[0]
        reach_virgin = reachable_domain(st0, full, p0=G.P0, use_gravity=True)["reachable"]
        print("=== phi %.2f (L_max %.1f m): virgin reach %d cells ===" % (phi_v, l_max(phi_v), int(reach_virgin.sum())))
        sys.stdout.flush()
        for name, depths in PLANS:
            label = "phi%.2f/%s" % (phi_v, name)
            row = S.run_sequence(label, phi, depths, None, reach_virgin, full, False)
            later = []
            for k, stg in enumerate(row["stages"]):
                if k == 0:
                    continue
                delta = depths[k] - depths[k - 1]
                later.append("%.2f->%.2f (%d/%d)" % (delta / l_max(phi_v), stg["v_in"], stg["buried"], stg["src_cells"]))
            lines.append("%-20s %6.1f | %-40s | %7.2f %6d %6d %6d %6.1f%%" % (
                label, l_max(phi_v), "; ".join(later) if later else "-", row["v_total"], row["filled"],
                row["reach_virgin"], row["shadowed"], 100 * row["over_reach_filled"]))
            print(lines[-1])
            sys.stdout.flush()
    lines.append("")
    lines.append("V = grout the pass took [m3/m]; buried/src = source cells already cemented; fill = cells")
    lines.append("cemented over the sequence; reach = virgin reach of the full hole; shadow = virgin-reachable,")
    lines.append("unfilled, and unreachable from the last pass (staging's permanent residual);")
    lines.append("reach% = share of virgin-reachable cells cemented.")
    text = "\n".join(lines)
    if not os.path.isdir(S.OUT_ROOT):
        os.makedirs(S.OUT_ROOT)
    open(os.path.join(S.OUT_ROOT, "summary.txt"), "w").write(text + "\n")
    print("")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
