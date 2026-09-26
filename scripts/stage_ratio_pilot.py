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


def extent_report():
    """
    After the runs: how far past its own toe did the first pass actually
    cement, along the hole (cells within one cell of the axis)? Compare with
    the next increment. Reads outputs/stage_ratio/<phi>/<plan>/cells.csv.

    RESULT (2026-09-26): Delta/L_max does NOT separate the runs (0.66 injects,
    0.74 does not); Delta against this measured extent x_f does, 12 of 12,
    where Delta is counted from the toe of the last pass that actually took
    grout (7 -> 10 -> 17 at phi 0.10: the 10 m pass is dead, so the 17 m pass
    sees Delta = 10). x_f / L_max = 0.34 .. 0.80: a pass stops on refusal
    while its front is still short of L_max (the Gustafson-Stille tail), and
    a short source spreads less along its own axis.
    """
    end, d, nrm = G.hole_geometry()
    ax = np.array([end[0] - G.MOUTH[0], end[1] - G.MOUTH[1]])
    ax /= np.hypot(ax[0], ax[1])
    lines = ["%-18s %6s %8s %8s %7s | %s" % ("run", "L_max", "d1", "x_f", "x_f/L", "next Delta (from last live toe) vs x_f")]
    for phi_v in PHIS:
        for name, depths in PLANS:
            p = os.path.join(S.OUT_ROOT, "phi%.2f" % phi_v, name, "cells.csv")
            for sub in (("c%g" % G.CELL,), ("pass2", "c%g" % G.CELL), ("pass2",)):
                if not os.path.exists(p):
                    p = os.path.join(*((S.OUT_ROOT,) + sub + ("phi%.2f" % phi_v, name, "cells.csv")))
            if not os.path.exists(p):
                continue
            h = np.loadtxt(os.path.join(os.path.dirname(p), "injection_rate.csv"), delimiter=",",
                           skiprows=1, ndmin=2)
            q_stop = [h[h[:, 2] == k][-1, 1] for k in sorted(set(h[:, 2].astype(int)))]
            a = np.loadtxt(p, delimiter=",", skiprows=1)
            x, y, by_stage = a[:, 0], a[:, 1], a[:, 5]
            along = (x - G.MOUTH[0]) * ax[0] + (y - G.MOUTH[1]) * ax[1]
            perp = np.abs((x - G.MOUTH[0]) * nrm[0] + (y - G.MOUTH[1]) * nrm[1])
            first = (by_stage == 0) & (perp < G.CELL)
            x_f = float(along[first].max()) - depths[0]
            nxt = ", ".join("%.0f m %s" % (dd - depths[0], ">" if dd - depths[0] > x_f else "<=")
                            for dd in depths[1:])
            lines.append("%-18s %6.1f %8.1f %8.1f %7.2f | %-28s | Q at each stop [m3/m/s]: %s" % (
                "phi%.2f/%s" % (phi_v, name), l_max(phi_v), depths[0], x_f, x_f / l_max(phi_v), nxt or "-",
                ", ".join("%.2e" % q for q in q_stop)))
    text = "\n".join(lines)
    open(os.path.join(S.OUT_ROOT, "extent.txt"), "w").write(text + "\n")
    print(text)
    return 0


def options(argv):
    """--cell X (default: the case's 2.5 m), --pass2 (second independent run,
    separate outputs), --plans a,b (subset of PLANS by name). A finer cell
    goes to outputs/stage_ratio_<cell>[_pass2]/ with labels prefixed by the
    cell, so no checkpoint or output of another grid is ever reused."""
    global PLANS, PHIS
    cell = float(argv[argv.index("--cell") + 1]) if "--cell" in argv else G.CELL
    G.CELL = cell
    root = "stage_ratio" if cell == 2.5 else "stage_ratio_%g" % cell
    if "--pass2" in argv:
        root += "_pass2"
    S.OUT_ROOT = os.path.join(REPO, "outputs", root)
    if "--plans" in argv:
        keep = argv[argv.index("--plans") + 1].split(",")
        PLANS = tuple(p for p in PLANS if p[0] in keep)
    if "--phi" in argv:
        PHIS = tuple(float(v) for v in argv[argv.index("--phi") + 1].split(","))
    prefix = "" if cell == 2.5 else "c%g/" % cell
    if "--pass2" in argv:
        prefix = "pass2/" + prefix        # distinct checkpoint tags: never resume pass 1
    return prefix


def main(argv):
    prefix = options(argv)
    if "--extent" in argv:
        return extent_report()
    phis = PHIS
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
            label = prefix + "phi%.2f/%s" % (phi_v, name)
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
    name = "summary.txt"
    if "--plans" in argv or "--phi" in argv:        # a subset: never overwrite the full table
        name = "summary_%s_%s.txt" % ("-".join(p[0] for p in PLANS), "-".join("%.2f" % v for v in phis))
    open(os.path.join(S.OUT_ROOT, name), "w").write(text + "\n")
    print("")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
