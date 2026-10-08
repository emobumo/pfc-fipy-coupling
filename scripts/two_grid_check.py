# -*- coding: utf-8 -*-
"""
Two-grid agreement (2.5 vs 1.25 m) and x_f / L_max under the rate rule, from
the rerun outputs only (both passes bit-identical; post-processing, no solver).
Replaces the stop-rule diagnostic's single-pass replay values ("V_in within
5 %, x_f within 0.9 m", "x_f / L_max 0.83 ~ 0.94") with current numbers.

    powershell -File scripts\run_local.ps1 scripts\two_grid_check.py

Pairs that exist on both grids (the resolved ones, L_max / dx >= 7 on 2.5 m):
  uniform 0.18 full hole      A/final_uniform_2.5 vs A/final_uniform_1.25
  gradient full hole          A/final_gradient_2.5 vs A/final_gradient_1.25
  uniform 0.18, 7 m segment   C_struct base 7_10_17 stage 1 (2.5 m) vs
                              C_p2 phi0.18 7_17 stage 1 (1.25 m)
phi 0.10 has no 2.5 m rerun (L_max / dx = 3.7 there, below the threshold).
x_f = furthest filled cell within one cell of the hole axis, measured along
the axis past the (nominal) end of the source. Also lists x_f / L_max of every
rate-stopped stage in layer C.
Output: outputs/rerun_rate/two_grid_check.txt
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

G = imp.load_source("inclined_hole_gradient", os.path.join(REPO, "cases", "inclined_hole_gradient.py"))
R = os.path.join(REPO, "outputs", "rerun_rate")


def l_max(phi):
    return 83.33 * phi / (1.0 - phi)


def x_f(x, y, filled, depth, cell):
    _, d, n = G.hole_geometry()
    along = (x - G.MOUTH[0]) * d[0] + (y - G.MOUTH[1]) * d[1]
    perp = np.abs((x - G.MOUTH[0]) * n[0] + (y - G.MOUTH[1]) * n[1])
    sel = filled & (perp < cell)
    return float(along[sel].max()) - depth


def full_hole(label, cell):
    f = np.load(os.path.join(R, "A", "final_%s.npz" % label))
    filled = np.asarray(f["s"]) >= 0.5
    filled[np.asarray(f["hole"], dtype=int)] = False
    return x_f(np.asarray(f["x"]), np.asarray(f["y"]), filled, G.HOLE_LEN, cell)


def stage1(sub, depth, cell):
    a = np.loadtxt(os.path.join(R, "C", sub, "cells.csv"), delimiter=",", skiprows=1)
    return x_f(a[:, 0], a[:, 1], a[:, 5] == 0, depth, cell)


def main():
    A = dict((r["label"], r) for r in json.load(open(os.path.join(R, "A", "results.json"))))
    C = dict((r["label"], r) for r in json.load(open(os.path.join(R, "C", "results.json"))))
    rows = []
    for name, (va, vb), (xa, xb), lm in (
            ("uniform 0.18, full hole", (A["uniform_2.5"]["v_in"], A["uniform_1.25"]["v_in"]),
             (full_hole("uniform_2.5", 2.5), full_hole("uniform_1.25", 1.25)), l_max(0.18)),
            ("gradient, full hole", (A["gradient_2.5"]["v_in"], A["gradient_1.25"]["v_in"]),
             (full_hole("gradient_2.5", 2.5), full_hole("gradient_1.25", 1.25)), None),
            ("uniform 0.18, 7 m segment (stage 1)",
             (C["C_struct/base_H17_c2.5/7_10_17"]["stages"][0]["v_in"],
              C["C_p2/phi0.18_H17_c1.25/7_17"]["stages"][0]["v_in"]),
             (stage1("C_struct/base_H17_c2.5/7_10_17", 7.0, 2.5),
              stage1("C_p2/phi0.18_H17_c1.25/7_17", 7.0, 1.25)), l_max(0.18))):
        rows.append("%-36s V_in %8.3f | %8.3f (%+.1f%%)   x_f %6.2f | %6.2f m (diff %.2f m)%s" % (
            name, va, vb, 100.0 * (vb - va) / va, xa, xb, abs(xb - xa),
            "   x_f/L_max %.3f | %.3f" % (xa / lm, xb / lm) if lm else ""))
    lines = ["Two-grid agreement under the rate rule (2.5 m | 1.25 m), from outputs/rerun_rate (two passes bit-identical)", ""]
    lines += rows
    lines += ["", "x_f / L_max at each rate-stopped stage (layer C, eps 0.1):"]
    vals = {}
    for lab, r in sorted(C.items()):
        for st in r["stages"]:
            if st["reason"] == "rate" and st["v_in"] > 1.0 and st.get("x_f_over_L") == st.get("x_f_over_L"):
                vals.setdefault(r["cell"], []).append(st["x_f_over_L"])
                lines.append("  %-40s stage %d  cell %.2f  x_f/L_max %.3f" % (lab, st["k"], r["cell"], st["x_f_over_L"]))
    for cell in sorted(vals):
        lines.append("  range at %.2f m: %.3f .. %.3f (%d stages)" % (cell, min(vals[cell]), max(vals[cell]), len(vals[cell])))
    text = "\n".join(lines)
    open(os.path.join(R, "two_grid_check.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
