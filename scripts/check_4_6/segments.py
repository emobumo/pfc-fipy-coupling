# -*- coding: utf-8 -*-
"""
4.6 check, anchors SG01 and SG02 (post-processing only, no solver):

    powershell -File scripts\\run_local.ps1 scripts\\check_4_6\\segments.py

SG01  x_f against L_max over the 19 independent segments of the 1.25 m P2/P3
      runs: least-squares slope through the origin (c), the 22-segment slope
      (the n 0.14 7 m stage 1 counted once per plan), the x_f/L_max range.
      19 = 9 full-hole runs + 7 first stages + 3 effective second stages;
      the critical n 0.12 second stage is left out.
SG02  the values x_f can take on the 1.25 m grid: spacing of the axial
      projections of the cell centres in the strip (perpendicular distance
      to the hole axis < dx), 5-40 m along the axis; each segment's
      deviation from c in cells, (x_f - c L_max) / dx. Also the offset of the
      nominal depth d_k from the last source-cell centre (x_f is measured
      from d_k).

Input: outputs/rerun_rate/C/results.json; C_pass2/results.json must agree.
Output: outputs/check_4_6/segments.txt
"""
from __future__ import print_function

import imp
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

S = imp.load_source("inclined_hole_staged", os.path.join(REPO, "cases", "inclined_hole_staged.py"))
G = S.G
RR = os.path.join(REPO, "outputs", "rerun_rate")
OUT = os.path.join(REPO, "outputs", "check_4_6")
CELL = 1.25

# (label, stage) -- the 19 independent segments
SEG = [("C_p2/phi0.10_H17_c1.25/single", 1), ("C_p2/phi0.14_H17_c1.25/single", 1),
       ("C_p2/phi0.18_H17_c1.25/single", 1), ("C_p3phi/phi0.12_H17_c1.25/single", 1),
       ("C_p3phi/phi0.16_H17_c1.25/single", 1), ("C_p3phi/phi0.22_H17_c1.25/single", 1),
       ("C_p3len/phi0.14_H10_c1.25/single", 1), ("C_p3len/phi0.14_H25_c1.25/single", 1),
       ("C_p3len/phi0.14_H32_c1.25/single", 1),
       ("C_p2/phi0.10_H17_c1.25/7_17", 1), ("C_p2/phi0.14_H17_c1.25/7_17", 1),
       ("C_p2/phi0.18_H17_c1.25/7_17", 1), ("C_p2/phi0.18_H17_c1.25/5_17", 1),
       ("C_p3phi/phi0.12_H17_c1.25/7_17", 1), ("C_p3phi/phi0.16_H17_c1.25/7_17", 1),
       ("C_p3phi/phi0.22_H17_c1.25/7_17", 1),
       ("C_p2/phi0.10_H17_c1.25/7_17", 2), ("C_p3len/phi0.14_H25_c1.25/7_25", 2),
       ("C_p3len/phi0.14_H32_c1.25/7_32", 2)]
# the same n 0.14 7 m first stage, repeated in the other three plans
REPEATS = [("C_p3len/phi0.14_H10_c1.25/7_10", 1), ("C_p3len/phi0.14_H25_c1.25/7_25", 1),
           ("C_p3len/phi0.14_H32_c1.25/7_32", 1)]


def load(name, suffix):
    return dict((r["label"].replace(suffix, ""), r) for r in json.load(open(os.path.join(RR, name, "results.json"))))


def seg(rows, lab, k):
    r = rows[lab]
    return r["L_max"], r["stages"][k - 1]["x_f"]


def slope(rows, segs):
    L = np.array([seg(rows, l, k)[0] for l, k in segs])
    X = np.array([seg(rows, l, k)[1] for l, k in segs])
    return float(np.sum(L * X) / np.sum(L * L)), X / L


def main():
    rows, rows2 = load("C", ""), load("C_pass2", "_pass2")
    same = all(seg(rows, l, k) == seg(rows2, l, k) for l, k in SEG + REPEATS)
    c19, q19 = slope(rows, SEG)
    c22, q22 = slope(rows, SEG + REPEATS)
    lines = ["4.6 check: x_f against L_max (SG01, SG02), 1.25 m, rate stop rule eps 0.1", "",
             "C and C_pass2 give the same L_max and x_f for every segment: %s" % same, "",
             "SG01", "  19 independent segments: slope through the origin c = %.6f; x_f/L_max %.6f .. %.6f" % (
                 c19, q19.min(), q19.max()),
             "  22 segments (n 0.14 7 m stage 1 once per plan): slope %.6f; x_f/L_max %.6f .. %.6f" % (
                 c22, q22.min(), q22.max()), "",
             "  %-34s %5s %5s %10s %10s %9s %10s" % ("segment", "stage", "n", "L_max m", "x_f m", "x_f/L", "dev cells")]
    devs = []
    for l, k in SEG:
        L, x = seg(rows, l, k)
        dev = (x - c19 * L) / CELL
        devs.append(dev)
        lines.append("  %-34s %5d %5.2f %10.6f %10.6f %9.6f %+10.3f" % (l, k, rows[l]["n_ref"], L, x, x / L, dev))
    i = int(np.argmax(np.abs(devs)))
    lines.append("  largest |deviation| from c: %.3f cells (%s, stage %d)" % (abs(devs[i]), SEG[i][0], SEG[i][1]))

    G.CELL = CELL
    try:
        _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, CELL)
        mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        nx = int(round((G.X_MAX - G.X_MIN) / CELL))
        ny = int(round((G.Y_MAX - G.Y_MIN) / CELL))
        end, dv, nv = G.hole_geometry()
        along = (mx - G.MOUTH[0]) * dv[0] + (my - G.MOUTH[1]) * dv[1]
        perp = np.abs((mx - G.MOUTH[0]) * nv[0] + (my - G.MOUTH[1]) * nv[1])
        vals = np.unique(np.round(along[perp < CELL], 6))
        gaps = np.diff(vals[(vals > 5) & (vals < 40)])
        lines += ["", "SG02",
                  "  strip (perpendicular distance < %.2f m), axial projections 5-40 m: spacing min %.4f median %.4f max %.4f m" % (
                      CELL, gaps.min(), np.median(gaps), gaps.max()),
                  "  segments deviate from c by at most %.3f cells (table above)" % abs(devs[i]), "",
                  "  nominal depth d_k vs last source-cell centre along the axis (x_f is measured from d_k):"]
        for d in (5.0, 7.0, 10.0, 17.0, 25.0, 32.0):
            hc = np.asarray(S.hole_cells_to_depth(mx, my, nx, ny, d), dtype=int)
            lines.append("    d_k %5.1f m: %2d source cells, last centre at %.4f m, offset %+.4f m" % (
                d, len(hc), along[hc].max(), along[hc].max() - d))
    finally:
        G.CELL = 2.5
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "segments.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
