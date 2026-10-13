# -*- coding: utf-8 -*-
"""
Inter-stage shadow by eq. (4-27), post-processing only (no solver):

    Omega_shadow = (Omega_design & Omega_reach^w) \\ U_k Omega_reach^(k)

    powershell -File scripts\\run_local.ps1 scripts\\shadow_eq427.py

The `shadowed` column the staged runs write (cases/inclined_hole_staged.py)
is NOT this: it is reach^w & unfilled & ~reach_final, where reach_final is a
hypothetical further pass from the full hole in the post-sequence ground,
and it is not intersected with the design domain. Do not cite it.

Here, for two-stage plans whose second stage failed (it took ~1e-3 m3/m):
  Omega_reach^(1)  stage-1 hole section, ORIGINAL porosity field (exact)
  Omega_reach^(2)  stage-2 hole section, FINAL porosity field, standing in
                   for the field at the start of stage 2 (the failed stage
                   changed it only where its ~1e-3 m3/m went)
  Omega_reach^w    the single-pass reach of the full hole (reachable_virgin)
  Omega_design     A/target_mask.npy: the 2.5 m uniform-0.18 stop footprint
                   (175 cells, hole included); on 1.25 m grids each 2.5 m
                   cell is split into its four 1.25 m cells.
Checks: the reach cell counts against the `reach_now` counts the runs logged
on the true stage-start fields; Omega_fill inside the union of the stage
reaches. Approximation error of Omega_reach^(2): for n 0.14 the 7->17 and
7->10 plans share a bit-identical stage 1, so the cells where their final
porosities differ are the cells either failed stage 2 touched.

Output: outputs/shadow_eq427/shadow_eq427.txt
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

from src.analysis.fill_diagnostics import reachable_domain

S = imp.load_source("inclined_hole_staged", os.path.join(REPO, "cases", "inclined_hole_staged.py"))
G = S.G
RR = os.path.join(REPO, "outputs", "rerun_rate")
OUT = os.path.join(REPO, "outputs", "shadow_eq427")

# (label in C/results.json, cell, hole length) -- two-stage plans, second stage failed
CASES = [("C_p2/phi0.14_H17_c1.25/7_17", 1.25, 17.0),
         ("C_p2/phi0.18_H17_c1.25/7_17", 1.25, 17.0),
         ("C_p2/phi0.18_H17_c1.25/5_17", 1.25, 17.0),
         ("C_p3phi/phi0.16_H17_c1.25/7_17", 1.25, 17.0),
         ("C_p3phi/phi0.22_H17_c1.25/7_17", 1.25, 17.0),
         ("C_p3len/phi0.14_H10_c1.25/7_10", 1.25, 10.0),
         ("C_struct/base_H17_c2.5/8.5_17", 2.5, 17.0),
         ("C_struct/random_H17_c2.5/8.5_17", 2.5, 17.0)]


def grid(cell):
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, cell)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return mx, my, int(round((G.X_MAX - G.X_MIN) / cell)), int(round((G.Y_MAX - G.Y_MIN) / cell))


def design_mask(mx, my):
    t = np.load(os.path.join(RR, "A", "target_mask.npy")).astype(bool)
    cx, cy, _, _ = grid(2.5)
    assert t.size == cx.size
    ix = np.floor((mx - G.X_MIN) / 2.5).astype(int)
    iy = np.floor((my - G.Y_MIN) / 2.5).astype(int)
    lookup = {}
    for k in range(cx.size):
        lookup[(int(np.floor((cx[k] - G.X_MIN) / 2.5)), int(np.floor((cy[k] - G.Y_MIN) / 2.5)))] = t[k]
    return np.array([lookup[(a, b)] for a, b in zip(ix, iy)], dtype=bool)


def reach(phi, src):
    st = S.C.build_from_phi(phi)[0]
    return np.asarray(reachable_domain(st, src, p0=G.P0, use_gravity=True)["reachable"], dtype=bool)


def one(label, cell, hole_len, logged):
    G.CELL, G.HOLE_LEN = cell, hole_len
    try:
        mx, my, nx, ny = grid(cell)
        a = np.loadtxt(os.path.join(RR, "C", label, "cells.csv"), delimiter=",", skiprows=1)
        assert np.allclose(a[:, 0], mx) and np.allclose(a[:, 1], my)
        phi0, n_final, cat = a[:, 2], a[:, 3], a[:, 9].astype(int)
        reach_w, code_shadow = a[:, 6] > 0.5, a[:, 8] > 0.5
        filled = (cat == 0) | (cat == 4)
        d1, d2 = [s["depth"] for s in logged["stages"]]
        src1 = np.asarray(S.hole_cells_to_depth(mx, my, nx, ny, d1), dtype=int)
        src2 = np.asarray(S.hole_cells_to_depth(mx, my, nx, ny, d2), dtype=int)
        r1, r2 = reach(phi0, src1), reach(n_final, src2)
        union = r1 | r2
        design = design_mask(mx, my)
        shadow = reach_w & np.logical_not(union)
        return {
            "label": label, "cell": cell, "area": cell * cell, "n_final": n_final, "phi0": phi0,
            "r1": int(r1.sum()), "r1_log": logged["stages"][0]["reach_now"],
            "r2": int(r2.sum()), "r2_log": logged["stages"][1]["reach_now"],
            "r2_is_src": bool(np.array_equal(np.sort(np.where(r2)[0]), np.sort(src2))),
            "v2": logged["stages"][1]["v_in"],
            "reach_w": int(reach_w.sum()), "union": int(union.sum()),
            "fill_outside_union": int(np.sum(filled & np.logical_not(union))),
            "shadow_all": int(shadow.sum()), "shadow_design": int(np.sum(shadow & design)),
            "design": int(design.sum()), "design_reach_w": int(np.sum(design & reach_w)),
            "code_shadow": int(code_shadow.sum()), "code_shadow_design": int(np.sum(code_shadow & design)),
            "code_minus_427": int(np.sum(code_shadow & np.logical_not(shadow))),
            "427_minus_code": int(np.sum(shadow & np.logical_not(code_shadow))),
        }
    finally:
        G.CELL, G.HOLE_LEN = 2.5, 17.0


def main():
    rows = dict((r["label"], r) for r in json.load(open(os.path.join(RR, "C", "results.json"))) if "stages" in r)
    res = [one(lab, cell, hl, rows[lab]) for lab, cell, hl in CASES]
    lines = ["Inter-stage shadow by eq. (4-27), two-stage plans with a failed second stage (post-processing)", "",
             "%-34s %5s | %5s %5s | %5s %5s %3s | %9s | %5s %5s | %4s | %-17s | %-17s | %s" % (
                 "plan", "cell", "R1", "log", "R2", "log", "src", "V2 m3/m", "R^w", "U", "fill\\U",
                 "(4-27) all/design", "code all/design", "code\\427 / 427\\code")]
    for r in res:
        lines.append("%-34s %5.2f | %5d %5d | %5d %5d %3s | %9.2e | %5d %5d | %4d | %7d / %-7d | %7d / %-7d | %d / %d" % (
            r["label"], r["cell"], r["r1"], r["r1_log"], r["r2"], r["r2_log"], "yes" if r["r2_is_src"] else "no",
            r["v2"], r["reach_w"], r["union"], r["fill_outside_union"], r["shadow_all"], r["shadow_design"],
            r["code_shadow"], r["code_shadow_design"], r["code_minus_427"], r["427_minus_code"]))
    lines += ["", "R1/R2: cells in Omega_reach^(1)/(2) here vs `reach_now` logged on the true stage-start field;",
              "src: Omega_reach^(2) is exactly the stage-2 source cells; U: union; fill\\U: Omega_fill cells outside U;",
              "design: %d cells on 2.5 m, %d on 1.25 m." % (
                  [r for r in res if r["cell"] == 2.5][0]["design"], [r for r in res if r["cell"] == 1.25][0]["design"])]
    by = dict((r["label"], r) for r in res)
    a, b = by["C_p2/phi0.14_H17_c1.25/7_17"], by["C_p3len/phi0.14_H10_c1.25/7_10"]
    G.CELL, G.HOLE_LEN = 1.25, 17.0
    try:
        diff = a["n_final"] != b["n_final"]
    finally:
        G.CELL, G.HOLE_LEN = 2.5, 17.0
    dn = np.abs(a["n_final"] - b["n_final"])
    lines += ["", "Approximation check, n 0.14: 7->17 and 7->10 share a bit-identical stage 1.",
              "  cells whose final porosity differs between the two plans: %d (max |dn| %.3e; porosity there %.3e .. %.3e)" % (
                  int(diff.sum()), float(dn.max()) if diff.any() else 0.0,
                  float(np.min(a["n_final"][diff])) if diff.any() else 0.0,
                  float(np.max(a["n_final"][diff])) if diff.any() else 0.0)]
    for r in res:
        lines.append("  %-34s cells changed from the original field anywhere: %d" % (
            r["label"], int(np.sum(r["n_final"] != r["phi0"]))))
    # Share of the coverage loss (single - staged, filled share of reach^w) that eq. (4-27) calls shadow.
    # Unfilled cells in reach^w are the runs' `shadowed` counts (equal to the shortfall on full-depth runs).
    lines += ["", "Coverage loss vs eq. (4-27) shadow (cells; loss cells = unfilled in reach^w, staged minus single):"]
    for r in res:
        st, si = rows[r["label"]], rows[r["label"].rsplit("/", 1)[0] + "/single"]
        loss_cells = st["shadowed"] - si["shadowed"]
        r["loss_points"] = 100.0 * (si["over_reach_filled"] - st["over_reach_filled"])
        r["loss_cells"] = loss_cells
        lines.append("  %-34s loss %5.1f pts = %3d cells | (4-27) shadow %3d = %5.1f%% of the loss cells | single quota-stopped: %s" % (
            r["label"], r["loss_points"], loss_cells, r["shadow_all"], 100.0 * r["shadow_all"] / loss_cells,
            "yes" if si["stages"][0]["reason"] == "quota" else "no"))
    for r in res:
        r.pop("n_final"); r.pop("phi0")
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "shadow_eq427.txt"), "w").write(text + "\n")
    json.dump(res, open(os.path.join(OUT, "shadow_eq427.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
