# -*- coding: utf-8 -*-
"""
Numbers for table 4.4-1 that the verification-ladder tests assert but do not
print. Reuses the test fixtures and scripts/conic_check.py unchanged.

    powershell -File scripts\\run_local.ps1 scripts\\ladder_extra_check.py

1. Step 7 (between-stage closure): per stage, the relative residual of
   sum (n_k - n_{k+1})V + floor_clip = sum n_k S_k V, and whether
   n_{k+1} == max(n_k(1-S_k), floor) holds on every cell bit for bit.
2. 2D point source vs the focal conic r = L_max/(1 + Pi_g sin theta), 36
   sectors, n 0.10/0.18/0.30, cells 1.25/0.625 m, gravity on/off: worst
   under-statement, the margin to the angle-dependent metric bound
   2.75 %/(1 + Pi_g sin theta), the area ratio, and the largest per-sector
   difference between the two grids.
3. Step 2b: the front-vs-analytic deviation with and without the first 20 %
   of the march that the test skips, and where the largest one sits.

Output: outputs/ladder_extra/ladder_extra.txt
"""
from __future__ import print_function

import imp
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (REPO, os.path.join(REPO, "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)

T = imp.load_source("test_verification_ladder", os.path.join(REPO, "tests", "test_verification_ladder.py"))
C = imp.load_source("conic_check", os.path.join(REPO, "scripts", "conic_check.py"))
OUT = os.path.join(REPO, "outputs", "ladder_extra")
BOUND = 0.0275


def step7():
    recs = T._run_step7_stages()
    vols = T._STEP7_CACHE["vols"]
    floor = T.get_stage_porosity_floor(T._STEP7_CACHE["state"]["slurry_parameters"])
    lines = ["1. Step 7 between-stage closure (1D, %.2f m, %d cells, dx %.3f m, %d stages, floor %g)"
             % (T.S7_L, T.S7_NX, T.S7_L / T.S7_NX, T.S7_STAGES, floor)]
    worst, exact_all = 0.0, True
    for r in recs:
        removed = float(np.sum((r["phi_before"] - r["phi_after"]) * vols))
        occ = r["report"]["occupied_this_stage"]
        clip = r["report"]["floor_clipped_step"]
        res = abs(removed + clip - occ) / max(abs(occ), 1.0e-30)
        exact = bool(np.array_equal(r["phi_after"], np.maximum(r["phi_before"] * (1.0 - r["s_before"]), floor)))
        worst, exact_all = max(worst, res), exact_all and exact
        lines.append("   stage %d: occupied %.10e  removed %.10e  floor clip %.10e  rel residual %.3e  cell-wise exact %s"
                     % (r["stage"] + 1, occ, removed, clip, res, exact))
    lines.append("   max relative residual %.3e; n_{k+1} == max(n_k(1-S_k), floor) on every cell in every stage: %s"
                 % (worst, exact_all))
    return lines


def conic():
    lines = ["", "2. Point source vs focal conic (36 sectors; err = r_meas/r_an - 1; bound = %.2f%%/(1 + Pi_g sin theta))"
             % (100 * BOUND),
             "   %4s %6s %5s | %8s | %9s %9s | %7s %7s | %s" % ("n", "cell", "grav", "area", "worst err", "at deg",
                                                              "margin", "at deg", "sectors in bound")]
    per = {}
    total, inside, min_margin = 0, 0, 1.0
    for n in C.NS:
        for g in (True, False):
            for cell in C.CELLS:
                st, mx, my, src, prm = C.build(n, cell)
                lam = C.lam_of(n, prm)
                L = C.P0 / lam
                pig = prm["slurry_density"] * 9.81 / lam if g else 0.0
                res = C.reachable_domain(st, [src], p0=C.P0, use_gravity=g)
                cost = np.asarray(res["cost"], dtype=float)
                area = float(np.sum(res["reachable"])) * cell * cell / (math.pi * L * L / (1.0 - pig * pig) ** 1.5)
                xs, ys = np.unique(mx), np.unique(my)
                errs, margins, ths = [], [], []
                for k in range(C.N_SECTORS):
                    th = 2 * math.pi * (k + 0.5) / C.N_SECTORS
                    rm = C.boundary_radius(cost, xs, ys, mx[src], my[src], th, C.HALF - 2 * cell, cell / 20.0)
                    e = rm / (L / (1.0 + pig * math.sin(th))) - 1.0
                    errs.append(e)
                    margins.append(BOUND / (1.0 + pig * math.sin(th)) + e)   # >= 0: under-statement within bound
                    ths.append(math.degrees(th))
                errs, margins = np.array(errs), np.array(margins)
                per[(n, g, cell)] = errs
                total += errs.size
                inside += int(np.sum((margins >= 0.0) & (errs <= 0.0)))
                min_margin = min(min_margin, float(margins.min()))
                iw, im = int(np.argmin(errs)), int(np.argmin(margins))
                lines.append("   %4.2f %6.3f %5s | %8.4f | %+8.3f%% %9.1f | %6.3f%% %7.1f | %d/%d"
                             % (n, cell, "on" if g else "off", area, 100 * errs[iw], ths[iw], 100 * margins[im],
                                ths[im], int(np.sum((margins >= 0.0) & (errs <= 0.0))), errs.size))
    lines.append("   all sectors: %d of %d within the bound and never over-stated; smallest margin %.3f%%"
                 % (inside, total, 100 * min_margin))
    worst_grid = 0.0
    for n in C.NS:
        for g in (True, False):
            worst_grid = max(worst_grid, float(np.max(np.abs(per[(n, g, 1.25)] - per[(n, g, 0.625)]))))
    lines.append("   largest per-sector difference between the 1.25 m and 0.625 m grids: %.3f%%" % (100 * worst_grid))
    for g in (False, True):
        lines.append("   worst under-statement, gravity %-3s: %s" % ("on" if g else "off", " / ".join(
            "n %.2f %.2f%%" % (n, -100 * min(per[(n, g, c)].min() for c in C.CELLS)) for n in C.NS)))
    return lines


def step2b():
    _, traj, _ = T._run_step2b_march()
    dx = T.S2_L / float(T.S2B_NX)
    start = max(int(0.2 * len(traj)), 1)
    rows = []
    for i, (t, f) in enumerate(traj):
        rows.append((i, t, f, T._fill_front(t), abs(f - T._fill_front(t)) / T.S2B_LMAX))
    full = max(rows, key=lambda r: r[4])
    kept = max(rows[start:], key=lambda r: r[4])
    head = max(rows[:start], key=lambda r: r[4])
    lines = ["", "3. Step 2b front vs analytic (%d steps, dx %.3f m = %.4f L_max; test skips steps 1..%d)"
             % (len(traj), dx, dx / T.S2B_LMAX, start),
             "   whole march : max dev %.4f L_max at step %d, t %.4f s, front %.4f m (%.2f cells), analytic %.4f m"
             % (full[4], full[0] + 1, full[1], full[2], full[2] / dx, full[3]),
             "   first 20%%   : max dev %.4f L_max at step %d, t %.4f s, front %.4f m (%.2f cells), analytic %.4f m"
             % (head[4], head[0] + 1, head[1], head[2], head[2] / dx, head[3]),
             "   after 20%%   : max dev %.4f L_max at step %d, t %.4f s, front %.4f m (%.2f cells), analytic %.4f m"
             % (kept[4], kept[0] + 1, kept[1], kept[2], kept[2] / dx, kept[3]),
             "   at the 20%% mark: step %d, t %.4f s, front %.4f m (%.2f cells, %.3f L_max)"
             % (start + 1, traj[start][0], traj[start][1], traj[start][1] / dx, traj[start][1] / T.S2B_LMAX),
             "   first 12 steps (step, t, front, analytic, dev/L_max):"]
    for i, t, f, a, d in rows[:12]:
        lines.append("     %3d  %.5f  %.5f  %.5f  %.4f" % (i + 1, t, f, a, d))
    return lines


def main():
    lines = step7() + conic() + step2b()
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "ladder_extra.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
