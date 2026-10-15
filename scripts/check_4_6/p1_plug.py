# -*- coding: utf-8 -*-
"""
4.6 check, anchor PL01: the pre-plugged collar-channel runs (P1, collar_v06,
2.5 m). Post-processing only (reach recomputed, no solver):

    powershell -File scripts\\run_local.ps1 scripts\\check_4_6\\p1_plug.py

For h = 0 (no plug), 2.5 and 10 m (the plug: scripts/stage_supplement.py
`plugged`, band cells within h above and below the hole crossing, hole cells
left open, n = 1e-3):
  - minimum required source pressure p_req/p0 over the channel band cells that
    are neither hole nor plug cells (the channel part outside the hole);
  - reach cells, reach cells outside the design domain;
  - design-domain cells (165, source cells excluded) that are unreachable, and
    whether they are exactly the plug cells;
  - the design-domain fill rate three ways: plug cells counted as filled
    (the chapter's rule), plug cells unfilled (old), plug cells taken out of
    the denominator.
Filled = S >= 0.5 in B/B_p1_preplug_h*/cells.csv; the reach column there must
equal the recomputed reach cell by cell.

Output: outputs/check_4_6/p1_plug.txt
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain

SS = imp.load_source("stage_supplement", os.path.join(REPO, "scripts", "stage_supplement.py"))
C, G = SS.C, SS.G
RR = os.path.join(REPO, "outputs", "rerun_rate")
OUT = os.path.join(REPO, "outputs", "check_4_6")
RUNS = {0.0: "B_p1_full_hole", 2.5: "B_p1_preplug_h2.50", 10.0: "B_p1_preplug_h10.00"}


def main():
    mx, my, nx, ny, hole, phi, band, _ = SS.setup()
    hm = np.zeros(mx.size, dtype=bool)
    hm[hole] = True
    target = np.load(os.path.join(RR, "A", "target_mask.npy")).astype(bool)
    T = target & ~hm
    lines = ["4.6 check: pre-plugged collar channel (PL01), 2.5 m, reach with gravity, 16-neighbour", "",
             "design domain without source cells: %d cells; channel band cells: %d, of which in the design domain: %d" % (
                 int(T.sum()), int(band.sum()), int(np.sum(T & band))), ""]
    for h in (0.0, 2.5, 10.0):
        if h == 0.0:
            field, plug = phi, np.zeros(mx.size, dtype=bool)
        else:
            field, plug = SS.plugged(phi, band, hole, my, h)
        st = C.build_from_phi(field)[0]
        r = reachable_domain(st, hole, p0=G.P0, use_gravity=True)
        reach = np.asarray(r["reachable"], dtype=bool)
        cost = np.asarray(r["cost"]) / G.P0
        a = np.loadtxt(os.path.join(RR, "B", RUNS[h], "cells.csv"), delimiter=",", skiprows=1)
        assert np.allclose(a[:, 0], mx) and np.allclose(a[:, 1], my)
        s, reach_csv = np.clip(a[:, 3], 0.0, 1.0), a[:, 5] > 0.5
        filled = s >= 0.5
        open_band = band & ~hm & ~plug
        un = T & ~reach
        f_old = int(np.sum(filled & T))
        f_new = int(np.sum((filled | plug) & T))
        den = int(np.sum(T & ~plug))
        f_out = int(np.sum(filled & T & ~plug))
        lines += ["h = %.2f m  (%s)" % (h, RUNS[h]),
                  "  plug cells %d, in the design domain %d; plug S max %.4f, plug cells filled %d" % (
                      int(plug.sum()), int(np.sum(plug & T)), float(s[plug].max()) if plug.any() else 0.0,
                      int(np.sum(filled & plug))),
                  "  channel part outside the hole (band, not hole, not plug): %d cells, reachable %d; min p_req/p0 %.4f, max %.4f" % (
                      int(open_band.sum()), int(np.sum(reach & open_band)), cost[open_band].min(), cost[open_band].max()),
                  "  reach %d cells, outside the design domain %d; cells.csv reach column identical: %s" % (
                      int(reach.sum()), int(np.sum(reach & ~target)), bool(np.array_equal(reach, reach_csv))),
                  "  design cells unreachable %d; identical to the plug cells in the design domain: %s" % (
                      int(un.sum()), bool(np.array_equal(un, plug & T))),
                  "  design fill, plug counted filled:      %3d/165 = %.6f%%" % (f_new, 100.0 * f_new / 165),
                  "  design fill, plug counted unfilled:    %3d/165 = %.6f%%   (old, not cited)" % (f_old, 100.0 * f_old / 165),
                  "  design fill, plug out of denominator:  %3d/%d = %.6f%%   (not cited)" % (f_out, den, 100.0 * f_out / den),
                  "  design front-shortfall cells (reachable, unfilled, not plug): %d" % int(np.sum(T & reach & ~filled & ~plug)),
                  ""]
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "p1_plug.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
