# -*- coding: utf-8 -*-
"""S1 re-run against the first S1 run (main line 2026-10-19): V_in and filled-cell count at every common
20-step record, and V_in / model time at every common step.

    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\s1_compare.py

First run: outputs/xz_solver/s1_first/final_S1.npz (archived after the replacement; records [step, t, dt, V_in, filled, loss]; default MKL threads for the
first 600 steps, no subnormal flush). Re-run: outputs/xz_solver/s1_rerun/final_S1.npz (= the S1 now in outputs/xz_solver; records [step, t, dt, V_in,
filled, filled in the reach domain, filled outside]; one thread, flush from step 1). "filled" = rubble cells with
S >= 0.5 in both. Pass criterion: largest relative difference <= 1e-6. Output: outputs/xz_solver/s1_compare.txt
"""
from __future__ import print_function

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(REPO, "outputs", "xz_solver")
TOL = 1e-6


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.abs(a - b) / np.maximum(np.abs(b), 1e-300)


def main():
    o = np.load(os.path.join(OUT, "s1_first", "final_S1.npz"))
    n = np.load(os.path.join(OUT, "s1_rerun", "final_S1.npz"))
    so, sn = o["series"], n["series"]
    ko = dict((int(r[0]), r) for r in so)
    common = [r for r in sn if int(r[0]) in ko]
    L = ["S1 re-run vs first run: %d common 20-step records (steps %d..%d); re-run %d records, first run %d"
         % (len(common), common[0][0], common[-1][0], len(sn), len(so))]
    rv = rel([r[3] for r in common], [ko[int(r[0])][3] for r in common])
    rf = rel([r[4] for r in common], [ko[int(r[0])][4] for r in common])
    rt = rel([r[1] for r in common], [ko[int(r[0])][1] for r in common])
    for name, d in (("V_in", rv), ("filled cells", rf), ("model time", rt)):
        i = int(np.argmax(d))
        L.append("  %-12s max relative difference %.3e at step %d; records with any difference %d"
                 % (name, d[i], common[i][0], int((d > 0).sum())))
    k = min(len(n["v_hist"]), len(o["v_hist"]))
    dv = rel(n["v_hist"][1:k], o["v_hist"][1:k])
    dt = rel(n["t_hist"][1:k], o["t_hist"][1:k])
    L.append("  every step 1..%d: V_in max relative difference %.3e (step %d); model time %.3e (step %d)"
             % (k - 1, dv.max(), int(np.argmax(dv)) + 1, dt.max(), int(np.argmax(dt)) + 1))
    worst = max(rv.max(), rf.max())
    L.append("  criterion (records, V_in and filled cells) <= %.0e: %s" % (TOL, "PASS" if worst <= TOL else "FAIL"))
    txt = "\n".join(L) + "\n"
    open(os.path.join(OUT, "s1_compare.txt"), "w").write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
