# -*- coding: utf-8 -*-
"""
4.8 pre-draft check, solver part: post-processing of the stored step-2B cases (outputs/xz_solver, written by
scripts/xz_solver/run_cases.py), no solver.

    powershell -File scripts\\run_local.ps1 scripts\\check_4_8\\runs.py

Per case (reach domain = same hole, same p0, 0.5 m; all fill counted inside it only):
  - arrival at the lower end of the basal layer (first lower-end cell with S >= 0.5, as recorded during the run at
    step resolution): model time, V_in, V_in / reach-domain pore volume, and the reach-fill fraction at the two
    20-step records around it (the fill count at the arrival step itself was not stored);
  - the 0.90 / 0.95 / 0.99 snapshots: time, V_in, V_in / pore, hanging-side fill, void contact, cells filled
    outside the reach domain (numerical creep), rate / threshold, downward basal storage (the former "loss"
    zone: basal cells with t <= collar t - 5 m) and the lower-end corner fill under three definitions
    (rubble below the drift-1 collar elevation; down-dip of the drift-1 collar; the old box X < -5, Z < 20 m).
Cases whose result is marked approximate (S1 before its re-run) are listed as such.

Output: outputs/check_4_8/runs.txt
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

RC = imp.load_source("xz_run_cases", os.path.join(REPO, "scripts", "xz_solver", "run_cases.py"))
SRC = RC.OUT
OUT = os.path.join(REPO, "outputs", "check_4_8")


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    L = []
    w = L.append
    D, C, srcs, collar_t = RC.load()
    box = RC.crop_box(C)
    j0, j1, i0, i1 = box
    cr = lambda a: a[j0:j1, i0:i1]
    rub, hang, low = cr(C["rub"]), cr(C["hang"]), cr(C["low"])
    X, Z, T = cr(C["X"]), cr(C["Z"]), cr(C["T"])
    zc1, tc1 = float(D["fan_line"][1][1]), float(D["fan_collar_t"][1])
    corners = [("below collar Z %.2f" % zc1, rub & (Z < zc1)), ("down-dip of collar t %.4f" % tc1, rub & (T < tc1)),
               ("old box X < -5, Z < 20", rub & (X < -5.0) & (Z < 20.0))]
    w("4.8 pre-draft check, solver part (runs.py): step-2B cases, scenario 1, K 1.25 v4, 0.5 m crop; metrics inside the"
      " reach domain only")
    w("  corners (0.5 m): %s" % "; ".join("%s %.2f m2 (hanging side %.2f)" % (n_, m.sum() * RC.CELL ** 2,
                                                                          (m & hang).sum() * RC.CELL ** 2)
                                          for n_, m in corners))
    w("")
    for tag, label, _, p0 in RC.CASES:
        M, st, src = RC.case_masks(tag, p0, C, srcs, collar_t, box)
        reach = M["reach"].reshape(rub.shape)
        pore = float((M["n"] * M["vols"])[M["reach"]].sum())
        fn = os.path.join(SRC, "result_%s.json" % tag)
        w("%s (%s): reach domain %d cells, %.2f m2, pore %.4f m3/m; lower-end cells in it %d of %d"
          % (tag, label, M["nreach"], M["nreach"] * RC.CELL ** 2, pore, (low & reach).sum(), low.sum()))
        if not os.path.isfile(fn):
            w("  not run")
            w("")
            continue
        r = json.load(open(fn))
        approx = bool(r.get("approx"))
        if approx:
            w("  APPROXIMATE (closed from its last checkpoint; snapshots from the 20-step records; to be re-run)")
        w("  end: %s; step %d, t %.2f s, V_in %.4f m3/m" % (r["reason"], r["steps"], r["t"], r["v_in"]))
        a = r.get("arrival") or {}
        series = np.load(os.path.join(SRC, "final_%s.npz" % tag))["series"]
        col = 4 if approx else 5     # S1's early records hold the total filled count (outside cells included)
        if a:
            k = int(a["step"])
            before = series[series[:, 0] <= k]
            after = series[series[:, 0] >= k]
            fb = ("step %d %.4f" % (before[-1][0], before[-1][col] / float(M["nreach"]))) if len(before) else "-"
            fa = ("step %d %.4f" % (after[0][0], after[0][col] / float(M["nreach"]))) if len(after) else "-"
            w("  lower end reached: step %d, t %.4f s, V_in %.4f m3/m, V_in / pore %.4f; reach fill at the 20-step records"
              " around it: %s; %s" % (k, a["t"], a["v_in"], a["v_in"] / pore, fb, fa))
        else:
            w("  lower end not reached%s" % (" (not in the reach domain)" if not (low & reach).any() else ""))
        for key in ("0.90", "0.95", "0.99"):
            if key not in r["snaps"]:
                w("  snapshot %s: not reached" % key)
                continue
            m = r["snaps"][key]
            line = ("  snapshot %s: step %d, t %.4f s, V_in %.4f m3/m, V_in / pore %.4f; downward basal storage %.4f m3/m"
                    " (%.4f of the stored); rate / threshold %.4f" % (key, m["step"], m["t"], m["v_in"], m["v_in"] / pore,
                                                                      m["loss"], m["loss_share"], m["rate_ratio"]))
            if "hang_fill" in m:
                line += "; reach fill %.4f; hanging side %.4f; void contact %.2f m; filled outside %d cells" % (
                    m["frac_in"], m["hang_fill"], m["void_contact"], m["outside"])
            w(line)
            sp = os.path.join(SRC, "snap_%s_%s.npz" % (tag, key))
            if os.path.isfile(sp):
                f = (np.load(sp)["s"] >= 0.5) & rub & reach
                w("     corners filled (share; hanging-side part): %s" % "; ".join(
                    "%s %.4f; %s" % (n_, (f & cm).sum() / float(cm.sum()),
                                     ("%.4f" % ((f & cm & hang).sum() / float((cm & hang).sum()))) if (cm & hang).any() else "-")
                    for n_, cm in corners))
            else:
                w("     no snapshot field stored")
        w("")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "runs.txt"), "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
