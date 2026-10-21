# -*- coding: utf-8 -*-
"""Stage-2 solver cases of a new end state (main line 2026-10-21): post-processing only, no solver.

    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\stage2_summary.py      ($env:XZ_CONFIG = '30m')

Reads outputs/xz_solver<suffix>/result_*.json, final_*.npz and the snapshot fields written by run_cases.py. Every
case is evaluated on its own grid (base cases: the K 1.25 scenario-1 grid; stage-2 cases: their stored grid), the
reach domain being the same hole at the same p0 on that grid; all fill metrics count reach-domain cells only.

Closed cases (A, B): end; arrival at the lower end of the basal layer (first lower-end cell S >= 0.5: step, time,
V_in, V_in / reach pore volume); the 0.90 / 0.95 / 0.99 snapshots (time, V_in, V_in / pore, reach fill,
hanging-side fill, void contact, lower-end corner fill = rubble down-dip of the drift-1 collar, filled outside the
reach domain, rate / threshold, downward basal storage for reference).
Outlet cases (C): cumulative outflow / V_in against time (20-step records), the state when V_in first equals the
closed case's 0.95 value, and the end state (rate / threshold, corner fill, reach fill); first outflow > 1e-3 m3/m;
when the rate rule (4-29) and the creep rule would have stopped the run.

Output: outputs/xz_solver<suffix>/stage2.txt
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
P = RC.P
OUT = RC.OUT
A2 = RC.CELL * RC.CELL
ORDER = ["S1", "S2", "S3", "S4", "S5", "K1", "K2", "K3", "K4", "K5", "O1", "O2"]


def setup(tag, base):
    if tag in RC.CASES2:
        c2 = RC.CASES2[tag]
        C, srcs, ct, box = RC.grid_setup(c2)
        K = float(np.load(os.path.join(OUT, "grid_%s.npz" % c2["grid"]))["K"])
        return c2["label"], c2["p0"], C, srcs, ct, box, K, c2.get("outlet")
    D, C, srcs, ct, box = base
    lab, p0 = [(c_[1], c_[3]) for c_ in RC.CASES if c_[0] == tag][0]
    return lab, p0, C, srcs, ct, box, P.K_BASE, None


def main():
    D, C, srcs, ct = RC.load()
    base = (D, C, srcs, ct, RC.crop_box(C))
    L = ["Stage-2 solver cases, configuration %s (v0.6, gravity, one thread, subnormal flush from step 1, dt_cap %.0f s);"
         " metrics inside the reach domain only" % (RC.CFG["name"], RC.DT_CAP), ""]
    w = L.append
    for tag in ORDER:
        fn = os.path.join(OUT, "result_%s.json" % tag)
        if not os.path.isfile(fn):
            w("%s: not run" % tag)
            w("")
            continue
        label, p0, Cc, sr, ctc, box, K, outlet = setup(tag, base)
        M, st, src = RC.case_masks(tag, p0, Cc, sr, ctc, box)
        j0, j1, i0, i1 = box
        shp = (j1 - j0, i1 - i0)
        rub = M["rub"].reshape(shp)
        reach = M["reach"].reshape(shp)
        hang = M["hang"].reshape(shp)
        Dk = np.load(os.path.join(P.DATA, "base_geometry_K%.2f_v4.npz" % K))
        tc1 = float(Dk["fan_collar_t"][1])
        corner = rub & (Cc["T"][j0:j1, i0:i1] < tc1)
        pore = float((M["n"] * M["vols"])[M["reach"]].sum())
        r = json.load(open(fn))

        def corner_fill(path):
            f = (np.load(path)["s"] >= 0.5) & rub & reach
            return ((f & corner).sum() / float(corner.sum()),
                    (f & corner & hang).sum() / float((corner & hang).sum()) if (corner & hang).any() else float("nan"))
        w("%s (%s; K %.2f grid %s): sources %d; reach domain %d cells = %.2f m2, pore %.4f m3/m; lower-end cells in it %d"
          " of %d; lower-end corner (t < %.4f m) %.2f m2 (hanging side %.2f m2)"
          % (tag, label, K, (r.get("opts") or {}).get("grid", "K1.25_s1"), len(src), M["nreach"], M["nreach"] * A2, pore,
             int((M["low"] & M["reach"]).sum()), int(M["low"].sum()), tc1, corner.sum() * A2, (corner & hang).sum() * A2))
        w("  end: %s; step %d, t %.2f s, V_in %.4f m3/m, wall %.2f h; end state: reach fill %.4f, filled outside %d cells,"
          " rate / threshold %.4f" % (r["reason"], r["steps"], r["t"], r["v_in"], r["wall"] / 3600.0, r["final"]["frac_in"],
                                      r["final"]["outside"], r["final"]["rate_ratio"]))
        a = r.get("arrival") or {}
        if a:
            ser = np.load(os.path.join(OUT, "final_%s.npz" % tag))["series"]
            k = int(a["step"])
            bef, aft = ser[ser[:, 0] <= k], ser[ser[:, 0] >= k]
            w("  lower end reached: step %d, t %.4f s, V_in %.4f m3/m, V_in / pore %.4f; reach fill at the 20-step records"
              " around it: %s; %s" % (a["step"], a["t"], a["v_in"], a["v_in"] / pore,
                                      ("step %d %.4f" % (bef[-1][0], bef[-1][5] / float(M["nreach"]))) if len(bef) else "-",
                                      ("step %d %.4f" % (aft[0][0], aft[0][5] / float(M["nreach"]))) if len(aft) else "-"))
        elif not outlet:
            w("  lower end not reached%s" % (" (not in the reach domain)" if not (M["low"] & M["reach"]).any() else ""))
        if outlet:
            ox = r["outlet"]
            ser = np.load(os.path.join(OUT, "final_%s.npz" % tag))["series"]
            fo = [row for row in ser if row[7] > 1e-3]
            w("  outlet: lower end of the basal layer (%d cells) kept at S = 0; target V_in %.1f x the 0.95 snapshot of %s"
              % (int(M["low"].sum()), outlet["factor"], outlet["ref"]))
            w("  first outflow > 1e-3 m3/m (20-step records): %s" % (
                ("step %d, t %.2f s, V_in %.4f" % (fo[0][0], fo[0][1], fo[0][3])) if fo else "none"))
            w("  rate rule (4-29) would have stopped it: %s; creep rule: %s" % tuple(
                ("step %d, t %.2f s, V_in %.4f" % (q["step"], q["t"], q["v_in"])) if q else "never"
                for q in (ox.get("rate_at"), ox.get("creep_at"))))
            e = ox["eq"]
            cf = corner_fill(os.path.join(OUT, "snap_%s_eq.npz" % tag))
            w("  at V_in = the closed case's 0.95 value: step %d, t %.4f s, V_in %.4f, outflow %.4f m3/m (%.4f of V_in);"
              " reach fill %.4f; hanging side %.4f; corner %.4f (hanging-side part %.4f); filled outside %d; rate /"
              " threshold %.4f" % (e["step"], e["t"], e["v_in"], e["v_out"], e["v_out"] / e["v_in"], e["frac_in"],
                                   e["hang_fill"], cf[0], cf[1], e["outside"], e["rate_ratio"]))
            cf = corner_fill(os.path.join(OUT, "final_%s.npz" % tag))
            w("  end: outflow %.4f m3/m (%.4f of V_in); reach fill %.4f; hanging side %.4f; corner %.4f (hanging-side part"
              " %.4f); rate / threshold %.4f" % (ox["v_out"], ox["v_out"] / r["v_in"], r["final"]["frac_in"],
                                                 r["final"]["hang_fill"], cf[0], cf[1], r["final"]["rate_ratio"]))
            w("  outflow / V_in against time (every 500 steps and the last record): step, t s, V_in, outflow, share")
            rows = [row for row in ser if int(row[0]) % 500 == 0]
            if not rows or rows[-1][0] != ser[-1][0]:
                rows.append(ser[-1])
            for row in rows:
                w("     %6d  %9.2f  %9.4f  %8.4f  %.4f" % (row[0], row[1], row[3], row[7], row[7] / row[3] if row[3] else 0))
            w("")
            continue
        for key in ("0.90", "0.95", "0.99"):
            if key not in r["snaps"]:
                w("  snapshot %s: not reached" % key)
                continue
            m = r["snaps"][key]
            cf = corner_fill(os.path.join(OUT, "snap_%s_%s.npz" % (tag, key)))
            w("  snapshot %s: step %d, t %.4f s, V_in %.4f m3/m, V_in / pore %.4f; reach fill %.4f; hanging side %.4f;"
              " void contact %.2f m; corner %.4f (hanging-side part %.4f); filled outside %d cells; rate / threshold"
              " %.4f; downward basal storage %.4f m3/m (%.4f of the stored)"
              % (key, m["step"], m["t"], m["v_in"], m["v_in"] / pore, m["frac_in"], m["hang_fill"], m["void_contact"],
                 cf[0], cf[1], m["outside"], m["rate_ratio"], m["loss"], m["loss_share"]))
        w("")
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "stage2.txt"), "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
