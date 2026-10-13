# -*- coding: utf-8 -*-
"""
Picard budget under the CURRENT rate stop rule (single pass): the gradient
line source of layer A, marched with at most 12 and at most 40 Picard
iterations per step, everything else unchanged.

    powershell -File scripts\\run_local.ps1 scripts\\picard_rate_check.py [--cell 2.5] [--iters 12,40]

The 1.25 m case (R_ud 1.220) took 2.55 h at 12 iterations; at 40 the 2.5 m
scan (picard_scan E1) cost 3.4x, i.e. about 8.7 h, so the default is the
2.5 m gradient case (layer A gradient_2.5, R_ud 1.193).

Per step the ratio last/first Picard residual of the step's final pressure
solve is recorded (as in scripts/picard_scan.py), with the iteration count.
Also the full conservation ledger at the stop (V_in - V_store - V_comp - V_clip).
At the stop: filled cells (S >= 0.5, hole excluded), overshoot cells (filled
and outside the reach of the original field), r_up / r_down / R_ud
(front_metrics, cell centres), stop time, V_in, steps. No checkpoints (the
residual record would not survive a resume); rerun the command after a crash.

Output: outputs/picard_rate_check/picard_rate_check.txt (+ .json)
"""
from __future__ import print_function

import imp
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain
from src.models.slurry_transport.equations import conservation_report

G = imp.load_source("inclined_hole_gradient", os.path.join(REPO, "cases", "inclined_hole_gradient.py"))
OUT = os.path.join(REPO, "outputs", "picard_rate_check")
RECORD = {"ratio": [], "iters": []}
_step = G.solve_transport_step


def _recording_step(state, dt_cap=0.01):
    dt = _step(state, dt_cap=dt_cap)
    hist = state.get("picard_history_last") or []
    if len(hist) >= 1 and hist[0] > 0.0:
        RECORD["ratio"].append(float(hist[-1]) / float(hist[0]))
    it = state.get("picard_iterations_last")
    if it is not None:
        RECORD["iters"].append(int(it))
    return dt


G.solve_transport_step = _recording_step


def one(cell, iters):
    RECORD["ratio"], RECORD["iters"] = [], []
    G.CELL = cell
    try:
        st, mx, my, nx, ny, phi, hc, k = G.build_case("gradient")
        st["slurry_parameters"]["picard_max_iters"] = int(iters)
        reach = np.asarray(reachable_domain(st, hc, p0=G.P0, use_gravity=True)["reachable"], dtype=bool)
        vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
        t0 = time.time()
        t, v_in, v_store = G.march(st, mx, my, hc, vols, "gradient_%g_iters%d" % (cell, iters),
                                   keeper=None, n_ref=G.n_ref_for("gradient"))
        wall = time.time() - t0
        fm = G.front_metrics(st, mx, my, hc)
        s = np.asarray(st["saturation"].value, dtype=float)
        hm = np.zeros(mx.size, dtype=bool)
        hm[hc] = True
        filled = (s >= 0.5) & np.logical_not(hm)
        r = np.asarray(RECORD["ratio"], dtype=float)
        it = np.asarray(RECORD["iters"], dtype=int)
        led = conservation_report(st)      # full ledger: V_in - V_store - V_comp - V_clip
        return {"ledger_v_in": led["v_in"], "ledger_v_store": led["v_store"], "ledger_v_comp": led["v_comp"],
                "ledger_v_comp_pressure": led["v_comp_pressure"], "ledger_v_clip": led["v_clip"],
                "ledger_drift_rel": led["drift_rel"],
                "case_drift_rel": abs(v_in - v_store) / v_in,"cell": cell, "iters_max": iters, "reason": G.LAST_REASON[0], "steps": int(it.size),
                "t": t, "v_in": v_in, "v_store": v_store, "filled": int(filled.sum()),
                "overshoot": int(np.sum(filled & np.logical_not(reach))),
                "up": fm["perp_up"], "down": fm["perp_down"], "ratio": fm["perp_up"] / fm["perp_down"],
                "resid_mean": float(r.mean()), "resid_p90": float(np.percentile(r, 90)),
                "resid_median": float(np.median(r)), "iters_mean": float(it.mean()),
                "cap_frac": float(np.mean(it >= iters)), "wall_s": wall,
                "s_digest": __import__("hashlib").sha1(s.tobytes()).hexdigest()}
    finally:
        G.CELL = 2.5


def main(argv):
    cell = float(argv[argv.index("--cell") + 1]) if "--cell" in argv else 2.5
    iters = [int(v) for v in (argv[argv.index("--iters") + 1] if "--iters" in argv else "12,40").split(",")]
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    res = []
    for n in iters:
        print("=== gradient %.2f m, at most %d Picard iterations ===" % (cell, n))
        sys.stdout.flush()
        res.append(one(cell, n))
        json.dump(res, open(os.path.join(OUT, "picard_rate_check.json"), "w"), indent=1)
        print(res[-1])
        sys.stdout.flush()
    keys = [("reason", "%s"), ("steps", "%d"), ("t", "%.2f"), ("v_in", "%.4f"), ("filled", "%d"), ("overshoot", "%d"),
            ("up", "%.4f"), ("down", "%.4f"), ("ratio", "%.4f"), ("resid_mean", "%.3f"), ("resid_median", "%.3f"),
            ("resid_p90", "%.3f"), ("iters_mean", "%.2f"), ("cap_frac", "%.3f"), ("wall_s", "%.0f"), ("s_digest", "%s"),
            ("case_drift_rel", "%.3e"), ("ledger_v_in", "%.4f"), ("ledger_v_store", "%.4f"), ("ledger_v_comp", "%.4f"),
            ("ledger_v_comp_pressure", "%.4f"), ("ledger_v_clip", "%.3e"), ("ledger_drift_rel", "%.3e")]
    lines = ["Picard budget under the rate stop rule (single pass): gradient line source, %.2f m" % cell, ""]
    lines.append("%-14s" % "" + "".join("%22s" % ("max %d iters" % r["iters_max"]) for r in res))
    for k, f in keys:
        lines.append("%-14s" % k + "".join("%22s" % (f % r[k]) for r in res))
    if len(res) == 2:
        a, b = res
        lines += ["", "differences (%d minus %d): t %+.2f s (%+.4f%%), V_in %+.4f m3/m (%+.4f%%), steps %+d, filled %+d, "
                  "overshoot %+d, up %+.4f m, down %+.4f m, R_ud %+.4f; saturation field %s" % (
                      b["iters_max"], a["iters_max"], b["t"] - a["t"], 100 * (b["t"] - a["t"]) / a["t"],
                      b["v_in"] - a["v_in"], 100 * (b["v_in"] - a["v_in"]) / a["v_in"], b["steps"] - a["steps"],
                      b["filled"] - a["filled"], b["overshoot"] - a["overshoot"], b["up"] - a["up"], b["down"] - a["down"],
                      b["ratio"] - a["ratio"], "bit-identical" if a["s_digest"] == b["s_digest"] else "differs")]
    text = "\n".join(lines)
    print(text)
    open(os.path.join(OUT, "picard_rate_check.txt"), "w").write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
