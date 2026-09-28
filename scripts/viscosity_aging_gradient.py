# -*- coding: utf-8 -*-
"""
Does the headline gradient result survive the slow end of the viscosity bracket?

    powershell -File scripts\run_local.ps1 scripts\viscosity_aging_gradient.py

scripts/viscosity_aging_bracket.py shows the gradient line-source case is the
one headline run that does NOT finish inside the slow end: with every parcel
as old as the stage (viscosity e-folding T = 74 min, Lv et al. 2021), the run
is the constant-viscosity run stopped at model time tau = T, and by then only
~91% of the volume is in. This re-runs cases/inclined_hole_gradient.py's main
pair (gradient 0.12->0.30 and uniform 0.18) with the SAME march and stop rule,
and books the front metrics at the model times that correspond to real times
30 / 60 / 120 / 240 min and to tau = T (t -> infinity), as well as at the
stall. The stall row must reproduce the anchors (1.157 / 0.91) bit for bit.

Crash-safe: checkpointed like the case itself.
"""
from __future__ import print_function

import imp
import json
import math
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.slurry_transport.equations import solve_transport_step
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint
from src.analysis import stop_rule

G = imp.load_source("inclined_hole_gradient", os.path.join(REPO, "cases", "inclined_hole_gradient.py"))
OUT = os.path.join(REPO, "outputs", "viscosity_aging")

T = 60.0 / 0.01350                      # s, as in viscosity_aging_bracket.py
REAL_MIN = (30.0, 60.0, 120.0, 240.0)


def tau_of(real_min):
    return T * (1.0 - math.exp(-real_min * 60.0 / T))


MARKS = [("real %.0f min" % m, tau_of(m)) for m in REAL_MIN] + [("tau = T (slow end)", T)]


def metrics(state, mx, my, hcells, t, v_in):
    fm = G.front_metrics(state, mx, my, hcells)
    return {"t": t, "v_in": v_in, "count": fm["count"], "up": fm["perp_up"],
            "down": fm["perp_down"], "ratio": fm["perp_up"] / fm["perp_down"]}


def run(kind, label):
    state, mx, my, nx, ny, phi, hcells, k = G.build_case(kind)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    hmask = np.zeros(mx.size, dtype=bool)
    hmask[hcells] = True
    # stop rule: the case's (rate rule since 2026-09-27; Q_ref from the INITIAL
    # mu_p -- the run is the constant-viscosity fast end, the slow end is its
    # time-rescaling, so the stop is decided on the constant-viscosity clock)
    n_ref = G.n_ref_for(kind)
    qref = stop_rule.q_ref(n_ref, G.A_CAL, G.P0, G.MU_P)
    keeper = Checkpointer(os.path.join(OUT, "ckpt_" + kind), every=200,
                          fingerprint=state_fingerprint(state, "aging|%s" % kind + stop_rule.rate_tag(n_ref)))
    t, v_in, best, stall, step0, snaps = 0.0, 0.0, -1, 0, 0, {}
    t_hist, v_hist = [0.0], [0.0]
    resumed = keeper.restore(state)
    if resumed:
        step0 = int(resumed["step"])
        t, v_in = float(resumed["t"]), float(resumed["v_in"])
        best, stall = int(resumed["best"]), int(resumed["stall"])
        snaps = json.loads(str(resumed["snaps_json"]))
        if "t_hist" in resumed:
            t_hist = list(np.atleast_1d(resumed["t_hist"]))
            v_hist = list(np.atleast_1d(resumed["v_hist"]))
        print("  [%s] resuming at step %d (t=%.1f s)" % (label, step0, t))
    else:
        print("  [%s] marching ..." % label)
    sys.stdout.flush()
    for step in range(step0, G.MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=G.DT_CAP)
        t += float(dt)
        div_q = np.asarray(state["last_div_q"], dtype=float)
        v_in += float(np.sum(div_q[hmask] * vols[hmask])) * float(dt)
        t_hist.append(t)
        v_hist.append(v_in)
        s = np.array(state["saturation"].value, copy=True)
        s[hmask] = 1.0
        state["saturation"].setValue(s)
        for name, tau in MARKS:
            if name not in snaps and t >= tau:
                snaps[name] = metrics(state, mx, my, hcells, t, v_in)
        s_arr = np.asarray(state["saturation"].value, dtype=float)
        count = int(np.sum((s_arr >= 0.5) & np.logical_not(hmask)))
        if count > best:            # same count-plateau rule as the case
            best, stall = count, 0
        else:
            stall += 1
        if stop_rule.legacy():
            if stall >= G.STALL_PATIENCE:
                break
        elif stop_rule.stop_rule_physical(t_hist, v_hist, qref):
            break
        keeper.maybe_save(state, step=step + 1, t=t, v_in=v_in, best=best, stall=stall,
                          snaps_json=json.dumps(snaps),
                          t_hist=np.asarray(t_hist), v_hist=np.asarray(v_hist))
    keeper.finish()
    snaps["stall"] = metrics(state, mx, my, hcells, t, v_in)
    return snaps


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    res = {}
    for kind, label in (("gradient", "A gradient 0.12->0.30"), ("uniform", "B uniform 0.18")):
        res[kind] = run(kind, label)
    json.dump(res, open(os.path.join(OUT, "gradient_marks.json"), "w"), indent=1)
    lines = ["viscosity e-folding T = %.0f s; marks are MODEL times at the slow end" % T, ""]
    lines.append("%-20s %9s | %-34s | %-34s" % ("", "", "A gradient", "B uniform"))
    lines.append("%-20s %9s | %6s %6s %6s %6s %6s | %6s %6s %6s %6s %6s"
                 % ("mark", "tau [s]", "V_in", "cells", "up", "down", "up/dn",
                    "V_in", "cells", "up", "down", "up/dn"))
    for name, tau in MARKS + [("stall (fast end)", float("nan"))]:
        key = "stall" if name.startswith("stall") else name
        row = "%-20s %9.0f |" % (name, res["gradient"][key]["t"] if key == "stall" else tau)
        for kind in ("gradient", "uniform"):
            m = res[kind].get(key)
            row += (" %6.1f %6d %6.2f %6.2f %6.3f |" % (m["v_in"], m["count"], m["up"], m["down"], m["ratio"])
                    if m else " %34s |" % "(stalled earlier: = stall)")
        lines.append(row)
    text = "\n".join(lines)
    open(os.path.join(OUT, "gradient_marks.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
