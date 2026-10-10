# -*- coding: utf-8 -*-
"""
Slow end of the viscosity bracket for the P2 first stages: when does the rate
rule stop each first stage in REAL time, if the grout thickens as
mu_p(t) = mu_p0 * exp(t/T)?  (Post-processing only; no solver run.)

    powershell -File scripts\run_local.ps1 scripts\slow_end_stage1.py [out.txt]

Only the relative growth of Lv et al. (2021, P0004) is used: f(t) = mu(t)/mu(0)
= exp(0.01350 t), t in min, so T = 1/0.01350 min = 4444 s; the absolute
869.75 mPa s (an apparent viscosity) is not used. A spatially uniform mu_p(t)
is the constant-viscosity run on the clock d tau = dt / f(t), i.e.
tau = T (1 - exp(-t/T)) (see scripts/viscosity_aging_bracket.py). Each stage
starts with fresh grout, so the clock starts at the stage start.

Reads outputs/rerun_rate/C/C_p2/<case>/injection_rate.csv (stage 0 rows, the
recorded Q(tau) of the constant-viscosity run), maps the cumulative volume
V(tau) to real time, and replays src/analysis/stop_rule.stop_rule_physical on a
real-time grid with Q_ref from the INITIAL mu_p (0.15 Pa s). Writes
outputs/rerun_rate/slow_end_stage1.txt (or the path given).

Reconstructed 2026-10-10 from the one-off session computation of 2026-10-08
that wrote the original file; same arithmetic, paths made repo-relative.
"""
from __future__ import print_function

import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from src.analysis import stop_rule

R = os.path.join(REPO, "outputs", "rerun_rate", "C", "C_p2")
T = 60.0 / 0.01350
CASES = (("phi0.10_H17_c1.25/7_17", 0.10), ("phi0.14_H17_c1.25/7_17", 0.14), ("phi0.18_H17_c1.25/7_17", 0.18),
         ("phi0.18_H17_c1.25/5_17", 0.18), ("phi0.18_H17_c1.25/single", 0.18))


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(REPO, "outputs", "rerun_rate", "slow_end_stage1.txt")
    OUTL = ["slow-end replay of the rate rule for P2 first stages (tau = T(1-exp(-t/T)), T = 4444 s; Q_ref from initial mu_p; initial set 240 min)", ""]
    for sub, phi in CASES:
        h = np.loadtxt(os.path.join(R, sub, "injection_rate.csv"), delimiter=",", skiprows=1, ndmin=2)
        h = h[h[:, 2] == 0]
        tau = np.concatenate([[0.0], h[:, 0]]); V = np.concatenate([[0.0], np.cumsum(h[:, 1] * np.diff(tau))])
        qref = stop_rule.q_ref(phi, 1.25e-7, 5e6, 0.15)
        # slow end: real time t for tau < T
        ok = tau < T * (1 - 1e-12)
        t_real = -T * np.log(1 - tau[ok] / T); Vr = V[ok]
        # replay the rate rule on a real-time grid
        stop = None
        grid = np.linspace(1.0, 6 * 3600.0, 4000)
        Vg = np.interp(grid, t_real, Vr, right=Vr[-1])
        tt, vv = [0.0], [0.0]
        for g, v in zip(grid, Vg):
            tt.append(g); vv.append(v)
            if stop_rule.stop_rule_physical(tt, vv, qref):
                stop = g; break
        OUTL.append("%-28s fast-end stop tau=%.0f s V=%.2f | slow end: V(tau->T)=%.2f (%.0f%%), rate-rule stop at real %s min, V there %.2f" % (
            sub, tau[-1], V[-1], np.interp(T, tau, V), 100 * np.interp(T, tau, V) / V[-1], "%.0f" % (stop / 60) if stop else "none<360", np.interp(stop, grid, Vg) if stop else float("nan")))
    open(out, "w").write("\n".join(OUTL) + "\n")
    print("\n".join(OUTL))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
