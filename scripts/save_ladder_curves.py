# -*- coding: utf-8 -*-
"""
Save the two verification-ladder curves the Ch.4 figure 4.4-2 needs, by
running the test module's own fixtures unchanged (the same computation every
full test run performs; nothing is re-implemented):

    step 2b  _run_step2b_march(): front position vs time, against the
             model's own 1D analytic curve _fill_front(t)
    step 3   _run_step3_march(): stalled pressure along the +x axis, against
             the model's own radial stall solution max(0, p0 - lambda (r - r0));
             stall radius vs I_max = r0 + p0/lambda

    powershell -File scripts\run_local.ps1 scripts\save_ladder_curves.py

Outputs: outputs/verification_ladder/step2b_front_time.csv,
step3_axis_profile.csv, summary.txt
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

T = imp.load_source("test_verification_ladder", os.path.join(REPO, "tests", "test_verification_ladder.py"))
OUT = os.path.join(REPO, "outputs", "verification_ladder")


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    _, traj, _ = T._run_step2b_march()
    rows = [(t, f, T._fill_front(t)) for t, f in traj]
    np.savetxt(os.path.join(OUT, "step2b_front_time.csv"), np.array(rows), delimiter=",", comments="",
               header="t_s,front_numeric_m,front_analytic_m  (L_max = %.3f m)" % T.S2B_LMAX)
    start = max(int(0.2 * len(rows)), 1)
    worst = max(abs(f - g) / T.S2B_LMAX for _, f, g in rows[start:])
    state, hist = T._run_step3_march()
    x, p = T._step3_axis_profile(state)
    exact = np.where(x > T.S3_R0, np.maximum(0.0, T.S3_P0 - T.S3_LAMBDA * (x - T.S3_R0)), np.nan)
    np.savetxt(os.path.join(OUT, "step3_axis_profile.csv"), np.column_stack([x, p, exact]), delimiter=",",
               comments="", header="r_m,p_numeric_Pa,p_exact_Pa  (p0 = %g Pa, r0 = %g m, I_max = %.3f m)"
               % (T.S3_P0, T.S3_R0, T.S3_IMAX))
    stall = T._step3_stagnation_from_gradient(x, p)
    out = x > T.S3_R0
    dev = float(np.max(np.abs(p[out] - exact[out]))) / T.S3_P0
    text = "\n".join([
        "step 2b: %d steps to front %.4f m (%.2f L_max); max |front - analytic| / L_max after the first 20%% = %.4f"
        % (len(rows), rows[-1][1], rows[-1][1] / T.S2B_LMAX, worst),
        "step 3: stall radius %.4f m vs I_max %.4f m (rel err %.4f); max |p - exact| / p0 outside r0 = %.4f"
        % (stall, T.S3_IMAX, abs(stall - T.S3_IMAX) / T.S3_IMAX, dev)])
    open(os.path.join(OUT, "summary.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
