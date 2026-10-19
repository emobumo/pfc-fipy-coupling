# -*- coding: utf-8 -*-
"""Controlled per-step cost comparison between cases, from their checkpoints (no checkpoint is written).

    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\diag_bench.py 30 S2 S1 S2

For each tag in turn: restore its newest checkpoint (S1 from outputs/xz_solver/s1_end), run N steps, and log per
step the wall time, dt, CFL-guard attempts, linear pressure solves (= Picard iterations summed over attempts) and
the time inside them; a fixed machine-speed probe (diag.probe) is timed before and after. Appends to
outputs/xz_solver/diag_bench.txt.
"""
from __future__ import print_function

import imp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = imp.load_source("xz_run_cases", os.path.join(REPO, "scripts", "xz_solver", "run_cases.py"))
DG = imp.load_source("xz_diag", os.path.join(REPO, "scripts", "xz_solver", "diag.py"))
from src.analysis.run_checkpoint import load as ck_load, state_fingerprint
from src.analysis import stop_rule
from src.models.slurry_transport import equations as EQ


def main():
    nsteps = int(sys.argv[1])
    tags = sys.argv[2:]
    DG.install()
    D, C, srcs, collar_t = R.load()
    box = R.crop_box(C)
    p0s = dict((c[0], c[3]) for c in R.CASES)
    out = open(os.path.join(R.OUT, "diag_bench.txt"), "a")
    for tag in tags:
        p0 = p0s[tag]
        st, src, _ = R.build(C["n"], srcs[tag], box, p0)
        fp = "xz2b|%s|p0=%g|src=%s|K1.25v4|s1" % (tag, p0, ",".join(str(q) for q in src)) + stop_rule.rate_tag(R.N_REF)
        prefix = os.path.join(R.OUT, "s1_end", "ckpt_S1") if tag == "S1" else os.path.join(R.OUT, "ckpt_" + tag)
        res = ck_load(prefix, st, fingerprint=state_fingerprint(st, fp))
        hm = np.zeros(st["mesh"].numberOfCells, bool)
        hm[src] = True
        pb = DG.probe()
        DG.take()
        rows = []
        for k in range(nsteps):
            w0 = time.time()
            dt = EQ.solve_transport_step(st, dt_cap=R.DT_CAP)
            w = time.time() - w0
            c = DG.take()
            s = np.array(st["saturation"].value, copy=True)
            s[hm] = 1.0
            st["saturation"].setValue(s)
            rows.append((w, dt, c["attempts"], c["solves"], c["solve_time"], st.get("picard_iterations_last", 0)))
        pa = DG.probe()
        a = np.array(rows, float)
        line = ("%s from step %d (%s): %d steps; wall/step %.3f s; dt mean %.3e s; attempts/step %.2f; linear solves/step"
                " %.2f; time per linear solve %.4f s; time outside the solves/step %.3f s; Picard iters of the last attempt"
                " mean %.2f; machine probe before / after %.3f / %.3f s"
                % (tag, int(res["step"]), time.strftime("%H:%M:%S"), nsteps, a[:, 0].mean(), a[:, 1].mean(),
                   a[:, 2].mean(), a[:, 3].mean(), a[:, 4].sum() / a[:, 3].sum(), (a[:, 0] - a[:, 4]).mean(),
                   a[:, 5].mean(), pb, pa))
        out.write(line + "\n")
        out.flush()
        print(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
