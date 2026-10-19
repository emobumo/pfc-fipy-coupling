# -*- coding: utf-8 -*-
"""Wall time per step vs the BLAS / OpenMP thread count, from the same S1 checkpoint (no checkpoint is written).

    $env:OMP_NUM_THREADS = 4; $env:MKL_NUM_THREADS = 4; $env:OPENBLAS_NUM_THREADS = 4
    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\thread_bench.py 40

Writes outputs/xz_solver/bench_<threads>.npz (saturation after the steps) and appends a line to bench.txt.
"""
from __future__ import print_function

import imp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = imp.load_source("xz_run_cases", os.path.join(REPO, "scripts", "xz_solver", "run_cases.py"))
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint
from src.analysis import stop_rule
from src.models.slurry_transport.equations import solve_transport_step


def main():
    nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    thr = os.environ.get("OMP_NUM_THREADS", "default")
    D, C, srcs, collar_t = R.load()
    box = R.crop_box(C)
    p0 = 2.0e6
    st, src, _ = R.build(C["n"], srcs["S1"], box, p0)
    fp = "xz2b|%s|p0=%g|src=%s|K1.25v4|s1" % ("S1", p0, ",".join(str(q) for q in src)) + stop_rule.rate_tag(R.N_REF)
    res = Checkpointer(os.path.join(R.OUT, "ckpt_S1"), fingerprint=state_fingerprint(st, fp)).restore(st)
    assert res, "no S1 checkpoint to start from"
    hm = np.zeros(st["mesh"].numberOfCells, bool)
    hm[src] = True
    walls = []
    for k in range(nsteps):
        w0 = time.time()
        solve_transport_step(st, dt_cap=R.DT_CAP)
        walls.append(time.time() - w0)
        s = np.array(st["saturation"].value, copy=True)
        s[hm] = 1.0
        st["saturation"].setValue(s)
    walls = np.array(walls)
    np.savez(os.path.join(R.OUT, "bench_%s.npz" % thr), s=np.asarray(st["saturation"].value, float),
             p=np.asarray(st["pressure"].value, float), walls=walls)
    line = ("threads %-7s from step %d: %d steps, wall per step mean %.3f, median %.3f s (process CPU %.0f s)"
            % (thr, int(res["step"]), nsteps, walls.mean(), np.median(walls), os.times()[0] + os.times()[1]))
    open(os.path.join(R.OUT, "bench.txt"), "a").write(line + "\n")
    print(line)


if __name__ == "__main__":
    main()
