# -*- coding: utf-8 -*-
"""Conjugate-gradient iterations, convergence flag and final residual of every pressure solve, for a few steps
from each case's checkpoint (no checkpoint is written; numerics unchanged -- the wrapper only adds a callback).

    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\pcg_count.py 5 S2 S1

FiPy's scipy LinearPCGSolver: scipy.sparse.linalg.cg, no preconditioner, tol 1e-15, maxiter 2000.
Appends to outputs/xz_solver/pcg_count.txt.
"""
from __future__ import print_function

import imp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R = imp.load_source("xz_run_cases", os.path.join(REPO, "scripts", "xz_solver", "run_cases.py"))
import fipy.solvers.scipy.linearPCGSolver as PCG
from src.analysis.run_checkpoint import load as ck_load, state_fingerprint
from src.analysis import stop_rule
from src.models.slurry_transport import equations as EQ

LOG = []
FTZ = []
_cg = PCG.cg


def cg(A, b, x0=None, tol=1e-5, maxiter=None, M=None, **k):
    n = [0]

    def cb(xk):
        n[0] += 1
    w0 = time.time()
    x, info = _cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, callback=cb)
    w = time.time() - w0
    bn = np.linalg.norm(b)
    tiny = 2.2250738585072014e-308
    sub = lambda v: int(((np.abs(v) < tiny) & (v != 0)).sum())
    if os.environ.get("PCG_FTZ_TEST"):
        # control: the same system with every subnormal flushed to zero (result discarded)
        A2 = A.tocsr().copy()
        A2.data[np.abs(A2.data) < tiny] = 0.0
        b2 = np.where(np.abs(b) < tiny, 0.0, b)
        x02 = None if x0 is None else np.where(np.abs(x0) < tiny, 0.0, x0)
        m = [0]

        def cb2(xk):
            m[0] += 1
        w1 = time.time()
        y, _ = _cg(A2, b2, x0=x02, tol=tol, maxiter=maxiter, M=M, callback=cb2)
        FTZ.append((m[0], time.time() - w1, float(np.abs(y - x).max() / max(np.abs(x).max(), 1e-300))))
    Ad = A.tocsr().data if hasattr(A, "tocsr") else np.asarray(A)
    LOG.append((n[0], info, np.linalg.norm(b - A * x) / bn if bn > 0 else 0.0, w, A.shape[0], sub(Ad), sub(b), sub(x),
                float(np.abs(Ad[Ad != 0]).min()), float(np.abs(Ad).max())))
    return x, info


def main():
    PCG.cg = cg
    nsteps = int(sys.argv[1])
    D, C, srcs, collar_t = R.load()
    box = R.crop_box(C)
    p0s = dict((c[0], c[3]) for c in R.CASES)
    out = open(os.path.join(R.OUT, "pcg_count.txt"), "a")
    for tag in sys.argv[2:]:
        p0 = p0s[tag]
        st, src, _ = R.build(C["n"], srcs[tag], box, p0)
        fp = "xz2b|%s|p0=%g|src=%s|K1.25v4|s1" % (tag, p0, ",".join(str(q) for q in src)) + stop_rule.rate_tag(R.N_REF)
        prefix = os.path.join(R.OUT, "s1_end", "ckpt_S1") if tag == "S1" else os.path.join(R.OUT, "ckpt_" + tag)
        res = ck_load(prefix, st, fingerprint=state_fingerprint(st, fp))
        hm = np.zeros(st["mesh"].numberOfCells, bool)
        hm[src] = True
        del LOG[:]
        for k in range(nsteps):
            EQ.solve_transport_step(st, dt_cap=R.DT_CAP)
            s = np.array(st["saturation"].value, copy=True)
            s[hm] = 1.0
            st["saturation"].setValue(s)
        a = np.array(LOG, float)
        it = a[:, 0]
        line = ("%s from step %d: %d steps, %d CG solves (n = %d); iterations min / median / mean / max %d / %d / %.0f / %d;"
                " solves hitting maxiter (info > 0) %d; relative residual median %.2e, max %.2e; time per solve %.3f s,"
                " per iteration %.2f ms"
                % (tag, int(res["step"]), nsteps, len(a), int(a[0, 4]), it.min(), np.median(it), it.mean(), it.max(),
                   int((a[:, 1] > 0).sum()), np.median(a[:, 2]), a[:, 2].max(), a[:, 3].mean(),
                   1e3 * a[:, 3].sum() / it.sum()))
        line += ("; subnormal entries per solve: matrix %d, rhs %d, solution %d (median); smallest / largest |A| %.2e / %.2e"
                 % (np.median(a[:, 5]), np.median(a[:, 6]), np.median(a[:, 7]), a[:, 8].min(), a[:, 9].max()))
        if FTZ:
            f = np.array(FTZ, float)
            line += ("; FLUSH-TO-ZERO CONTROL (same systems): iterations mean %.0f, time per solve %.3f s, per iteration"
                     " %.2f ms; max relative difference of the solution %.2e"
                     % (f[:, 0].mean(), f[:, 1].mean(), 1e3 * f[:, 1].sum() / f[:, 0].sum(), f[:, 2].max()))
            del FTZ[:]
        out.write(line + "\n")
        out.flush()
        print(line)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
