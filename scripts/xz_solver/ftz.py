# -*- coding: utf-8 -*-
"""Flush subnormal numbers to zero before every conjugate-gradient pressure solve (main line, 2026-10-18).

Why: in the step-2B cases the pressure matrices carry ~7e4 subnormal entries (|a| < 2.2e-308, down to 1.98e-323)
and the right-hand sides up to ~9e3; CG on such data runs up to ~9x slower per iteration (outputs/xz_solver/
pcg_count.txt). With them set to zero the same systems take the same number of iterations and the solutions differ
by < 1e-313 relative.

Rule: before each call of scipy.sparse.linalg.cg made by FiPy's LinearPCGSolver, OFF-DIAGONAL matrix entries, the
right-hand side and the initial guess with |v| < 1e-308 are set to 0 (on a copy of the matrix). Diagonal entries are
never changed; a subnormal diagonal entry is counted, with the step it occurred in. Nothing in src/, the parameters,
the algorithm or the regularisation changes. Only active while set_active(True); the wrapper is installed once per
process.
"""
from __future__ import print_function

import numpy as np

import fipy.solvers.scipy.linearPCGSolver as PCG

TINY = 1.0e-308
STATE = {"active": False, "step": 0, "calls": 0, "diag_subnormal": 0, "diag_steps": [], "flushed": 0}
_cg = PCG.cg


def _cg_ftz(A, b, x0=None, tol=1e-5, maxiter=None, M=None, **k):
    if not STATE["active"]:
        return _cg(A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, **k)
    A2 = A.tocsr().copy()
    rows = np.repeat(np.arange(A2.shape[0]), np.diff(A2.indptr))
    small = (np.abs(A2.data) < TINY) & (A2.data != 0)
    diag = A2.indices == rows
    nd = int((small & diag).sum())
    if nd:
        STATE["diag_subnormal"] += nd
        if len(STATE["diag_steps"]) < 1000:
            STATE["diag_steps"].append(STATE["step"])
    off = small & ~diag
    STATE["flushed"] += int(off.sum())
    A2.data[off] = 0.0
    b2 = np.where(np.abs(b) < TINY, 0.0, b)
    x02 = None if x0 is None else np.where(np.abs(x0) < TINY, 0.0, x0)
    STATE["calls"] += 1
    return _cg(A2, b2, x0=x02, tol=tol, maxiter=maxiter, M=M, **k)


def install():
    PCG.cg = _cg_ftz


def set_active(flag):
    STATE["active"] = bool(flag)
