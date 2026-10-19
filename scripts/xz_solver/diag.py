# -*- coding: utf-8 -*-
"""Per-step cost diagnostics for the step-2B solver runs (no change to src/ or to the numerics).

install() wraps equations.solve_pressure_step (one call = one attempt of the fill-CFL guard in
solve_transport_step) and equations._solve_pressure_once (one call = one linear pressure solve, i.e. one Picard
iteration) with counters and timers. take() returns and resets the counters of the last step. probe() times a fixed
dense workload (independent of the case) to tell machine speed apart from case cost.
"""
from __future__ import print_function

import time

import numpy as np

from src.models.slurry_transport import equations as EQ

C = {"attempts": 0, "solves": 0, "solve_time": 0.0}
_A = np.random.RandomState(0).rand(300, 300)


def install():
    if getattr(EQ, "_diag_installed", False):
        return
    sps, spo = EQ.solve_pressure_step, EQ._solve_pressure_once

    def solve_pressure_step(*a, **k):
        C["attempts"] += 1
        return sps(*a, **k)

    def _solve_pressure_once(*a, **k):
        w0 = time.time()
        r = spo(*a, **k)
        C["solves"] += 1
        C["solve_time"] += time.time() - w0
        return r
    EQ.solve_pressure_step = solve_pressure_step
    EQ._solve_pressure_once = _solve_pressure_once
    EQ._diag_installed = True


def take():
    out = dict(C)
    C.update({"attempts": 0, "solves": 0, "solve_time": 0.0})
    return out


def probe(reps=20):
    """Seconds for a fixed single-thread-sized workload (300 x 300 matrix products)."""
    w0 = time.time()
    x = _A
    for _ in range(reps):
        x = np.dot(_A, x)
        x /= np.abs(x).max()
    return time.time() - w0
