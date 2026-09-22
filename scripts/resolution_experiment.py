# -*- coding: utf-8 -*-
"""
Does refining the mesh buy accuracy, or just creep?

    powershell -File scripts\run_local.ps1 scripts\resolution_experiment.py [--cells 2.5,1.25,1.0] [--kinds gradient,uniform]

Runs inclined_hole_gradient's case at several cell sizes -- same domain,
hole, parameters and stall rule (patience and step cap scaled with 2.5/cell
so the plateau test spans comparable physical time) -- and measures, per run:

    wall-clock time, steps, physical time at stall
    reachable cells vs filled cells, OVERSHOOT (filled beyond the analytic
        reach) in cells and in m^2, and the worst path-cost/p0 among them
    up/down spread ratio from front_metrics
    Picard: max iterations and the share of solves that hit the cap
    Q(t) tail/peak

The question this answers: on main, a finer mesh does not fail (the solver
carries a non-converged Picard forward rather than rejecting the step), so
the issue is whether it can be TRUSTED. If overshoot stays about one cell --
i.e. shrinks in metres as the cell shrinks -- refinement is a net gain and
the resolution limits in cases/README.md lift without touching the solver.
If overshoot grows in cells, creep is winning and the mesh cannot be refined
without a solver change.

Outputs to outputs/resolution_experiment/.
"""
import imp
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.models.slurry_transport.equations import solve_transport_step
from src.analysis.fill_diagnostics import (
    OVERSHOOT, FRONT_SHORTFALL,
    reachable_domain, classify_unfilled, record_injection_rate,
    injection_rate_diagnostics,
)

G = imp.load_source("inclined_hole_gradient",
                    os.path.join(REPO, "cases", "inclined_hole_gradient.py"))
OUT = os.path.join(REPO, "outputs", "resolution_experiment")

BASE_CELL = 2.5
BASE_PATIENCE = G.STALL_PATIENCE
BASE_MAX_STEPS = G.MAX_STEPS


def run(kind, cell):
    G.CELL = float(cell)
    scale = BASE_CELL / float(cell)
    G.STALL_PATIENCE = int(round(BASE_PATIENCE * scale))
    G.MAX_STEPS = int(round(BASE_MAX_STEPS * scale * scale))

    t0 = time.time()
    state, mx, my, nx, ny, phi, hcells, k = G.build_case(kind)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    hmask = np.zeros(mx.size, dtype=bool)
    hmask[hcells] = True
    params = state["slurry_parameters"]
    cap = int(params["picard_max_iters"])

    t, best, stall, count = 0.0, -1, 0, 0
    history, iters = [], []
    for step in range(G.MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=G.DT_CAP)
        t += float(dt)
        record_injection_rate(state, hcells, dt, t, history)
        it = state.get("picard_iterations_last")
        if it is not None:
            iters.append(int(it))
        s = np.array(state["saturation"].value, copy=True)
        s[hmask] = 1.0
        state["saturation"].setValue(s)
        count = int(np.sum((s >= 0.5) & np.logical_not(hmask)))
        if count > best:
            best, stall = count, 0
        else:
            stall += 1
        if step % 100 == 0:
            print("    [%s %.3g] step %d t=%.1f filled=%d picard=%s" % (kind, cell, step, t, count, it)); sys.stdout.flush()
        if stall >= G.STALL_PATIENCE:
            break
    wall = time.time() - t0

    reach = reachable_domain(state, hcells, p0=G.P0, use_gravity=True)
    cls = classify_unfilled(state, hcells, reach["reachable"], s_c=0.5)
    cat = cls["category"]
    over = cat == OVERSHOOT
    fm = G.front_metrics(state, mx, my, hcells)
    q = injection_rate_diagnostics(history, tail_fraction=0.2, stall_ratio=0.05)
    iters = np.array(iters, dtype=int) if iters else np.zeros(1, dtype=int)

    row = {
        "kind": kind, "cell": float(cell), "n_cells": int(mx.size),
        "wall_s": wall, "steps": step + 1, "t_stall": t,
        "reach": int(np.sum(reach["reachable"])), "filled": count,
        "over_cells": int(np.sum(over)), "over_m2": float(np.sum(vols[over])),
        "over_cost_max": float(np.max(reach["cost"][over]) / G.P0) if np.any(over) else 0.0,
        "short_cells": int(np.sum(cat == FRONT_SHORTFALL)),
        "up": fm["perp_up"] if fm else 0.0, "down": fm["perp_down"] if fm else 0.0,
        "picard_max": int(iters.max()), "picard_cap_frac": float(np.mean(iters >= cap)),
        "q_tail": q.get("decay_ratio", 0.0),
    }
    return row


def main(argv):
    cells = [2.5, 1.25, 1.0]
    kinds = ["gradient", "uniform"]
    i = 0
    while i < len(argv):
        if argv[i] == "--cells":
            cells = [float(c) for c in argv[i + 1].split(",")]; i += 2
        elif argv[i] == "--kinds":
            kinds = argv[i + 1].split(","); i += 2
        else:
            i += 1
    if not os.path.isdir(OUT):
        os.makedirs(OUT)

    rows = []
    hdr = "%-9s %5s %6s %8s %6s %7s %6s %6s %5s %7s %6s %6s %6s %6s %5s %6s %6s" % (
        "kind", "cell", "cells", "wall[s]", "steps", "t[s]", "reach", "fill",
        "over", "over m2", "cost", "short", "up", "down", "u/d", "Pmax", "cap%")
    print(hdr); print("-" * len(hdr)); sys.stdout.flush()
    for kind in kinds:
        for cell in cells:
            r = run(kind, cell)
            rows.append(r)
            print("%-9s %5.2f %6d %8.0f %6d %7.0f %6d %6d %5d %7.1f %6.3f %6d %6.2f %6.2f %5.2f %6d %5.0f%%" % (
                r["kind"], r["cell"], r["n_cells"], r["wall_s"], r["steps"], r["t_stall"],
                r["reach"], r["filled"], r["over_cells"], r["over_m2"], r["over_cost_max"],
                r["short_cells"], r["up"], r["down"],
                (r["up"] / r["down"]) if r["down"] > 0 else 0.0,
                r["picard_max"], 100.0 * r["picard_cap_frac"]))
            sys.stdout.flush()
            with open(os.path.join(OUT, "rows.txt"), "a") as fh:
                fh.write(repr(r) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
