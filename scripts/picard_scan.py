# -*- coding: utf-8 -*-
"""
Separate REGULARIZATION creep from UNCONVERGED-ITERATION creep.

    powershell -File scripts\run_local.ps1 scripts\picard_scan.py --tag E1 \
        --cells 2.5 --kinds gradient --iters 12,40,120

The engineering cases overshoot the analytic reachable domain -- 1-9 cells at
2.5 m, 48-61 at 1.25 m -- and the standing explanation is "yield-edge Picard
does not converge, so every step creeps outward". That explanation lumps two
different mechanisms together, and they need opposite fixes:

  (a) REGULARIZATION creep. The Papanastasiou truncation is deliberately
      non-zero at the threshold: T(|gradPhi| = lambda) = ln2 / m, which is
      1.7% of the fully-yielded mobility at the default m = 40. This creep
      survives a perfect solver and shrinks like 1/m.
  (b) ITERATION creep. omega = 0.15 with a cap of 12 iterations cannot reach
      the declared 1e-3 relative tolerance: a damped Picard contracts by at
      best |1 - omega + omega*rho| per sweep, so 12 sweeps buy a factor 7 to
      70 while the criterion asks for 1000. Every step near the front
      therefore exits on the cap, not on the tolerance.

Nobody has measured which dominates. If (b) dominates, Anderson acceleration
is worth building; if (a) dominates, raising m is the lever and accelerating
the iteration fixes the cheap half. Raising m also stiffens the map, so the
two interact -- hence a scan rather than an argument.

Design (one axis per --tag, everything else at case defaults):

  E1  --iters 12,40,120     does more iteration budget remove the overshoot,
                            and does it converge to a NON-ZERO floor? That
                            floor is (a).
  E2  --regm 40,120,400     does that floor shrink like 1/m, as (a) predicts?
  E3  --dtcap 5,20          does creep scale with STEP COUNT rather than
                            physical time? Fewer, larger steps should creep
                            less if the mechanism is per-step.

Headline metric: OVERSHOOT AREA FRACTION, filled-but-unreachable area over
filled area. It is solver-independent (the reachable domain is a path
integral, not a march) and it is the number that has to go to ~0.

Crash resilience is two-level, because this machine has a history: finished
points are appended to outputs/picard_scan/results.csv and skipped on a
re-run, and the march inside each point is checkpointed. Re-running the same
command after a crash is always the right move.
"""
from __future__ import print_function

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
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint

G = imp.load_source("inclined_hole_gradient",
                    os.path.join(REPO, "cases", "inclined_hole_gradient.py"))

OUT = os.path.join(REPO, "outputs", "picard_scan")
CSV = os.path.join(OUT, "results.csv")
CKPT = os.path.join(OUT, "checkpoints")

BASE_CELL = 2.5
BASE_PATIENCE = G.STALL_PATIENCE
BASE_MAX_STEPS = G.MAX_STEPS

COLUMNS = (
    "tag", "kind", "cell", "iters", "regm", "dtcap",
    "n_cells", "wall_s", "steps", "t_stall",
    "reach", "filled", "over_cells", "over_frac", "over_m2", "over_cost_max",
    "short_cells", "up", "down", "updown",
    "picard_mean", "picard_max", "cap_frac", "resid_mean", "resid_p90",
    "q_tail",
)


def key_of(cfg):
    return "%s|%s|%.4g|%d|%.4g|%.4g" % (
        cfg["tag"], cfg["kind"], cfg["cell"], cfg["iters"], cfg["regm"],
        cfg["dtcap"])


def done_keys():
    if not os.path.exists(CSV):
        return set()
    keys = set()
    handle = open(CSV, "r")
    try:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 6 or parts[0] == "tag":
                continue
            try:
                keys.add("%s|%s|%.4g|%d|%.4g|%.4g" % (
                    parts[0], parts[1], float(parts[2]), int(parts[3]),
                    float(parts[4]), float(parts[5])))
            except ValueError:
                continue
    finally:
        handle.close()
    return keys


def append_row(row):
    fresh = not os.path.exists(CSV)
    handle = open(CSV, "a")
    try:
        if fresh:
            handle.write(",".join(COLUMNS) + "\n")
        handle.write(",".join(
            ("%.6g" % row[c]) if isinstance(row[c], float) else str(row[c])
            for c in COLUMNS) + "\n")
    finally:
        handle.close()


def run_point(cfg):
    """One (cell, iters, regm, dtcap) point, checkpointed."""
    G.CELL = float(cfg["cell"])
    scale = BASE_CELL / float(cfg["cell"])
    # Patience and step cap scale with the refinement so the plateau test
    # spans comparable physical time -- same convention as
    # scripts/resolution_experiment.py, so the rows stay comparable.
    G.STALL_PATIENCE = int(round(BASE_PATIENCE * scale))
    G.MAX_STEPS = int(round(BASE_MAX_STEPS * scale * scale))
    if cfg.get("max_steps"):
        G.MAX_STEPS = int(cfg["max_steps"])      # smoke tests only
    dt_cap = float(cfg["dtcap"])

    t0 = time.time()
    state, mx, my, nx, ny, phi, hcells, k = G.build_case(cfg["kind"])
    params = state["slurry_parameters"]
    params["picard_max_iters"] = int(cfg["iters"])
    params["yield_reg_m"] = float(cfg["regm"])
    cap = int(cfg["iters"])

    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    hmask = np.zeros(mx.size, dtype=bool)
    hmask[hcells] = True

    tag = ("%s_%s_c%.4g_i%d_m%.4g_dt%.4g"
           % (cfg["tag"], cfg["kind"], cfg["cell"], cfg["iters"], cfg["regm"],
              cfg["dtcap"])).replace(".", "p")
    keeper = Checkpointer(
        os.path.join(CKPT, tag), every=200,
        fingerprint=state_fingerprint(state, "picard_scan|" + key_of(cfg)))

    t, best, stall, count, step0 = 0.0, -1, 0, 0, 0
    history, iters, resid = [], [], []
    resumed = keeper.restore(state)
    if resumed:
        step0 = int(resumed["step"])
        t = float(resumed["t"])
        best = int(resumed["best"])
        stall = int(resumed["stall"])
        history = [(float(a), float(b))
                   for a, b in np.atleast_2d(resumed["history"])]
        iters = [int(v) for v in np.atleast_1d(resumed["iters"])]
        resid = [float(v) for v in np.atleast_1d(resumed["resid"])]
        print("    resuming at step %d (t=%.1f s)" % (step0, t))
        sys.stdout.flush()

    step = step0
    for step in range(step0, G.MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=dt_cap)
        t += float(dt)
        record_injection_rate(state, hcells, dt, t, history)

        it = state.get("picard_iterations_last")
        if it is not None:
            iters.append(int(it))
        # How unconverged did the step actually end? The cap fraction says
        # "ran out of budget"; this says "by how much it missed", which is
        # what distinguishes a cap that barely mattered from one that did.
        hist = state.get("picard_history_last") or []
        if len(hist) >= 1 and hist[0] > 0.0:
            resid.append(float(hist[-1]) / float(hist[0]))

        s = np.array(state["saturation"].value, copy=True)
        s[hmask] = 1.0
        state["saturation"].setValue(s)
        count = int(np.sum((s >= 0.5) & np.logical_not(hmask)))
        if count > best:
            best, stall = count, 0
        else:
            stall += 1
        if step % 200 == 0:
            print("    [%s c=%.3g i=%d m=%.3g dt=%.3g] step %d t=%.0f filled=%d picard=%s"
                  % (cfg["kind"], cfg["cell"], cfg["iters"], cfg["regm"],
                     dt_cap, step, t, count, it))
            sys.stdout.flush()
        if stall >= G.STALL_PATIENCE:
            break
        keeper.maybe_save(state, step=step + 1, t=t, best=best, stall=stall,
                          history=np.asarray(history, dtype=float),
                          iters=np.asarray(iters, dtype=int),
                          resid=np.asarray(resid, dtype=float))
    keeper.finish()
    wall = time.time() - t0

    reach = reachable_domain(state, hcells, p0=G.P0, use_gravity=True)
    cls = classify_unfilled(state, hcells, reach["reachable"], s_c=0.5)
    cat = cls["category"]
    over = cat == OVERSHOOT
    fm = G.front_metrics(state, mx, my, hcells)
    q = injection_rate_diagnostics(history, tail_fraction=0.2, stall_ratio=0.05)
    it_arr = np.array(iters, dtype=int) if iters else np.zeros(1, dtype=int)
    rs_arr = np.array(resid, dtype=float) if resid else np.zeros(1, dtype=float)
    filled_m2 = float(np.sum(vols[(np.asarray(state["saturation"].value,
                                              dtype=float) >= 0.5)
                                  & np.logical_not(hmask)]))
    over_m2 = float(np.sum(vols[over]))

    return {
        "tag": cfg["tag"], "kind": cfg["kind"], "cell": float(cfg["cell"]),
        "iters": int(cfg["iters"]), "regm": float(cfg["regm"]),
        "dtcap": float(cfg["dtcap"]),
        "n_cells": int(mx.size), "wall_s": float(wall), "steps": int(step + 1),
        "t_stall": float(t),
        "reach": int(np.sum(reach["reachable"])), "filled": int(count),
        "over_cells": int(np.sum(over)),
        "over_frac": float(over_m2 / filled_m2) if filled_m2 > 0 else 0.0,
        "over_m2": over_m2,
        "over_cost_max": (float(np.max(reach["cost"][over]) / G.P0)
                          if np.any(over) else 0.0),
        "short_cells": int(np.sum(cat == FRONT_SHORTFALL)),
        "up": float(fm["perp_up"]) if fm else 0.0,
        "down": float(fm["perp_down"]) if fm else 0.0,
        "updown": float(fm["perp_up"] / max(fm["perp_down"], 1e-9)) if fm else 0.0,
        "picard_mean": float(it_arr.mean()), "picard_max": int(it_arr.max()),
        "cap_frac": float(np.mean(it_arr >= cap)),
        "resid_mean": float(rs_arr.mean()),
        "resid_p90": float(np.percentile(rs_arr, 90)),
        "q_tail": float(q.get("decay_ratio", 0.0)),
    }


def parse_list(text, cast):
    return [cast(v) for v in text.split(",") if v != ""]


def main(argv):
    tag = "E1"
    cells = [2.5]
    kinds = ["gradient"]
    iters = [12]                      # the case default
    regm = [40.0]
    dtcap = [float(G.DT_CAP)]
    max_steps = 0                     # 0 = the case's own cap
    i = 0
    while i < len(argv):
        if argv[i] == "--tag":
            tag = argv[i + 1]; i += 2
        elif argv[i] == "--cells":
            cells = parse_list(argv[i + 1], float); i += 2
        elif argv[i] == "--kinds":
            kinds = parse_list(argv[i + 1], str); i += 2
        elif argv[i] == "--iters":
            iters = parse_list(argv[i + 1], int); i += 2
        elif argv[i] == "--regm":
            regm = parse_list(argv[i + 1], float); i += 2
        elif argv[i] == "--dtcap":
            dtcap = parse_list(argv[i + 1], float); i += 2
        elif argv[i] == "--max-steps":
            max_steps = int(argv[i + 1]); i += 2
        elif argv[i] in ("-h", "--help"):
            print(__doc__)
            return 0
        else:
            i += 1

    for d in (OUT, CKPT):
        if not os.path.isdir(d):
            os.makedirs(d)

    already = done_keys()
    points = []
    for kind in kinds:
        for cell in cells:
            for it in iters:
                for m in regm:
                    for dtc in dtcap:
                        points.append({"tag": tag, "kind": kind, "cell": cell,
                                       "iters": it, "regm": m, "dtcap": dtc,
                                       "max_steps": max_steps})

    print("=== picard scan %s: %d point(s), %d already done ==="
          % (tag, len(points), sum(1 for p in points if key_of(p) in already)))
    sys.stdout.flush()

    for cfg in points:
        if key_of(cfg) in already:
            print("  skip (done): %s" % key_of(cfg))
            continue
        print("  [%s]" % key_of(cfg))
        sys.stdout.flush()
        row = run_point(cfg)
        append_row(row)
        print("    -> steps %d  t %.0f s  filled %d  reach %d  OVERSHOOT %d cells"
              " = %.1f%% of filled area  |  picard mean %.1f max %d cap %.0f%%"
              "  resid mean %.2g p90 %.2g  |  wall %.0f s"
              % (row["steps"], row["t_stall"], row["filled"], row["reach"],
                 row["over_cells"], 100.0 * row["over_frac"],
                 row["picard_mean"], row["picard_max"], 100.0 * row["cap_frac"],
                 row["resid_mean"], row["resid_p90"], row["wall_s"]))
        sys.stdout.flush()

    print("")
    print("results appended to %s" % CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
