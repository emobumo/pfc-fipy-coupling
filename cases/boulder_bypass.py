# -*- coding: utf-8 -*-
"""
Can a bypass void form under a large block? (site report: "巨大骨架下")

    powershell -File scripts\run_local.ps1 cases\boulder_bypass.py [--only A,B,C]

A single impermeable block does NOT make one: behind an obstacle the
path-integral reach is an open notch, never a pocket sealed off by grout.
Something has to wrap around a slow region before it fills. This case tests
the two candidates the continuum model can represent (trapped air, the third,
it cannot -- unfilled cells sit at air pressure, i.e. air always vents):

    A  bridged block, no rim                        control
    B  bridged block + a high-porosity rim           wall effect: loose packing
                                                    against the block faces is a
                                                    fast path that wraps round
    C  as B, compacted fines under the block         the pocket itself is dense

Geometry (a "架空" bridge): a lid block resting on two leg blocks leaves a
pocket under the lid that opens DOWNWARD. Grout comes from a source above.
Every block cell is solid (phi 1e-3, like cement/rock elsewhere); the rim is
the pore space within RIM of the blocks, excluding the pocket.

The run marches to stall under constant pressure (5 MPa, gravity on,
parameter set v0.6) and classifies the unfilled space with the solid-aware
classifier (fill_diagnostics.classify_unfilled(..., solid_phi)) along the
way. A stall fills everything reachable, so a bypass void of type B is
TRANSIENT: what the run reports is the window of injected volume in which
stopping would leave one -- the field stops on refusal OR on volume, and
either can land in that window. A type-C void is permanent (unreachable).

FIRST RESULTS (2026-09-26). All three variants seal a bypass void under the
lid -- the rim is NOT needed: the bridge's downward-opening concavity is
enough (the fronts close under the opening before the pocket fills). A
single convex block cannot do this; a concave, downward-open bridge can.
The rim moves the window earlier (A: 33.6..38.9 of 44.6 m3/m injected;
B: 23.9..31.5 of 48.3); peak 88% of the pocket in both. In C the dense cells
under the lid stay sealed for most of the run and are filled only thousands
of seconds after everything else has stalled: that is the regularized
yield creeping pressure into a sealed pocket (true Bingham grout at rest
does not transmit it), so a permanent dense inclusion is read from the
reach diagnostic, not from the solver's long-time state.

Crash-safe: checkpointed; finished variants are skipped on re-run.
"""
from __future__ import print_function

import imp
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from fipy import CellVariable

from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.models.slurry_transport.variables import initialize_slurry_variables
from src.models.slurry_transport.equations import solve_transport_step
from src.analysis.fill_diagnostics import (
    reachable_domain, classify_unfilled, BYPASS_VOID, ENCLOSED_UNREACHABLE, FILLED,
    OVERSHOOT,
)
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint
from src.analysis import stop_rule

# Same slurry physics as the Zhaojin section (v0.6, face form, threshold).
Z = imp.load_source("zhaojin_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
P0 = Z.P0
OUT = os.path.join(REPO, "outputs", "boulder_bypass")

X_MIN, X_MAX, Y_MIN, Y_MAX = -8.0, 8.0, 0.0, 16.0
CELL = 0.5
PHI_MATRIX = 0.18
PHI_SOLID = 1.0e-3
SOLID_PHI = 1.0e-2           # classifier: at or below this is not pore space
RIM = 1.0                    # wall-effect band thickness [m]
LID = (-2.0, 2.0, 8.0, 9.0)  # x0, x1, y0, y1
LEGS = [(-2.0, -1.0, 6.0, 8.0), (1.0, 2.0, 6.0, 8.0)]
POCKET = (-1.0, 1.0, 6.0, 8.0)
SOURCE = (-0.5, 0.5, 12.5, 13.5)     # a short vertical hole segment above

VARIANTS = [
    ("A", "bridged block, no rim (control)", dict(phi_rim=PHI_MATRIX, phi_pocket=PHI_MATRIX)),
    ("B", "bridged block + loose rim 0.45", dict(phi_rim=0.45, phi_pocket=PHI_MATRIX)),
    ("C", "rim 0.45 + compacted fines 0.03 under the lid", dict(phi_rim=0.45, phi_pocket=0.03)),
]

DT_CAP = 5.0
MAX_STEPS = 40000
CLASSIFY_EVERY = 25
STALL_WINDOW = 400            # legacy stop rule only (STOP_RULE=legacy)
STALL_REL = 1.0e-5            # default: rate rule, n_ref = the matrix porosity


def _in(box, x, y):
    x0, x1, y0, y1 = box
    return (x > x0) & (x < x1) & (y > y0) & (y < y1)


def fields(mx, my, phi_rim, phi_pocket):
    """Porosity plus masks: solid (blocks), rim, pocket, source."""
    solid = _in(LID, mx, my)
    for leg in LEGS:
        solid |= _in(leg, mx, my)
    pocket = _in(POCKET, mx, my)
    sx, sy = mx[solid], my[solid]
    d = np.full(mx.size, np.inf)
    for i in range(mx.size):
        d[i] = np.min(np.hypot(sx - mx[i], sy - my[i]))
    rim = (d <= RIM + 1e-9) & np.logical_not(solid) & np.logical_not(pocket)
    phi = np.full(mx.size, PHI_MATRIX)
    phi[rim] = phi_rim
    phi[pocket] = phi_pocket
    phi[solid] = PHI_SOLID
    src = np.where(_in(SOURCE, mx, my))[0]
    return phi, solid, rim, pocket, src


def build_state(phi, src):
    mesh, x, y, fx, fy = build_mesh_for_domain(X_MIN, X_MAX, Y_MIN, Y_MAX, CELL)
    state = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    state.update(initialize_slurry_variables(mesh, params=Z.slurry_params()))
    Z.set_porosity(state, phi)
    mask = np.zeros(phi.size)
    mask[src] = 1.0
    state["interior_dirichlet_mask"] = CellVariable(mesh=mesh, value=mask)
    state["interior_dirichlet_value"] = P0
    s0 = np.zeros(phi.size)
    s0[src] = 1.0
    state["saturation"].setValue(s0)
    state["pressure"].setValue(np.where(mask > 0.5, P0, 0.0))
    return state


def pocket_record(state, src, reach, pocket, vols, t, v_in, step):
    cls = classify_unfilled(state, src, reach, s_c=0.5, solid_phi=SOLID_PHI)
    cat = cls["category"]
    pv = float(np.sum(vols[pocket]))
    return {
        "t": t, "v_in": v_in, "step": step,
        # grout in a cell the reach says is out of budget still fills it;
        # it is booked separately as creep, not dropped from "filled"
        "pocket_filled": float(np.sum(vols[pocket & ((cat == FILLED) | (cat == OVERSHOOT))])) / pv,
        "pocket_overshoot": float(np.sum(vols[pocket & (cat == OVERSHOOT)])) / pv,
        "pocket_bypass": float(np.sum(vols[pocket & (cat == BYPASS_VOID)])) / pv,
        "pocket_enclosed_unreachable": float(np.sum(vols[pocket & (cat == ENCLOSED_UNREACHABLE)])) / pv,
        "n_bypass_voids": cls["n_bypass_voids"],
        "n_enclosed_unreachable": cls["n_enclosed_unreachable"],
        "bypass_area": float(np.sum(vols[cat == BYPASS_VOID])),
    }


def run_one(tag, label, cfg):
    mesh, x, y, fx, fy = build_mesh_for_domain(X_MIN, X_MAX, Y_MIN, Y_MAX, CELL)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    phi, solid, rim, pocket, src = fields(mx, my, cfg["phi_rim"], cfg["phi_pocket"])
    state = build_state(phi, src)
    vols = np.asarray(state["mesh"].cellVolumes, dtype=float)
    hm = np.zeros(mx.size, dtype=bool)
    hm[src] = True
    reach = reachable_domain(state, src, p0=P0, use_gravity=True)["reachable"]
    cap = float(np.sum(phi[reach] * vols[reach]))

    sp = Z.slurry_params()
    qref = stop_rule.q_ref(PHI_MATRIX, sp["calibrated_permeability_coefficient"], P0, sp["plastic_viscosity"])
    keeper = Checkpointer(os.path.join(OUT, "ckpt_" + tag), every=500,
                          fingerprint=state_fingerprint(state, "boulder|%s|%r" % (tag, sorted(cfg.items()))
                                                        + stop_rule.rate_tag(PHI_MATRIX)))
    t, v_in, step0, series, snaps = 0.0, 0.0, 0, [], {}
    resumed = keeper.restore(state)
    if resumed:
        step0 = int(resumed["step"])
        t, v_in = float(resumed["t"]), float(resumed["v_in"])
        series = json.loads(str(resumed["series_json"]))
        snaps = dict((k, np.asarray(v)) for k, v in json.loads(str(resumed["snaps_json"])).items())
        print("  [%s] resuming at step %d (t=%.1f s)" % (tag, step0, t))
    else:
        print("  [%s] %s -- %d cells, reachable pore volume %.2f m3/m, pocket reachable %d/%d"
              % (tag, label, mx.size, cap, int(np.sum(reach & pocket)), int(np.sum(pocket))))
    sys.stdout.flush()

    history = []
    t_hist, v_hist = [0.0], [0.0]
    if resumed and "t_hist" in resumed:
        t_hist = list(np.atleast_1d(resumed["t_hist"]))
        v_hist = list(np.atleast_1d(resumed["v_hist"]))
    reason = "max_steps"
    w0 = time.time()
    for step in range(step0, MAX_STEPS):
        dt = solve_transport_step(state, dt_cap=DT_CAP)
        t += float(dt)
        dq = np.asarray(state["last_div_q"], dtype=float)
        v_in += float(np.sum(dq[hm] * vols[hm])) * float(dt)
        t_hist.append(t)
        v_hist.append(v_in)
        s = np.array(state["saturation"].value, copy=True)
        s[hm] = 1.0
        state["saturation"].setValue(s)
        history.append(v_in)
        if step % CLASSIFY_EVERY == 0:
            rec = pocket_record(state, src, reach, pocket, vols, t, v_in, step + 1)
            series.append(rec)
            # keep the field at the first moment the pocket is sealed in
            if rec["pocket_bypass"] > 0 and "first_bypass" not in snaps:
                snaps["first_bypass"] = s.copy()
            if rec["pocket_bypass"] > 0:
                snaps["last_bypass"] = s.copy()
        if step % 1000 == 0:
            r = series[-1] if series else {}
            print("    step %6d  t=%9.2f s  dt=%.2e  V_in=%7.2f / %.2f  pocket filled %.0f%% bypass %.0f%% dense %.0f%%"
                  % (step, t, dt, v_in, cap, 100 * r.get("pocket_filled", 0), 100 * r.get("pocket_bypass", 0),
                     100 * r.get("pocket_enclosed_unreachable", 0)))
            sys.stdout.flush()
        if stop_rule.legacy():
            if len(history) > STALL_WINDOW and history[-1] - history[-1 - STALL_WINDOW] < STALL_REL * cap:
                reason = "legacy_stall"
                break
        elif stop_rule.stop_rule_physical(t_hist, v_hist, qref):
            reason = "rate"
            break
        keeper.maybe_save(state, step=step + 1, t=t, v_in=v_in, series_json=json.dumps(series),
                          snaps_json=json.dumps(dict((k, v.tolist()) for k, v in snaps.items())),
                          t_hist=np.asarray(t_hist), v_hist=np.asarray(v_hist))
    keeper.finish()

    final = pocket_record(state, src, reach, pocket, vols, t, v_in, step + 1)
    series.append(final)
    s = np.asarray(state["saturation"].value, dtype=float)
    np.savez(os.path.join(OUT, "final_%s.npz" % tag), s=s, phi=phi, x=mx, y=my, src=src,
             solid=solid, rim=rim, pocket=pocket, reach=reach,
             first_bypass=snaps.get("first_bypass", np.zeros(0)),
             last_bypass=snaps.get("last_bypass", np.zeros(0)))
    by = [r for r in series if r["pocket_bypass"] > 0]
    return {
        "tag": tag, "label": label, "cfg": cfg, "reason": reason, "steps": step + 1, "t": t,
        "wall": time.time() - w0, "v_in": v_in, "cap": cap,
        "pocket_reachable": int(np.sum(reach & pocket)), "pocket_cells": int(np.sum(pocket)),
        "final": final,
        "bypass_window_v": [by[0]["v_in"], by[-1]["v_in"]] if by else None,
        "bypass_window_t": [by[0]["t"], by[-1]["t"]] if by else None,
        "bypass_peak": max(r["pocket_bypass"] for r in by) if by else 0.0,
        "series": series,
    }


def main(argv):
    only = None
    if "--only" in argv:
        only = set(argv[argv.index("--only") + 1].split(","))
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    done_path = os.path.join(OUT, "results.jsonl")
    done = set()
    if os.path.exists(done_path):
        for line in open(done_path):
            done.add(json.loads(line)["tag"])
    for tag, label, cfg in VARIANTS:
        if (only is not None and tag not in only) or tag in done:
            continue
        r = run_one(tag, label, cfg)
        with open(done_path, "a") as h:
            h.write(json.dumps(r) + "\n")
        f = r["final"]
        print("  [%s] %s after %d steps, t=%.1f s (wall %.0f s): V_in %.2f / reachable %.2f | pocket at end: "
              "filled %.0f%%, bypass %.0f%%, dense-enclosed %.0f%% | bypass window V %s, peak %.0f%% of pocket"
              % (tag, r["reason"], r["steps"], r["t"], r["wall"], r["v_in"], r["cap"],
                 100 * f["pocket_filled"], 100 * f["pocket_bypass"], 100 * f["pocket_enclosed_unreachable"],
                 ("%.2f..%.2f" % tuple(r["bypass_window_v"])) if r["bypass_window_v"] else "none",
                 100 * r["bypass_peak"]))
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
