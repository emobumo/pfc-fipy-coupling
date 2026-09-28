# -*- coding: utf-8 -*-
"""
Anchor re-runs under the physical stop rule (src/analysis/stop_rule.py),
layer by layer.

    powershell -File scripts\run_local.ps1 scripts\rerun_rate.py A [--pass2]

Layer A (defines the baseline, must come first):
    base_2.5      uniform 0.18 line source through the channel case's run():
                  its stop volume is the NEW QUOTA and its footprint the NEW
                  DESIGN TARGET (replacing 166.15 m3/m and 155 cells)
    uniform / gradient line source, 2.5 and 1.25 m (up/down ratios)
    no-gravity uniform / gradient, 1.25 m
    gradient-strength sweep phi_top 0.25 / 0.35, 1.25 m
n_ref per family: the uniform porosity; the gradient field's porosity at the
hole's mid-point (each sweep field its own). --pass2 re-runs with separate
labels and outputs for the bit-for-bit comparison.
Outputs: outputs/rerun_rate/<layer>[_pass2]/ (results.json + the cases' own files).
"""
from __future__ import print_function

import imp
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis import stop_rule
from src.analysis.run_checkpoint import Checkpointer, state_fingerprint
from src.analysis.fill_diagnostics import reachable_domain

C = imp.load_source("inclined_hole_channel", os.path.join(REPO, "cases", "inclined_hole_channel.py"))
G = C.G
ROOT = os.path.join(REPO, "outputs", "rerun_rate")


def g_run(kind, cell, label, out, gravity=True, phi_top=None):
    """One gradient-case line-source run (G.march) at `cell`; returns scalars."""
    G.CELL = cell
    base_top = G.PHI_TOP
    if phi_top is not None:
        G.PHI_TOP = phi_top
    try:
        st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
        if not gravity:
            st["slurry_parameters"]["gravity_y"] = 0.0
        n_ref = G.n_ref_for(kind)
        vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
        keeper = Checkpointer(os.path.join(out, "ckpt_" + re.sub(r"[^A-Za-z0-9._-]+", "_", label)), every=200,
                              fingerprint=state_fingerprint(st, "rerunA|%s|%s|%g|g=%s|top=%s" % (
                                  label, kind, cell, gravity, phi_top) + stop_rule.rate_tag(n_ref)))
        t, v_in, v_store = G.march(st, mx, my, hc, vols, label, keeper=keeper, n_ref=n_ref)
        fm = G.front_metrics(st, mx, my, hc)
        s = np.asarray(st["saturation"].value, dtype=float)
        np.savez(os.path.join(out, "final_%s.npz" % re.sub(r"[^A-Za-z0-9._-]+", "_", label)),
                 s=s, x=mx, y=my, phi=phi, hole=hc)
        return {"label": label, "kind": kind, "cell": cell, "gravity": gravity, "phi_top": phi_top,
                "n_ref": n_ref, "reason": G.LAST_REASON[0], "t": t, "v_in": v_in, "v_store": v_store,
                "cells": fm["count"], "up": fm["perp_up"], "down": fm["perp_down"],
                "ratio": fm["perp_up"] / fm["perp_down"]}
    finally:
        G.PHI_TOP = base_top
        G.CELL = 2.5


def layer_a(pass2):
    sfx = "_pass2" if pass2 else ""
    out = os.path.join(ROOT, "A" + sfx)
    if not os.path.isdir(out):
        os.makedirs(out)
    done_path = os.path.join(out, "results.json")
    rows = json.load(open(done_path)) if os.path.exists(done_path) else []
    done = set(r["label"] for r in rows)

    def save():
        json.dump(rows, open(done_path, "w"), indent=1, default=float)

    # 1. the baseline through the channel case: new quota and target
    lab = "base_uniform_0.18_2.5" + sfx
    if lab not in done:
        G.CELL = 2.5
        C.OUT_ROOT = out
        _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, 2.5)
        row = C.run(lab, C.field("base", np.asarray(x), np.asarray(y)), v_quota=None)
        tgt = row.pop("filled_mask")
        row = dict((k, v) for k, v in row.items() if not isinstance(v, np.ndarray))
        row["target_cells"] = int(np.sum(tgt))
        np.save(os.path.join(out, "target_mask.npy"), tgt)
        rows.append(row)
        save()
    # 2-4. ratios
    jobs = [("uniform", 2.5, True, None), ("gradient", 2.5, True, None),
            ("uniform", 1.25, True, None), ("gradient", 1.25, True, None),
            ("uniform", 1.25, False, None), ("gradient", 1.25, False, None),
            ("gradient", 1.25, True, 0.25), ("gradient", 1.25, True, 0.35)]
    for kind, cell, grav, top in jobs:
        lab = "%s_%g%s%s%s" % (kind, cell, "" if grav else "_nograv",
                               "" if top is None else "_top%.2f" % top, sfx)
        if lab in done:
            continue
        rows.append(g_run(kind, cell, lab, out, gravity=grav, phi_top=top))
        save()
    for r in rows:
        print("%-34s %-9s t=%8.1f V=%8.3f cells=%s %s" % (
            r["label"], r.get("reason"), r["t"], r["v_in"], r.get("cells", r.get("filled_cells")),
            ("up/down %.3f" % r["ratio"]) if "ratio" in r else ("quota %.3f target %d" % (r["v_in"], r["target_cells"]))))
    return 0


def _scalars(r):
    out = dict((k, v) for k, v in r.items() if not isinstance(v, np.ndarray))
    if "stages" in out:
        out["stages"] = [dict((k, v) for k, v in st.items() if not isinstance(v, np.ndarray))
                         for st in out["stages"]]
    return out


def layer_b(pass2):
    """2.5 m, relative comparisons against layer A's new quota and target
    (both read from pass 1 of A): channel 5 fields, random seeds 1-3, P1 on
    collar_v06 (full hole, staged 7->10->17 and 8.5->17, pre-plug h 2.5 / 10)."""
    SS = imp.load_source("stage_supplement", os.path.join(REPO, "scripts", "stage_supplement.py"))
    S = SS.S
    C, G = S.C, S.G          # stage_supplement re-loads the case modules; use its copies
    sfx = "_pass2" if pass2 else ""
    out = os.path.join(ROOT, "B" + sfx)
    if not os.path.isdir(out):
        os.makedirs(out)
    base = [r for r in json.load(open(os.path.join(ROOT, "A", "results.json")))
            if r["label"] == "base_uniform_0.18_2.5"][0]
    quota = float(base["v_in"])
    target = np.load(os.path.join(ROOT, "A", "target_mask.npy"))
    done_path = os.path.join(out, "results.json")
    rows = json.load(open(done_path)) if os.path.exists(done_path) else []
    done = set(r["label"] for r in rows)

    def save():
        json.dump(rows, open(done_path, "w"), indent=1, default=float)

    G.CELL = 2.5
    for mod in (C, S):
        mod.OUT_ROOT = out
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, 2.5)
    mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    assert target.size == mx.size
    singles = []
    for pb in C.BAND_PHIS:                                  # porosity at the toe position
        singles.append(("B_band_toe_phi%.2f" % pb, C.field("band", mx, my, phi_band=pb)))
    for name, bx in C.BAND_POSITIONS:                       # position at phi 0.45 (toe shared)
        if name != "toe":
            singles.append(("B_band_%s_phi%.2f" % (name, C.BAND_PHI_POS),
                            C.field("band", mx, my, phi_band=C.BAND_PHI_POS, band_x=bx)))
    for sd in C.RANDOM_SEEDS:
        singles.append(("B_random_seed%d" % sd, C.field("random", mx, my, seed=sd)))
    mx_, my_, nx, ny, hole, phi_c, band, _ = SS.setup()
    singles.append(("B_p1_full_hole", phi_c))
    for h in (2.5, 10.0):
        singles.append(("B_p1_preplug_h%.2f" % h, SS.plugged(phi_c, band, hole, my, h)[0]))
    for lab, phi in singles:
        if lab + sfx in done:
            continue
        r = _scalars(C.run(lab + sfx, phi, v_quota=quota, target_mask=target))
        r["quota"] = quota
        rows.append(r)
        save()
    st0 = C.build_from_phi(phi_c)[0]
    reach_virgin = reachable_domain(st0, hole, p0=G.P0, use_gravity=True)["reachable"]
    for name, depths in S.SEQUENCES:
        lab = "B_p1_collar_v06/%s%s" % (name, sfx)
        if lab in done:
            continue
        r = _scalars(S.run_sequence(lab, phi_c, depths, quota, reach_virgin, hole, False))
        r["quota"] = quota
        rows.append(r)
        save()
    for r in rows:
        if "stages" in r:
            print("%-34s V_tot=%8.3f filled=%d shadow=%d | %s" % (
                r["label"], r["v_total"], r["filled"], r["shadowed"],
                "; ".join("%s t=%.0f V=%.2f" % (st["reason"], st["t"], st["v_in"]) for st in r["stages"])))
        else:
            print("%-34s %-6s t=%8.1f V=%8.3f fill=%d tgt=%.1f%% dsf=%.1f%%" % (
                r["label"], r["reason"], r["t"], r["v_in"], r["filled_cells"],
                100 * r["target_filled_fraction"], 100 * r["design_shortfall_fraction"]))
    return 0


def main(argv):
    if argv and argv[0] == "A":
        return layer_a("--pass2" in argv)
    if argv and argv[0] == "B":
        return layer_b("--pass2" in argv)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
