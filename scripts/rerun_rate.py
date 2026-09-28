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


def l_max(phi):
    return 83.33 * phi / (1.0 - phi)


P2_PHIS = (0.10, 0.14, 0.18)
C_JOBS = (
    # (group, field, cell, H, plan name, depths, budget=quota?)
    [("p2", phi, 1.25, 17.0, "single", (17.0,), False) for phi in P2_PHIS]
    + [("p2", phi, 1.25, 17.0, "7_17", (7.0, 17.0), False) for phi in P2_PHIS]
    + [("p2", 0.18, 1.25, 17.0, "5_17", (5.0, 17.0), False)]
    + [("struct", fld, 2.5, 17.0, name, d, True) for fld in ("base", "random")
       for name, d in (("single", (17.0,)), ("7_10_17", (7.0, 10.0, 17.0)), ("8.5_17", (8.5, 17.0)))]
    + [("p3", phi, 1.25, 17.0, name, d, False) for phi in (0.12, 0.16, 0.22)
       for name, d in (("single", (17.0,)), ("7_17", (7.0, 17.0)))]
    + [("p3", 0.14, 1.25, h, name, d, False) for h in (10.0, 25.0, 32.0)
       for name, d in (("single", (h,)), ("7_%g" % h, (7.0, h)))]
    + [("p2x", phi, 1.25, 17.0, name, d, False) for phi in P2_PHIS        # the optional rest of the 12
       for name, d in (("5_17", (5.0, 17.0)), ("7_10_17", (7.0, 10.0, 17.0))) if not (phi == 0.18 and name == "5_17")]
)


def _stage_extras(S, G, out, label, depths, n_ref):
    """Per stage: x_f (how far past its own toe the stage cemented along the
    hole axis, cells within one cell of it), x_f / L_max, and Qbar/Q_ref at the
    stage's stop, the window rate rebuilt from the recorded Q(t) exactly as the
    march accumulates V_in."""
    d = os.path.join(out, label)
    a = np.loadtxt(os.path.join(d, "cells.csv"), delimiter=",", skiprows=1)
    h = np.loadtxt(os.path.join(d, "injection_rate.csv"), delimiter=",", skiprows=1, ndmin=2)
    end, dv, nv = G.hole_geometry()
    along = (a[:, 0] - G.MOUTH[0]) * dv[0] + (a[:, 1] - G.MOUTH[1]) * dv[1]
    perp = np.abs((a[:, 0] - G.MOUTH[0]) * nv[0] + (a[:, 1] - G.MOUTH[1]) * nv[1])
    qref = stop_rule.q_ref(n_ref, G.A_CAL, G.P0, G.MU_P)
    res, t0 = [], 0.0
    for k, dep in enumerate(depths):
        sel = (a[:, 5] == k) & (perp < G.CELL)
        x_f = float(along[sel].max()) - dep if np.any(sel) else float("nan")
        hk = h[h[:, 2] == k]
        if hk.shape[0]:
            t = [0.0] + list(hk[:, 0] - t0)
            v = [0.0]
            for i in range(hk.shape[0]):
                v.append(v[-1] + hk[i, 1] * (t[i + 1] - t[i]))
            qr = stop_rule.window_rate(t, v) / qref
            t0 = float(hk[-1, 0])
        else:
            qr = float("nan")
        res.append({"x_f": x_f, "x_f_over_L": x_f / l_max(n_ref), "q_over_qref": qr})
    return res


def layer_c(pass2, groups):
    """Staged runs: P2 (uniform, 1.25 m, to refusal), structural staged
    (base / random, 2.5 m, budget = layer A's quota), P3 (1.25 m; porosity
    group at H 17, length group at phi 0.14 with H 10 / 25 / 32 -- the hole
    length is set at run time through G.HOLE_LEN, which every hole routine
    reads). n_ref = the uniform porosity (P2 / P3) or 0.18 (structural)."""
    S = imp.load_source("inclined_hole_staged", os.path.join(REPO, "cases", "inclined_hole_staged.py"))
    C, G = S.C, S.G
    sfx = "_pass2" if pass2 else ""
    out = os.path.join(ROOT, "C" + sfx)
    if not os.path.isdir(out):
        os.makedirs(out)
    S.OUT_ROOT = out
    quota = float([r for r in json.load(open(os.path.join(ROOT, "A", "results.json")))
                   if r["label"] == "base_uniform_0.18_2.5"][0]["v_in"])
    done_path = os.path.join(out, "results.json")
    rows = json.load(open(done_path)) if os.path.exists(done_path) else []
    done = set(r["label"] for r in rows)
    for grp, fld, cell, hl, name, depths, use_quota in C_JOBS:
        if grp not in groups:
            continue
        fname = fld if isinstance(fld, str) else "phi%.2f" % fld
        label = "C_%s/%s_H%g_c%g/%s%s" % (grp, fname, hl, cell, name, sfx)
        if label in done:
            continue
        G.CELL, G.HOLE_LEN = cell, hl
        try:
            _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, cell)
            mx, my = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
            nx = int(round((G.X_MAX - G.X_MIN) / cell))
            ny = int(round((G.Y_MAX - G.Y_MIN) / cell))
            full = np.asarray(G.hole_cells(mx, my, nx, ny), dtype=int)
            if isinstance(fld, str):
                phi, n_ref = S.make_field(fld, mx, my), C.PHI_BASE
            else:
                phi, n_ref = np.full(mx.size, float(fld)), float(fld)
            st0 = C.build_from_phi(phi)[0]
            reach_virgin = reachable_domain(st0, full, p0=G.P0, use_gravity=True)["reachable"]
            r = _scalars(S.run_sequence(label, phi, depths, quota if use_quota else None,
                                        reach_virgin, full, False, n_ref=n_ref))
            ext = _stage_extras(S, G, out, label, depths, n_ref)
            for st, e in zip(r["stages"], ext):
                st.update(e)
            end = G.hole_geometry()[0]
            r.update({"group": grp, "field": fname, "cell": cell, "H": hl, "plan": name,
                      "depths": list(depths), "n_ref": n_ref, "budget": quota if use_quota else None,
                      "L_max": l_max(n_ref), "toe": list(end),
                      "toe_to_wall": min(end[0] - G.X_MIN, G.X_MAX - end[0], end[1] - G.Y_MIN, G.Y_MAX - end[1])})
            rows.append(r)
            json.dump(rows, open(done_path, "w"), indent=1, default=float)
        finally:
            G.CELL, G.HOLE_LEN = 2.5, 17.0
    for r in rows:
        print("%-44s V_tot=%8.2f reach%%=%5.1f shadow=%4d | %s" % (
            r["label"], r["v_total"], 100 * r["over_reach_filled"], r["shadowed"],
            "; ".join("%s V=%.2f x_f/L=%.2f Q/Qref=%.3f" % (st["reason"], st["v_in"], st["x_f_over_L"],
                                                            st["q_over_qref"]) for st in r["stages"])))
    return 0


def layer_d(pass2, groups):
    """The scripts' own runs, redirected to outputs/rerun_rate/D[_pass2]/ so no
    old (legacy-rule) result or checkpoint is skipped, resumed or overwritten:
    zj       Zhaojin D2, E2 (to arrival, 1 m) and E2_20 (--seal20)
    boulder  boulder_bypass A / B / C (0.5 m)
    aging    viscosity_aging_gradient at 1.25 m (the grid ratio anchors are
             quoted on); its stop row must equal layer A's uniform / gradient 1.25."""
    sfx = "_pass2" if pass2 else ""
    out = os.path.join(ROOT, "D" + sfx)
    flag = ["--pass2"] if pass2 else []
    if "zj" in groups:
        ZR = imp.load_source("zhaojin_runs", os.path.join(REPO, "scripts", "zhaojin_runs.py"))
        ZR.OUT = os.path.join(out, "zhaojin")
        ZR.main(["--arrival", "--only", "E2,D2"] + flag)
        ZR.main(["--arrival", "--seal20"] + flag)
    if "boulder" in groups:
        BB = imp.load_source("boulder_bypass", os.path.join(REPO, "cases", "boulder_bypass.py"))
        BB.OUT = os.path.join(out, "boulder")
        BB.main([])
    if "aging" in groups:
        VA = imp.load_source("viscosity_aging_gradient",
                             os.path.join(REPO, "scripts", "viscosity_aging_gradient.py"))
        VA.G.CELL = 1.25
        VA.OUT = os.path.join(out, "aging_1.25")
        VA.main()
    return 0


def main(argv):
    if argv and argv[0] == "D":
        return layer_d("--pass2" in argv, [a for a in argv[1:] if not a.startswith("--")]
                       or ["zj", "boulder", "aging"])
    if argv and argv[0] == "A":
        return layer_a("--pass2" in argv)
    if argv and argv[0] == "B":
        return layer_b("--pass2" in argv)
    if argv and argv[0] == "C":
        groups = [a for a in argv[1:] if not a.startswith("--")] or ["p2", "struct", "p3"]
        return layer_c("--pass2" in argv, groups)
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
