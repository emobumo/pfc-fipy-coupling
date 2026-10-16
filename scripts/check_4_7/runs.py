# -*- coding: utf-8 -*-
"""
4.7 check, part 3: the solver runs of the case-study section (items 1.6,
3.1-3.4, 4.1, 4.3, 4.4 of the 4.7 pre-draft check). Post-processing of the
stored results only, no solver.

    powershell -File scripts\\run_local.ps1 scripts\\check_4_7\\runs.py

  - run codes and settings, from the run script's own tables;
  - stop volume, time, stop reason and arrivals (stored-ore bottom = lowest
    2 m of the stored ore, outlet drift; "arrived" = some cell of the target
    with S >= 0.5) of every run, with the second pass compared field by field;
  - the share below the grouting level two ways: over the stored grout
    (n S V summed over fill, contacts, stored ore and drift, source cells
    excluded) and over V_in;
  - arrival volume / 28.6 (the original design volume per metre of strike);
  - the current rate stop rule (eq. 4-29, eps 0.1, n_ref = the zone's fill
    porosity) replayed on the stored series (one row per 50 steps, linear
    interpolation inside the window): would it have stopped a run that was
    computed under the earlier rules? Checked against the three runs that
    were recomputed under the rate rule.
  - the seal runs: largest volume below the grouting level over every
    stored series row and in the final field.
Elevations relative to the grouting level.

Output: outputs/check_4_7/runs.txt
"""
from __future__ import print_function

import imp
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis import stop_rule

Z = imp.load_source("case_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
R = imp.load_source("case_runs", os.path.join(REPO, "scripts", "zhaojin_runs.py"))
OUT = os.path.join(REPO, "outputs", "check_4_7")
OLD = os.path.join(REPO, "outputs", "zhaojin_runs")
NEW = os.path.join(REPO, "outputs", "rerun_rate", "D", "zhaojin")
NEW2 = os.path.join(REPO, "outputs", "rerun_rate", "D_pass2", "zhaojin")
ARR2 = os.path.join(REPO, "outputs", "zhaojin_arrival_pass2")
ZN = {"z1_cuts1-6": "I", "z2_cuts6-10": "II", "z3_cuts10-15": "III"}
Q = float(Z.ORIGINAL_DESIGN["grout_per_m"])


def load(path):
    return dict((json.loads(l)["tag"], json.loads(l)) for l in open(path) if l.strip())


def fields_equal(a, b):
    fa, fb = np.load(a), np.load(b)
    return all(np.array_equal(fa[k], fb[k]) for k in fa.files)


def replay(series, n_ref):
    """Rate rule on the stored series: first row where the window-mean rate < eps Q_ref."""
    sp = Z.slurry_params()
    q = stop_rule.q_ref(n_ref, sp["calibrated_permeability_coefficient"], Z.P0, sp["plastic_viscosity"])
    t = [0.0] + list(series[:, 0]); v = [0.0] + list(series[:, 1])
    ratios = []
    for i in range(2, len(t)):
        ratios.append((stop_rule.window_rate(t[:i + 1], v[:i + 1]) / q, t[i], v[i]))
    first = [r for r in ratios if r[0] < stop_rule.EPS]
    return q, min(ratios), (first[0] if first else None)


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    L = []
    w = L.append
    old, old2 = load(os.path.join(OLD, "results.jsonl")), load(os.path.join(OLD, "results_pass2.jsonl"))
    new, new2 = load(os.path.join(NEW, "results.jsonl")), load(os.path.join(NEW2, "results_pass2.jsonl"))
    arr2 = load(os.path.join(ARR2, "results_pass2.jsonl"))
    w("4.7 check, part 3: solver runs (1 m grid; elevations relative to the grouting level)")
    w("")
    w("1.6 Run codes (scripts/zhaojin_runs.py): common settings %s" % ", ".join(
        "%s %s" % (k, v) for k, v in sorted(R.COMMON.items())))
    for group, rows in (("stop at the changed-design volume", R.RUNS), ("run to arrival / stall / cap %.0f m3/m" % R.ARRIVAL_CAP, R.ARRIVAL_RUNS),
                        ("stop at the original design volume %.1f m3/m" % Q, R.RUNS_ORIGINAL),
                        ("20 deg seal, changed-design volume", R.RUNS_SEAL20), ("20 deg seal, to arrival", R.ARRIVAL_SEAL20)):
        w("  %s:" % group)
        for tag, _, ov in rows:
            cfg = Z.config(**dict(R.COMMON, **ov))
            w("    %-6s zone %-3s contacts %-5s seal %-5s %2.0f deg" % (tag, ZN[cfg["zone"]], cfg["contacts"], cfg["seal"], cfg["hole_angle"]))
    w("  stop rules in the code: the volume (quota) once V_in >= it; with --arrival also the first step a drift cell has S >= 0.5;")
    w("  otherwise the default rule of the run's date: the 400-step stall window (STOP_RULE=legacy, all runs in outputs/zhaojin_runs,")
    w("  and the second pass of A2-C2) or the rate rule (eq. 4-29, n_ref = fill porosity; D2, E2, E2_20 in outputs/rerun_rate/D)")
    w("  V_in = cumulative net outflow of the source cells (div q at the source x cell volume x dt): grout delivered into the")
    w("  ground, compressive storage included; the source cells themselves start full and are not counted")
    w("")

    w("3.1/3.2/4.1 Results")
    w("  %-6s %-4s %-13s %6s %11s %12s | %9s %9s %9s | %8s %8s | %11s %11s | %s" % (
        "run", "zone", "stop", "steps", "t s", "V_in m3/m", "layer", "above", "below", "below/st", "below/V",
        "V at ore bt", "V at drift", "pass 2"))
    rows = []
    for tag in ("A3", "B3", "C3", "D3", "A2", "B2", "C2", "E", "E20"):
        rows.append((tag, old[tag], "outputs/zhaojin_runs"))
    for tag in ("D2", "E2", "E2_20"):
        rows.append((tag, new[tag], "outputs/rerun_rate/D"))
    for tag, r, src in rows:
        st = r["design"] + r["above"] + r["below"]
        sn = r.get("snaps", {})
        if tag in ("A3", "B3", "C3", "D3"):
            p2 = "scalars %s, field %s" % (old2[tag] == dict(old[tag], wall=old2[tag]["wall"]),
                                          fields_equal(os.path.join(OLD, "final_%s.npz" % tag), os.path.join(OLD, "final_%s_pass2.npz" % tag)))
        elif tag in ("A2", "B2", "C2"):
            # the second pass also records design_mode / layer / design_capacity, added to the run script later
            common = (set(old[tag]) & set(arr2[tag])) - set(["wall"])
            same = all(old[tag][k] == arr2[tag][k] for k in common)
            p2 = "scalars %s (%d shared keys), field %s" % (same, len(common), fields_equal(os.path.join(OLD, "final_%s.npz" % tag),
                                                                 os.path.join(ARR2, "final_%s_pass2.npz" % tag)))
        elif tag in new:
            p2 = "scalars %s, field %s" % (dict(new[tag], wall=0) == dict(new2[tag], wall=0),
                                          fields_equal(os.path.join(NEW, "final_%s.npz" % tag), os.path.join(NEW2, "final_%s_pass2.npz" % tag)))
        else:
            p2 = "no second pass (see the design-volume snapshot check below)"
        w("  %-6s %-4s %-13s %6d %11.4f %12.6f | %9.4f %9.4f %9.6f | %8.4f%% %7.4f%% | %11s %11s | %s" % (
            tag, ZN[r["zone"]], r["reason"], r["steps"], r["t"], r["v_in"], r["design"], r["above"], r["below"],
            100 * r["below"] / st, 100 * r["below"] / r["v_in"],
            ("%.6f" % sn["reach_-56m"]["v_in"]) if "reach_-56m" in sn else "never",
            ("%.6f" % sn["reach_-63m"]["v_in"]) if "reach_-63m" in sn else "never", p2))
    w("  'layer' = the run's design layer (A3-D3: the original 22 m layer; others: the changed-design layer); below = z <= 0;")
    w("  below/st = below over the stored grout (the run script's below_share); below/V = below over V_in")
    w("")
    w("  arrival multiples (V_in at arrival / %.1f):" % Q)
    for tag in ("A2", "B2", "C2", "D2"):
        r = old[tag] if tag != "D2" else new[tag]
        sn = r["snaps"]
        w("    %-3s zone %-3s ore bottom %s, drift %s" % (
            tag, ZN[r["zone"]],
            ("%.6f -> %.6f x at t %.4f s" % (sn["reach_-56m"]["v_in"], sn["reach_-56m"]["v_in"] / Q, sn["reach_-56m"]["t"])) if "reach_-56m" in sn else "never",
            ("%.6f -> %.6f x at t %.4f s" % (sn["reach_-63m"]["v_in"], sn["reach_-63m"]["v_in"] / Q, sn["reach_-63m"]["t"])) if "reach_-63m" in sn else "never"))
    v = [old[t]["v_in"] for t in ("A2", "B2", "C2")]
    w("  3.3 V_in at the drift, zones II / I / III: %s; max/min %.6f, (max - min)/mean %.4f" % (
        " / ".join("%.6f" % x for x in v), max(v) / min(v), (max(v) - min(v)) / np.mean(v)))
    w("  design-volume snapshots: E2 (to arrival) vs E (changed-design quota) at V_in >= 50.4: t %s, V_in %s, below %s" % (
        old["E2"]["snaps"]["design_volume"]["t"] == old["E"]["t"], old["E2"]["snaps"]["design_volume"]["v_in"] == old["E"]["v_in"],
        old["E2"]["snaps"]["design_volume"]["below"] == old["E"]["below"]))
    w("")

    w("3.2 The current rate rule replayed on the stored series (eps %.2f, window %.2f t, n_ref = fill porosity)" % (stop_rule.EPS, stop_rule.BETA))
    for tag, path in (("A2", OLD), ("B2", OLD), ("C2", OLD), ("A3", OLD), ("B3", OLD), ("C3", OLD), ("D3", OLD),
                      ("D2", NEW), ("E2", NEW), ("E2_20", NEW)):
        r = (old if path == OLD else new)[tag]
        ser = np.load(os.path.join(path, "final_%s.npz" % tag))["series"]
        n_ref = float(Z.ZONES[r["zone"]]["phi"])
        q, mn, first = replay(ser, n_ref)
        w("  %-6s n_ref %.2f Q_ref %.6g m2/s: rows %4d (to t %.1f s); min Qbar/Q_ref %.4f at t %.1f s; first row below eps: %s; run stopped: %s at t %.4f s, V_in %.6f" % (
            tag, n_ref, q, len(ser), ser[-1, 0], mn[0], mn[1],
            ("t %.2f s, V_in %.4f" % (first[1], first[2])) if first else "none", r["reason"], r["t"], r["v_in"]))
    w("")

    w("4.3/4.4 Seal runs: volume below the grouting level")
    for tag, path, res in (("E", OLD, old), ("E2 (old rule)", OLD, old), ("E20", OLD, old), ("E2_20 (old rule)", OLD, old),
                           ("E2", NEW, new), ("E2_20", NEW, new)):
        key = tag.split()[0]
        f = np.load(os.path.join(path, "final_%s.npz" % key))
        ser = f["series"]
        region, s, phi, y = f["region"], f["s"], f["phi"], f["y"]
        below = ((region == Z.FILL) | (region == Z.CONTACT) | (region == Z.ORE) | (region == Z.OUTLET)) & (y <= Z.Y_LEVEL)
        mask = np.zeros(s.size, dtype=bool); mask[f["hole"]] = True
        below &= ~mask
        seal = region == Z.SEAL
        w("  %-17s series rows %3d, max below %.3e m3/m; final below %.3e m3/m, cells below with S >= 0.5: %d, max S below %.3e; "
          "seal cells %d (z %+.1f .. %+.1f), max S in the seal %.3e" % (
              tag, len(ser), float(np.max(ser[:, 4])), float(np.sum(phi[below] * s[below])), int(np.sum(s[below] >= 0.5)),
              float(np.max(s[below])), int(seal.sum()), float(y[seal].min() - Z.Y_LEVEL), float(y[seal].max() - Z.Y_LEVEL),
              float(np.max(s[seal]))))
    text = "\n".join(L)
    print(text)
    open(os.path.join(OUT, "runs.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
