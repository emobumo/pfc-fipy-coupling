# -*- coding: utf-8 -*-
"""
Connectivity of the unfilled set: 4- vs 8-adjacency, and enclosed components
outside the boulder case. Post-processing only (no solver).

    powershell -File scripts\run_local.ps1 scripts\connectivity_check.py

classify_unfilled groups unfilled pore cells through shared faces
(4-adjacency on the structured grid) over the WHOLE computational domain; a
component owning any exterior face is not enclosed. Here the same rule is
re-implemented on the grid index with a 4- or 8-neighbour structure:

  filled    S >= 0.5, plus the source cells
  pore      unfilled and (solid_phi given) porosity > solid_phi
  enclosed  a pore component with no cell on the domain border
  unified   enclosed: reachable cells -> bypass void, unreachable -> dense
            inclusion (the solid_phi branch of classify_unfilled)

Part 1 (boulder A/B/C, solid_phi 0.01): pocket fractions at the stored
first-closure snapshot and at the final state, 4 vs 8. 8-enclosed implies
4-enclosed, so the first 8-closure cannot precede the first 4-closure; if the
4-first snapshot is already 8-closed, the two first-closure volumes coincide.
The 4-neighbour numbers must reproduce results.jsonl (self-check).

Part 2 (non-boulder final states): number of enclosed components, split into
all-reachable / all-unreachable / mixed, for 4 and 8, with solids off (the
current setting of these cases) and on (porosity <= 0.01).

Output: outputs/connectivity_check/connectivity_check.txt
"""
from __future__ import print_function

import glob
import imp
import json
import os
import sys

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain

OUT = os.path.join(REPO, "outputs", "connectivity_check")
RR = os.path.join(REPO, "outputs", "rerun_rate")
S_C = 0.5
SOLID_PHI = 1.0e-2
STRUCT = {4: ndimage.generate_binary_structure(2, 1), 8: ndimage.generate_binary_structure(2, 2)}


def grid(x, y):
    xs, ys = np.unique(np.round(x, 6)), np.unique(np.round(y, 6))
    ix = np.searchsorted(xs, np.round(x, 6))
    iy = np.searchsorted(ys, np.round(y, 6))
    assert xs.size * ys.size == x.size, "not a full rectangular grid"
    return ix, iy, xs.size, ys.size


def components(x, y, pore, conn):
    """Per-cell component label (0 = not pore) and the set of labels touching the border."""
    ix, iy, nx, ny = grid(x, y)
    g = np.zeros((ny, nx), dtype=bool)
    g[iy, ix] = pore
    lab, n = ndimage.label(g, structure=STRUCT[conn])
    border = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))) - set([0])
    return lab[iy, ix], n, border


def enclosed_stats(x, y, filled, reach, phi, conn, solid_on):
    pore = np.logical_not(filled)
    if solid_on:
        pore &= phi > SOLID_PHI
    lab, n, border = components(x, y, pore, conn)
    out = {"enclosed": 0, "all_reach": 0, "all_unreach": 0, "mixed": 0, "cells": 0}
    for c in range(1, n + 1):
        if c in border:
            continue
        m = lab == c
        out["enclosed"] += 1
        out["cells"] += int(np.sum(m))
        r = reach[m]
        if np.all(r):
            out["all_reach"] += 1
        elif not np.any(r):
            out["all_unreach"] += 1
        else:
            out["mixed"] += 1
    return out


def unified_categories(x, y, filled, reach, phi, conn):
    pore = np.logical_not(filled) & (phi > SOLID_PHI)
    lab, n, border = components(x, y, pore, conn)
    enc = (lab > 0) & np.logical_not(np.in1d(lab, list(border)))
    return enc & reach, enc & np.logical_not(reach)          # bypass void, dense inclusion


# --- part 1: boulder ----------------------------------------------------------

def boulder():
    rec = {}
    for line in open(os.path.join(RR, "D", "boulder", "results.jsonl")):
        r = json.loads(line)
        first = [s for s in r["series"] if s["pocket_bypass"] > 0][0]
        rec[r["tag"]] = (r["cap"], first, r["series"][-1])
    lines = ["PART 1  boulder A/B/C (0.5 m, solid_phi 0.01): pocket fractions, 4- vs 8-adjacency", "",
             "%-3s %-14s %4s | %8s %8s %8s | %s" % ("var", "snapshot", "conn", "filled", "bypass", "dense", "recorded (4)")]
    for tag in ("A", "B", "C"):
        f = np.load(os.path.join(RR, "D", "boulder", "final_%s.npz" % tag))
        x, y, phi, reach, pocket = f["x"], f["y"], f["phi"], f["reach"].astype(bool), f["pocket"].astype(bool)
        cap, first, last = rec[tag]
        for snap, key, ref in (("first closure", "first_bypass", first), ("final", "s", last)):
            s = np.asarray(f[key], dtype=float)
            filled = s >= S_C
            filled[f["src"]] = True
            for conn in (4, 8):
                by, dense = unified_categories(x, y, filled, reach, phi, conn)
                pv = float(np.sum(pocket))
                vals = (np.sum(pocket & filled) / pv, np.sum(pocket & by) / pv, np.sum(pocket & dense) / pv)
                note = ""
                if conn == 4:
                    want = (ref["pocket_filled"], ref["pocket_bypass"], ref["pocket_enclosed_unreachable"])
                    note = "%.3f %.3f %.3f %s" % (want + ("ok" if np.allclose(vals, want) else "MISMATCH",))
                lines.append("%-3s %-14s %4d | %8.3f %8.3f %8.3f | %s" % ((tag, snap, conn) + vals + (note,)))
        lines.append("    first 4-closure at V_in %.4f m3/m = %.1f%% of reachable pore %.2f m3/m"
                     % (first["v_in"], 100.0 * first["v_in"] / cap, cap))
    return lines


# --- part 2: non-boulder final states -------------------------------------------

def cases():
    """(label, x, y, filled, reach, phi)"""
    out = []
    G = imp.load_source("inclined_hole_gradient", os.path.join(REPO, "cases", "inclined_hole_gradient.py"))
    G.CELL = 1.25
    for kind in ("uniform", "gradient"):
        st, mx, my, nx, ny, phi, hc, k = G.build_case(kind)
        f = np.load(os.path.join(RR, "A", "final_%s_1.25.npz" % kind))
        assert np.allclose(f["x"], mx) and np.allclose(f["y"], my) and np.allclose(f["phi"], phi)
        reach = reachable_domain(st, hc, p0=G.P0, use_gravity=True)["reachable"]
        filled = np.asarray(f["s"], dtype=float) >= S_C
        filled[np.asarray(hc, dtype=int)] = True
        out.append(("A " + kind + " 1.25", mx, my, filled, reach, np.asarray(phi, dtype=float)))
    for p in sorted(glob.glob(os.path.join(RR, "B", "*", "cells.csv")) + glob.glob(os.path.join(RR, "B", "*", "*", "cells.csv"))
                    + glob.glob(os.path.join(RR, "C", "*", "*", "*", "cells.csv"))):
        head = open(p).readline().strip().split(",")
        a = np.loadtxt(p, delimiter=",", skiprows=1)
        cat = a[:, -1].astype(int)
        filled = (cat == 0) | (cat == 4)            # FILLED or OVERSHOOT, exactly as classified
        if head[3] == "S":                           # single-pass layout
            reach, phi = a[:, 5] > 0.5, a[:, 2]
        else:                                        # staged layout: phi_final, reachable_virgin
            reach, phi = a[:, 6] > 0.5, a[:, 3]
        lab = os.path.relpath(os.path.dirname(p), RR).replace("\\", "/")
        out.append((lab, a[:, 0], a[:, 1], filled, reach, phi))
    Z = imp.load_source("zhaojin_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
    ZR = imp.load_source("zhaojin_runs", os.path.join(REPO, "scripts", "zhaojin_runs.py"))
    over = dict((t, o) for t, _, o in ZR.RUNS + ZR.ARRIVAL_RUNS + ZR.RUNS_ORIGINAL + ZR.RUNS_SEAL20 + ZR.ARRIVAL_SEAL20)
    files = [(t, os.path.join(REPO, "outputs", "zhaojin_runs", "final_%s.npz" % t)) for t in ("A2", "B2", "C2", "A3", "B3", "C3", "D3", "E")]
    files += [(t + " (rate)", os.path.join(RR, "D", "zhaojin", "final_%s.npz" % t)) for t in ("D2", "E2", "E2_20")]
    for tag, p in files:
        cfg = Z.config(**dict(ZR.COMMON, **over[tag.split()[0]]))
        st, mx, my, phi, region, hc = Z.build_state(cfg, 1.0)
        f = np.load(p)
        assert np.allclose(f["x"], mx) and np.allclose(f["y"], my) and np.allclose(f["phi"], phi)
        assert np.array_equal(np.sort(f["hole"]), np.sort(hc))
        reach = reachable_domain(st, hc, p0=Z.P0, use_gravity=True)["reachable"]
        filled = np.asarray(f["s"], dtype=float) >= S_C
        filled[np.asarray(hc, dtype=int)] = True
        out.append(("ZJ " + tag, mx, my, filled, reach, np.asarray(phi, dtype=float)))
    return out


def part2():
    lines = ["", "PART 2  enclosed unfilled components in the non-boulder final states",
             "        enclosed = no cell on the domain border; split: all reachable / all unreachable / mixed", "",
             "%-40s | %-23s | %-23s | %-23s | %-23s" % ("case", "4, solids off", "8, solids off", "4, solids on", "8, solids on")]
    tot = {}
    for lab, x, y, filled, reach, phi in cases():
        row = []
        for solid_on in (False, True):
            for conn in (4, 8):
                r = enclosed_stats(x, y, filled, reach, phi, conn, solid_on)
                key = (conn, solid_on)
                t = tot.setdefault(key, dict((k, 0) for k in r))
                for k in r:
                    t[k] += r[k]
                row.append((conn, solid_on, "%2d (%2d/%2d/%2d) %4d c" % (r["enclosed"], r["all_reach"], r["all_unreach"],
                                                                         r["mixed"], r["cells"])))
        row = dict(((c, s), v) for c, s, v in row)
        lines.append("%-40s | %-23s | %-23s | %-23s | %-23s" % (lab, row[(4, False)], row[(8, False)], row[(4, True)], row[(8, True)]))
    lines.append("")
    for key in sorted(tot):
        t = tot[key]
        lines.append("total %d-adjacency, solids %-3s: enclosed %d (all reachable %d, all unreachable %d, mixed %d), %d cells"
                     % (key[0], "on" if key[1] else "off", t["enclosed"], t["all_reach"], t["all_unreach"], t["mixed"], t["cells"]))
    return lines


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    lines = boulder() + part2()
    text = "\n".join(lines)
    print(text)
    open(os.path.join(OUT, "connectivity_check.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
