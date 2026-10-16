# -*- coding: utf-8 -*-
"""
4.7 check, part 2: reach-only results of the case-study section (items 2.1-2.7,
4.2, 4.4 of the 4.7 pre-draft check). No solver.

    powershell -File scripts\\run_local.ps1 scripts\\check_4_7\\reach.py

  A  the 144 screening configurations recomputed at full precision on the
     0.5 m grid (contact width 1 m, the screening default) and compared with
     the stored screening table (6 significant digits); per configuration the
     required-source-pressure ratio to the outlet drift, to the stored-ore
     bottom (lowest 2 m of the stored ore) and to the top of the original
     22 m layer. From them: the contact values (2.2), the spread over fill
     porosity (2.7), the no-contact values, the stored-ore preference of C2
     (4.2) and the seal (4.4).
  B  ratio to the outlet drift against n_c (contacts only: fill base shut,
     35 deg, no seal, stored ore 0.35), widths 0.5 / 1 / 2 m on 0.5 m and
     width 2 m on 1 m; crossing of 1 by linear interpolation between samples
     and by bisection (14 steps, as C1); largest difference between zones at
     the same n_c; the 1D along-contact estimate.
  C  why 2D is below 1D: the optimal path to the outlet drift traced back
     (16-neighbour graph with a zero-cost super source, predecessors from
     Dijkstra) and counted by material; and the crossing recomputed with the
     stored ore made impassable (n 1e-3) -- a diagnostic, not a new case.
Elevations relative to the grouting level.

Output: outputs/check_4_7/reach.txt
"""
from __future__ import print_function

import csv
import imp
import itertools
import math
import os
import sys

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain, _stencil_edges

Z = imp.load_source("case_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
OUT = os.path.join(REPO, "outputs", "check_4_7")
SCREEN = os.path.join(REPO, "outputs", "zhaojin_screen")
ZONES = [("z1_cuts1-6", "I"), ("z2_cuts6-10", "II"), ("z3_cuts10-15", "III")]
ZN = dict(ZONES)
PHIS = (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.45)
DOWN_DIP = 42.7


class Section(object):
    def __init__(self, cell):
        self.cell = cell
        bundle = Z.build_mesh(cell)
        self.state, self.mx, self.my, _, _, _ = Z.build_state(Z.config(), cell, mesh_bundle=bundle)

    def cost(self, cfg, ore_closed=False):
        phi, region = Z.section_fields(self.mx, self.my, cfg)
        if ore_closed:
            phi = np.where(region == Z.ORE, Z.NO_GROUT_PHI, phi)
        Z.set_porosity(self.state, phi)
        src = Z.hole_cells(self.mx, self.my, cfg, cell=self.cell)
        c = np.asarray(reachable_domain(self.state, src, p0=Z.P0, use_gravity=True)["cost"], dtype=float) / Z.P0
        return c, region, src

    def ratios(self, cfg, ore_closed=False):
        c, region, src = self.cost(cfg, ore_closed)
        tg = Z.targets(self.mx, self.my, region, cfg)
        top = Z.Y_LEVEL + Z.ORIGINAL_DESIGN["layer"]
        up = (region == Z.FILL) & (self.my > top - 1.0) & (self.my <= top)
        return {"out": float(np.min(c[tg["outlet"]])), "ore": float(np.min(c[tg["ore_bottom"]])),
                "up": float(np.min(c[up])) if np.any(up) else float("inf")}

    def path(self, cfg):
        """Optimal path from the source to the cheapest outlet-drift cell, as a list of cells."""
        phi, region = Z.section_fields(self.mx, self.my, cfg)
        Z.set_porosity(self.state, phi)
        src = Z.hole_cells(self.mx, self.my, cfg, cell=self.cell)
        rows, cols, data = _stencil_edges(self.state, 16, True)
        n = self.mx.size
        rows = np.concatenate([rows, np.full(len(src), n)])
        cols = np.concatenate([cols, src])
        data = np.concatenate([data, np.zeros(len(src))])
        g = csr_matrix((data, (rows, cols)), shape=(n + 1, n + 1))
        d, pred = dijkstra(g, directed=True, indices=n, return_predecessors=True)
        out = np.where(Z.targets(self.mx, self.my, region, cfg)["outlet"])[0]
        j = int(out[np.argmin(d[out])])
        cells = [j]
        while pred[cells[-1]] != n and pred[cells[-1]] >= 0:
            cells.append(int(pred[cells[-1]]))
        return cells[::-1], region, float(d[j]) / Z.P0


def crossing_interp(phis, vals):
    for (p0, v0), (p1, v1) in zip(zip(phis[:-1], vals[:-1]), zip(phis[1:], vals[1:])):
        if v0 > 1.0 >= v1:
            return p0 + (v0 - 1.0) * (p1 - p0) / (v0 - v1)
    return float("nan")


def bisect(sec, cfg, ore_closed=False):
    lo, hi = 0.06, 0.60
    cfg = dict(cfg, contact_phi=hi)
    if sec.ratios(cfg, ore_closed)["out"] > 1.0:
        return float("inf")
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        cfg["contact_phi"] = mid
        if sec.ratios(cfg, ore_closed)["out"] > 1.0:
            lo = mid
        else:
            hi = mid
    return hi


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    L = []
    w = L.append
    s05 = Section(0.5)

    # ---- A: the 144 configurations -------------------------------------------
    sweep = [("zone", [k for k, _ in sorted(ZONES)]), ("contacts", [False, True]), ("base_connected", [True, False]),
             ("ore_phi", [0.25, 0.35, 0.45]), ("hole_angle", [35.0, 20.0]), ("seal", [False, True])]
    keys = [k for k, _ in sweep]
    rows = []
    for values in itertools.product(*[v for _, v in sweep]):
        cfg = Z.config(**dict(zip(keys, values)))
        r = s05.ratios(cfg)
        r.update(cfg)
        rows.append(r)
    stored = list(csv.DictReader(open(os.path.join(SCREEN, "screen_cell0.50.csv"))))
    same = len(stored) == len(rows) and all(
        "%.6g" % r["out"] == s["out"] and "%.6g" % r["ore"] == s["ore"] for r, s in zip(rows, stored))
    w("4.7 check, part 2: reach-only results (0.5 m unless stated; ratio = minimum p_req over the target cells / p0)")
    w("")
    w("A. 144 screening configurations = zone 3 x contacts 2 x fill base 2 x stored-ore n 3 x hole angle 2 x seal 2")
    w("   recomputed: %d configurations; equal to outputs/zhaojin_screen/screen_cell0.50.csv at its 6 digits: %s" % (len(rows), same))
    w("   contact width in the screening = %.1f m (case default); n_c %.2f; targets: outlet drift cells, stored-ore bottom (lowest 2 m)," % (
        Z.DEFAULT_CONFIG["contact_width"], Z.DEFAULT_CONFIG["contact_phi"]))
    w("   top of the original layer (fill cells in the 1 m row below z +22)")
    w("")
    sel = lambda **kw: [r for r in rows if all(r[k] == v for k, v in kw.items())]
    w("2.2 with contacts, 35 deg, no seal: ratio to the outlet drift")
    w("   %-4s %-5s | %s" % ("base", "ore n", "   ".join("%-14s" % ("zone " + ZN[k]) for k, _ in ZONES)))
    for base in (True, False):
        for ore in (0.25, 0.35, 0.45):
            vals = [sel(zone=k, contacts=True, base_connected=base, ore_phi=ore, hole_angle=35.0, seal=False)[0]["out"]
                    for k, _ in ZONES]
            w("   %-4s %-5.2f | %s   spread %.6f" % ("open" if base else "shut", ore, "   ".join("%-14.6f" % v for v in vals),
                                                     max(vals) - min(vals)))
    w("   20 deg, base open, ore 0.35: %s" % ", ".join("%s %.6f" % (ZN[k], sel(zone=k, contacts=True, base_connected=True,
                                                                          ore_phi=0.35, hole_angle=20.0, seal=False)[0]["out"])
                                                     for k, _ in ZONES))
    w("")
    w("2.7 without contacts, no seal: ratio to the outlet drift (all > 1) and to the stored-ore bottom")
    for k, _ in ZONES:
        o = [r["out"] for r in sel(zone=k, contacts=False, seal=False)]
        b = [(r["base_connected"], r["ore_phi"], r["hole_angle"], r["ore"]) for r in sel(zone=k, contacts=False, seal=False)]
        w("   zone %-3s outlet %.3f .. %.3f; ore bottom (base open, 35 deg) %s" % (
            ZN[k], min(o), max(o), ", ".join("n %.2f: %.6f" % (x[1], x[3]) for x in b if x[0] and x[2] == 35.0)))
    spreads = []
    for base in (True, False):
        for ore in (0.25, 0.35, 0.45):
            for ang in (35.0, 20.0):
                v = [sel(zone=k, contacts=True, base_connected=base, ore_phi=ore, hole_angle=ang, seal=False)[0]["out"]
                     for k, _ in ZONES]
                spreads.append(max(v) - min(v))
    w("   with contacts, no seal: spread of the outlet ratio over the three fill porosities, all 12 (base, ore, angle) sets:"
      " %.6f .. %.6f" % (min(spreads), max(spreads)))
    w("")
    w("4.2 C2: stored-ore preference = ratio to the ore bottom < 1 and below the ratio upward (to z +22)")
    pref = [r for r in rows if r["ore"] < 1.0 and r["ore"] < r["up"]]
    w("   %d of %d configurations:" % (len(pref), len(rows)))
    for r in pref:
        w("     zone %-3s contacts %-5s base %-4s ore n %.2f %2.0f deg seal %-5s | ore %.6f  up %.6f" % (
            ZN[r["zone"]], r["contacts"], "open" if r["base_connected"] else "shut", r["ore_phi"], r["hole_angle"], r["seal"],
            r["ore"], r["up"]))
    ups = sorted(set(round(r["up"], 6) for r in sel(zone="z2_cuts6-10", seal=False)))
    w("   zone II, no seal: upward ratio values %s" % ", ".join("%.6f" % u for u in ups))
    w("")
    w("4.4 seal on: ratios to the outlet drift and to the stored-ore bottom")
    for ang in (35.0, 20.0):
        for c in (False, True):
            ss = sel(seal=True, hole_angle=ang, contacts=c)
            w("   %2.0f deg, contacts %-5s: outlet %.3f .. %.3f, ore bottom %.3f .. %.3f (%d configurations)" % (
                ang, c, min(r["out"] for r in ss), max(r["out"] for r in ss), min(r["ore"] for r in ss),
                max(r["ore"] for r in ss), len(ss)))
    ss = sel(seal=True)
    w("   all 72 seal configurations: min over (outlet, ore bottom) %.3f .. %.3f" % (
        min(min(r["out"], r["ore"]) for r in ss), max(min(r["out"], r["ore"]) for r in ss)))
    w("")

    # ---- B: threshold curves ---------------------------------------------------
    w("B. Ratio to the outlet drift against n_c (contacts only: base shut, 35 deg, no seal, stored ore 0.35)")
    s10 = Section(1.0)
    curves = {}
    for sec, cw in ((s05, 0.5), (s05, 1.0), (s05, 2.0), (s10, 2.0)):
        for k, _ in ZONES:
            cfg = Z.config(zone=k, contacts=True, contact_width=cw, base_connected=False, seal=False, hole_angle=35.0,
                           ore_phi=0.35)
            vals = [sec.ratios(dict(cfg, contact_phi=p))["out"] for p in PHIS]
            curves[(sec.cell, cw, k)] = vals
    w("   %-9s %-4s | %s" % ("grid/w", "zone", " ".join("%9.2f" % p for p in PHIS)))
    for (cell, cw, k), vals in sorted(curves.items()):
        w("   %.1f / %.1f   %-4s | %s" % (cell, cw, ZN[k], " ".join("%9.4f" % min(v, 999.0) for v in vals)))
    w("")
    for cell, cw in ((0.5, 1.0), (0.5, 2.0), (1.0, 2.0)):
        diffs = [max(curves[(cell, cw, k)][i] for k, _ in ZONES) - min(curves[(cell, cw, k)][i] for k, _ in ZONES)
                 for i in range(len(PHIS))]
        w("   %.1f m grid, width %.1f m: crossing of 1 (linear interpolation) %s; largest zone difference at one n_c %.4f "
          "(at n_c %.2f), over n_c >= 0.20 %.4f" % (
              cell, cw, ", ".join("%s %.4f" % (ZN[k], crossing_interp(PHIS, curves[(cell, cw, k)])) for k, _ in ZONES),
              max(diffs), PHIS[int(np.argmax(diffs))], max(d for p, d in zip(PHIS, diffs) if p >= 0.20)))
    w("   width 1 vs 2 m on 0.5 m, |difference| per zone over all n_c: %s" % ", ".join(
        "%s %.4f" % (ZN[k], max(abs(a - b) for a, b in zip(curves[(0.5, 1.0, k)], curves[(0.5, 2.0, k)]))) for k, _ in ZONES))
    w("   width 1 vs 2 m on 0.5 m at n_c 0.35: %s" % ", ".join(
        "%s %.6f / %.6f" % (ZN[k], curves[(0.5, 1.0, k)][7], curves[(0.5, 2.0, k)][7]) for k, _ in ZONES))
    w("   0.5 m width 2 vs 1 m width 2 at n_c 0.35: %s" % ", ".join(
        "%s %.6f / %.6f" % (ZN[k], curves[(0.5, 2.0, k)][7], curves[(1.0, 2.0, k)][7]) for k, _ in ZONES))
    for cw in (1.0, 2.0):
        w("   bisection (0.06..0.60, 14 steps), 0.5 m, width %.1f m: %s" % (cw, ", ".join(
            "%s %.4f" % (ZN[k], bisect(s05, Z.config(zone=k, contacts=True, contact_width=cw, base_connected=False, seal=False,
                                                     hole_angle=35.0, ore_phi=0.35))) for k, _ in ZONES)))
    rho, g = 1830.0, 9.81
    lam_star = Z.P0 / DOWN_DIP + rho * g * math.sin(math.radians(Z.DIP_DEG))
    phi_1d = 1.0 / (1.0 + lam_star / 6.0e4)
    cx, cy = Z.collar()
    d = Z.hole_direction(Z.config())
    ey = cy + Z.ORIFICE_DEPTH * d[1]
    w("   1D along the contact: reach down-dip = p0 / (lambda(n_c) - rho g sin 70), i.e. eq. (4-25) with T = 0 and the start at the")
    w("   point where the hole enters the footwall contact (z %+.3f m); %.1f m down-dip = %.3f m vertical, to z %+.3f m (the drift floor)" % (
        ey - Z.Y_LEVEL, DOWN_DIP, DOWN_DIP * math.sin(math.radians(Z.DIP_DEG)),
        ey - Z.Y_LEVEL - DOWN_DIP * math.sin(math.radians(Z.DIP_DEG))))
    w("   critical n_c = 1 / (1 + lambda* / 6.00e4), lambda* = p0/%.1f + rho g sin 70 = %.1f Pa/m -> n_c %.6f" % (
        DOWN_DIP, lam_star, phi_1d))
    w("")

    # ---- C: why 2D is below 1D ----------------------------------------------------
    w("C. Optimal path to the outlet drift (0.5 m, 35 deg, no seal, stored ore 0.35), cells by material")
    for k, _ in ZONES:
        for cw, base, nc in ((1.0, False, 0.35), (1.0, True, 0.35), (2.0, False, 0.35), (1.0, False, 0.25)):
            cfg = Z.config(zone=k, contacts=True, contact_width=cw, base_connected=base, seal=False, hole_angle=35.0,
                           ore_phi=0.35, contact_phi=nc)
            cells, region, ratio = s05.path(cfg)
            cnt = dict((Z.REGION_NAMES[c], int(np.sum(region[cells] == c))) for c in Z.REGION_NAMES)
            ore = [i for i in cells if region[i] == Z.ORE]
            w("   zone %-3s width %.1f base %-4s n_c %.2f: ratio %.6f; %d cells: %s%s" % (
                ZN[k], cw, "open" if base else "shut", nc, ratio, len(cells),
                ", ".join("%s %d" % (n_, v) for n_, v in sorted(cnt.items()) if v),
                ("; stored-ore cells at z %+.2f .. %+.2f" % (s05.my[ore].min() - Z.Y_LEVEL, s05.my[ore].max() - Z.Y_LEVEL)) if ore else ""))
    w("   crossing with the stored ore impassable (diagnostic), 0.5 m, width 1.0 m, base shut: %s" % ", ".join(
        "%s %.4f" % (ZN[k], bisect(s05, Z.config(zone=k, contacts=True, contact_width=1.0, base_connected=False, seal=False,
                                                 hole_angle=35.0, ore_phi=0.35), ore_closed=True)) for k, _ in ZONES))
    text = "\n".join(L)
    print(text)
    open(os.path.join(OUT, "reach.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
