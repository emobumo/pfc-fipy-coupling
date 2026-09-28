# -*- coding: utf-8 -*-
"""
Case-study section, solver-free supplements C1 and C2.

    powershell -File scripts\run_local.ps1 scripts\zhaojin_c1c2.py

C1  the contact's minimum openness phi_c (cost ratio to the -63 m drift = 1)
    for each zone, under the 10 parameter groups of reach_sensitivity_1.25.txt
    whose uniform and gradient reach both stay inside the box. Same set-up as
    the original threshold table: 0.5 m cells, contacts 1 m wide, fill base
    shut, 35 deg, no seal, stored ore 0.35. Bisection on phi_c. Alongside,
    the 1D along-contact estimate: the reach down the 70 deg contact is
    p0 / (lambda - rho g sin 70), and the drift is 42.7 m down-dip from the hole.
C2  the 144 screening configurations (zone x contacts x base x ore phi x
    angle x seal) re-priced to add the UPWARD cost (to the top of the
    original 22 m design layer) next to the cost to the stored-ore bottom;
    grout "prefers the stored ore" where the ore cost is < 1 and below the
    upward cost. Sideways the vein is walled by host rock (phi 1e-3, not
    passable); the only lateral path is a contact, which leads down.
Outputs: outputs/zhaojin_c1c2/.
"""
from __future__ import print_function

import imp
import math
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain

Z = imp.load_source("zhaojin_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
OUT = os.path.join(REPO, "outputs", "zhaojin_c1c2")
CELL = 0.5
DOWN_DIP = 42.7          # m, hole to the -63 m drift along the contact (original table)


def groups():
    rows = []
    p = os.path.join(REPO, "outputs", "reach_sensitivity", "reach_sensitivity_1.25.txt")
    for line in open(p):
        m = re.match(r"\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)(\*?)\s+([\d.]+)(\*?)", line)
        if m and not m.group(6) and not m.group(8):
            rows.append((float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))
    return rows


def set_params(state, tau0, a_factor, rho, base):
    sp = state["slurry_parameters"]
    sp["yield_stress"] = tau0
    sp["calibrated_permeability_coefficient"] = base["calibrated_permeability_coefficient"] * a_factor
    sp["slurry_density"] = rho


def out_ratio(state, mx, my, cfg):
    phi, region = Z.section_fields(mx, my, cfg)
    Z.set_porosity(state, phi)
    src = Z.hole_cells(mx, my, cfg, cell=CELL)
    cost = np.asarray(reachable_domain(state, src, p0=Z.P0, use_gravity=True)["cost"], dtype=float)
    tg = Z.targets(mx, my, region, cfg)
    return float(np.min(cost[tg["outlet"]])) / Z.P0, cost, region


def c1(state, mx, my, base):
    lines = ["C1: contact threshold phi_c (cost ratio to the -63 m drift = 1); 0.5 m, contacts 1 m, base shut, 35 deg",
             "", "%6s %5s %4s %5s | %-9s %-9s %-9s | %s" % ("s", "tau0", "A x", "rho", "z1 (10%)", "z2 (45%)",
                                                            "z3 (25%)", "1D along-contact")]
    res = []
    for s_, tau0, af, rho in groups():
        set_params(state, tau0, af, rho, base)
        row = []
        for zone in ("z1_cuts1-6", "z2_cuts6-10", "z3_cuts10-15"):
            cfg = Z.config(zone=zone, contacts=True, contact_width=1.0, base_connected=False, seal=False,
                           hole_angle=35.0, ore_phi=0.35)
            lo, hi = 0.06, 0.60
            cfg["contact_phi"] = hi
            if out_ratio(state, mx, my, cfg)[0] > 1.0:
                row.append(float("inf"))
                continue
            for _ in range(14):
                mid = 0.5 * (lo + hi)
                cfg["contact_phi"] = mid
                if out_ratio(state, mx, my, cfg)[0] > 1.0:
                    lo = mid
                else:
                    hi = mid
            row.append(hi)
        lam_star = Z.P0 / DOWN_DIP + rho * 9.81 * math.sin(math.radians(Z.DIP_DEG))
        c = 2.0 * tau0 / math.sqrt(8.0 * base["calibrated_permeability_coefficient"] * af)
        phi_1d = 1.0 / (1.0 + lam_star / c)
        res.append((s_, tau0, af, rho, row, phi_1d))
        vals = tuple(["%.3f" % v if v < 1 else ">0.60" for v in row])
        diff = ", ".join("%+.3f" % (v - phi_1d) if v < 1 else "-" for v in row)
        lines.append("%6.3f %5.0f %4.1f %5.0f | %-9s %-9s %-9s | %.3f (2D - 1D: %s)"
                     % ((s_, tau0, af, rho) + vals + (phi_1d, diff)))
    set_params(state, base["yield_stress"], 1.0, base["slurry_density"], base)
    allv = [v for r in res for v in r[4] if v < 1]
    lines.append("")
    for k, zone in enumerate(("z1 (10%)", "z2 (45%)", "z3 (25%)")):
        v = [r[4][k] for r in res if r[4][k] < 1]
        lines.append("  %s: phi_c %.3f .. %.3f" % (zone, min(v), max(v)))
    d = [r[4][k] - r[5] for r in res for k in range(3) if r[4][k] < 1]
    lines.append("  2D minus 1D: %.3f .. %.3f (all groups, all zones)" % (min(d), max(d)))
    return lines


def c2(state, mx, my):
    lines = ["C2: the 144 screening configurations re-priced (0.5 m; default parameters); ore = cost to the stored-ore",
             "bottom / p0, up = cost to the top of the original 22 m layer / p0; 'prefers ore' = ore < 1 and ore < up", ""]
    rows = []
    for zone in ("z1_cuts1-6", "z2_cuts6-10", "z3_cuts10-15"):
        for contacts in (False, True):
            for base_c in (True, False):
                for ore_phi in (0.25, 0.35, 0.45):
                    for ang in (35.0, 20.0):
                        for seal in (False, True):
                            cfg = Z.config(zone=zone, contacts=contacts, base_connected=base_c, ore_phi=ore_phi,
                                           hole_angle=ang, seal=seal)
                            r_out, cost, region = out_ratio(state, mx, my, cfg)
                            tg = Z.targets(mx, my, region, cfg)
                            top = Z.Y_LEVEL + Z.ORIGINAL_DESIGN["layer"]
                            up = (region == Z.FILL) & (my > top - 1.0) & (my <= top)
                            ore = float(np.min(cost[tg["ore_bottom"]])) / Z.P0
                            upc = float(np.min(cost[up])) / Z.P0 if np.any(up) else float("inf")
                            rows.append((zone, contacts, base_c, ore_phi, ang, seal, ore, upc, r_out))
    for zone in ("z1_cuts1-6", "z2_cuts6-10", "z3_cuts10-15"):
        lines.append("%s  (35 deg, no seal)" % zone)
        lines.append("  %-9s %-6s | %-24s %-24s %-24s" % ("contacts", "base", "ore 0.25 (ore/up)", "ore 0.35", "ore 0.45"))
        for contacts in (False, True):
            for base_c in (True, False):
                cells = []
                for ore_phi in (0.25, 0.35, 0.45):
                    r = [x for x in rows if x[:6] == (zone, contacts, base_c, ore_phi, 35.0, False)][0]
                    pref = r[6] < 1.0 and r[6] < r[7]
                    cells.append("%6.2f / %5.2f %s" % (min(r[6], 99), min(r[7], 99), "PREF" if pref else "    "))
                lines.append("  %-9s %-6s | %-24s %-24s %-24s" % ((contacts, "open" if base_c else "shut") + tuple(cells)))
        lines.append("")
    pref = [x for x in rows if x[6] < 1.0 and x[6] < x[7]]
    lines.append("configurations where grout prefers the stored ore: %d of %d" % (len(pref), len(rows)))
    lines.append("  all of them have: seal = %s; base open = %s; ore phi in %s; contacts in %s" % (
        sorted(set(x[5] for x in pref)), sorted(set(x[2] for x in pref)), sorted(set(x[3] for x in pref)),
        sorted(set(x[1] for x in pref))))
    return lines


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    mesh = Z.build_mesh(CELL)
    state, mx, my, phi, region, hc = Z.build_state(Z.config(), CELL, mesh_bundle=mesh)
    base = dict(state["slurry_parameters"])
    text = "\n".join(c1(state, mx, my, base) + [""] + c2(state, mx, my))
    open(os.path.join(OUT, "c1c2.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
