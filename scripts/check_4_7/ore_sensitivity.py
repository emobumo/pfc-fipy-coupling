# -*- coding: utf-8 -*-
"""Critical contact porosity against the stored-ore assumptions (reach only, no solver).

    powershell -File scripts\\run_local.ps1 scripts\\check_4_7\\ore_sensitivity.py

Writes outputs/check_4_7/ore_sensitivity.txt.

All runs: 0.5 m grid, contact width 1 m, hole 35 deg, no seal. Critical n_c = the
contact porosity at which the required source pressure to the outlet drift equals p0,
found by bisection as in C1 (14 steps, upper end returned) but on 0.02..0.60.

Step 1: v0.6 parameters; fill base shut / connected x stored ore 0.25 / 0.35 / 0.45
        x zones I, II, III (18 values). The impassable-ore check is R28 (not recomputed).
Step 2: the configuration with the lowest step-1 value, under the 10 parameter groups
        of table 4.5-2 (the rows of reach_sensitivity_1.25.txt that do not touch the
        domain box, as C1), all three zones.
"""
from __future__ import print_function

import imp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

R = imp.load_source("check_reach", os.path.join(REPO, "scripts", "check_4_7", "reach.py"))
C = imp.load_source("case_c1c2", os.path.join(REPO, "scripts", "zhaojin_c1c2.py"))
Z = R.Z
LO, HI, STEPS = 0.02, 0.60, 14
ORES = (0.25, 0.35, 0.45)
BASES = ((False, "shut"), (True, "connected"))


def bisect(sec, cfg):
    """Critical n_c; inf when not reachable at HI, LO when reachable at LO."""
    cfg = dict(cfg, contact_phi=HI)
    if sec.ratios(cfg)["out"] > 1.0:
        return float("inf")
    cfg["contact_phi"] = LO
    if sec.ratios(cfg)["out"] <= 1.0:
        return LO
    lo, hi = LO, HI
    for _ in range(STEPS):
        mid = 0.5 * (lo + hi)
        cfg["contact_phi"] = mid
        if sec.ratios(cfg)["out"] > 1.0:
            lo = mid
        else:
            hi = mid
    return hi


def fmt(v):
    if v == float("inf"):
        return "> %.2f" % HI
    if v == LO:
        return "<= %.2f" % LO
    return "%.6f" % v


def cfg_of(zone, base_connected, ore):
    return Z.config(zone=zone, contacts=True, contact_width=1.0, base_connected=base_connected,
                    ore_phi=ore, hole_angle=35.0, seal=False)


def main():
    sec = R.Section(0.5)
    base = dict(sec.state["slurry_parameters"])
    L = ["Critical contact porosity n_c (required source pressure to the outlet drift = p0); 0.5 m grid,",
         "contact width 1 m, 35 deg, no seal; bisection on %.2f..%.2f, %d steps, upper end (as C1)" % (LO, HI, STEPS), ""]

    L.append("Step 1: v0.6 parameters (tau0 30 Pa, A x 1, rho 1830)")
    L.append("   base       ore   |  %-10s %-10s %-10s | min" % ("I", "II", "III"))
    step1 = {}
    for bc, bname in BASES:
        for ore in ORES:
            row = [bisect(sec, cfg_of(k, bc, ore)) for k, _ in R.ZONES]
            step1[(bc, ore)] = row
            L.append("   %-10s %.2f  |  %-10s %-10s %-10s | %s" % ((bname, ore) + tuple(fmt(v) for v in row) + (fmt(min(row)),)))
    allv = [v for r in step1.values() for v in r]
    L.append("   all 18: %s .. %s" % (fmt(min(allv)), fmt(max(allv))))
    for j, (k, zn) in enumerate(R.ZONES):
        v = [step1[(bc, o)][j] for bc, _ in BASES for o in ORES]
        L.append("   zone %-3s: %s .. %s" % (zn, fmt(min(v)), fmt(max(v))))
    L.append("   stored ore impassable (n 1e-3): R28, 0.3050 / 0.2983 / 0.3050 (bisection 0.06..0.60, base shut)")
    key = min(step1, key=lambda q: (min(step1[q]), sum(step1[q])))
    bname = dict(BASES)[key[0]]
    L.append("   lowest configuration: base %s, stored ore %.2f (min over zones %s)" % (bname, key[1], fmt(min(step1[key]))))
    L.append("")

    L.append("Step 2: base %s, stored ore %.2f, the 10 parameter groups of table 4.5-2" % (bname, key[1]))
    L.append("   %6s %5s %4s %5s |  %-10s %-10s %-10s" % ("s", "tau0", "A x", "rho", "I", "II", "III"))
    res = []
    for s_, tau0, af, rho in C.groups():
        C.set_params(sec.state, tau0, af, rho, base)
        row = [bisect(sec, cfg_of(k, key[0], key[1])) for k, _ in R.ZONES]
        res.append(((s_, tau0, af, rho), row))
        L.append("   %6.3f %5.0f %4.1f %5.0f |  %-10s %-10s %-10s" % ((s_, tau0, af, rho) + tuple(fmt(v) for v in row)))
    C.set_params(sec.state, base["yield_stress"], 1.0, base["slurry_density"], base)
    fin = [(v, g, R.ZONES[j][1]) for g, row in res for j, v in enumerate(row) if v != float("inf")]
    vmin, gmin, zmin = min(fin, key=lambda t: t[0])
    L.append("   minimum over all groups and zones: %s (tau0 %.0f, A x %.1f, rho %.0f, zone %s)"
             % (fmt(vmin), gmin[1], gmin[2], gmin[3], zmin))
    for j, (k, zn) in enumerate(R.ZONES):
        v = [row[j] for _, row in res if row[j] != float("inf")]
        L.append("   zone %-3s: %s .. %s over %d reachable groups" % (zn, fmt(min(v)), fmt(max(v)), len(v)))
    nr = ["tau0 %.0f A x %.1f rho %.0f zone %s" % (g[1], g[2], g[3], R.ZONES[j][1])
          for g, row in res for j, v in enumerate(row) if v == float("inf")]
    L.append("   not reachable at n_c %.2f: %s" % (HI, "; ".join(nr) if nr else "none"))
    L.append("")
    below = [("step 1", dict(BASES)[q[0]], q[1], R.ZONES[j][1], v) for q, r in sorted(step1.items()) for j, v in enumerate(r) if v < 0.10]
    below += [("step 2", "tau0 %.0f A x %.1f rho %.0f" % g[1:], "", R.ZONES[j][1], v)
              for g, row in res for j, v in enumerate(row) if v < 0.10]
    L.append("Below 0.10: %s" % ("; ".join("%s %s %s zone %s %s" % (a, b, c, d, fmt(e)) for a, b, c, d, e in below) if below else "none"))

    if not os.path.isdir(R.OUT):
        os.makedirs(R.OUT)
    text = "\n".join(L) + "\n"
    open(os.path.join(R.OUT, "ore_sensitivity.txt"), "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
