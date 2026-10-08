# -*- coding: utf-8 -*-
"""
Second pass of the solver-free (reach-only) results quoted in Ch.4, into
outputs/rerun_reach/, then a byte / value comparison against the originals.

    powershell -File scripts\run_local.ps1 scripts\rerun_reach.py [run|compare]

The scripts are imported unchanged and only their OUT directory is redirected:
gravity_control (reach part, no --solver), reach_sensitivity --cell 1.25,
zhaojin_screen --cell 0.5 / 1.0, channel_entry_cost --cell 1.25 / 2.5,
zhaojin_c1c2. The contact-threshold table (outputs/zhaojin_screen/
contact_threshold.txt) was written by a scratch script that no longer exists;
it is rebuilt here from cases/zhaojin_section.py with the setting printed in
its header (contacts only, fill base shut, 35 deg, no seal) and compared at
the printed precision.
Outputs: outputs/rerun_reach/ (+ compare.txt).
"""
from __future__ import print_function

import filecmp
import imp
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain

ROOT = os.path.join(REPO, "outputs", "rerun_reach")
OLD = os.path.join(REPO, "outputs")
WIDTHS = (0.5, 1.0, 2.0)
PHIS = (0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30, 0.35, 0.45)


def _load(name):
    return imp.load_source(name, os.path.join(REPO, "scripts", name + ".py"))


def contact_threshold(out):
    Z = imp.load_source("zhaojin_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
    cell = 0.5
    bundle = Z.build_mesh(cell)
    state, mx, my, _, _, _ = Z.build_state(Z.config(), cell, mesh_bundle=bundle)
    lines = ["contacts only (fill base shut), 35 deg, no seal. out = min cost to -63 m / p0",
             "zone          width " + "".join("%7.2f" % p for p in PHIS)]
    for zone in ("z1_cuts1-6", "z2_cuts6-10", "z3_cuts10-15"):
        for w in WIDTHS:
            vals = []
            for ph in PHIS:
                cfg = Z.config(zone=zone, contacts=True, contact_width=w, contact_phi=ph,
                               base_connected=False, seal=False, hole_angle=35.0)
                phi, region = Z.section_fields(mx, my, cfg)
                Z.set_porosity(state, phi)
                src = Z.hole_cells(mx, my, cfg, cell=cell)
                cost = np.asarray(reachable_domain(state, src, p0=Z.P0, use_gravity=True)["cost"], dtype=float)
                r = float(np.min(cost[Z.targets(mx, my, region, cfg)["outlet"]])) / Z.P0
                vals.append(">99" if r > 99 else ("%.2f%s" % (r, "*" if r <= 1.0 else "")))
            lines.append("%-13s %-5s " % (zone, w) + " ".join("%6s" % v for v in vals))
    open(os.path.join(out, "contact_threshold.txt"), "w").write("\n".join(lines) + "\n")


def run():
    jobs = [("gravity_control", "gravity_control", []),
            ("reach_sensitivity", "reach_sensitivity", ["--cell", "1.25"]),
            ("zhaojin_screen", "zhaojin_screen", ["--cell", "0.5"]),
            ("zhaojin_screen", "zhaojin_screen", ["--cell", "1.0"]),
            ("channel_entry_cost", "channel_entry_cost", ["--cell", "1.25"]),
            ("channel_entry_cost", "channel_entry_cost", ["--cell", "2.5"])]
    for name, sub, argv in jobs:
        m = _load(name)
        m.OUT = os.path.join(ROOT, sub)
        print("=== %s %s" % (name, " ".join(argv)))
        sys.stdout.flush()
        m.main(argv)
    c = _load("zhaojin_c1c2")
    c.OUT = os.path.join(ROOT, "zhaojin_c1c2")
    c.main()
    out = os.path.join(ROOT, "zhaojin_screen")
    contact_threshold(out)
    return 0


def _tokens(path, stop=None):
    rows = []
    for line in open(path):
        if stop and line.startswith(stop):
            break
        if line.strip():                     # blank lines are layout, not data
            rows.append([t.rstrip("*") for t in line.split()])   # '*' marks were not applied consistently
    return rows


def compare():
    res = []

    def same_file(rel):
        a, b = os.path.join(OLD, rel), os.path.join(ROOT, rel)
        res.append((rel, "byte-identical" if filecmp.cmp(a, b, shallow=False) else "DIFFERS"))

    for rel in ("reach_sensitivity/reach_sensitivity_1.25.txt",
                "zhaojin_screen/screen_cell0.50.csv", "zhaojin_screen/screen_cell0.50.txt",
                "zhaojin_screen/screen_cell1.00.csv", "zhaojin_screen/screen_cell1.00.txt",
                "channel_entry_cost/entry_cost_1.25.txt", "channel_entry_cost/entry_cost_2.50.txt",
                "zhaojin_c1c2/c1c2.txt"):
        same_file(rel)
    # gravity_control: the original also carries the --solver section; compare the reach part
    a = json.load(open(os.path.join(OLD, "gravity_control", "gravity_control.json")))["reach"]
    b = json.load(open(os.path.join(ROOT, "gravity_control", "gravity_control.json")))["reach"]
    ta = _tokens(os.path.join(OLD, "gravity_control", "gravity_control.txt"), stop="solver")
    tb = _tokens(os.path.join(ROOT, "gravity_control", "gravity_control.txt"), stop="solver")
    res.append(("gravity_control/gravity_control.json [reach] + .txt (reach part)",
                "identical" if (a == b and ta == tb) else "DIFFERS"))
    # contact threshold: values at the printed precision (the original table had no generator kept)
    ta = [r for r in _tokens(os.path.join(OLD, "zhaojin_screen", "contact_threshold.txt"))
          if r and r[0].startswith("z") and r[0] != "zone" and len(r) == 2 + len(PHIS)]
    tb = [r for r in _tokens(os.path.join(ROOT, "zhaojin_screen", "contact_threshold.txt"))
          if r and r[0].startswith("z") and r[0] != "zone" and len(r) == 2 + len(PHIS)]
    res.append(("zhaojin_screen/contact_threshold.txt (9 rows x 9 values, printed precision)",
                "identical (%d rows)" % len(ta) if ta == tb and len(ta) == 9 else "DIFFERS (%d vs %d rows)" % (len(ta), len(tb))))
    text = "\n".join("%-80s %s" % r for r in res)
    open(os.path.join(ROOT, "compare.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    a = sys.argv[1:] or ["run", "compare"]
    rc = 0
    if "run" in a:
        rc = run()
    if "compare" in a:
        rc = compare()
    sys.exit(rc)
