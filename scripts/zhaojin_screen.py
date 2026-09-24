# -*- coding: utf-8 -*-
"""
Step 0 for the Zhaojin section: which configurations can put grout into the
-63 m level at 5 MPa -- without running the solver.

    powershell -File scripts\run_local.ps1 scripts\zhaojin_screen.py [--cell 0.5]

The reachable domain (fill_diagnostics.reachable_domain, 16-neighbour stencil,
tunnel-proof) answers "can grout get there under the stall condition" as a
path integral, in seconds. That is exactly the question the site record
poses -- runaway appeared at -63 m under cuts 5#, 7#, 9#, 15#, one or more in
every porosity zone -- and it lets every assumption the document leaves open
be swept before a single hour of solver time is spent.

Swept (see cases/zhaojin_section.py for the geometry and defaults):
    zone             10% / 45% / 25% fill (cuts 1-6 / 6-10 / 10-15)
    contacts         through-going gaps along both walls, or none
    base_connected   fill base open to the stored ore below, or a tight layer
    ore_phi          stored broken ore, 0.25 / 0.35 / 0.45
    hole_angle       35 (when the runaway happened) / 20 (after the change)
    seal             the 2 m double-liquid bottom seal, off / on

Reported per configuration:
    out   min path cost to the -63 m drift / p0       (<= 1: grout gets there)
    ore   min path cost to the bottom of the stored ore / p0
          (-56 m draw points; if ore passes connect them to -63 m, this is
          runaway too -- the document does not say)
    dsg%  share of the changed-design reinforcement layer that is reachable

A 2D section cannot see along-strike connectivity, which the document itself
flags ("充填体长度方面可能存在未封闭可能"); a configuration that cannot reach
-63 m in-plane is therefore "not by this route", not "never".
"""
from __future__ import print_function

import imp
import itertools
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain

Z = imp.load_source("zhaojin_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
OUT = os.path.join(REPO, "outputs", "zhaojin_screen")

SWEEP = [
    ("zone", sorted(Z.ZONES)),
    ("contacts", [False, True]),
    ("base_connected", [True, False]),
    ("ore_phi", [0.25, 0.35, 0.45]),
    ("hole_angle", [35.0, 20.0]),
    ("seal", [False, True]),
]


def screen_one(state, mx, my, cfg, cell):
    phi, region = Z.section_fields(mx, my, cfg)
    Z.set_porosity(state, phi)
    src = Z.hole_cells(mx, my, cfg, cell=cell)
    r = reachable_domain(state, src, p0=Z.P0, use_gravity=True)
    cost = np.asarray(r["cost"], dtype=float)
    tg = Z.targets(mx, my, region, cfg)

    def min_ratio(mask):
        return float(np.min(cost[mask])) / Z.P0 if np.any(mask) else float("inf")

    design = tg["design"]
    return {
        "out": min_ratio(tg["outlet"]),
        "ore": min_ratio(tg["ore_bottom"]),
        "dsg": float(np.mean(r["reachable"][design])) if np.any(design) else 0.0,
        "solver": r["solver"],
    }


def fmt_ratio(v):
    if v == float("inf") or v > 99:
        return "  >99"
    return "%5.2f" % v


def main(argv):
    cell = 0.5
    i = 0
    while i < len(argv):
        if argv[i] == "--cell":
            cell = float(argv[i + 1]); i += 2
        else:
            i += 1
    if not os.path.isdir(OUT):
        os.makedirs(OUT)

    t0 = time.time()
    bundle = Z.build_mesh(cell)
    state, mx, my, phi, region, hc = Z.build_state(Z.config(), cell, mesh_bundle=bundle)
    print("mesh: %d cells at %.2f m (x %.0f..%.0f, y %.0f..%.0f)"
          % (mx.size, cell, Z.X_MIN, Z.X_MAX, Z.Y_MIN, Z.Y_MAX))
    sys.stdout.flush()

    keys = [k for k, _ in SWEEP]
    rows = []
    for values in itertools.product(*[v for _, v in SWEEP]):
        cfg = Z.config(**dict(zip(keys, values)))
        res = screen_one(state, mx, my, cfg, cell)
        res.update(cfg)
        rows.append(res)

    # --- table, grouped the way the questions are asked -------------------
    hdr = "%-13s %-4s %-4s %-5s %-4s %-4s | %5s %5s %5s" % (
        "zone", "cont", "base", "ore", "ang", "seal", "out", "ore", "dsg%")
    lines = [hdr, "-" * len(hdr)]
    for r in rows:
        lines.append("%-13s %-4s %-4s %-5.2f %-4.0f %-4s | %s %s %5.0f%s" % (
            r["zone"], "yes" if r["contacts"] else "no",
            "open" if r["base_connected"] else "shut", r["ore_phi"], r["hole_angle"],
            "yes" if r["seal"] else "no", fmt_ratio(r["out"]), fmt_ratio(r["ore"]),
            100.0 * r["dsg"], "  <- -63 m" if r["out"] <= 1.0 else ""))
    text = "\n".join(lines)
    print(text)

    csv = os.path.join(OUT, "screen_cell%.2f.csv" % cell)
    handle = open(csv, "w")
    try:
        handle.write(",".join(keys + ["out", "ore", "dsg"]) + "\n")
        for r in rows:
            handle.write(",".join([str(r[k]) for k in keys] +
                                  ["%.6g" % r["out"], "%.6g" % r["ore"], "%.6g" % r["dsg"]]) + "\n")
    finally:
        handle.close()
    handle = open(os.path.join(OUT, "screen_cell%.2f.txt" % cell), "w")
    try:
        handle.write(text + "\n")
    finally:
        handle.close()
    print("\n%d configurations in %.0f s -> %s" % (len(rows), time.time() - t0, csv))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
