# -*- coding: utf-8 -*-
"""
The case-study section against the design that was in force when the runaway
happened (the ORIGINAL design), not the changed design written afterwards.

    powershell -File scripts\run_local.ps1 scripts\zhaojin_original_design.py

Runs A-D were stopped at the CHANGED design's grout per metre (9.75 / 50.4 /
19.5 m3/m, layers 12 / 16 / 12 m, from the post-seal porosity estimates), but
the runaway occurred under the original design: one 22 m reinforcement layer
above the grouting level, 20% porosity assumed throughout, 86 m x 6.5 m x
22 m x 0.2 = 2460 m3 of grout, i.e. 28.6 m3 per metre of strike, the same in
every zone. The ground itself stays at the zone estimates (10/45/25%): only
the design intent -- stop volume and design layer -- changes.

T1 (this script, no solver): read the A2-E2 series, recorded every
RECORD_EVERY = 50 steps (row k = state after step 50k+1), at 28.6 m3/m:
  - `below` (under the grouting level) does not depend on any layer, so it is
    interpolated directly; `design + above` (everything above the level) is
    reported only as a sum, since the split used the changed-design layers;
  - arrival multiples: V_in at reaching -56 / -63 m divided by 28.6;
  - sensitivity: the zone widths give 7.5 / 7.0 / 6.5 m x 22 x 0.2 =
    33.0 / 30.8 / 28.6 m3/m.
T2 cross-check (once final_A3..D3 exist): A3-D3 march exactly as A2-D2 until
they stop, so every series row they share must be bit-identical.
"""
from __future__ import print_function

import imp
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = imp.load_source("zhaojin_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
OUT = os.path.join(REPO, "outputs", "zhaojin_runs")
Q = Z.ORIGINAL_DESIGN["grout_per_m"]
RECORD_EVERY = 50


def by_width(zone):
    return Z.ZONES[zone]["width"] * Z.ORIGINAL_DESIGN["layer"] * Z.ORIGINAL_DESIGN["phi"]


def at_volume(series, v):
    """Linear interpolation of the series at V_in = v, plus the bracketing rows."""
    vin = series[:, 1]
    i = int(np.searchsorted(vin, v))
    if i == 0 or i >= len(vin):
        return None
    a, b = series[i - 1], series[i]
    w = (v - a[1]) / (b[1] - a[1])
    row = a + w * (b - a)
    return {"below": float(row[4]), "above_level": float(row[2] + row[3]),
            "steps": ((i - 1) * RECORD_EVERY + 1, i * RECORD_EVERY + 1),
            "v_bracket": (float(a[1]), float(b[1])),
            "below_bracket": (float(a[4]), float(b[4]))}


def main():
    rows = dict((json.loads(l)["tag"], json.loads(l)) for l in open(os.path.join(OUT, "results.jsonl")))
    lines = ["T1: original design %.1f m3/m (layer %.0f m, phi %.2f uniform); series rows every %d steps"
             % (Q, Z.ORIGINAL_DESIGN["layer"], Z.ORIGINAL_DESIGN["phi"], RECORD_EVERY), ""]
    for tag in ("A2", "B2", "C2", "D2", "E2"):
        if tag not in rows:
            continue
        r = rows[tag]
        ser = np.load(os.path.join(OUT, "final_%s.npz" % tag))["series"]
        for v, name in ((Q, "28.6 (uniform)"), (by_width(r["zone"]), "zone width x 22 x 0.2")):
            p = at_volume(ser, v)
            if p is None:
                lines.append("%s  %-24s V=%.1f outside the record" % (tag, name, v))
                continue
            lines.append("%s  %-24s V=%5.1f: below %6.2f (%4.1f%% of injected), above level %6.2f | "
                         "rows at steps %d/%d, V %.2f/%.2f, below %.2f/%.2f"
                         % (tag, name, v, p["below"], 100 * p["below"] / v, p["above_level"],
                            p["steps"][0], p["steps"][1], p["v_bracket"][0], p["v_bracket"][1],
                            p["below_bracket"][0], p["below_bracket"][1]))
        mult = []
        for key, lab in (("reach_-56m", "-56 m"), ("reach_-63m", "-63 m")):
            if key in r["snaps"]:
                vv = r["snaps"][key]["v_in"]
                mult.append("%s at V %.1f = %.2fx of 28.6 (%.2fx of zone-width %.1f)"
                            % (lab, vv, vv / Q, vv / by_width(r["zone"]), by_width(r["zone"])))
            else:
                mult.append("%s never" % lab)
        lines.append("%s  arrival: %s" % (tag, "; ".join(mult)))
        lines.append("")

    # T2 cross-check: shared series rows must be bit-identical.
    pairs = [("A3", "A2"), ("B3", "B2"), ("C3", "C2"), ("D3", "D2")]
    have = [(a, b) for a, b in pairs if os.path.exists(os.path.join(OUT, "final_%s.npz" % a))]
    if have:
        lines.append("T2 cross-check (shared series rows, bit for bit):")
        for a, b in have:
            sa = np.load(os.path.join(OUT, "final_%s.npz" % a))["series"]
            sb = np.load(os.path.join(OUT, "final_%s.npz" % b))["series"]
            n = min(len(sa), len(sb))
            # t, V_in and `below` are layer independent; design/above use
            # different layers (22 m vs the changed design) and must differ
            cols = [0, 1, 4]
            same = bool(np.array_equal(sa[:n][:, cols], sb[:n][:, cols]))
            # the same cells summed in a different split: equal up to round-off
            above = float(np.max(np.abs((sa[:n][:, 2] + sa[:n][:, 3]) - (sb[:n][:, 2] + sb[:n][:, 3]))))
            lines.append("  %s vs %s: %d shared rows, t/V_in/below identical: %s; design+above max |diff| %.1e"
                         % (a, b, n, same, above))
    text = "\n".join(lines)
    open(os.path.join(OUT, "original_design.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
