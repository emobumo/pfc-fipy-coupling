# -*- coding: utf-8 -*-
"""
Overshoot cells and filled / reach of every layer A/B/C run under the rate
rule, read from the results.json files (post-processing only; no solver run).

    powershell -File scripts\run_local.ps1 scripts\overshoot_fill_reach.py [out.txt]

Writes outputs/rerun_rate/overshoot_fill_reach.txt (or the path given). The
path cost of the overshoot cells is in scripts/overshoot_cost.py.

Reconstructed 2026-10-10 from the one-off session computation of 2026-10-08
that wrote the original file; same arithmetic, paths made repo-relative.
"""
from __future__ import print_function

import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(REPO, "outputs", "rerun_rate")


def main(argv):
    out = argv[1] if len(argv) > 1 else os.path.join(R, "overshoot_fill_reach.txt")
    lines = ["overshoot cells and filled/reach under the rate rule (read from layer A/B/C results.json, pass 1; pass 2 bit-identical)", ""]
    for lay in ("A", "B", "C"):
        for r in json.load(open(os.path.join(R, lay, "results.json"))):
            if "overshoot" in r:
                f = r.get("filled_cells", r.get("filled")); re_ = r.get("reach_cells", r.get("reach_virgin"))
                lines.append("%-42s overshoot %d  filled %d  reach %d  filled/reach %.3f  over_reach_filled %s" % (
                    r["label"], r["overshoot"], f, re_, float(f) / re_, ("%.3f" % r["over_reach_filled"]) if r.get("over_reach_filled") is not None else "-"))
    open(out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-3:]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
