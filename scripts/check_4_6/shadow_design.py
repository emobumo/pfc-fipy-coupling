# -*- coding: utf-8 -*-
"""
4.6 check, anchor C14b: the eq. (4-27) shadow with and without the design
domain, as a share of the coverage-loss cells. Post-processing only (reach
recomputed through scripts/shadow_eq427.py `one`, no solver):

    powershell -File scripts\\run_local.ps1 scripts\\check_4_6\\shadow_design.py

C14  = |reach^w \\ U reach^(k)| / loss cells      (no design domain; cited)
C14b = |design & reach^w \\ U reach^(k)| / loss cells (record only)
The design domain is the 2.5 m uniform-0.18 stop footprint (175 cells, 700 on
1.25 m); it has no meaning for n != 0.18, which is why the chapter sets no
design domain for the uniform staged runs. Loss cells = unfilled cells in
reach^w, staged minus single. The counts must equal
outputs/shadow_eq427/shadow_eq427.json.

Output: outputs/check_4_6/shadow_design.txt
"""
from __future__ import print_function

import imp
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

SH = imp.load_source("shadow_eq427", os.path.join(REPO, "scripts", "shadow_eq427.py"))
OUT = os.path.join(REPO, "outputs", "check_4_6")


def main():
    rows = dict((r["label"], r) for r in json.load(open(os.path.join(SH.RR, "C", "results.json"))) if "stages" in r)
    saved = dict((r["label"], r) for r in json.load(open(os.path.join(REPO, "outputs", "shadow_eq427", "shadow_eq427.json"))))
    lines = ["4.6 check: eq. (4-27) shadow vs the design domain (C14, C14b)", "",
             "%-34s %5s | %6s %6s | %5s | %10s %10s | %s" % (
                 "plan", "cell", "shadow", "&dsgn", "loss", "C14 %", "C14b %", "same as shadow_eq427.json")]
    for lab, cell, hl in SH.CASES:
        r = SH.one(lab, cell, hl, rows[lab])
        st, si = rows[lab], rows[lab.rsplit("/", 1)[0] + "/single"]
        loss = st["shadowed"] - si["shadowed"]
        ok = (r["shadow_all"] == saved[lab]["shadow_all"] and r["shadow_design"] == saved[lab]["shadow_design"]
              and loss == saved[lab]["loss_cells"])
        quota = si["stages"][0]["reason"] == "quota"
        lines.append("%-34s %5.2f | %6d %6d | %5d | %10.6f %10.6f | %s%s" % (
            lab, cell, r["shadow_all"], r["shadow_design"], loss, 100.0 * r["shadow_all"] / loss,
            100.0 * r["shadow_design"] / loss, ok, "  (single quota-stopped, not comparable)" if quota else ""))
    text = "\n".join(lines)
    print(text)
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    open(os.path.join(OUT, "shadow_design.txt"), "w").write(text + "\n")


if __name__ == "__main__":
    main()
