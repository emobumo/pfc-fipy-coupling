# -*- coding: utf-8 -*-
"""
Second pass of the Zhaojin arrival runs A2 / B2 / C2 (1 m, to the -63 m drift),
the source of "124.8 ~ 147.9 m3/m, 4.4 ~ 5.2 x the original design volume".

    $env:STOP_RULE = "legacy"; powershell -File scripts\run_local.ps1 scripts\zhaojin_arrival_pass2.py

Pass 1 (2026-09-25, outputs/zhaojin_runs/results.jsonl) ran before the rate
rule existed, i.e. under the legacy 400-step window, which never fired: the
runs stop on arrival. Run with STOP_RULE=legacy so pass 2 is the same
computation; zhaojin_runs.py is imported unchanged, only OUT is redirected.
Then compares t_out, V_in at arrival, the arrival snapshots and the final
fields with pass 1.
Outputs: outputs/zhaojin_arrival_pass2/ (+ compare.txt).
"""
from __future__ import print_function

import imp
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis import stop_rule

OLD = os.path.join(REPO, "outputs", "zhaojin_runs")
OUT = os.path.join(REPO, "outputs", "zhaojin_arrival_pass2")
TAGS = ("A2", "B2", "C2")


def main():
    assert stop_rule.legacy(), "run with STOP_RULE=legacy (pass 1's rule)"
    ZR = imp.load_source("zhaojin_runs", os.path.join(REPO, "scripts", "zhaojin_runs.py"))
    ZR.OUT = OUT
    ZR.main(["--arrival", "--only", ",".join(TAGS), "--pass2"])
    old = dict((json.loads(l)["tag"], json.loads(l)) for l in open(os.path.join(OLD, "results.jsonl")) if l.strip())
    new = dict((json.loads(l)["tag"], json.loads(l)) for l in open(os.path.join(OUT, "results_pass2.jsonl")) if l.strip())
    lines = []
    for tag in TAGS:
        a, b = old[tag], new[tag]
        keys = [k for k in a if k not in ("wall", "label", "design_mode", "design_capacity", "layer")]
        diff = [k for k in keys if json.dumps(a[k], sort_keys=True) != json.dumps(b.get(k), sort_keys=True)]
        fa = np.load(os.path.join(OLD, "final_%s.npz" % tag))
        fb = np.load(os.path.join(OUT, "final_%s_pass2.npz" % tag))
        same = all(np.array_equal(fa[k], fb[k]) for k in fa.files)
        lines.append("%s: reason %s / %s, t_out %.6f / %.6f, V at -63 m %.6f / %.6f | scalars+snaps %s | final fields %s" % (
            tag, a["reason"], b["reason"], a["t_out"], b["t_out"],
            a["snaps"]["reach_-63m"]["v_in"], b["snaps"]["reach_-63m"]["v_in"],
            "identical" if not diff else "DIFF %s" % diff, "identical" if same else "DIFFER"))
    text = "\n".join(lines)
    open(os.path.join(OUT, "compare.txt"), "w").write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
