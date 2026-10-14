# -*- coding: utf-8 -*-
"""
Critical F* of the flip criterion on 0.625 m for all four parameter groups
(reach only, no solver):

    powershell -File scripts\\run_local.ps1 scripts\\fstar_0625.py

Same F family, groups, box and read-outs as scripts/fstar_readout.py
(case_box): each p_req field is read at cell centres (the recorded way) and
on the bilinear sub-grid (dx/10). Groups 2 and 3 at 0.625 m repeat the rows
of outputs/fstar_readout/fstar_readout.txt (R17, R17b) as a check; groups 1
and 4 at 0.625 m are new. The 1.25 m rows are in fstar_readout.txt.

Output: outputs/fstar_0625/fstar_0625.txt
"""
from __future__ import print_function

import imp
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

FR = imp.load_source("fstar_readout", os.path.join(REPO, "scripts", "fstar_readout.py"))


def main():
    out = []
    FR.case_box(("1", "2", "3", "4"), 0.625, out)
    dest = os.path.join(REPO, "outputs", "fstar_0625")
    if not os.path.isdir(dest):
        os.makedirs(dest)
    open(os.path.join(dest, "fstar_0625.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
