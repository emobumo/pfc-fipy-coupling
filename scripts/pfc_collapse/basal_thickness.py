# -*- coding: utf-8 -*-
"""Thickness of the loose footwall (basal) layer of the PFC rubble, on fine slabs (main line 2026-10-20: the new
basal thickness sets the length of the extended-standpipe case).

    powershell -File scripts\\run_local.ps1 scripts\\pfc_collapse\\basal_thickness.py

Reads outputs/pfc_collapse<suffix>/fields_final.npz (analyze.py, same pixel classification). Particle-scale porosity
of the rubble region in slabs of DS normal to the orebody (s from the footwall contact), for all rubble pixels and
for the pixels kept by the v4 profile (farther than 3 d50 from the residual void, or s <= basal_s). Reference: the
interior porosity of analysis.txt (> 3 d50 from rock and void). Thickness = upper edge of the last slab of the
contiguous run from the floor whose porosity exceeds the reference by at least DN (two thresholds), and the slab
where the profile first falls to the reference.

Output: outputs/pfc_collapse<suffix>/basal_thickness.txt
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A = imp.load_source("pc_analyze", os.path.join(REPO, "scripts", "pfc_collapse", "analyze.py"))
DS = 0.5
S_LO, S_HI = -1.5, 10.0
DN = (0.02, 0.03)
EXCL_D = 3.0


def main():
    F = np.load(os.path.join(A.OUT, "fields_final.npz"))
    A.BOX = tuple(F["box"])
    region, pore, void, d50 = F["region"], F["pore"], F["void"], float(F["d50"])
    S = A.s_coord(*np.meshgrid(*A.grid()))
    inner = A.interior_mask({"rock": F["rock"], "void": void}, d50)
    n_ref = (pore & region & inner).sum() / float((region & inner).sum())
    keep = (ndimage.distance_transform_edt(~void) * A.H > EXCL_D * d50) | (S <= A.CFG["basal_s"])
    L = ["Loose footwall layer, configuration %s: particle-scale porosity in %.1f m slabs of s; interior reference %.4f"
         " (> 3 d50 from rock and void)" % (A.CFG["name"], DS, n_ref),
         "  s slab        all: n / area m2       v4-kept: n / area m2"]
    rows = []
    for a in np.arange(S_LO, S_HI - 1e-9, DS):
        out = [a, a + DS]
        for m in (region, region & keep):
            sl = m & (S >= a) & (S < a + DS)
            ar = sl.sum() * A.H * A.H
            out += [(pore & sl).sum() / float(sl.sum()) if sl.sum() else float("nan"), ar]
        rows.append(out)
        L.append("  %5.1f..%5.1f   %.4f / %6.1f       %.4f / %6.1f" % tuple(out))
    for col, name in ((2, "all"), (4, "v4-kept")):
        for dn in DN:
            top = None
            for r in rows:
                if r[1] <= 0.0 or r[col + 1] < 2.0:
                    continue
                if r[col] >= n_ref + dn:
                    top = r[1]
                elif top is not None:
                    break
            L.append("  %s: contiguous run from the floor with n >= reference + %.2f ends at s = %s m"
                     % (name, dn, "%.1f" % top if top is not None else "-"))
        first = [r[0] for r in rows if r[0] >= 0.0 and r[col + 1] >= 2.0 and r[col] <= n_ref]
        L.append("  %s: first slab above the floor at or below the reference starts at s = %s m"
                 % (name, "%.1f" % first[0] if first else "-"))
    text = "\n".join(L) + "\n"
    open(os.path.join(A.OUT, "basal_thickness.txt"), "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
