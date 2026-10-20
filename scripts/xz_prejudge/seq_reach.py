# -*- coding: utf-8 -*-
"""Sequential reach domains in construction order (main line 2026-10-20; deterministic, reach graph only).

    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\seq_reach.py

0.25 m, p0 = 2 MPa, v4 fields of the configured end state (data/xz_prejudge<suffix>/base_geometry_K*_v4.npz), the
8 fan holes (-30 / 0 deg per drift). Holes are grouted one after another; after each hole its reach domain ON THE
CURRENT FIELD is cemented (n = 1e-3, impermeable) before the next hole is computed. Grouting cells already
cemented by an earlier hole do not inject (the cemented pipe section self-blocks, 4.6). Orders:
    none       -- no order: union of the single-hole reach domains (the current zoning at 2 MPa);
    bottom-up  -- drift 1 -> 4, -30 deg before 0 deg in each drift;
    top-down   -- drift 4 -> 1, -30 deg before 0 deg in each drift.
Per order: final coverage (area, share of the rubble and of the hanging side); per hole the stepwise reach
domain, its ratio to the single-hole reach domain, cemented grouting cells and whether it contains the lower end of
the basal layer; the area lost against the union (blocks with area, centroid and extent). K 1.15 / 1.25 / 1.40 x
scenarios 1 / 2.

Output: outputs/xz_prejudge<suffix>/seq_reach.txt; seq_reach.npz (coverage masks per K, scenario and order, for the figures)
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

S2A = imp.load_source("xz_step2a", os.path.join(REPO, "scripts", "xz_prejudge", "step2a.py"))
P = S2A.P
from src.fipy_adapter.mesh_init import build_mesh_for_domain

P0 = 2.0e6
ORDERS = (("bottom-up", "drift 1 -> %d, -30 deg first"), ("top-down", "drift %d -> 1, -30 deg first"))


def order_of(name, drift, angle):
    idx = range(len(drift))
    if name == "bottom-up":
        return sorted(idx, key=lambda i: (drift[i], angle[i]))
    return sorted(idx, key=lambda i: (-drift[i], angle[i]))


def main():
    a2 = P.CELL * P.CELL
    mesh = build_mesh_for_domain(P.DOM[0], P.DOM[1], P.DOM[2], P.DOM[3], P.CELL)[0]
    L = ["Sequential reach domains (0.25 m, p0 = %.0f MPa, v4, configuration %s); cemented = n %.0e"
         % (P0 / 1e6, P.CFG["name"], P.ROCK_N), ""]
    w = L.append
    masks = {}
    for K in P.KS:
        D = np.load(os.path.join(P.DATA, "base_geometry_K%.2f_v4.npz" % K))
        cls = D["cls"]
        rub = cls == P.RUB
        ny, nx = rub.shape
        XR, ZR = D["X"], D["Z"]
        bas, hang, low = D["basal"], D["hanging"], D["low_end"]
        inner = rub & ~bas & ~hang
        fc, fp = D["fan_cells"], D["fan_ptr"]
        drift, angle = [int(v) for v in D["fan_drift"]], [float(v) for v in D["fan_angle"]]
        name = ["d%d %+.0f" % (drift[i], angle[i]) for i in range(len(drift))]
        cells = [fc[fp[i]:fp[i + 1]] for i in range(len(drift))]
        for k in (1, 2):
            n0 = D["n%d" % k].copy()
            st0 = P.make_state(mesh, n0.ravel())
            single = [rub & (S2A.cost_of(st0, c_, ny, nx) <= P0) for c_ in cells]
            union = np.zeros_like(rub)
            for r_ in single:
                union |= r_
            masks["K%.2f_s%d_none" % (K, k)] = union
            zmap = D["zone%d" % k]
            w("K %.2f scenario %d: rubble %.2f m2, hanging side %.2f m2; union of single-hole reach domains %.2f m2"
              " (%.4f; hanging side %.4f); equals the stored zoning (zone <= 1): %s"
              % (K, k, rub.sum() * a2, hang.sum() * a2, union.sum() * a2, union.sum() / float(rub.sum()),
                 (union & hang).sum() / float(hang.sum()), bool(np.array_equal(union, rub & (zmap >= 0) & (zmap <= 1)))))
            for oname, odesc in ORDERS:
                odesc = odesc % max(drift)
                ncur = n0.copy()
                sealed = np.zeros_like(rub)
                cover = np.zeros_like(rub)
                rows = []
                for i in order_of(oname, drift, angle):
                    live = np.array([q for q in cells[i] if not sealed.ravel()[q]], dtype=int)
                    if live.size:
                        cost = S2A.cost_of(P.make_state(mesh, ncur.ravel()), live, ny, nx)
                        reach = rub & ~sealed & (cost <= P0)
                    else:
                        reach = np.zeros_like(rub)
                    rows.append((name[i], reach.sum() * a2, single[i].sum() * a2,
                                 reach.sum() / float(single[i].sum()) if single[i].any() else float("nan"),
                                 len(cells[i]) - live.size, len(cells[i]), bool((reach & low).any()),
                                 bool((single[i] & low).any())))
                    cover |= reach
                    sealed |= reach
                    ncur[reach] = P.ROCK_N
                lost = union & ~cover
                masks["K%.2f_s%d_%s" % (K, k, oname)] = cover
                w("  %s (%s): coverage %.2f m2 = %.4f of the rubble (hanging side %.4f, interior %.4f); lost against the"
                  " union %.2f m2 (%.4f of the union); outside the union %.2f m2"
                  % (oname, odesc, cover.sum() * a2, cover.sum() / float(rub.sum()), (cover & hang).sum() / float(hang.sum()),
                     (cover & inner).sum() / float(inner.sum()), lost.sum() * a2, lost.sum() / float(union.sum()),
                     (cover & ~union).sum() * a2))
                w("     hole    step reach m2  single m2  ratio   cemented grouting cells  lower end: step / single")
                for r_ in rows:
                    w("     %-6s  %9.2f   %9.2f   %.4f   %3d of %3d               %s / %s" % r_)
                comp = S2A.components(lost, XR, ZR, (hang, inner, bas), a2)
                w("     lost blocks (%d; largest 5): area, centroid (X, Z), X range, Z range, hanging side / interior / basal"
                  % len(comp))
                for c_ in comp[:5]:
                    w("        %7.2f m2  (%.2f, %.2f)  X %.2f..%.2f  Z %.2f..%.2f  %.2f / %.2f / %.2f" % c_[:10])
            w("")
    text = "\n".join(L) + "\n"
    if not os.path.isdir(P.OUT):
        os.makedirs(P.OUT)
    open(os.path.join(P.OUT, "seq_reach.txt"), "w").write(text)
    np.savez_compressed(os.path.join(P.OUT, "seq_reach.npz"), **masks)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
