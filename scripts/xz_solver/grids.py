# -*- coding: utf-8 -*-
"""0.5 m solver grids for stage 2 of a new end state (main line 2026-10-20; deterministic, reach graph only).

    powershell -File scripts\\run_local.ps1 scripts\\xz_solver\\grids.py

Grids: K 1.25 scenario 1 (base), K 1.15 scenario 1, K 1.40 scenario 1, K 1.25 scenario 2 -- each aggregated from
data/xz_prejudge<suffix>/base_geometry_K*_v4.npz exactly as run_cases.coarse() does for the base grid (the
scenario-2 grid aggregates n2 instead of n1), cropped by run_cases.crop_box() on its own rubble. Lower end of the
basal layer: t <= run_cases.LOW_T (derived from the K 1.25 basal layer when the configuration leaves it null;
the per-K value is printed as a check).

On every grid, for each hole set of the configured case list (fan hole of a drift, or the extended-standpipe
section "s2", which only the K 1.25 file carries): source cells; reach domain at the case's p0 (rubble cells with
p_req <= p0, same crop) -- cells / m2 / pore m3/m; lower-end cells inside it; p_req from the grouting section to
the lower end; and the lower-end corner (rubble down-dip of the drift-1 collar, t < t_collar) area and its
hanging-side part, with the share of it in the reach domain.

Outputs: outputs/xz_solver<suffix>/grids.txt; grid_K<K>_s<k>.npz per grid (n, rub, cls, S, T, X, Z, basal, hang,
low, box, the source cells of every hole set) for the stage-2 runs.
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

RC = imp.load_source("xz_run_cases", os.path.join(REPO, "scripts", "xz_solver", "run_cases.py"))
P = RC.P
from src.analysis.fill_diagnostics import reachable_domain

GRIDS = ((1.25, 1), (1.15, 1), (1.40, 1), (1.25, 2))


class _D(dict):
    pass


def main():
    RC.load()                                   # sets RC.LOW_T from the K 1.25 basal layer if null
    a2 = RC.CELL * RC.CELL
    L = ["0.5 m solver grids (configuration %s; aggregation and crop as run_cases.py; lower end t <= %.4f m)"
         % (RC.CFG["name"], RC.LOW_T), ""]
    w = L.append
    for K, k in GRIDS:
        Dk = np.load(os.path.join(P.DATA, "base_geometry_K%.2f_v4.npz" % K))
        D = _D((q, Dk[q]) for q in Dk.files)
        D["n1"] = Dk["n%d" % k]                 # coarse() aggregates n1
        C = RC.coarse(D)
        box = RC.crop_box(C)
        j0, j1, i0, i1 = box
        cr = lambda a: a[j0:j1, i0:i1]
        rub5, hang5, low5, T5, X5, Z5 = (cr(C[q]) for q in ("rub", "hang", "low", "T", "X", "Z"))
        nx_f = D["cls"].shape[1]
        tc1 = float(D["fan_collar_t"][1])
        corner = rub5 & (T5 < tc1)
        w("K %.2f scenario %d: grid %d x %d, crop rows %d..%d, cols %d..%d -> %d x %d = %d cells (X %.2f..%.2f, Z %.2f..%.2f m);"
          " rubble %d cells = %.2f m2; basal %.2f m2; hanging side %.2f m2; lower end %d cells (0.25 m basal min t + %.1f"
          " on this K: %.4f)" % (K, k, C["cx"], C["cy"], j0, j1, i0, i1, i1 - i0, j1 - j0, (i1 - i0) * (j1 - j0),
                                X5.min() - RC.CELL / 2, X5.max() + RC.CELL / 2, Z5.min() - RC.CELL / 2, Z5.max() + RC.CELL / 2,
                                rub5.sum(), rub5.sum() * a2, cr(C["basal"]).sum() * a2, hang5.sum() * a2, low5.sum(),
                                P.LOW_DT, float(D["T"][D["basal"]].min()) + P.LOW_DT))
        w("  lower-end corner (rubble with t < %.4f m, down-dip of the drift-1 collar): %d cells = %.2f m2 (hanging side"
          " %.2f m2)" % (tc1, corner.sum(), corner.sum() * a2, (corner & hang5).sum() * a2))
        w("  hole set (cases): sources; reach cells / m2 / pore m3/m at p0; lower-end cells in the reach (of %d);"
          " p_req to the lower end (MPa); corner share in the reach" % low5.sum())
        keep = {}
        done = {}
        for tag, label, key, p0 in RC.CASES:
            if key == "s2" and "s2_cells" not in D:
                w("   %-4s %s: no extended-standpipe section on this K (step2b.py writes it for K 1.25 only)" % (tag, label))
                continue
            cells = D["s2_cells"] if key == "s2" else D["fan_cells"][D["fan_ptr"][key]:D["fan_ptr"][key + 1]]
            src_ji = RC.sources(cells, nx_f, C)
            keep["src_%s" % tag] = np.array(src_ji, int).reshape(-1, 2)
            st, src, _ = RC.build(C["n"], src_ji, box, p0)
            cost = np.asarray(reachable_domain(st, src, p0=p0)["cost"], float).reshape(rub5.shape)
            reach = rub5 & (cost <= p0)
            vols = np.asarray(st["mesh"].cellVolumes, dtype=float).reshape(rub5.shape)
            w("   %-4s %s: %d; %d / %.2f / %.2f; %d; %.3f; %.4f" % (
                tag, label, len(src), reach.sum(), reach.sum() * a2, (cr(C["n"]) * vols)[reach].sum(),
                (reach & low5).sum(), cost[low5].min() / 1e6, (reach & corner).sum() / float(corner.sum())))
        w("")
        np.savez_compressed(os.path.join(RC.OUT, "grid_K%.2f_s%d.npz" % (K, k)), K=K, scenario=k, box=np.array(box),
                            low_t=RC.LOW_T, n=C["n"], rub=C["rub"], cls=C["cls"], S=C["S"], T=C["T"], X=C["X"], Z=C["Z"],
                            basal=C["basal"], hang=C["hang"], low=C["low"], **keep)
    text = "\n".join(L) + "\n"
    if not os.path.isdir(RC.OUT):
        os.makedirs(RC.OUT)
    open(os.path.join(RC.OUT, "grids.txt"), "w").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
