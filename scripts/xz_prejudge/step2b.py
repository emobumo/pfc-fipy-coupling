# -*- coding: utf-8 -*-
"""Section 4.8, step 2B (deterministic part): area-weighted smoothing (v4) and groutability zoning by reach.

    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\step2b.py

v4 = v3 (step2a.py) with one change in the s-profile smoothing: slabs whose area (after the near-void cut) is
below 20 m2 are dropped, and the 5-point moving average is weighted by slab area (window shrinks at the ends).
Near-void exclusion, stretch, proportional mapping (area mean 1 - 1/K), scenario-2 Delta n(t) and clipping as
v3. Geometry and porosity for K = 1.15 / 1.25 / 1.40.

Holes: per drift crossing the rubble, -30 deg and 0 deg from +s (collar as v3; +30 deg dropped by decision),
re-cut on each K geometry. Zoning from the union of the 8 single-hole reach domains (0.25 m, 16-neighbour,
gravity): reliable = in the union at p0 1 MPa; conditional = not at 1 but at 2 MPa; hard = not at 2 MPa.

Outputs: outputs/xz_prejudge/step2b.txt (deterministic), step2b.npz (figures);
data/xz_prejudge/base_geometry_K{1.15,1.25,1.40}_v4.npz (same format as v3, plus zone maps; the K 1.25 file also
carries the S2 grouting cells: drift 1, 0 deg, standpipe to s = 3.5 m on the hole line).
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

S2A = imp.load_source("xz_step2a", os.path.join(REPO, "scripts", "xz_prejudge", "step2a.py"))
P, A = S2A.P, S2A.A
from src.fipy_adapter.mesh_init import build_mesh_for_domain

OUT, DATA, CELL = P.OUT, P.DATA, P.CELL
RUB, VOID = P.RUB, P.VOID
MIN_SLAB = 20.0                  # m2
ANGLES = (-30.0, 0.0)
S2_PIPE_S = 3.5                  # S2: standpipe up to s = 3.5 m on the hole line
P0S = P.P0S


def smooth_weighted(ns, ar):
    out = []
    for i in range(len(ns)):
        sl = slice(max(i - 2, 0), i + 3)
        out.append((ns[sl] * ar[sl]).sum() / ar[sl].sum())
    return np.array(out)


def local_f(rub, n):
    return P.window_gamma(rub, np.log(P.lam(np.where(rub, n, 0.5))))


def main():
    for dd in (OUT, DATA):
        if not os.path.isdir(dd):
            os.makedirs(dd)
    L = []
    w = L.append
    G = S2A.setup()
    S, T, XR, ZR, ny, nx, geo = G["S"], G["T"], G["XR"], G["ZR"], G["ny"], G["nx"], G["geo"]
    c, d50, ctrs = G["c"], G["d50"], G["ctrs"]
    a2 = CELL * CELL
    w("Step 2B (4.8), deterministic part: v4 area-weighted smoothing, groutability zoning (reach; v0.6; 2D; %.2f m)" % CELL)
    w("")

    # ---- 1. v4 smoothing --------------------------------------------------------------------------------------------
    Sp = A.s_coord(*np.meshgrid(*A.grid()))
    dv = ndimage.distance_transform_edt(~c["void"]) * A.H
    keep = (dv > S2A.EXCL_D * d50) | (Sp <= P.BASAL)
    zs_x, ns_x, ar_x = A.profile(c, Sp, A.PROF_S[0], A.PROF_S[1], A.SLAB_D * d50, mask=keep)
    s_pfc_max = S[G["pfc_rub"]].max()
    sm3 = P.smooth5(ns_x)
    big = ar_x >= MIN_SLAB
    zs4, ns4, ar4 = zs_x[big], ns_x[big], ar_x[big]
    sm4 = smooth_weighted(ns4, ar4)
    w("1. v4 s-profile: near-void pixels (%.0f d50) left out as v3; slabs < %.0f m2 dropped (%s); area-weighted"
      " 5-point average" % (S2A.EXCL_D, MIN_SLAB, ", ".join("s %.2f: %.1f m2" % (z, a) for z, a in
                                                              zip(zs_x[~big], ar_x[~big]))))
    rub = geo[P.K_BASE][1]
    sm3_, zst3 = S2A.mapping_v3(zs_x, ns_x, s_pfc_max, S[rub].max())
    _, zst4 = S2A.mapping_v3(zs4, ns4, s_pfc_max, S[rub].max())
    F3 = {}
    F4 = {}
    for K in P.KS:
        rk = geo[K][1]
        F3[K] = S2A.fields_v3(rk, S, T, S2A.mapping_v3(zs_x, ns_x, s_pfc_max, S[rk].max())[1], sm3, K)
        F4[K] = S2A.fields_v3(rk, S, T, S2A.mapping_v3(zs4, ns4, s_pfc_max, S[rk].max())[1], sm4, K)
    w("  smoothed min / max: v3 %.4f / %.4f, v4 %.4f (s %.2f) / %.4f (s %.2f); c (K 1.15 / 1.25 / 1.40): v3 %s, v4 %s"
      % (sm3.min(), sm3.max(), sm4.min(), zs4[np.argmin(sm4)], sm4.max(), zs4[np.argmax(sm4)],
         " / ".join("%.4f" % F3[K]["c"] for K in P.KS), " / ".join("%.4f" % F4[K]["c"] for K in P.KS)))
    w("  slab s / kept area / kept n / v3 smoothed / v4 smoothed / stretched s (K 1.25) / v3 n1 / v4 n1 (K 1.25)")
    for i, zv in enumerate(zs_x):
        j = np.where(np.abs(zs4 - zv) < 1e-9)[0]
        if j.size:
            j = j[0]
            w("    %6.2f  %6.1f  %.4f   %.4f  %.4f   %6.2f   %.4f  %.4f" % (zv, ar_x[i], ns_x[i], sm3[i], sm4[j], zst4[j],
                                                                         F3[P.K_BASE]["c"] * sm3[i],
                                                                         F4[P.K_BASE]["c"] * sm4[j]))
        else:
            w("    %6.2f  %6.1f  %.4f   %.4f  (dropped)        %6.2f   %.4f" % (zv, ar_x[i], ns_x[i], sm3[i], zst3[i],
                                                                         F3[P.K_BASE]["c"] * sm3[i]))
    w("")
    w("  n distribution (rubble): version K scen: min / median / mean / max; medians basal / interior / hanging side;"
      " area < 0.13 / > 0.29 m2; scen-2 clipped < 0.10 / > 0.35 cells")
    zone_k = {}
    for K in P.KS:
        rk = geo[K][1]
        zone_k[K] = P.zones(rk, S, T)
    for ver, FF in (("v3", F3), ("v4", F4)):
        for K in P.KS:
            rk = geo[K][1]
            bas, inn, hng = zone_k[K]
            for k, key in ((1, "n1"), (2, "n2")):
                n = FF[K][key]
                nv = n[rk]
                w("   %s %.2f %d: %.4f / %.4f / %.4f / %.4f; %.4f / %.4f / %.4f; %.2f / %.2f%s" % (
                    ver, K, k, nv.min(), np.median(nv), nv.mean(), nv.max(), np.median(n[bas]), np.median(n[inn]),
                    np.median(n[hng]), (rk & (n < P.N_LO)).sum() * a2, (rk & (n > P.N_HI)).sum() * a2,
                    ("; clipped %d / %d of %d" % (FF[K]["clo"], FF[K]["chi"], rk.sum())) if k == 2 else ""))
    w("")
    w("  local F (3 m window): version K scen: F < 0 share all; basal / interior / hanging side; global F p0 1 / 2 MPa")
    for ver, FF in (("v3", F3), ("v4", F4)):
        for K in P.KS:
            rk = geo[K][1]
            bas, inn, hng = zone_k[K]
            for k, key in ((1, "n1"), (2, "n2")):
                g = local_f(rk, FF[K][key])
                ok = rk & np.isfinite(g)
                gz = S2A.global_f(rk, FF[K][key], S, T)[0]
                w("   %s %.2f %d: %.4f; %.4f / %.4f / %.4f; %+.3f / %+.3f" % (
                    ver, K, k, (g[ok] < 0).mean(), (g[ok & bas] < 0).mean(), (g[ok & inn] < 0).mean(),
                    (g[ok & hng] < 0).mean(), gz * P0S[0] / (2 * P.RHO * P.G), gz * P0S[1] / (2 * P.RHO * P.G)))
    w("")

    # ---- 2. zoning --------------------------------------------------------------------------------------------------
    mesh = build_mesh_for_domain(P.DOM[0], P.DOM[1], P.DOM[2], P.DOM[3], CELL)[0]
    w("2. Groutability zoning: union of the %d single-hole reach domains (-30 / 0 deg per drift); reliable = in at 1 MPa,"
      " conditional = in at 2 MPa only, hard = out at 2 MPa" % (len(ANGLES) * len(ctrs)))
    zones_out = {}
    holes_k = {}
    lowk = {}
    for K in P.KS:
        cls_k, rk, _ = geo[K]
        hs = []
        for di, (xc, yc) in enumerate(ctrs):
            for ang in ANGLES:
                h = S2A.hole_fan(cls_k, yc, ang, ctrs)
                h["drift"] = di
                assert h["status"] == "ok", (K, di, ang, h["status"])
                hs.append(h)
        holes_k[K] = hs
        bas = zone_k[K][0]
        lowk[K] = bas & (T <= T[bas].min() + P.LOW_DT)
        w("  K %.2f: lower end of the basal layer = basal cells with t <= %.2f m (%d cells); hole ends (m from collar):"
          " %s" % (K, T[bas].min() + P.LOW_DT, lowk[K].sum(), ", ".join("d%d %+.0f %.2f" % (h["drift"] + 1, h["angle"],
                                                                                    h["u_end"]) for h in hs)))
    w("")
    w("  K     scen  reliable m2 (share)    conditional m2 (share)  hard m2 (share)     hanging side: reliable /"
      " conditional / hard")
    preq = {}
    for K in P.KS:
        cls_k, rk, _ = geo[K]
        hng = zone_k[K][2]
        for k, key in ((1, "n1"), (2, "n2")):
            st = P.make_state(mesh, F4[K][key].ravel())
            u1 = np.zeros((ny, nx), bool)
            u2 = np.zeros((ny, nx), bool)
            for hi, h in enumerate(holes_k[K]):
                cost = S2A.cost_of(st, h["cells"], ny, nx)
                u1 |= cost <= P0S[0]
                u2 |= cost <= P0S[1]
                preq[(K, k, hi)] = cost[lowk[K]].min()
            u1 &= rk
            u2 &= rk
            zmap = np.full((ny, nx), -1, int)
            zmap[rk] = 2
            zmap[u2] = 1
            zmap[u1] = 0
            zones_out[(K, k)] = zmap
            ar = [(zmap == q).sum() * a2 for q in (0, 1, 2)]
            tot = rk.sum() * a2
            hs_ = [(hng & (zmap == q)).sum() / float(hng.sum()) for q in (0, 1, 2)]
            w("  %.2f   %d    %8.2f (%.4f)     %8.2f (%.4f)      %8.2f (%.4f)    %.4f / %.4f / %.4f"
              % (K, k, ar[0], ar[0] / tot, ar[1], ar[1] / tot, ar[2], ar[2] / tot, hs_[0], hs_[1], hs_[2]))
    w("")
    w("  p_req to the lower end of the basal layer (MPa): K scen: " + ", ".join(
        "drift %d -30 / 0" % (d + 1) for d in range(len(ctrs))))
    for K in P.KS:
        for k in (1, 2):
            w("   %.2f %d: %s" % (K, k, ", ".join("%.3f" % (preq[(K, k, hi)] / 1e6) for hi in range(len(holes_k[K])))))
    w("")
    # hard-to-grout components, base
    for k in (1, 2):
        hard = zones_out[(P.K_BASE, k)] == 2
        comp = S2A.components(hard, XR, ZR, (zone_k[P.K_BASE][2], zone_k[P.K_BASE][1], zone_k[P.K_BASE][0]), a2)
        w("  K 1.25 scenario %d hard-to-grout components (largest 4 of %d): area, centroid, X range, Z range, shares"
          " hanging side / interior / basal" % (k, len(comp)))
        for r_ in comp[:4]:
            w("     %7.2f m2  (%.2f, %.2f)  X %.2f..%.2f  Z %.2f..%.2f  %.2f / %.2f / %.2f" % r_[:10])
    w("")

    # ---- S2 cells ---------------------------------------------------------------------------------------------------
    cls0 = geo[P.K_BASE][0]
    h0 = S2A.hole_fan(cls0, ctrs[0][1], 0.0, ctrs)
    ds = P.s_of(h0["collar"][0] + h0["dir"][0], h0["collar"][1] + h0["dir"][1]) - h0["s_collar"]
    u_pipe = (S2_PIPE_S - h0["s_collar"]) / ds
    hs2 = S2A.hole_fan(cls0, ctrs[0][1], 0.0, ctrs, pipe=u_pipe)
    w("  S2 hole (drift 1, 0 deg): standpipe to s %.2f m = %.3f m from the collar; grouting cells %d (2 m standpipe: %d)"
      % (S2_PIPE_S, u_pipe, hs2["n_cells"], h0["n_cells"]))
    w("")

    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "step2b.txt"), "w").write(text)

    def pack(hs):
        cells = np.concatenate([h["cells"] for h in hs])
        ptr = np.cumsum([0] + [len(h["cells"]) for h in hs])
        line = np.array([[h["collar"][0] - P.ORIGIN[0], h["collar"][1] - P.ORIGIN[1], h["end"][0] - P.ORIGIN[0],
                          h["end"][1] - P.ORIGIN[1]] for h in hs])
        return cells, ptr, line
    for K in P.KS:
        cls_k, rk, _ = geo[K]
        bas, _, hng = zone_k[K]
        hs = holes_k[K]
        f_cells, f_ptr, f_line = pack(hs)
        n_cells, n_ptr, n_line = pack([h for h in hs if h["angle"] == 0.0])
        extra = {}
        if K == P.K_BASE:
            extra = {"s2_cells": hs2["cells"], "s2_pipe": u_pipe}
        np.savez_compressed(
            os.path.join(DATA, "base_geometry_K%.2f_v4.npz" % K), cls=cls_k, X=XR, Z=ZR, S=S, T=T, n1=F4[K]["n1"],
            n2=F4[K]["n2"], cell=CELL, origin_model=np.array(P.ORIGIN), domain_model=np.array(P.DOM), basal=bas,
            hanging=hng, void_boundary_faces=P.neighbours_count(rk, cls_k == VOID), hole_cells=n_cells,
            hole_ptr=n_ptr, hole_line=n_line, fan_cells=f_cells, fan_ptr=f_ptr, fan_line=f_line,
            fan_drift=np.array([h["drift"] + 1 for h in hs]), fan_angle=np.array([h["angle"] for h in hs]),
            fan_collar_t=np.array([h["t"] for h in hs]), low_end=lowk[K], mapping_c=F4[K]["c"],
            zone1=zones_out[(K, 1)], zone2=zones_out[(K, 2)], K=K,
            classes=np.array(["rubble", "footwall", "hanging wall", "ore pillar", "void", "drift"]),
            note=np.array("v4 (step 2B): area-weighted 5-point smoothing of the near-void-cut s-profile, slabs < 20 m2"
                          " dropped; proportional mapping n1 = c n_smooth(s), mean n1 = 1 - 1/K; n2 = clip(n1 + Delta"
                          " n(t), 0.10, 0.35). cls codes 0..5 as in classes; rock, void, drift n = 1e-3. fan_* = 8"
                          " holes (-30 / 0 deg per drift), hole_* = the 0 deg holes; cells[ptr[i]:ptr[i+1]] = flat"
                          " indices iy * nx + ix. zone1 / zone2 (scenario 1 / 2): 0 reliable, 1 conditional, 2 hard,"
                          " -1 not rubble. low_end = basal cells with t <= min t + 2 m."), **extra)
    np.savez_compressed(os.path.join(OUT, "step2b.npz"), X=XR, Z=ZR, zs=zs_x, ns=ns_x, ar=ar_x, sm3=sm3, zst3=zst3,
                        zs4=zs4, sm4=sm4, zst4=zst4, c3=F3[P.K_BASE]["c"], c4=F4[P.K_BASE]["c"],
                        cls=np.array([geo[K][0] for K in P.KS]), hang=np.array([zone_k[K][2] for K in P.KS]),
                        zones=np.array([zones_out[(K, k)] for k in (1, 2) for K in P.KS]),
                        fan=np.array([pack(holes_k[K])[2] for K in P.KS]),
                        drifts=np.array([[xc - P.ORIGIN[0], yc - P.ORIGIN[1]] for xc, yc in ctrs]), ks=np.array(P.KS))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
