# -*- coding: utf-8 -*-
"""Section 4.8, step 2A: proportional porosity mapping, fan holes, and a solver timing probe.

    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\step2a.py            (mapping, closed forms, reach)
    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\step2a.py --probe    (solver timing only)

Geometry, drifts, hole rule, collar, scenario-2 Delta n(t), zones and the reach graph are those of step 1 v2
(prejudge.py, imported). What changes is the porosity:

- s-profile: PFC particle scale, 1.5 m slabs as in v2, but pixels of the rubble region within 3 d50 of a PFC
  residual void are left out of the statistics (pixels with s <= 3 m, the footwall band, are kept whatever
  their distance). 5-point moving average, the v2 stretch (s <= 3 m unchanged, the PFC rubble's maximum s to the
  new rubble's maximum s).
- proportional mapping: n1 = c n_smooth(s), c such that the area mean of n1 on the corrected rubble is
  1 - 1/K. No clipping; 0.13..0.29 is only a check. Scenario 2: n2 = n1 + Delta n(t), clipped to 0.10..0.35.
  K sets both the geometry and the porosity level (1.15 / 1.25 / 1.40).

Fan holes: three holes per drift from the v2 collar (mid-point of the -x wall), at -30 / 0 / +30 deg from +s
(negative = towards down-dip), in the section; same standpipe and stop rule as v2. A hole whose rock section
touches a drift (closed square, 1e-6 m tolerance) or leaves the domain before reaching the rubble is dropped.

Outputs: outputs/xz_prejudge/step2a.txt (deterministic), step2a.npz (figures), data/xz_prejudge/
base_geometry_K1.25_v3.npz (step-2 input); --probe writes outputs/xz_prejudge/step2a_probe.txt (wall times,
not reproducible by nature; no fill results are kept).
"""
from __future__ import print_function

import imp
import math
import os
import sys
import time

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

P = imp.load_source("xz_prejudge", os.path.join(REPO, "scripts", "xz_prejudge", "prejudge.py"))
A, Z = P.A, P.Z
from fipy import CellVariable
from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.analysis.fill_diagnostics import reachable_domain
from src.analysis import stop_rule
from src.models.slurry_transport.equations import solve_transport_step
from src.models.slurry_transport.variables import initialize_slurry_variables

OUT, DATA, CELL = P.OUT, P.DATA, P.CELL
RUB, VOID, DRIFT = P.RUB, P.VOID, P.DRIFT
EXCL_D = 3.0                     # x d50: rubble pixels this close to a residual void are left out of the profile
FAN = (-30.0, 0.0, 30.0)
P0S = P.P0S
ROCK_N = P.ROCK_N


def mapping_v3(zs, ns, s_pfc_max, s_new_max):
    sm = P.smooth5(ns)
    zst = np.where(zs <= P.BASAL, zs, P.BASAL + (zs - P.BASAL) * (s_new_max - P.BASAL) / (s_pfc_max - P.BASAL))
    return sm, zst


def fields_v3(rub, S, T, zst, sm, K):
    nsm = np.interp(S, zst, sm)
    c = (1.0 - 1.0 / K) / nsm[rub].mean()
    n1r = c * nsm
    tmin, tmax = T[rub].min(), T[rub].max()
    raw2 = n1r + (-P.DELTA + 2 * P.DELTA * (T - tmin) / (tmax - tmin))
    n1 = np.where(rub, n1r, ROCK_N)
    n2 = np.where(rub, np.clip(raw2, P.CLIP[0], P.CLIP[1]), ROCK_N)
    return {"n1": n1, "n2": n2, "c": c, "clo": int((rub & (raw2 < P.CLIP[0])).sum()),
            "chi": int((rub & (raw2 > P.CLIP[1])).sum()), "tmin": tmin, "tmax": tmax}


def hole_dir(ang):
    a = math.radians(ang)
    return (math.cos(a) * -P.SN + math.sin(a) * P.CS, math.cos(a) * P.CS + math.sin(a) * P.SN)


def hole_fan(cls, yd, ang, ctrs):
    """Hole at angle ang from +s, collar on the -x wall of the drift at level yd. Returns a dict with 'status'
    'ok', 'misses', 'drift' (the rock section touches a drift) or 'domain'."""
    ny, nx = cls.shape
    xc = P.x_fw(yd) + P.DRIFT_DX
    x0, y0 = xc - P.DRIFT_HALF, yd
    ex, ey = hole_dir(ang)
    u = np.arange(0.0, 90.0, 0.01)
    px, py = x0 + u * ex, y0 + u * ey
    ix = np.floor((px - P.DOM[0]) / CELL).astype(int)
    iy = np.floor((py - P.DOM[2]) / CELL).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cl = np.where(inside, cls[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)], -9)
    base = {"collar": (x0, y0), "angle": ang, "dir": (ex, ey), "t": P.t_of(x0, y0), "s_collar": P.s_of(x0, y0)}
    hit = np.where((cl == RUB) & (u > 0.0))[0]
    first = hit[0] if hit.size else len(u)
    out = np.where(~inside[:first])[0]
    if out.size:
        base.update(status="domain", u_conf=u[out[0]])
        return base
    seg = (u > 1e-9) & (np.arange(len(u)) <= min(first, len(u) - 1))
    for xd, ydd in ctrs:
        ind = seg & (np.abs(px - xd) <= P.DRIFT_HALF + 1e-6) & (np.abs(py - ydd) <= P.DRIFT_HALF + 1e-6)
        if ind.any():
            base.update(status="drift", u_conf=u[ind].max(), drift_hit=(xd, ydd))
            return base
    if not hit.size:
        base.update(status="misses")
        return base
    after = np.where((np.arange(len(u)) > first) & (cl != RUB))[0]
    u_out = u[after[0]]
    u_end = u_out - P.STOP_GAP
    sel = (u >= P.PIPE) & (u <= u_end)
    cells = sorted(set(zip(iy[sel], ix[sel])))
    flat = np.array([j * nx + i for j, i in cells])
    pe = (x0 + P.PIPE * ex, y0 + P.PIPE * ey)
    pipe_cell = cls[int((pe[1] - P.DOM[2]) / CELL), int((pe[0] - P.DOM[0]) / CELL)]
    base.update(status="ok", end=(x0 + u_end * ex, y0 + u_end * ey), u_in=u[first], u_out=u_out, u_end=u_end,
                stop=P.NAME[int(cl[after[0]])], cells=flat, s_pipe_end=P.s_of(*pe), t_pipe_end=P.t_of(*pe),
                pipe_cell=P.NAME[int(pipe_cell)], n_cells=len(cells),
                rubble_cells=int(sum(cls[j, i] == RUB for j, i in cells)))
    return base


def setup():
    cav, pfc_rub, side, zs, ns, ar, solid, d50, c = P.pfc_cells()
    ny, nx = cav.shape
    xs = P.DOM[0] + (np.arange(nx) + 0.5) * CELL
    ys = P.DOM[2] + (np.arange(ny) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys)
    S, T = P.s_of(X, Y), P.t_of(X, Y)
    A0 = solid / (1.0 - P.N_HW)
    colr = pfc_rub.any(axis=0)
    ytop = np.full(nx, np.nan)
    for i in np.where(colr)[0]:
        ytop[i] = ys[np.where(pfc_rub[:, i])[0].max()] + CELL / 2
    i_lo, i_hi = np.where(colr)[0].min(), np.where(colr)[0].max()
    ytop[:i_lo] = ytop[i_lo]
    ytop[i_hi + 1:] = ytop[i_hi]
    for i in range(i_lo, i_hi + 1):
        if not colr[i]:
            ytop[i] = np.interp(i, [j for j in range(nx) if colr[j]], [ytop[j] for j in range(nx) if colr[j]])
    cls0 = P.geometry(cav, pfc_rub, side, ytop, X, Y, A0, P.K_BASE, [])[0]
    yd, drifts, ctrs = P.ORIGIN[1], [], []
    while True:
        dm, ctr = P.drift_mask(X, Y, yd)
        cls_try = cls0.copy()
        for m in drifts + [dm]:
            cls_try[m] = DRIFT
        if P.hole(cls_try, yd) is None:
            break
        drifts.append(dm)
        ctrs.append(ctr)
        yd += P.DRIFT_DY
    geo = {}
    for K in P.KS:
        geo[K] = P.geometry(cav, pfc_rub, side, ytop, X, Y, A0, K, drifts)
    return {"cav": cav, "pfc_rub": pfc_rub, "zs": zs, "ns": ns, "ar": ar, "d50": d50, "c": c, "X": X, "Y": Y,
            "S": S, "T": T, "XR": X - P.ORIGIN[0], "ZR": Y - P.ORIGIN[1], "A0": A0, "ytop": ytop, "geo": geo,
            "ctrs": ctrs, "xs": xs, "ny": ny, "nx": nx}


def cost_of(st, cells, ny, nx):
    return np.asarray(reachable_domain(st, cells, p0=P0S[-1])["cost"], float).reshape(ny, nx)


def global_f(rub, n, S, T):
    ll = np.log(P.lam(n[rub]))
    Mx = np.column_stack([np.ones(ll.size), T[rub], S[rub]])
    cf = np.linalg.lstsq(Mx, ll)[0]
    gz = -(cf[1] * P.SN + cf[2] * P.CS)
    nv = n[rub]
    cn = np.linalg.lstsq(Mx, nv)[0]
    r = nv - Mx.dot(cn)
    r2 = 1 - (r ** 2).sum() / ((nv - nv.mean()) ** 2).sum()
    return gz, cn, r2


def components(miss, XR, ZR, masks, a2):
    lab, nl = ndimage.label(miss, structure=np.ones((3, 3), int))
    comp = []
    for q in range(1, nl + 1):
        mk = lab == q
        comp.append((mk.sum() * a2, XR[mk].mean(), ZR[mk].mean(), XR[mk].min(), XR[mk].max(), ZR[mk].min(),
                     ZR[mk].max()) + tuple((mk & m).sum() / float(mk.sum()) for m in masks) + (mk,))
    comp.sort(key=lambda r: -r[0])
    return comp


def main():
    for dd in (OUT, DATA):
        if not os.path.isdir(dd):
            os.makedirs(dd)
    L = []
    w = L.append
    G = setup()
    S, T, XR, ZR, ny, nx, geo = G["S"], G["T"], G["XR"], G["ZR"], G["ny"], G["nx"], G["geo"]
    a2 = CELL * CELL
    c, d50 = G["c"], G["d50"]
    w("Step 2A (4.8): proportional porosity mapping, fan holes (reach and closed forms; v0.6; 2D; grid %.2f m)" % CELL)
    w("geometry, drifts, holes, scenario-2 Delta n(t), zones and reach graph as step 1 v2 (prejudge.py)")
    w("")

    # ---- 1.1 profile with the near-void pixels left out --------------------------------------------------------
    w("1.1 PFC s-profile (particle scale, 1.5 m slabs) with the rubble pixels within %.0f d50 = %.3f m of a residual void"
      " left out (footwall band s <= %.0f m kept)" % (EXCL_D, EXCL_D * d50, P.BASAL))
    Sp = A.s_coord(*np.meshgrid(*A.grid()))
    dv = ndimage.distance_transform_edt(~c["void"]) * A.H
    near = dv <= EXCL_D * d50
    keep = ~near | (Sp <= P.BASAL)
    zs_x, ns_x, ar_x = A.profile(c, Sp, -2.0, 44.0, A.SLAB_D * d50, mask=keep)
    reg = c["region"]
    w("  rubble region %.2f m2; left out %.2f m2 (porosity %.4f); near-void pixels kept in the footwall band %.2f m2"
      % (reg.sum() * A.H ** 2, (reg & ~keep).sum() * A.H ** 2, A.porosity(c, ~keep)[0],
         (reg & near & (Sp <= P.BASAL)).sum() * A.H ** 2))
    w("  whole-region porosity: all %.4f, kept %.4f" % (A.porosity(c)[0], A.porosity(c, keep)[0]))
    s_pfc_max = S[G["pfc_rub"]].max()
    rub = geo[P.K_BASE][1]
    cls = geo[P.K_BASE][0]
    sm_all = P.smooth5(G["ns"])
    sm_x, zst_x = mapping_v3(zs_x, ns_x, s_pfc_max, S[rub].max())
    _, mapped_v2, zst_v2 = P.mapping(G["zs"], G["ns"], s_pfc_max, S[rub].max())
    F0 = fields_v3(rub, S, T, zst_x, sm_x, P.K_BASE)
    w("  smoothed profile: all slabs min %.4f (s %.2f) max %.4f (s %.2f); near-void left out min %.4f (s %.2f)"
      " max %.4f (s %.2f); max / min %.3f vs %.3f" % (sm_all.min(), G["zs"][np.argmin(sm_all)], sm_all.max(),
                                                       G["zs"][np.argmax(sm_all)], sm_x.min(), zs_x[np.argmin(sm_x)],
                                                       sm_x.max(), zs_x[np.argmax(sm_x)], sm_all.max() / sm_all.min(),
                                                       sm_x.max() / sm_x.min()))
    w("  slab s / area all / n all / smoothed all / area kept / n kept / smoothed kept / stretched s / n1 = c x smoothed"
      " (K %.2f) / v2 mapped" % P.K_BASE)
    for i, zv in enumerate(G["zs"]):
        j = np.where(np.abs(zs_x - zv) < 1e-9)[0]
        if j.size:
            j = j[0]
            w("    %6.2f  %6.1f  %.4f  %.4f   %6.1f  %.4f  %.4f   %6.2f  %.4f   %.4f"
              % (zv, G["ar"][i], G["ns"][i], sm_all[i], ar_x[j], ns_x[j], sm_x[j], zst_x[j], F0["c"] * sm_x[j],
                 mapped_v2[i]))
        else:
            w("    %6.2f  %6.1f  %.4f  %.4f   (slab < 2 m2 after the cut)                       %.4f"
              % (zv, G["ar"][i], G["ns"][i], sm_all[i], mapped_v2[i]))
    near_hw = ndimage.distance_transform_edt(~(c["rock"] & (c["rock_side"] == 2))) * A.H <= EXCL_D * d50
    w("  kept pixels of the upper slabs split by the distance to the hanging-wall rock (3 d50):")
    for zv in zs_x[zs_x >= 21.0]:
        m = keep & (Sp >= zv - 0.75) & (Sp < zv + 0.75)
        na, aa = A.porosity(c, m & near_hw)
        nf, af = A.porosity(c, m & ~near_hw)
        w("    slab %.2f: within 3 d50 of the hanging wall %.1f m2, n %.4f; farther %.1f m2, n %.4f" % (zv, aa, na, af, nf))
    w("")

    # ---- 1.2 proportional mapping per K ------------------------------------------------------------------------
    w("1.2 Proportional mapping n1 = c n_smooth(s), area mean of n1 = 1 - 1/K; check range %.2f..%.2f (not clipped)"
      % (P.N_LO, P.N_HI))
    w("  K     c       mean n1  min n1   max n1   area < %.2f  area > %.2f (m2, of rubble m2)   scen. 2: clipped < %.2f"
      " / > %.2f (cells, of)   mean n2" % (P.N_LO, P.N_HI, P.CLIP[0], P.CLIP[1]))
    FK = {}
    for K in P.KS:
        cls_k, rub_k, _ = geo[K]
        sm_k, zst_k = mapping_v3(zs_x, ns_x, s_pfc_max, S[rub_k].max())
        FK[K] = fields_v3(rub_k, S, T, zst_k, sm_k, K)
        f = FK[K]
        nv = f["n1"][rub_k]
        w("  %.2f  %.4f  %.4f   %.4f   %.4f   %8.2f     %8.2f   (%.2f)                      %d / %d (%d)           %.4f"
          % (K, f["c"], nv.mean(), nv.min(), nv.max(), (rub_k & (f["n1"] < P.N_LO)).sum() * a2,
             (rub_k & (f["n1"] > P.N_HI)).sum() * a2, rub_k.sum() * a2, f["clo"], f["chi"], rub_k.sum(),
             f["n2"][rub_k].mean()))
    w("")

    # ---- 1.3 closed forms on the base geometry -----------------------------------------------------------------
    n1, n2 = FK[P.K_BASE]["n1"], FK[P.K_BASE]["n2"]
    sc = {1: n1, 2: n2}
    w("1.3 Base geometry K = %.2f: closed forms" % P.K_BASE)
    basal, interior, hang = P.zones(rub, S, T)
    zl = (("basal", basal), ("interior", interior), ("hanging side", hang))
    w("  zones: basal %.2f m2, interior %.2f m2, hanging side %.2f m2" % (basal.sum() * a2, interior.sum() * a2,
                                                                        hang.sum() * a2))
    for k in (1, 2):
        nv = sc[k][rub]
        w("  scenario %d: n min / median / mean / max %.4f / %.4f / %.4f / %.4f; medians basal / interior / hanging side"
          " %.4f / %.4f / %.4f" % ((k, nv.min(), np.median(nv), nv.mean(), nv.max()) +
                                   tuple(np.median(sc[k][zm]) for _, zm in zl)))
    w("  L_max (m): scenario, p0: min / median / max; medians basal / interior / hanging side; min L_max / dx")
    for k in (1, 2):
        for p0 in P0S + (P.P0_REF,):
            Lm = p0 / P.lam(sc[k])
            row = (Lm[rub].min(), np.median(Lm[rub]), Lm[rub].max()) + tuple(np.median(Lm[zm]) for _, zm in zl)
            w("    %d  %.0f MPa: %.3f / %.3f / %.3f; %.3f / %.3f / %.3f; %.2f%s" % ((k, p0 / 1e6) + row + (
                row[0] / CELL, "  (< 7)" if row[0] / CELL < 7 else "")))
    gam = {}
    for k in (1, 2):
        gam[k] = P.window_gamma(rub, np.log(P.lam(np.where(rub, sc[k], 0.5))))
    okg = rub & np.isfinite(gam[1]) & np.isfinite(gam[2])
    for k in (1, 2):
        gv = gam[k][okg]
        w("  scenario %d: gamma (3 m window, %d cells) percentiles 5/25/50/75/95 %s 1/m" % (
            k, gv.size, " / ".join("%+.4f" % q for q in np.percentile(gv, [5, 25, 50, 75, 95]))))
        for p0 in P0S + (P.P0_REF,):
            F = gv * p0 / (2 * P.RHO * P.G)
            w("    p0 %.0f MPa: F percentiles 5/25/50/75/95 %s; share F > 1 %.4f, F < 0 %.4f"
              % (p0 / 1e6, " / ".join("%+.3f" % q for q in np.percentile(F, [5, 25, 50, 75, 95])), (F > 1).mean(),
                 (F < 0).mean()))
        w("    F < 0 share by zone: " + "; ".join("%s %.4f" % (zn, (gam[k][zm & okg] < 0).mean()) for zn, zm in zl))
    w("  cell-wise sign agreement of F between the scenarios: %.4f of %d cells" % (
        (np.sign(gam[1][okg]) == np.sign(gam[2][okg])).mean(), okg.sum()))
    for k in (1, 2):
        gz, cn, r2 = global_f(rub, sc[k], S, T)
        w("  scenario %d: n = a + b_t t + b_s s: b_t %+.5f, b_s %+.5f /m, R2 %.3f; b_s cos30 %+.5f vs b_t sin30 %+.5f;"
          " vertical %+.5f /m. Whole-rubble ln(lambda) plane: vertical gamma %+.5f 1/m -> F %s (p0 1 / 2 / 5 MPa)"
          % (k, cn[1], cn[2], r2, cn[2] * P.CS, cn[1] * P.SN, cn[2] * P.CS + cn[1] * P.SN, gz,
             " / ".join("%+.3f" % (gz * p0 / (2 * P.RHO * P.G)) for p0 in P0S + (P.P0_REF,))))
    w("")

    # ---- 1.3 reach, normal holes ------------------------------------------------------------------------------------
    mesh = build_mesh_for_domain(P.DOM[0], P.DOM[1], P.DOM[2], P.DOM[3], CELL)[0]
    ctrs = G["ctrs"]
    vb_faces = P.neighbours_count(rub, cls == VOID)
    top_face = rub & (vb_faces > 0)
    top_band = rub & (ndimage.distance_transform_edt(~(cls == VOID)) * CELL <= 3.0)
    tb = T[basal]
    low_end = basal & (T <= tb.min() + 2.0)
    w("1.3 Base geometry: drifts, normal holes, required source pressure and reach")
    w("  lower end of the basal layer: basal cells with t <= %.2f m: %d cells" % (tb.min() + 2.0, low_end.sum()))
    fan = []
    for di, (xc, yc) in enumerate(ctrs):
        for ang in FAN:
            h = hole_fan(cls, yc, ang, ctrs)
            h["drift"] = di
            fan.append(h)
    normal = [h for h in fan if h["angle"] == 0.0]
    for i, h in enumerate(normal):
        ref = P.hole(cls, ctrs[i][1])
        assert np.array_equal(ref["cells"], h["cells"])
    costs, costs_b = {}, {}
    for k in (1, 2):
        st = P.make_state(mesh, sc[k].ravel())
        st_b = P.make_state(mesh, np.where(basal, sc[k], ROCK_N).ravel())
        for fi, h in enumerate(fan):
            if h["status"] != "ok":
                continue
            costs[(k, fi)] = cost_of(st, h["cells"], ny, nx)
            if h["angle"] == 0.0:
                srcb = np.array([q for q in h["cells"] if basal.ravel()[q]])
                costs_b[(k, fi)] = cost_of(st_b, srcb, ny, nx) if srcb.size else np.full((ny, nx), np.inf)
    nidx = [fi for fi, h in enumerate(fan) if h["angle"] == 0.0]
    w("  scen hole p0     reach/rubble  basal entry  lower end p_req/p0 (any; basal only)  hanging-side reach  void contact")
    for k in (1, 2):
        for hi, fi in enumerate(nidx):
            h, cost, cb = fan[fi], costs[(k, fi)], costs_b[(k, fi)]
            hc = np.zeros(ny * nx, bool)
            hc[h["cells"]] = True
            hc = hc.reshape(ny, nx)
            entry = cost[basal & hc].min() if (basal & hc).any() else float("nan")
            lowv, lowb = cost[low_end].min(), cb[low_end].min()
            for p0 in P0S:
                rch = (cost <= p0) & rub
                vc = (vb_faces * rch).sum() * CELL
                w("   %d   %d   %.0f MPa   %.4f       %.3f MPa    %.3f ; %.3f                       %.4f            %s"
                  % (k, hi + 1, p0 / 1e6, rch.sum() / float(rub.sum()), entry / 1e6, lowv / p0, lowb / p0,
                     (rch & hang).sum() / float(hang.sum()), ("yes, %.2f m" % vc) if vc > 0 else "no"))
            w("          p_req: lower end of the basal layer %.3f MPa (any path) / %.3f MPa (basal only); rubble top"
              " surface min %.3f MPa" % (lowv / 1e6, lowb / 1e6, cost[top_face].min() / 1e6))
    w("")
    masks = (hang, interior, top_band, basal)

    def union_block(title, idx, tag):
        w("  union of the single-hole reach domains: %s" % title)
        res = {}
        for k in (1, 2):
            for p0 in P0S:
                u = np.zeros((ny, nx), bool)
                for fi in idx:
                    u |= costs[(k, fi)] <= p0
                u &= rub
                miss = rub & ~u
                res[(k, p0)] = (u, miss)
                w("   scenario %d, %.0f MPa: union %.4f of the rubble; not covered %.2f m2: hanging side %.4f, top band"
                  " %.4f, basal %.4f, interior %.4f of each zone uncovered"
                  % (k, p0 / 1e6, u.sum() / float(rub.sum()), miss.sum() * a2, (miss & hang).sum() / float(hang.sum()),
                     (miss & top_band).sum() / float(top_band.sum()), (miss & basal).sum() / float(basal.sum()),
                     (miss & interior).sum() / float(interior.sum())))
                comp = components(miss, XR, ZR, masks, a2)
                res[(k, p0, "comp")] = comp
                for r_ in comp[:5]:
                    w("      uncovered %7.2f m2  centroid (%.2f, %.2f)  X %.2f..%.2f  Z %.2f..%.2f  shares: hanging side"
                      " %.2f, interior %.2f, top band %.2f, basal %.2f" % r_[:11])
                w("      (%d uncovered components in all)" % len(comp))
        return res

    U_n = union_block("normal holes (step 1 v2 layout)", nidx, "normal")
    w("  top band = rubble within 3 m of the residual void; shares of a component sum to 1 except where the top band"
      " overlaps the other zones")
    w("")

    # ---- 1.4 K sensitivity -------------------------------------------------------------------------------------------
    w("1.4 K sensitivity, scenario 1 (geometry and porosity level both from K; same drifts; holes re-cut per geometry)")
    w("  K     hole 1 to the basal lower end (MPa)   union 1 / 2 MPa   hanging side uncovered 1 / 2 MPa   global F 1 / 2 MPa"
      "   hole 2 hanging-side reach 1 / 2 MPa")
    ksens = []
    for K in P.KS:
        cls_k, rub_k, _ = geo[K]
        n1k = FK[K]["n1"]
        bas_k, _, hang_k = P.zones(rub_k, S, T)
        lo_k = bas_k & (T <= T[bas_k].min() + 2.0)
        st = P.make_state(mesh, n1k.ravel())
        hk = [hole_fan(cls_k, yc, 0.0, ctrs) for _, yc in ctrs]
        ck = [costs[(1, nidx[i])] if K == P.K_BASE else cost_of(st, h["cells"], ny, nx) for i, h in enumerate(hk)]
        gz = global_f(rub_k, n1k, S, T)[0]
        row = [K, ck[0][lo_k].min() / 1e6]
        for p0 in P0S:
            u = np.zeros((ny, nx), bool)
            for cc in ck:
                u |= cc <= p0
            u &= rub_k
            row += [u.sum() / float(rub_k.sum()), (hang_k & ~u).sum() / float(hang_k.sum())]
        row = row[:2] + [row[2], row[4], row[3], row[5]]
        row += [gz * p0 / (2 * P.RHO * P.G) for p0 in P0S]
        row += [((ck[1] <= p0) & hang_k).sum() / float(hang_k.sum()) for p0 in P0S]
        ksens.append(row)
        w("  %.2f  %.3f                                 %.4f / %.4f   %.4f / %.4f                    %+.3f / %+.3f"
          "      %.4f / %.4f" % tuple(row))
    w("")

    # ---- 2. fan holes ---------------------------------------------------------------------------------------------
    w("2. Fan holes (base geometry K = %.2f): three per drift at %s deg from +s (negative = down-dip)"
      % (P.K_BASE, " / ".join("%+.0f" % a for a in FAN)))
    w("  drift  angle  direction (dX, dZ)  status")
    for fi, h in enumerate(fan):
        d = h["dir"]
        if h["status"] == "ok":
            w("   %d     %+3.0f    (%+.4f, %+.4f)   ok: collar X %.2f Z %.2f; standpipe end s %+.3f t %.2f (cell: %s);"
              " enters rubble at %.2f m, rubble ends at %.2f m (%s), grouting to %.2f m, end X %.2f, Z %.2f; cells %d"
              " (rubble %d)" % (h["drift"] + 1, h["angle"], d[0], d[1], h["collar"][0] - P.ORIGIN[0],
                                h["collar"][1] - P.ORIGIN[1], h["s_pipe_end"], h["t_pipe_end"], h["pipe_cell"],
                                h["u_in"], h["u_out"], h["stop"], h["u_end"], h["end"][0] - P.ORIGIN[0],
                                h["end"][1] - P.ORIGIN[1], h["n_cells"], h["rubble_cells"]))
        elif h["status"] == "drift":
            w("   %d     %+3.0f    (%+.4f, %+.4f)   DROPPED: the rock section touches a drift (centre X %.2f, Z %.2f)"
              " up to %.2f m from the collar" % (h["drift"] + 1, h["angle"], d[0], d[1],
                                                 h["drift_hit"][0] - P.ORIGIN[0], h["drift_hit"][1] - P.ORIGIN[1],
                                                 h["u_conf"]))
        else:
            w("   %d     %+3.0f    (%+.4f, %+.4f)   DROPPED: %s" % (h["drift"] + 1, h["angle"], d[0], d[1], h["status"]))
    ok = [fi for fi, h in enumerate(fan) if h["status"] == "ok"]
    w("  holes kept: %d of %d" % (len(ok), len(fan)))
    w("  scen drift angle  basal entry p_req (MPa)  lower end p_req (MPa)  reach/rubble 1 / 2 MPa  hanging-side reach 1 / 2 MPa")
    for k in (1, 2):
        for fi in ok:
            h, cost = fan[fi], costs[(k, fi)]
            hc = np.zeros(ny * nx, bool)
            hc[h["cells"]] = True
            hc = hc.reshape(ny, nx)
            entry = cost[basal & hc].min() / 1e6 if (basal & hc).any() else float("nan")
            w("   %d    %d    %+3.0f    %s                    %.3f                  %.4f / %.4f          %.4f / %.4f"
              % ((k, h["drift"] + 1, h["angle"], ("%.3f" % entry) if entry == entry else "  - (no basal cell)",
                  cost[low_end].min() / 1e6) +
                 tuple(((cost <= p0) & rub).sum() / float(rub.sum()) for p0 in P0S) +
                 tuple(((cost <= p0) & hang).sum() / float(hang.sum()) for p0 in P0S)))
    w("")
    U_f = union_block("fan holes (all kept holes)", ok, "fan")
    w("")
    w("  corner check: the largest uncovered component of the normal-hole union (per scenario and p0) and how much of it"
      " the -30 deg hole of drift 1 alone, and the whole fan, covers")
    i30 = [fi for fi in ok if fan[fi]["drift"] == 0 and fan[fi]["angle"] == -30.0]
    v2 = None
    v2p = os.path.join(OUT, "prejudge.npz")
    if os.path.isfile(v2p):
        v2 = np.load(v2p)["union"]
    for k in (1, 2):
        for p0 in P0S:
            comp = U_n[(k, p0, "comp")]
            mk = comp[0][-1]
            line = "   scenario %d, %.0f MPa: component %.2f m2, centroid (%.2f, %.2f), X %.2f..%.2f, Z %.2f..%.2f" % (
                (k, p0 / 1e6) + comp[0][:7])
            if i30:
                c30 = costs[(k, i30[0])] <= p0
                line += "; covered by the drift-1 -30 deg hole %.4f, by the fan union %.4f" % (
                    (mk & c30).sum() / float(mk.sum()), (mk & U_f[(k, p0)][0]).sum() / float(mk.sum()))
            w(line)
            if v2 is not None and i30 and p0 == P0S[-1]:
                mv2 = rub & ~v2[(k - 1) * 2 + list(P0S).index(p0)]
                lab, nl = ndimage.label(mv2, structure=np.ones((3, 3), int))
                big = lab == (1 + int(np.argmax([(lab == q).sum() for q in range(1, nl + 1)])))
                c30 = costs[(k, i30[0])] <= p0
                w("      step-1 v2 largest uncovered component at 2 MPa (%.2f m2, v2 porosity; same base geometry): covered"
                  " under v3 by the drift-1 -30 deg hole %.4f, by the fan union %.4f, by the v3 normal union %.4f"
                  % (big.sum() * a2, (big & c30).sum() / float(big.sum()),
                     (big & U_f[(k, p0)][0]).sum() / float(big.sum()), (big & U_n[(k, p0)][0]).sum() / float(big.sum())))
    w("")

    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "step2a.txt"), "w").write(text)

    def pack(hs):
        cells = np.concatenate([h["cells"] for h in hs])
        ptr = np.cumsum([0] + [len(h["cells"]) for h in hs])
        line = np.array([[h["collar"][0] - P.ORIGIN[0], h["collar"][1] - P.ORIGIN[1], h["end"][0] - P.ORIGIN[0],
                          h["end"][1] - P.ORIGIN[1]] for h in hs])
        return cells, ptr, line
    hn = [fan[fi] for fi in nidx]
    hf = [fan[fi] for fi in ok]
    n_cells, n_ptr, n_line = pack(hn)
    f_cells, f_ptr, f_line = pack(hf)
    np.savez_compressed(os.path.join(DATA, "base_geometry_K1.25_v3.npz"), cls=cls, X=XR, Z=ZR, S=S, T=T, n1=n1, n2=n2,
                        cell=CELL, origin_model=np.array(P.ORIGIN), domain_model=np.array(P.DOM), basal=basal,
                        hanging=hang, void_boundary_faces=vb_faces, hole_cells=n_cells, hole_ptr=n_ptr,
                        hole_line=n_line, fan_cells=f_cells, fan_ptr=f_ptr, fan_line=f_line,
                        fan_drift=np.array([h["drift"] + 1 for h in hf]), fan_angle=np.array([h["angle"] for h in hf]),
                        mapping_c=FK[P.K_BASE]["c"],
                        classes=np.array(["rubble", "footwall", "hanging wall", "ore pillar", "void", "drift"]),
                        note=np.array("v3 (step 2A): proportional mapping n1 = c n_smooth(s), mean n1 = 1 - 1/K = 0.20;"
                                      " n2 = clip(n1 + Delta n(t), 0.10, 0.35). cls codes 0..5 as in classes; rock, void,"
                                      " drift n = 1e-3. hole_* = the normal holes (step 1 v2 layout), fan_* = all kept"
                                      " fan holes (drift no. fan_drift, angle fan_angle deg from +s, negative down-dip);"
                                      " cells[ptr[i]:ptr[i+1]] = flat cell indices iy * nx + ix; X, Z cell centres"
                                      " relative to the origin"))
    np.savez_compressed(os.path.join(OUT, "step2a.npz"), cls=cls, X=XR, Z=ZR, S=S, n1=n1, n2=n2, gam1=gam[1],
                        gam2=gam[2], zs=G["zs"], ns=G["ns"], ar=G["ar"], sm_all=sm_all, zst_v2=zst_v2,
                        mapped_v2=mapped_v2, zs_x=zs_x, ns_x=ns_x, ar_x=ar_x, sm_x=sm_x, zst_x=zst_x,
                        c_base=FK[P.K_BASE]["c"], basal=basal, hang=hang,
                        cost=np.array([costs[(k, fi)] for k in (1, 2) for fi in nidx]),
                        holes=n_line, fan=f_line, fan_angle=np.array([h["angle"] for h in hf]),
                        drifts=np.array([[xc - P.ORIGIN[0], yc - P.ORIGIN[1]] for xc, yc in ctrs]),
                        union=np.array([U_n[(k, p0)][0] for k in (1, 2) for p0 in P0S]),
                        union_fan=np.array([U_f[(k, p0)][0] for k in (1, 2) for p0 in P0S]),
                        ksens=np.array(ksens))
    print(text)
    return 0


# ---- 3. solver timing probe ------------------------------------------------------------------------------------------
PROBE_P0 = 2.0e6
PROBE_DTCAP = 5.0
N_REF = 0.20                      # 1 - 1/K for K = 1.25: the area mean of n1


def aggregate(cls, n1, cells, f):
    """0.25 m -> 0.25 f m: a coarse cell is rubble when at least half of its f x f fine cells are rubble; its porosity
    is the mean n1 of those rubble cells; every other coarse cell gets n = 1e-3. Source = the coarse cells that
    contain a fine source cell and are rubble."""
    ny, nx = cls.shape
    cy, cx = ny // f, nx // f
    r = (cls == RUB)[:cy * f, :cx * f].reshape(cy, f, cx, f)
    nr = r.sum(axis=(1, 3))
    nsum = np.where(r, n1[:cy * f, :cx * f].reshape(cy, f, cx, f), 0.0).sum(axis=(1, 3))
    rub_c = nr * 2 >= f * f
    n_c = np.where(rub_c, nsum / np.maximum(nr, 1), ROCK_N)
    fy, fx = np.asarray(cells) // nx, np.asarray(cells) % nx
    src = sorted(set((j // f) * cx + (i // f) for j, i in zip(fy, fx) if rub_c[j // f, i // f]))
    return n_c, rub_c, np.array(src, int)


def probe_state(cell, n_c, src):
    mesh, x, y, fx, fy = build_mesh_for_domain(P.DOM[0], P.DOM[1], P.DOM[2], P.DOM[3], cell)
    params = Z.slurry_params()
    params["inlet_pressure_core_value"] = PROBE_P0
    st = {"mesh": mesh, "x": x, "y": y, "fx": fx, "fy": fy}
    st.update(initialize_slurry_variables(mesh, params=params))
    Z.set_porosity(st, n_c.ravel())
    m = np.zeros(mesh.numberOfCells)
    m[src] = 1.0
    st["interior_dirichlet_mask"] = CellVariable(mesh=mesh, value=m)
    st["interior_dirichlet_value"] = PROBE_P0
    st["saturation"].setValue(m.copy())
    st["pressure"].setValue(np.where(m > 0.5, PROBE_P0, 0.0))
    return st, params


def run_steps(st, src, nsteps, w, label):
    vols = np.asarray(st["mesh"].cellVolumes, dtype=float)
    hm = np.zeros(vols.size, bool)
    hm[src] = True
    t, v_in, rec = 0.0, 0.0, []
    for step in range(nsteps):
        w0 = time.time()
        dt = solve_transport_step(st, dt_cap=PROBE_DTCAP)
        wall = time.time() - w0
        t += float(dt)
        dq = np.asarray(st["last_div_q"], dtype=float)
        v_in += float(np.sum(dq[hm] * vols[hm])) * float(dt)
        s = np.array(st["saturation"].value, copy=True)
        s[hm] = 1.0
        st["saturation"].setValue(s)
        nf = int(((s >= 0.5) & ~hm).sum())
        rec.append((step + 1, wall, float(dt), t, v_in, nf))
        print("  [%s] step %4d  wall %7.2f s  dt %.3e s  t %.3e s" % (label, step + 1, wall, dt, t))
        sys.stdout.flush()
    return rec


def probe():
    L = []

    def w(s):
        L.append(s)
        print(s)
        sys.stdout.flush()
    D = np.load(os.path.join(DATA, "base_geometry_K1.25_v3.npz"))
    cls, n1 = D["cls"], D["n1"]
    hp = D["hole_ptr"]
    cells2 = D["hole_cells"][hp[1]:hp[2]]
    sp = Z.slurry_params()
    w("Step 2A, solver timing probe: hole 2 normal, scenario 1, p0 %.0f MPa, v0.6, stop rule (4-29) eps %.2g beta %.2g"
      " n_ref %.2f (not reached in a probe), dt_cap %.1f s, FIPY_SOLVERS=%s, linear_solver %s"
      % (PROBE_P0 / 1e6, stop_rule.EPS, stop_rule.BETA, N_REF, PROBE_DTCAP, os.environ.get("FIPY_SOLVERS", "?"),
         sp.get("linear_solver", "pcg (default)")))
    qref = stop_rule.q_ref(N_REF, sp["calibrated_permeability_coefficient"], PROBE_P0, sp["plastic_viscosity"])
    w("Q_ref = k(n_ref) p0 / mu_p = %.4e m3/(m s); stop when the window-mean rate < %.4e" % (qref, stop_rule.EPS * qref))
    res = {}
    for cell, f, nsteps in ((0.5, 2, 100), (0.25, 1, 10)):
        n_c, rub_c, src = aggregate(cls, n1, cells2, f) if f > 1 else (n1, cls == RUB, np.asarray(cells2))
        st, params = probe_state(cell, n_c, src)
        ncell = st["mesh"].numberOfCells
        cost = np.asarray(reachable_domain(st, src, p0=PROBE_P0)["cost"], float).reshape(n_c.shape)
        reach = (cost <= PROBE_P0) & rub_c
        a2 = cell * cell
        w("")
        w("grid %.2f m: %d cells (%d rubble, mean n %.4f); source %d cells; reach at %.0f MPa %d cells = %.2f m2, pore"
          " %.2f m3/m" % (cell, ncell, rub_c.sum(), n_c[rub_c].mean(), len(src), PROBE_P0 / 1e6, reach.sum(),
                         reach.sum() * a2, (n_c * reach).sum() * a2))
        w0 = time.time()
        rec = run_steps(st, src, nsteps, w, "%.2f m" % cell)
        tot = time.time() - w0
        walls = np.array([r[1] for r in rec])
        w("  %d steps in %.1f s: wall per step mean %.2f, median %.2f, min %.2f, max %.2f s; last-half mean %.2f s"
          % (nsteps, tot, walls.mean(), np.median(walls), walls.min(), walls.max(), walls[nsteps // 2:].mean()))
        w("  step  wall(s)  dt(s)      t(s)")
        for r in rec:
            if nsteps <= 10 or r[0] in (1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
                w("  %4d  %7.2f  %.3e  %.3e" % r[:4])
        res[cell] = {"rec": rec, "reach": int(reach.sum()), "ncell": ncell, "pore": (n_c * reach).sum() * a2}
    r5 = res[0.5]["rec"]
    w("")
    w("extrapolation inputs (0.5 m): filled cells (S >= 0.5, sources excluded) per step over steps 51..100: %.3f;"
      " volume per step %.4e m3/m; dt mean over steps 51..100 %.3e s"
      % ((r5[-1][5] - r5[49][5]) / 50.0, (r5[-1][4] - r5[49][4]) / 50.0, np.mean([r[2] for r in r5[50:]])))
    w("  at step 100: t %.3e s, filled %d cells of %d reachable" % (r5[-1][3], r5[-1][5], res[0.5]["reach"]))
    a_cal, mu = sp["calibrated_permeability_coefficient"], sp["plastic_viscosity"]
    for n in (0.18, 0.20, 0.25):
        M = a_cal * n ** 3 / (1 - n) ** 2 / mu
        lm = PROBE_P0 / P.lam(n)
        for chi in (0.87,):
            tt = n / (M * P.lam(n)) * (lm * math.log(1.0 / (1.0 - chi)) - chi * lm)
            w("  1D front time to x_f = %.2f L_max (step 2b closed form), n %.2f: L_max %.2f m, t %.4e s" % (chi, n, lm, tt))
    open(os.path.join(OUT, "step2a_probe.txt"), "w").write("\n".join(L) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(probe() if "--probe" in sys.argv[1:] else main())
