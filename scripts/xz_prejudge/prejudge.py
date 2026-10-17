# -*- coding: utf-8 -*-
"""Structure pre-judgement of the caved rubble for section 4.8, step 1 (reach and closed forms only, no solver).

    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\prejudge.py

Input: the PFC2D final state (save 50) through scripts/pfc_collapse/analyze.py (same pixel classification;
export directory from $PFC_COLLAPSE_EXPORT). Output: outputs/xz_prejudge/prejudge.txt and prejudge.npz
(for plot.py).

Grid 0.5 m. Coordinates: X = x + 14, Z = y - 40 (origin = the footwall corner of the original stope's lower
end, Z up); s = normal distance from the footwall contact (towards the hanging wall), t = up-dip distance
from the same corner. A 0.5 m cell takes the class of its 10 x 10 pixels: void if >= half void, otherwise
rubble if the rubble-region share >= the rock share, otherwise rock (side of the largest rock share).
Void components touching no rock cell (cavities enclosed by rubble) are turned into rubble.
Rock (footwall, hanging wall, ore pillar) n = 1e-3; residual void n = 1e-3 too, i.e. effectively a no-flow
region for the reach graph (a 0.5 m step through it costs 30 MPa).

Porosity: the PFC particle-scale s-profile (1.5 m slabs, whole rubble region, as analyze.py) smoothed by a
5-point moving average (window shrinks at the ends), its min / max mapped linearly to 0.13 / 0.29 -> n_s(s)
(ends held beyond the profile). Scenario 1: n = n_s(s). Scenario 2: n = n_s(s) + Delta(t), Delta linear
from -0.08 at the lower end to +0.08 at the upper end of the rubble (t range of rubble cells), clipped to
0.10..0.35.

v0.6 grout: tau0 30 Pa, mu_p 0.15 Pa s, rho 1830 kg/m3, A 1.25e-7 m2 -> lambda(n) = 6.00e4 (1 - n) / n Pa/m,
L_max = p0 / lambda. Local F = gamma p0 / (2 rho g), gamma = -d ln(lambda) / dZ from a least-squares plane on
the rubble cells of a 3 m window (7 x 7 cell centres). p_req by the 16-neighbour reach graph with gravity
(src.analysis.fill_diagnostics.reachable_domain); it does not depend on p0, so each scenario x hole is
priced once and thresholded at p0 = 1, 2, 3, 5 MPa.
"""
from __future__ import print_function

import imp
import math
import os
import sys

import numpy as np
from scipy import ndimage

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

A = imp.load_source("pc_analyze", os.path.join(REPO, "scripts", "pfc_collapse", "analyze.py"))
Z = imp.load_source("case_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.models.slurry_transport.variables import initialize_slurry_variables
from src.analysis.fill_diagnostics import reachable_domain

OUT = os.path.join(REPO, "outputs", "xz_prejudge")
CELL = 0.5
DOM = (-53.0, 64.0, 29.0, 92.0)         # model x, y; >= 10 m of rock around the rubble region (checked below)
ORIGIN = (-14.0, 40.0)
P0S = (1.0e6, 2.0e6, 3.0e6, 5.0e6)
N_LO, N_HI = 0.13, 0.29
DELTA = 0.08
CLIP = (0.10, 0.35)
ROCK_N = 1.0e-3
BASAL = 3.0                               # loose footwall layer, s <= 3 m
PIPE = 2.0
COLLAR_S = -5.0
STOP_GAP = 1.0
HOLE_F = (1.0 / 6.0, 0.5, 5.0 / 6.0)
RHO, G = 1830.0, 9.81
LAM0 = 6.00e4
SN, CS = math.sin(math.radians(30.0)), math.cos(math.radians(30.0))
CLS = {"rubble": 0, "footwall": 1, "hanging wall": 2, "ore pillar": 3, "void": 4}


def lam(n):
    return LAM0 * (1.0 - n) / n


def t_coord(x, y):
    return (x - ORIGIN[0]) * CS + (y - ORIGIN[1]) * SN


def cells_from_pfc():
    A.BOX = DOM
    b, m = A.load("local_stope_result_50")
    d50 = float(np.median(2 * b["radius"]))
    rub = np.hypot(b["dx"], b["dy"]) > A.DISP_RUBBLE
    img = A.raster(b, rub)
    c = A.classify(img, d50)
    k = int(round(CELL / A.H))
    ny, nx = c["region"].shape[0] // k, c["region"].shape[1] // k

    def frac(mask):
        return mask[:ny * k, :nx * k].reshape(ny, k, nx, k).mean(axis=(1, 3))
    f_void = frac(c["void"])
    f_reg = frac(c["region"])
    f_side = [frac(c["rock"] & (c["rock_side"] == sd)) for sd in (1, 2, 3)]
    f_rock = f_side[0] + f_side[1] + f_side[2]
    cls = np.full((ny, nx), -1, dtype=int)
    cls[f_void >= 0.5] = CLS["void"]
    rest = cls < 0
    cls[rest & (f_reg >= f_rock)] = CLS["rubble"]
    side = np.argmax(np.array(f_side), axis=0) + 1
    rest = cls < 0
    cls[rest] = side[rest]
    # void components not touching rock -> rubble
    lab, nl = ndimage.label(cls == CLS["void"])
    rock = (cls >= 1) & (cls <= 3)
    rock_adj = ndimage.binary_dilation(rock)
    enclosed = []
    for q in range(1, nl + 1):
        mk = lab == q
        if not (mk & rock_adj).any():
            enclosed.append(mk.sum())
            cls[mk] = CLS["rubble"]
    xs = DOM[0] + (np.arange(nx) + 0.5) * CELL
    ys = DOM[2] + (np.arange(ny) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys)
    Sp = A.s_coord(*np.meshgrid(*A.grid()))
    zs, ns, ar = A.profile(c, Sp, -2.0, 44.0, A.SLAB_D * d50)
    return cls, X, Y, zs, ns, ar, enclosed, d50, c


def smooth5(v):
    out = np.empty_like(v)
    for i in range(len(v)):
        out[i] = v[max(i - 2, 0):i + 3].mean()
    return out


def window_gamma(cls, lnlam, half=3):
    """gamma = -d ln(lambda)/dZ from a plane fitted to the rubble cells of a (2 half + 1)^2 window."""
    M = (cls == CLS["rubble"]).astype(float)
    off = np.arange(-half, half + 1) * CELL
    DX, DZ = np.meshgrid(off, off)
    corr = lambda a, kern: ndimage.correlate(a, kern, mode="constant", cval=0.0)
    L = np.where(M > 0, lnlam, 0.0)
    S1, Sx, Sz = corr(M, np.ones_like(DX)), corr(M, DX), corr(M, DZ)
    Sxx, Szz, Sxz = corr(M, DX * DX), corr(M, DZ * DZ), corr(M, DX * DZ)
    Sy, Sxy, Szy = corr(L, np.ones_like(DX)), corr(L, DX), corr(L, DZ)
    def det3(a, b, c, d, e, f, g, h, i):
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    det = det3(S1, Sx, Sz, Sx, Sxx, Sxz, Sz, Sxz, Szz)
    ok = (M > 0) & (S1 >= 10) & (np.abs(det) > 1e-6)
    # Cramer's rule for the Z coefficient (third unknown)
    dz = det3(S1, Sx, Sy, Sx, Sxx, Sxy, Sz, Sxz, Szy)
    gam = np.where(ok, -dz / np.where(ok, det, 1.0), np.nan)
    return gam, ok


def hole(cls, X, Y, tk):
    """Cells of the grouting section; collar at s = COLLAR_S, t = tk, drilled along +s."""
    ny, nx = cls.shape
    ex, ey = -SN, CS
    # collar: point with t = tk, s = COLLAR_S; s grows along (ex, ey) from the footwall line
    xc0 = ORIGIN[0] + tk * CS + COLLAR_S * ex
    yc0 = ORIGIN[1] + tk * SN + COLLAR_S * ey
    u = np.arange(0.0, 80.0, 0.02)
    px, py = xc0 + u * ex, yc0 + u * ey
    ix = np.floor((px - DOM[0]) / CELL).astype(int)
    iy = np.floor((py - DOM[2]) / CELL).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cl = np.where(inside, cls[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)], -9)
    first_rub = np.argmax(cl == CLS["rubble"])
    after = np.where((np.arange(len(u)) > first_rub) & (cl != CLS["rubble"]))[0]
    u_out = u[after[0]]
    u_end = u_out - STOP_GAP
    sel = (u >= PIPE) & (u <= u_end)
    cells = sorted(set(zip(iy[sel], ix[sel])))
    flat = np.array([j * nx + i for j, i in cells])
    stop_cls = [k for k, v in CLS.items() if v == cl[after[0]]][0]
    return {"collar": (xc0, yc0), "end": (xc0 + u_end * ex, yc0 + u_end * ey), "u_in": u[first_rub],
            "u_out": u_out, "u_end": u_end, "stop": stop_cls, "cells": flat,
            "rubble_cells": int(sum(cls[j, i] == CLS["rubble"] for j, i in cells)), "n_cells": len(cells)}


def make_state(mesh, phi):
    params = Z.slurry_params()
    st = {"mesh": mesh}
    st.update(initialize_slurry_variables(mesh, params=params))
    Z.set_porosity(st, phi)
    return st


def main():
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    L = []
    w = L.append
    cls, X, Y, zs, ns, ar, enclosed, d50, c = cells_from_pfc()
    ny, nx = cls.shape
    rubm = cls == CLS["rubble"]
    S = A.s_coord(X, Y)
    T = t_coord(X, Y)
    XR, ZR = X - ORIGIN[0], Y - ORIGIN[1]
    w("Rubble structure pre-judgement, step 1 (reach and closed forms; v0.6 grout; 2D)")
    w("grid %.1f m, %d x %d cells; domain X %.1f..%.1f, Z %.1f..%.1f m (model x %.1f..%.1f, y %.1f..%.1f)"
      % ((CELL, nx, ny, DOM[0] - ORIGIN[0], DOM[1] - ORIGIN[0], DOM[2] - ORIGIN[1], DOM[3] - ORIGIN[1]) + DOM))
    w("origin = footwall corner of the original stope lower end (model (%.1f, %.1f)); Z up; s normal from the"
      " footwall contact; t up-dip" % ORIGIN)
    w("")
    w("1. Masks")
    for nm in ("rubble", "footwall", "hanging wall", "ore pillar", "void"):
        w("  %-12s %6d cells  %8.2f m2" % (nm, (cls == CLS[nm]).sum(), (cls == CLS[nm]).sum() * CELL * CELL))
    w("  enclosed cavities turned into rubble: %d component(s), %s cells" % (len(enclosed), ", ".join(str(q) for q in enclosed)))
    jj, ii = np.where(rubm)
    w("  rubble cells: X %.2f..%.2f, Z %.2f..%.2f m; margin to the domain edge: left %.2f, right %.2f, bottom %.2f,"
      " top %.2f m" % (XR[rubm].min(), XR[rubm].max(), ZR[rubm].min(), ZR[rubm].max(),
                       XR[rubm].min() - (DOM[0] - ORIGIN[0]) - CELL / 2, (DOM[1] - ORIGIN[0]) - XR[rubm].max() - CELL / 2,
                       ZR[rubm].min() - (DOM[2] - ORIGIN[1]) - CELL / 2, (DOM[3] - ORIGIN[1]) - ZR[rubm].max() - CELL / 2))
    w("  rubble s %.2f..%.2f m, t %.2f..%.2f m" % (S[rubm].min(), S[rubm].max(), T[rubm].min(), T[rubm].max()))
    vmask = cls == CLS["void"]
    vb = np.zeros_like(rubm)
    faces = np.zeros(rubm.shape, dtype=int)
    for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.roll(np.roll(vmask, dj, 0), di, 1)
        if dj == 1:
            nb[0, :] = False
        if dj == -1:
            nb[-1, :] = False
        if di == 1:
            nb[:, 0] = False
        if di == -1:
            nb[:, -1] = False
        faces += (rubm & nb).astype(int)
    vb = faces > 0
    w("  rubble cells on the void boundary: %d cells, %d faces = %.1f m of boundary" % (vb.sum(), faces.sum(), faces.sum() * CELL))
    w("  domain void cells are given n = %.0e (no-flow for the reach graph)" % ROCK_N)
    w("")

    # ---- 2. porosity scenarios -----------------------------------------------------------------------
    w("2. Porosity mapping")
    sm = smooth5(ns)
    mmin, mmax = sm.min(), sm.max()
    mapped = N_LO + (sm - mmin) / (mmax - mmin) * (N_HI - N_LO)
    w("  PFC particle-scale s-profile (1.5 m slabs, whole rubble region), 5-point moving average, linear map"
      " min %.4f -> %.2f, max %.4f -> %.2f" % (mmin, N_LO, mmax, N_HI))
    w("  slab s / area m2 / PFC / smoothed / mapped:")
    for q in zip(zs, ar, ns, sm, mapped):
        w("    %6.2f  %6.1f  %.4f  %.4f  %.4f" % q)
    w("  smoothed minimum at s = %.2f m, maximum at s = %.2f m" % (zs[np.argmin(sm)], zs[np.argmax(sm)]))
    ns_cell = np.interp(S, zs, mapped)
    n1 = np.where(rubm, ns_cell, ROCK_N)
    tmin, tmax = T[rubm].min(), T[rubm].max()
    delta = -DELTA + 2 * DELTA * (T - tmin) / (tmax - tmin)
    raw2 = ns_cell + delta
    n2 = np.where(rubm, np.clip(raw2, CLIP[0], CLIP[1]), ROCK_N)
    lo_c = (rubm & (raw2 < CLIP[0])).sum()
    hi_c = (rubm & (raw2 > CLIP[1])).sum()
    w("  scenario 2: Delta from -%.2f at t %.2f to +%.2f at t %.2f m; clipped cells: %d below %.2f, %d above %.2f"
      " (of %d rubble cells)" % (DELTA, tmin, DELTA, tmax, lo_c, CLIP[0], hi_c, CLIP[1], rubm.sum()))
    scen = {1: n1, 2: n2}
    s13 = S[rubm].min() + 2.0 / 3.0 * (S[rubm].max() - S[rubm].min())
    zone = {"basal (s <= 3 m)": rubm & (S <= BASAL), "interior": rubm & (S > BASAL) & (S <= s13),
            "hanging side (s > %.2f m)" % s13: rubm & (S > s13)}
    for k in (1, 2):
        nv = scen[k][rubm]
        w("  scenario %d: rubble porosity min / median / mean / max %.4f / %.4f / %.4f / %.4f; " % (k, nv.min(), np.median(nv), nv.mean(), nv.max())
          + "; ".join("%s median %.4f" % (zn, np.median(scen[k][zm])) for zn, zm in sorted(zone.items())))
    w("")

    # ---- 3. closed forms -------------------------------------------------------------------------------
    w("3. L_max, local F, gradient decomposition")
    res3 = {}
    for k in (1, 2):
        lmb = lam(scen[k])
        for p0 in P0S:
            Lm = p0 / lmb[rubm]
            row = [Lm.min(), np.median(Lm), Lm.max()] + [np.median(p0 / lmb[zm]) for zn, zm in sorted(zone.items())]
            res3[(k, p0)] = row
            w("  scenario %d, p0 %.0f MPa: L_max min / median / max %.3f / %.3f / %.3f m; zone medians (%s) %s;"
              " min L_max / dx %.2f%s" % ((k, p0 / 1e6) + tuple(row[:3]) + (", ".join(sorted(zone)),
                                          " / ".join("%.3f" % v for v in row[3:]), row[0] / CELL,
                                          "  < 7" if row[0] / CELL < 7 else "")))
    gam = {}
    for k in (1, 2):
        g_, ok = window_gamma(cls, np.log(lam(np.where(rubm, scen[k], 0.5))))
        gam[k] = g_
        gv = g_[rubm & ok]
        w("  scenario %d: gamma = -dln(lambda)/dZ over %d cells (3 m window): percentiles 5/25/50/75/95 %s 1/m;"
          " F < 0 share %.4f" % (k, gv.size, " / ".join("%+.4f" % q for q in np.percentile(gv, [5, 25, 50, 75, 95])),
                                 (gv < 0).mean()))
        for p0 in P0S:
            F = gv * p0 / (2 * RHO * G)
            w("    p0 %.0f MPa: F percentiles 5/25/50/75/95 %s; area share F > 1 %.4f, F < 0 %.4f"
              % (p0 / 1e6, " / ".join("%+.3f" % q for q in np.percentile(F, [5, 25, 50, 75, 95])), (F > 1).mean(), (F < 0).mean()))
    both = rubm & np.isfinite(gam[1]) & np.isfinite(gam[2])
    w("  cell-wise sign agreement of F between the scenarios: %.4f of %d cells (same sign or both zero)"
      % ((np.sign(gam[1][both]) == np.sign(gam[2][both])).mean(), both.sum()))
    for zn, zm in sorted(zone.items()):
        w("    %-26s F < 0 share: scenario 1 %.4f, scenario 2 %.4f; median gamma %+.4f / %+.4f 1/m"
          % (zn, (gam[1][zm & both] < 0).mean(), (gam[2][zm & both] < 0).mean(), np.median(gam[1][zm & both]),
             np.median(gam[2][zm & both])))
    for k in (1, 2):
        ll = np.log(lam(scen[k][rubm]))
        Mx = np.column_stack([np.ones(ll.size), T[rubm], S[rubm]])
        cf = np.linalg.lstsq(Mx, ll)[0]
        gz = -(cf[1] * SN + cf[2] * CS)
        w("  scenario %d: whole-rubble plane fit of ln(lambda) on t, s: vertical gamma %+.5f 1/m -> F %s (p0 1/2/3/5 MPa)"
          % (k, gz, " / ".join("%+.3f" % (gz * p0 / (2 * RHO * G)) for p0 in P0S)))
    for k in (1, 2):
        nv = scen[k][rubm]
        Mx = np.column_stack([np.ones(nv.size), T[rubm], S[rubm]])
        coef, _, _, _ = np.linalg.lstsq(Mx, nv)
        r = nv - Mx.dot(coef)
        r2 = 1 - (r ** 2).sum() / ((nv - nv.mean()) ** 2).sum()
        w("  scenario %d: n = a + b_t t + b_s s over rubble cells: b_t %+.5f /m, b_s %+.5f /m, R2 %.3f;"
          " b_s cos30 %+.5f vs b_t sin30 %+.5f; vertical gradient %+.5f /m"
          % (k, coef[1], coef[2], r2, coef[2] * CS, coef[1] * SN, coef[2] * CS + coef[1] * SN))
    w("  (PFC particle scale, analyze.py: b_s -0.00182, b_t -0.00051)")
    w("")

    # ---- 4. holes and reach ----------------------------------------------------------------------------
    w("4. Holes, required source pressure and reach")
    basal = zone["basal (s <= 3 m)"]
    tb = T[basal]
    tlo, thi = tb.min(), tb.max()
    w("  hole positions along the footwall: t range of the basal layer %.2f..%.2f m; holes at 1/6, 1/2, 5/6" % (tlo, thi))
    low_end = basal & (T <= tlo + 2.0)
    w("  lower end of the basal layer (target): basal cells with t <= %.2f m: %d cells, Z %.2f..%.2f m"
      % (tlo + 2.0, low_end.sum(), ZR[low_end].min(), ZR[low_end].max()))
    hang = zone[[q for q in zone if q.startswith("hanging")][0]]
    holes = []
    for f in HOLE_F:
        h = hole(cls, X, Y, tlo + f * (thi - tlo))
        holes.append(h)
        w("  hole t %.2f m: collar (X %.2f, Z %.2f), end (X %.2f, Z %.2f); enters rubble at %.2f m, rubble ends at %.2f m"
          " (%s), stops at %.2f m; grouting cells %d (rubble %d)"
          % (tlo + f * (thi - tlo), h["collar"][0] - ORIGIN[0], h["collar"][1] - ORIGIN[1], h["end"][0] - ORIGIN[0],
             h["end"][1] - ORIGIN[1], h["u_in"], h["u_out"], h["stop"], h["u_end"], h["n_cells"], h["rubble_cells"]))
    mesh = build_mesh_for_domain(DOM[0], DOM[1], DOM[2], DOM[3], CELL)[0]
    costs = {}
    w("  scen hole  p0   reach/rubble  basal entry p_req  to lower end p_req/p0 (any path; basal only)  hanging-side reach"
      "  void contact")
    for k in (1, 2):
        st = make_state(mesh, scen[k].ravel())
        nb_only = np.where(basal, scen[k], ROCK_N)
        st_b = make_state(mesh, nb_only.ravel())
        for hi, h in enumerate(holes):
            cost = np.asarray(reachable_domain(st, h["cells"], p0=P0S[-1], use_gravity=True)["cost"], float).reshape(ny, nx)
            src_b = np.array([q for q in h["cells"] if basal.ravel()[q]])
            cost_b = np.asarray(reachable_domain(st_b, src_b, p0=P0S[-1], use_gravity=True)["cost"], float).reshape(ny, nx)
            costs[(k, hi)] = cost
            hc = np.zeros(ny * nx, bool)
            hc[h["cells"]] = True
            hc = hc.reshape(ny, nx)
            entry = cost[basal & hc].min() if (basal & hc).any() else float("nan")
            lowv = cost[low_end].min()
            lowb = cost_b[low_end].min()
            for p0 in P0S:
                rch = (cost <= p0) & rubm
                vc = (faces * (rch & vb)).sum() * CELL
                w("  %d    %d   %.0f MPa   %.4f       %.3f MPa            %.3f ; %.3f                        %.4f"
                  "            %s" % (k, hi + 1, p0 / 1e6, rch.sum() / float(rubm.sum()), entry / 1e6, lowv / p0, lowb / p0,
                                     ((cost <= p0) & hang).sum() / float(hang.sum()),
                                     ("yes, %.1f m" % vc) if vc > 0 else "no"))
            w("          p_req to the lower end of the basal layer: %.3f MPa (any path), %.3f MPa (basal layer only)"
              % (lowv / 1e6, lowb / 1e6))
    w("")
    w("  resolution: min L_max / dx over the rubble (dx = %.1f m); < 7 marked" % CELL)
    for k in (1, 2):
        w("   scenario %d: " % k + "; ".join("%.0f MPa %.2f%s" % (p0 / 1e6, res3[(k, p0)][0] / CELL,
                                                                 " (<7)" if res3[(k, p0)][0] / CELL < 7 else "") for p0 in P0S))
    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "prejudge.txt"), "w").write(text)
    np.savez_compressed(os.path.join(OUT, "prejudge.npz"), cls=cls, X=XR, Z=ZR, S=S, T=T, n1=n1, n2=n2,
                        gam1=gam[1], gam2=gam[2], zs=zs, ns=ns, sm=sm, mapped=mapped, ar=ar,
                        cost=np.array([costs[(k, hi)] for k in (1, 2) for hi in range(3)]),
                        holes=np.array([[h["collar"][0] - ORIGIN[0], h["collar"][1] - ORIGIN[1], h["end"][0] - ORIGIN[0],
                                         h["end"][1] - ORIGIN[1]] for h in holes]), vb=vb, basal=basal, low_end=low_end,
                        hang=hang, s13=s13)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
