# -*- coding: utf-8 -*-
"""Structure pre-judgement of the caved rubble for section 4.8, step 1, request v2 + drift supplement
(reach and closed forms only, no solver).

    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\prejudge.py

Input: the PFC2D final state (save 50) through scripts/pfc_collapse/analyze.py (same pixel classification;
export directory from $PFC_COLLAPSE_EXPORT). Output: outputs/xz_prejudge/prejudge.txt, prejudge.npz (figures)
and data/xz_prejudge/base_geometry_K1.25.npz (base geometry + both porosity fields, for step 2).

Grid 0.25 m. Coordinates X = x + 14, Z = y - 40 (origin = the footwall corner of the original stope's lower end,
Z up); s = normal distance from the footwall contact (towards the hanging wall), t = up-dip distance from the
same corner. A 0.25 m cell (5 x 5 pixels) belongs to the cavity when the PFC rubble-region and void pixels make
at least half of it, otherwise it is rock of the side with the largest share. Cavity = PFC rubble region + all
residual voids.

Rubble extent by bulking: A0 = PFC rubble solid area / (1 - 0.1416). y_top(x) = top of the PFC rubble in each
0.25 m column of the cavity; new rubble = cavity cells with y <= y_top(x) + D, D bisected so that the area is
the first one >= K A0 (K = 1.25 base; 1.15 and 1.40 for the sensitivity check). The rest of the cavity is
residual void: n = 1e-3, i.e. a no-flow region for the reach graph; rock and drifts also n = 1e-3.

Porosity: PFC particle-scale s-profile (1.5 m slabs, analyze.py), 5-point moving average (window shrinks at
the ends). Slabs with centre s <= 3 m keep their s; the rest are stretched linearly in s so that the PFC
rubble's maximum s maps to the new rubble's maximum s. Min / max of the smoothed profile -> 0.13 / 0.29 =
n_s(s); ends held. Scenario 1 n = n_s(s); scenario 2 adds Delta n(t), linear from -0.08 at the lower end to
+0.08 at the upper end of the new rubble, clipped to 0.10..0.35.

Drifts: footwall drifts every 10 m from the original stope floor (y = 40) upwards; centre 5 m horizontally
from the footwall contact at its own level, 2.5 m x 2.5 m impermeable square. Collar = mid-point of the drift
wall facing the orebody (the -x wall); hole along +s (normal to the orebody, up towards the hanging wall);
2 m standpipe, grouting from 2 m to 1 m short of the first non-rubble cell after the hole has entered the
rubble. Drifts stop at the first one whose hole misses the rubble.

v0.6 grout: lambda(n) = 6.00e4 (1 - n) / n Pa/m, L_max = p0 / lambda; F = gamma p0 / (2 rho g),
gamma = -d ln(lambda)/dZ from a least-squares plane on the rubble cells of a 3 m window (13 x 13 centres,
>= 40 rubble cells). p_req from the 16-neighbour reach graph with gravity (independent of p0).
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
XC, CFG = A.XC, A.CFG              # geometry configuration ($XZ_CONFIG, default 40m)
Z = imp.load_source("case_section", os.path.join(REPO, "cases", "zhaojin_section.py"))
from src.fipy_adapter.mesh_init import build_mesh_for_domain
from src.models.slurry_transport.variables import initialize_slurry_variables
from src.analysis.fill_diagnostics import reachable_domain

OUT = XC.out(CFG, "outputs", "xz_prejudge")
DATA = XC.out(CFG, "data", "xz_prejudge")
CELL = 0.25
DOM = tuple(CFG["domain"])               # model x, y: >= 10 m of rock around the cavity (checked below)
ORIGIN = tuple(CFG["stope"]["origin"])    # footwall corner of the stope floor
N_HW = CFG["n_hw"]                        # PFC report: intact hanging-wall disc-assembly porosity
KS = (1.15, 1.25, 1.40)
K_BASE = 1.25
P0S = (1.0e6, 2.0e6)
P0_REF = 5.0e6
N_LO, N_HI = 0.13, 0.29
DELTA = 0.08
CLIP = (0.10, 0.35)
ROCK_N = 1.0e-3
BASAL = CFG["basal_s"]
LOW_DT = CFG["low_end_dt"]                # lower end of the basal layer: t <= min t + LOW_DT
PIPE = 2.0
STOP_GAP = 1.0
DRIFT_DX = CFG["drifts"]["dx"]            # drift centre to footwall contact, horizontal
DRIFT_HALF = CFG["drifts"]["half"]
DRIFT_DY = CFG["drifts"]["spacing"]
DRIFT_N = CFG["drifts"]["count"]          # None: up to the first drift whose hole misses the rubble
RHO, G = 1830.0, 9.81
LAM0 = 6.00e4
SN, CS = math.sin(math.radians(CFG["stope"]["dip_deg"])), math.cos(math.radians(CFG["stope"]["dip_deg"]))
COT = CS / SN
RUB, FW, HW, PIL, VOID, DRIFT = 0, 1, 2, 3, 4, 5
NAME = {RUB: "rubble", FW: "footwall", HW: "hanging wall", PIL: "ore pillar", VOID: "void", DRIFT: "drift"}


def lam(n):
    return LAM0 * (1.0 - n) / n


def s_of(x, y):
    return A.s_coord(x, y)


def t_of(x, y):
    return (x - ORIGIN[0]) * CS + (y - ORIGIN[1]) * SN


def x_fw(y):
    return ORIGIN[0] + (y - ORIGIN[1]) * COT


def pfc_cells():
    A.BOX = DOM
    b, _ = A.load(XC.state(CFG, "final"))
    d50 = float(np.median(2 * b["radius"]))
    rub = np.hypot(b["dx"], b["dy"]) > A.DISP_RUBBLE
    c = A.classify(A.raster(b, rub), d50)
    k = int(round(CELL / A.H))
    ny, nx = c["region"].shape[0] // k, c["region"].shape[1] // k

    def frac(mask):
        return mask[:ny * k, :nx * k].reshape(ny, k, nx, k).mean(axis=(1, 3))
    f_reg, f_void = frac(c["region"]), frac(c["void"])
    f_side = [frac(c["rock"] & (c["rock_side"] == sd)) for sd in (1, 2, 3)]
    cav = (f_reg + f_void) >= 0.5
    pfc_rub = cav & (f_reg >= f_void)
    side = np.argmax(np.array(f_side), axis=0) + 1
    Sp = A.s_coord(*np.meshgrid(*A.grid()))
    zs, ns, ar = A.profile(c, Sp, A.PROF_S[0], A.PROF_S[1], A.SLAB_D * d50)
    solid = float(np.sum(np.pi * b["radius"][rub] ** 2))
    return cav, pfc_rub, side, zs, ns, ar, solid, d50, c


def smooth5(v):
    return np.array([v[max(i - 2, 0):i + 3].mean() for i in range(len(v))])


def rubble_for(cav, ytop, Y, target):
    lo, hi = -5.0, 80.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if (cav & (Y <= ytop[None, :] + mid)).sum() * CELL * CELL >= target:
            hi = mid
        else:
            lo = mid
    return cav & (Y <= ytop[None, :] + hi), hi


def neighbours_count(a, b):
    """Number of 4-neighbour faces between mask a and mask b (per cell of a)."""
    f = np.zeros(a.shape, dtype=int)
    for dj, di in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.zeros_like(b)
        sj = slice(max(dj, 0), b.shape[0] + min(dj, 0))
        si = slice(max(di, 0), b.shape[1] + min(di, 0))
        tj = slice(max(-dj, 0), b.shape[0] + min(-dj, 0))
        ti = slice(max(-di, 0), b.shape[1] + min(-di, 0))
        sh[sj, si] = b[tj, ti]
        f += (a & sh).astype(int)
    return f


def mapping(zs, ns, s_pfc_max, s_new_max):
    sm = smooth5(ns)
    mapped = N_LO + (sm - sm.min()) / (sm.max() - sm.min()) * (N_HI - N_LO)
    zst = np.where(zs <= BASAL, zs, BASAL + (zs - BASAL) * (s_new_max - BASAL) / (s_pfc_max - BASAL))
    return sm, mapped, zst


def zones(rub, S, T):
    basal = rub & (S <= BASAL)
    tb = np.floor(T / 0.5).astype(int)
    hang = np.zeros_like(rub)
    for q in np.unique(tb[rub]):
        m = rub & (tb == q)
        lo, hi = S[m].min(), S[m].max()
        hang |= m & (S > lo + 2.0 / 3.0 * (hi - lo))
    interior = rub & ~basal & ~hang
    return basal, interior, hang


def window_gamma(rub, lnlam, half=6):
    M = rub.astype(float)
    off = np.arange(-half, half + 1) * CELL
    DX, DZ = np.meshgrid(off, off)
    corr = lambda a, kern: ndimage.correlate(a, kern, mode="constant", cval=0.0)
    L = np.where(rub, lnlam, 0.0)
    one = np.ones_like(DX)
    S1, Sx, Sz = corr(M, one), corr(M, DX), corr(M, DZ)
    Sxx, Szz, Sxz = corr(M, DX * DX), corr(M, DZ * DZ), corr(M, DX * DZ)
    Sy, Sxy, Szy = corr(L, one), corr(L, DX), corr(L, DZ)

    def det3(a, b, c, d, e, f, g, h, i):
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    det = det3(S1, Sx, Sz, Sx, Sxx, Sxz, Sz, Sxz, Szz)
    ok = rub & (S1 >= 40) & (np.abs(det) > 1e-9)
    dz = det3(S1, Sx, Sy, Sx, Sxx, Sxy, Sz, Sxz, Szy)
    return np.where(ok, -dz / np.where(ok, det, 1.0), np.nan)


def drift_mask(X, Y, yd):
    xc = x_fw(yd) + DRIFT_DX
    return (np.abs(X - xc) <= DRIFT_HALF) & (np.abs(Y - yd) <= DRIFT_HALF), (xc, yd)


def hole(cls, yd):
    """Hole from the -x wall of the drift at level yd; None if it misses the rubble."""
    ny, nx = cls.shape
    xc = x_fw(yd) + DRIFT_DX
    x0, y0 = xc - DRIFT_HALF, yd
    ex, ey = -SN, CS
    u = np.arange(0.0, 90.0, 0.01)
    px, py = x0 + u * ex, y0 + u * ey
    ix = np.floor((px - DOM[0]) / CELL).astype(int)
    iy = np.floor((py - DOM[2]) / CELL).astype(int)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    cl = np.where(inside, cls[np.clip(iy, 0, ny - 1), np.clip(ix, 0, nx - 1)], -9)
    hit = np.where((cl == RUB) & (u > 0.0))[0]
    if hit.size == 0:
        return None
    first = hit[0]
    after = np.where((np.arange(len(u)) > first) & (cl != RUB))[0]
    u_out = u[after[0]]
    u_end = u_out - STOP_GAP
    sel = (u >= PIPE) & (u <= u_end)
    cells = sorted(set(zip(iy[sel], ix[sel])))
    flat = np.array([j * nx + i for j, i in cells])
    s0 = s_of(x0, y0)
    pipe_cell = cls[int((y0 + PIPE * ey - DOM[2]) / CELL), int((x0 + PIPE * ex - DOM[0]) / CELL)]
    return {"collar": (x0, y0), "end": (x0 + u_end * ex, y0 + u_end * ey), "u_in": u[first], "u_out": u_out,
            "u_end": u_end, "stop": NAME[int(cl[after[0]])], "cells": flat, "t": t_of(x0, y0), "s_collar": s0,
            "s_pipe_end": s0 + PIPE, "pipe_cell": NAME[int(pipe_cell)],
            "rubble_cells": int(sum(cls[j, i] == RUB for j, i in cells)), "n_cells": len(cells)}


def make_state(mesh, phi):
    st = {"mesh": mesh}
    st.update(initialize_slurry_variables(mesh, params=Z.slurry_params()))
    Z.set_porosity(st, phi)
    return st


def geometry(cav, pfc_rub, side, ytop, X, Y, A0, K, drifts):
    rub, d = rubble_for(cav, ytop, Y, K * A0)
    cls = np.where(cav, VOID, side)
    cls[rub] = RUB
    for m in drifts:
        cls[m] = DRIFT
    return cls, rub, d


def main():
    for dd in (OUT, DATA):
        if not os.path.isdir(dd):
            os.makedirs(dd)
    L = []
    w = L.append
    cav, pfc_rub, side, zs, ns, ar, solid, d50, c = pfc_cells()
    ny, nx = cav.shape
    xs = DOM[0] + (np.arange(nx) + 0.5) * CELL
    ys = DOM[2] + (np.arange(ny) + 0.5) * CELL
    X, Y = np.meshgrid(xs, ys)
    S, T = s_of(X, Y), t_of(X, Y)
    XR, ZR = X - ORIGIN[0], Y - ORIGIN[1]
    a2 = CELL * CELL
    w("Rubble structure pre-judgement, step 1 (request v2 + drift supplement; reach and closed forms; v0.6; 2D)")
    w("grid %.2f m, %d x %d cells; domain X %.2f..%.2f, Z %.2f..%.2f m (model x %.1f..%.1f, y %.1f..%.1f)"
      % ((CELL, nx, ny, DOM[0] - ORIGIN[0], DOM[1] - ORIGIN[0], DOM[2] - ORIGIN[1], DOM[3] - ORIGIN[1]) + DOM))
    w("origin = footwall corner of the original stope lower end, model (%.1f, %.1f); Z up; s normal from the"
      " footwall contact; t up-dip" % ORIGIN)
    w("")

    # ---- 1. cavity and masks ---------------------------------------------------------------------------
    w("1. Cavity and masks")
    w("  cavity (PFC rubble region + residual voids) %d cells = %.2f m2 (PFC report at 0.05 m: 1923.16 + 990.07 ="
      " 2913.23 m2); PFC rubble cells %.2f m2, PFC void cells %.2f m2"
      % (cav.sum(), cav.sum() * a2, pfc_rub.sum() * a2, (cav & ~pfc_rub).sum() * a2))
    for sd in (FW, HW, PIL):
        w("  %-12s %.2f m2 (rock, n = 1e-3)" % (NAME[sd], (~cav & (side == sd)).sum() * a2))
    w("  cavity: X %.2f..%.2f, Z %.2f..%.2f m; margin to the domain edge: left %.2f, right %.2f, bottom %.2f, top %.2f m"
      % (XR[cav].min(), XR[cav].max(), ZR[cav].min(), ZR[cav].max(),
         XR[cav].min() - (DOM[0] - ORIGIN[0]) - CELL / 2, (DOM[1] - ORIGIN[0]) - XR[cav].max() - CELL / 2,
         ZR[cav].min() - (DOM[2] - ORIGIN[1]) - CELL / 2, (DOM[3] - ORIGIN[1]) - ZR[cav].max() - CELL / 2))
    w("")

    # ---- 2. bulking -------------------------------------------------------------------------------------
    w("2. Rubble extent by bulking")
    A0 = solid / (1.0 - N_HW)
    w("  PFC rubble solid area %.2f m2; A0 = %.2f / (1 - %.4f) = %.2f m2 (PFC rubble region %.2f m2 = %.3f A0)"
      % (solid, solid, N_HW, A0, pfc_rub.sum() * a2, pfc_rub.sum() * a2 / A0))
    colr = pfc_rub.any(axis=0)
    ytop = np.full(nx, np.nan)
    for i in np.where(colr)[0]:
        ytop[i] = ys[np.where(pfc_rub[:, i])[0].max()] + CELL / 2
    i_lo, i_hi = np.where(colr)[0].min(), np.where(colr)[0].max()
    cav_cols = np.where(cav.any(axis=0))[0]
    ext_cols = [i for i in cav_cols if not colr[i]]
    ytop[:i_lo] = ytop[i_lo]
    ytop[i_hi + 1:] = ytop[i_hi]
    inner_gap = [i for i in range(i_lo, i_hi + 1) if not colr[i]]
    w("  y_top(x) = top of the PFC rubble in each 0.25 m column; PFC rubble spans X %.2f..%.2f m; cavity columns"
      " outside that span: %d (held at the end value); columns inside the span without PFC rubble: %d"
      % (xs[i_lo] - ORIGIN[0], xs[i_hi] - ORIGIN[0], len(ext_cols), len(inner_gap)))
    for i in inner_gap:
        ytop[i] = np.interp(i, [j for j in range(nx) if np.isfinite(ytop[j]) and colr[j]],
                            [ytop[j] for j in range(nx) if colr[j]])
    w("  K     rubble m2   void m2    D (m)    rubble / A0")
    drift_levels = []
    geo = {}
    for K in KS:
        cls_k, rub_k, d_k = geometry(cav, pfc_rub, side, ytop, X, Y, A0, K, [])
        geo[K] = (cls_k, rub_k, d_k)
        w("  %.2f  %9.2f  %8.2f  %+7.3f   %.4f" % (K, rub_k.sum() * a2, (cav & ~rub_k).sum() * a2, d_k, rub_k.sum() * a2 / A0))
    # drifts (on the base geometry)
    cls0, rub0, _ = geo[K_BASE]
    holes = []
    yd = ORIGIN[1]
    drifts = []
    while DRIFT_N is None or len(holes) < DRIFT_N:
        dm, ctr = drift_mask(X, Y, yd)
        cls_try = cls0.copy()
        for m in drifts + [dm]:
            cls_try[m] = DRIFT
        h = hole(cls_try, yd)
        drift_levels.append((yd, ctr, h is not None))
        if h is None:
            if DRIFT_N is not None:
                raise ValueError("configured %d drifts, but the hole of the drift at y %.2f misses the rubble" % (DRIFT_N, yd))
            break
        drifts.append(dm)
        holes.append(h)
        yd += DRIFT_DY
    for K in KS:
        cls_k, rub_k, d_k = geometry(cav, pfc_rub, side, ytop, X, Y, A0, K, drifts)
        geo[K] = (cls_k, rub_k, d_k)
    cls, rub, _ = geo[K_BASE]
    vb_faces = neighbours_count(rub, cls == VOID)
    w("  base geometry K = %.2f: rubble cells on the void boundary %d, %d faces = %.2f m" % (K_BASE, (vb_faces > 0).sum(),
                                                                                         vb_faces.sum(), vb_faces.sum() * CELL))
    w("  base rubble: X %.2f..%.2f, Z %.2f..%.2f, s %.2f..%.2f, t %.2f..%.2f m"
      % (XR[rub].min(), XR[rub].max(), ZR[rub].min(), ZR[rub].max(), S[rub].min(), S[rub].max(), T[rub].min(), T[rub].max()))
    w("")

    # ---- 3. porosity and closed forms ---------------------------------------------------------------------
    w("3. Porosity fields (base geometry) and closed forms")
    s_pfc_max = S[pfc_rub].max()
    fields = {}
    for K in KS:
        cls_k, rub_k, _ = geo[K]
        sm, mapped, zst = mapping(zs, ns, s_pfc_max, S[rub_k].max())
        nsc = np.interp(S, zst, mapped)
        n1 = np.where(rub_k, nsc, ROCK_N)
        tmin, tmax = T[rub_k].min(), T[rub_k].max()
        raw2 = nsc + (-DELTA + 2 * DELTA * (T - tmin) / (tmax - tmin))
        n2 = np.where(rub_k, np.clip(raw2, CLIP[0], CLIP[1]), ROCK_N)
        fields[K] = (n1, n2, sm, mapped, zst, (rub_k & (raw2 < CLIP[0])).sum(), (rub_k & (raw2 > CLIP[1])).sum(), tmin, tmax)
    n1, n2, sm, mapped, zst, clo, chi, tmin, tmax = fields[K_BASE]
    w("  PFC s-profile smoothed (5-point): min %.4f at s %.2f -> %.2f, max %.4f at s %.2f -> %.2f"
      % (sm.min(), zs[np.argmin(sm)], N_LO, sm.max(), zs[np.argmax(sm)], N_HI))
    w("  stretch for s > %.0f m: PFC rubble max s %.2f -> new rubble max s %.2f (factor %.4f)"
      % (BASAL, s_pfc_max, S[rub].max(), (S[rub].max() - BASAL) / (s_pfc_max - BASAL)))
    w("  slab: PFC s / stretched s / area m2 / PFC n / smoothed / mapped")
    for q in zip(zs, zst, ar, ns, sm, mapped):
        w("    %6.2f  %6.2f  %6.1f  %.4f  %.4f  %.4f" % q)
    w("  scenario 2: Delta n from -%.2f at t %.2f to +%.2f at t %.2f m; clipped %d cells below %.2f, %d above %.2f"
      " (of %d rubble cells)" % (DELTA, tmin, DELTA, tmax, clo, CLIP[0], chi, CLIP[1], rub.sum()))
    basal, interior, hang = zones(rub, S, T)
    zl = (("basal", basal), ("interior", interior), ("hanging side", hang))
    w("  zones: basal (s <= 3 m) %.2f m2, interior %.2f m2, hanging side (s above 2/3 of the local thickness,"
      " 0.5 m t-bins) %.2f m2" % (basal.sum() * a2, interior.sum() * a2, hang.sum() * a2))
    sc = {1: n1, 2: n2}
    for k in (1, 2):
        nv = sc[k][rub]
        w("  scenario %d: n min / median / mean / max %.4f / %.4f / %.4f / %.4f; medians basal / interior / hanging side"
          " %.4f / %.4f / %.4f" % ((k, nv.min(), np.median(nv), nv.mean(), nv.max()) +
                                   tuple(np.median(sc[k][zm]) for _, zm in zl)))
    w("  L_max (m): scenario, p0: min / median / max; medians basal / interior / hanging side; min L_max / dx")
    lres = {}
    for k in (1, 2):
        for p0 in P0S + (P0_REF,):
            Lm = p0 / lam(sc[k])
            row = (Lm[rub].min(), np.median(Lm[rub]), Lm[rub].max()) + tuple(np.median(Lm[zm]) for _, zm in zl)
            lres[(k, p0)] = row
            w("    %d  %.0f MPa: %.3f / %.3f / %.3f; %.3f / %.3f / %.3f; %.2f%s" % ((k, p0 / 1e6) + row + (row[0] / CELL,
                                                                                    "  (< 7)" if row[0] / CELL < 7 else "")))
    gam = {}
    for k in (1, 2):
        gam[k] = window_gamma(rub, np.log(lam(np.where(rub, sc[k], 0.5))))
    okg = rub & np.isfinite(gam[1]) & np.isfinite(gam[2])
    for k in (1, 2):
        gv = gam[k][okg]
        w("  scenario %d: gamma (3 m window, %d cells) percentiles 5/25/50/75/95 %s 1/m" % (
            k, gv.size, " / ".join("%+.4f" % q for q in np.percentile(gv, [5, 25, 50, 75, 95]))))
        for p0 in P0S + (P0_REF,):
            F = gv * p0 / (2 * RHO * G)
            w("    p0 %.0f MPa: F percentiles 5/25/50/75/95 %s; share F > 1 %.4f, F < 0 %.4f"
              % (p0 / 1e6, " / ".join("%+.3f" % q for q in np.percentile(F, [5, 25, 50, 75, 95])), (F > 1).mean(), (F < 0).mean()))
        w("    F < 0 share by zone: " + "; ".join("%s %.4f" % (zn, (gam[k][zm & okg] < 0).mean()) for zn, zm in zl))
    w("  cell-wise sign agreement of F between the scenarios: %.4f of %d cells" % (
        (np.sign(gam[1][okg]) == np.sign(gam[2][okg])).mean(), okg.sum()))
    for k in (1, 2):
        ll = np.log(lam(sc[k][rub]))
        Mx = np.column_stack([np.ones(ll.size), T[rub], S[rub]])
        cf = np.linalg.lstsq(Mx, ll)[0]
        gz = -(cf[1] * SN + cf[2] * CS)
        nv = sc[k][rub]
        cn = np.linalg.lstsq(Mx, nv)[0]
        r = nv - Mx.dot(cn)
        r2 = 1 - (r ** 2).sum() / ((nv - nv.mean()) ** 2).sum()
        w("  scenario %d: n = a + b_t t + b_s s: b_t %+.5f, b_s %+.5f /m, R2 %.3f; b_s cos30 %+.5f vs b_t sin30 %+.5f;"
          " vertical %+.5f /m. Whole-rubble ln(lambda) plane: vertical gamma %+.5f 1/m -> F %s (p0 1 / 2 / 5 MPa)"
          % (k, cn[1], cn[2], r2, cn[2] * CS, cn[1] * SN, cn[2] * CS + cn[1] * SN, gz,
             " / ".join("%+.3f" % (gz * p0 / (2 * RHO * G)) for p0 in P0S + (P0_REF,))))
    w("")

    # ---- 4. drifts, holes, reach ------------------------------------------------------------------------------
    w("4. Drifts, holes, required source pressure and reach (base geometry K = %.2f)" % K_BASE)
    for yd, ctr, ok in drift_levels:
        w("  drift at y %.1f (Z %.1f): centre X %.2f, Z %.2f (s %.2f); hole %s the rubble"
          % (yd, yd - ORIGIN[1], ctr[0] - ORIGIN[0], ctr[1] - ORIGIN[1], s_of(ctr[0], ctr[1]),
             "crosses" if ok else "misses"))
    for i, h in enumerate(holes):
        w("  hole %d: collar X %.2f, Z %.2f (s %.3f, t %.2f); standpipe end s %+.3f m (cell: %s); enters rubble at"
          " %.2f m, rubble ends at %.2f m (%s), grouting to %.2f m, end X %.2f, Z %.2f; grouting cells %d (rubble %d)"
          % (i + 1, h["collar"][0] - ORIGIN[0], h["collar"][1] - ORIGIN[1], h["s_collar"], h["t"], h["s_pipe_end"],
             h["pipe_cell"], h["u_in"], h["u_out"], h["stop"], h["u_end"], h["end"][0] - ORIGIN[0],
             h["end"][1] - ORIGIN[1], h["n_cells"], h["rubble_cells"]))
    tb = T[basal]
    low_end = basal & (T <= tb.min() + LOW_DT)
    w("  lower end of the basal layer: basal cells with t <= %.2f m: %d cells, X %.2f..%.2f, Z %.2f..%.2f"
      % (tb.min() + LOW_DT, low_end.sum(), XR[low_end].min(), XR[low_end].max(), ZR[low_end].min(), ZR[low_end].max()))
    top_face = rub & (vb_faces > 0)
    mesh = build_mesh_for_domain(DOM[0], DOM[1], DOM[2], DOM[3], CELL)[0]
    costs = {}
    w("  scen hole p0     reach/rubble  basal entry  lower end p_req/p0 (any; basal only)  hanging-side reach  void contact")
    for k in (1, 2):
        st = make_state(mesh, sc[k].ravel())
        st_b = make_state(mesh, np.where(basal, sc[k], ROCK_N).ravel())
        for hi, h in enumerate(holes):
            cost = np.asarray(reachable_domain(st, h["cells"], p0=P0S[-1])["cost"], float).reshape(ny, nx)
            srcb = np.array([q for q in h["cells"] if basal.ravel()[q]])
            cb = np.asarray(reachable_domain(st_b, srcb, p0=P0S[-1])["cost"], float).reshape(ny, nx) if srcb.size else \
                np.full((ny, nx), np.inf)
            costs[(k, hi)] = cost
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
              " surface (rubble cells on the void boundary) min %.3f MPa" % (lowv / 1e6, lowb / 1e6, cost[top_face].min() / 1e6))
    w("")
    w("  union of the single-hole reach domains")
    top_band = rub & (ndimage.distance_transform_edt(~(cls == VOID)) * CELL <= 3.0)
    union = {}
    for k in (1, 2):
        for p0 in P0S:
            u = np.zeros((ny, nx), bool)
            for hi in range(len(holes)):
                u |= costs[(k, hi)] <= p0
            u &= rub
            union[(k, p0)] = u
            miss = rub & ~u
            w("   scenario %d, %.0f MPa: union %.4f of the rubble; not covered %.2f m2: hanging side %.4f of its zone"
              " uncovered, top band (rubble within 3 m of the void) %.4f uncovered, basal layer %.4f uncovered, interior %.4f"
              % (k, p0 / 1e6, u.sum() / float(rub.sum()), miss.sum() * a2, (miss & hang).sum() / float(hang.sum()),
                 (miss & top_band).sum() / float(top_band.sum()), (miss & basal).sum() / float(basal.sum()),
                 (miss & interior).sum() / float(interior.sum())))
            lab, nl = ndimage.label(miss, structure=np.ones((3, 3), int))
            comp = []
            for q in range(1, nl + 1):
                mk = lab == q
                comp.append((mk.sum() * a2, XR[mk].mean(), ZR[mk].mean(), XR[mk].min(), XR[mk].max(), ZR[mk].min(),
                             ZR[mk].max(), (mk & hang).sum() / float(mk.sum()), (mk & top_band).sum() / float(mk.sum())))
            comp.sort(key=lambda r: -r[0])
            for r_ in comp[:4]:
                w("      uncovered %7.2f m2  centroid (%.2f, %.2f)  X %.2f..%.2f  Z %.2f..%.2f  hanging-side share %.2f,"
                  " top-band share %.2f" % r_)
            w("      (%d uncovered components in all)" % len(comp))
    w("")
    # ---- 4.3 geometry sensitivity -----------------------------------------------------------------------------
    tm = 0.5 * (T[rub].min() + T[rub].max())
    hm = int(np.argmin([abs(h["t"] - tm) for h in holes]))
    w("  geometry sensitivity: scenario 1, hole %d (t %.2f m; middle of the base rubble t %.2f m)" % (hm + 1, holes[hm]["t"], tm))
    w("   K     D (m)   hole end (m)  hanging-side reach 1 / 2 MPa   p_req to the top surface / p0 (1 / 2 MPa)   void contact 1 / 2 MPa")
    sens = []
    for K in KS:
        cls_k, rub_k, d_k = geo[K]
        n1k = fields[K][0]
        hk = hole(cls_k, drift_levels[hm][0])
        vbk = neighbours_count(rub_k, cls_k == VOID)
        _, _, hang_k = zones(rub_k, S, T)
        st = make_state(mesh, n1k.ravel())
        ck = np.asarray(reachable_domain(st, hk["cells"], p0=P0S[-1])["cost"], float).reshape(ny, nx)
        topk = rub_k & (vbk > 0)
        row = [K, d_k, hk["u_end"]]
        for p0 in P0S:
            row.append(((ck <= p0) & hang_k).sum() / float(hang_k.sum()))
        for p0 in P0S:
            row.append(ck[topk].min() / p0)
        for p0 in P0S:
            row.append((vbk * ((ck <= p0) & rub_k)).sum() * CELL)
        sens.append(row)
        w("   %.2f  %+6.3f   %6.2f        %.4f / %.4f                %.3f / %.3f                               %.2f / %.2f m" % tuple(row))
    w("")

    text = "\n".join(L) + "\n"
    open(os.path.join(OUT, "prejudge.txt"), "w").write(text)
    hole_cells = np.concatenate([h["cells"] for h in holes])
    hole_ptr = np.cumsum([0] + [len(h["cells"]) for h in holes])
    hole_line = np.array([[h["collar"][0] - ORIGIN[0], h["collar"][1] - ORIGIN[1], h["end"][0] - ORIGIN[0],
                           h["end"][1] - ORIGIN[1]] for h in holes])
    np.savez_compressed(os.path.join(DATA, "base_geometry_K1.25.npz"), cls=cls, X=XR, Z=ZR, S=S, T=T, n1=n1, n2=n2,
                        cell=CELL, origin_model=np.array(ORIGIN), domain_model=np.array(DOM), basal=basal, hanging=hang,
                        void_boundary_faces=vb_faces, hole_cells=hole_cells, hole_ptr=hole_ptr, hole_line=hole_line,
                        classes=np.array(["rubble", "footwall", "hanging wall", "ore pillar", "void", "drift"]),
                        note=np.array("cls codes 0..5 as in classes; n1, n2 = scenario porosity (rock, void, drift 1e-3);"
                                      " hole_cells[hole_ptr[i]:hole_ptr[i+1]] = flat cell indices (iy * nx + ix) of the"
                                      " grouting section of hole i; X, Z cell centres relative to the origin"))
    np.savez_compressed(os.path.join(OUT, "prejudge.npz"), cls=cls, X=XR, Z=ZR, S=S, n1=n1, n2=n2, gam1=gam[1],
                        gam2=gam[2], zs=zs, zst=zst, ns=ns, sm=sm, mapped=mapped, ar=ar, basal=basal, hang=hang,
                        cost=np.array([costs[(k, hi)] for k in (1, 2) for hi in range(len(holes))]),
                        holes=hole_line, ytop=ytop - ORIGIN[1], xs=xs - ORIGIN[0],
                        dks=np.array([geo[K][2] for K in KS]), ks=np.array(KS), cav=cav,
                        union=np.array([union[(k, p0)] for k in (1, 2) for p0 in P0S]),
                        drifts=np.array([[d[1][0] - ORIGIN[0], d[1][1] - ORIGIN[1]] for d in drift_levels if d[2]]))
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
