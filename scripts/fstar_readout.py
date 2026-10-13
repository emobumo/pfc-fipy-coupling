# -*- coding: utf-8 -*-
"""
Critical F* of the flip criterion with the 2D R_ud read two ways (reach only,
no solver):

    powershell -File scripts\run_local.ps1 scripts\fstar_readout.py

Same F family, groups and boxes as scripts/flip_criterion.py (V3) and
scripts/flip_large_domain.py; each p_req field is computed once and read
  cell -- the recorded way: largest normal distance of reachable cell centres
  sub  -- bilinear sub-grid (dx/10) on p_req, points with p_req <= p0.
The cell read-out reproduces the recorded F* (flip_criterion.txt,
flip_criterion_refine_0.625.txt, flip_large_domain.txt) digit for digit.
Factor = R_ud / geometric baseline (uniform n_mid, no gravity, same read-out);
F* = where the factor crosses 1 (linear interpolation; the large box skips
the excluded or box-clipped F as flip_large_domain does). Amplitude =
(factor - 1)/L_max at F1 = 2.

Output: outputs/fstar_readout/fstar_readout.txt
"""
from __future__ import print_function
import imp, math, os, sys
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
FC = imp.load_source("flip_criterion", os.path.join(REPO, "scripts", "flip_criterion.py"))
FL = imp.load_source("flip_large_domain", os.path.join(REPO, "scripts", "flip_large_domain.py"))
C, G = FC.C, FC.G
SUB = 10

def readouts(cost, mx, my, hc, cell):
    _, _, nrm = G.hole_geometry()
    reach = cost <= G.P0
    hm = np.zeros(mx.size, bool); hm[hc] = True
    m = reach & ~hm
    perp = (mx[m] - G.MOUTH[0]) * nrm[0] + (my[m] - G.MOUTH[1]) * nrm[1]
    r_cell = float(-perp.min() / perp.max())
    xs, ys = np.unique(mx), np.unique(my)
    g = np.minimum(cost, 10 * G.P0).reshape(len(ys), len(xs))
    ix = np.searchsorted(xs, mx[reach]); iy = np.searchsorted(ys, my[reach])
    i0, i1 = max(ix.min() - 2, 0), min(ix.max() + 2, len(xs) - 1)
    j0, j1 = max(iy.min() - 2, 0), min(iy.max() + 2, len(ys) - 1)
    h = cell / SUB
    fx = np.arange(xs[i0], xs[i1] + 1e-9, h); fy = np.arange(ys[j0], ys[j1] + 1e-9, h)
    X, Y = np.meshgrid(fx, fy); X, Y = X.ravel(), Y.ravel()
    i = np.clip(np.searchsorted(xs, X) - 1, 0, len(xs) - 2); j = np.clip(np.searchsorted(ys, Y) - 1, 0, len(ys) - 2)
    tx = (X - xs[i]) / (xs[i + 1] - xs[i]); ty = (Y - ys[j]) / (ys[j + 1] - ys[j])
    v = (1 - tx) * (1 - ty) * g[j, i] + tx * (1 - ty) * g[j, i + 1] + (1 - tx) * ty * g[j + 1, i] + tx * ty * g[j + 1, i + 1]
    k = v <= G.P0
    p2 = (X[k] - G.MOUTH[0]) * nrm[0] + (Y[k] - G.MOUTH[1]) * nrm[1]
    return r_cell, float(-p2.min() / p2.max()), m

def crossing(xs, ys):
    return FC.crossing(xs, ys) if hasattr(FC, "crossing") else FL.crossing(xs, ys)

def case_box(groups, cell, out):
    FC.CELL = cell
    end, d, nrm = G.hole_geometry()
    y_mid = 0.5 * (G.MOUTH[1] + end[1]); n_mid = float(FC.gradient_n(y_mid))
    unit = 2.0 * 1830.0 * FC.G_ACC * n_mid * (1 - n_mid) / G.P0
    f_cur = ((G.PHI_TOP - G.PHI_BOTTOM) / G.H) / unit
    my = FC.mesh_y()
    for name, tau0, af, rho in FC.GROUPS:
        if name.split()[0] not in groups: continue
        FC.set_params(tau0, af, rho)
        try:
            G.CELL = cell
            lmax = G.P0 / FC.lam(n_mid)
            def run(phi, grav):
                st, mx, my2, nx, ny, _, hc, _ = C.build_from_phi(phi)
                cost = np.asarray(FC.reachable_domain(st, hc, p0=G.P0, use_gravity=grav)["cost"], float)
                return readouts(cost, mx, my2, np.asarray(hc, int), cell)
            bc, bs, _ = run(np.full(my.size, n_mid), False)
            rows = []
            for f1 in sorted(set(FC.F_LIST + (round(f_cur, 3),))):
                nf = lambda y, sl=f1 * unit: n_mid + sl * (y - y_mid)
                rc, rs, _ = run(nf(my), True)
                rows.append((f1 * 1830.0 / rho, f1, rc / bc, rs / bs))
            rows.sort()
            fs = [r[0] for r in rows]
            fc, fsub = crossing(fs, [r[2] for r in rows]), crossing(fs, [r[3] for r in rows])
            at2 = [r for r in rows if abs(r[1] - 2.0) < 1e-9][0]
            out.append("case box %.3f m  %-11s base %.4f / %.4f | F*_2D cell %.3f  sub %.3f  (diff %+.3f) | "
                       "amp (2D-1)/L_max at F1=2: cell %.4f  sub %.4f /m (L_max %.2f m)" % (
                           cell, name, bc, bs, fc, fsub, fsub - fc, (at2[2] - 1) / lmax, (at2[3] - 1) / lmax, lmax))
            for r in rows:
                out.append("    F %6.3f  factor cell %.4f  sub %.4f" % (r[0], r[2], r[3]))
            print(out[-len(rows) - 1]); sys.stdout.flush()
        finally:
            FC.restore()

def large_box(out):
    end, _, _ = G.hole_geometry()
    y_mid = 0.5 * (G.MOUTH[1] + end[1]); n_mid = float(FC.gradient_n(y_mid))
    unit = 2.0 * 1830.0 * FC.G_ACC * n_mid * (1 - n_mid) / G.P0
    saved = (G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL)
    try:
        G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX = FL.big_box(); G.CELL = FL.CELL
        _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, FL.CELL)
        my = np.asarray(y, float)
        for tau0 in FL.TAU0S:
            FC.set_params(tau0, 1.0, 1830.0)
            try:
                def run(phi, grav):
                    st, cx, cy, _, _, _, hc, _ = C.build_from_phi(phi)
                    cost = np.asarray(FL.reachable_domain(st, hc, p0=G.P0, use_gravity=grav)["cost"], float)
                    rc, rs, m = readouts(cost, cx, cy, np.asarray(hc, int), FL.CELL)
                    box = bool(cx[m].min() < G.X_MIN + FL.CELL or cx[m].max() > G.X_MAX - FL.CELL
                               or cy[m].min() < G.Y_MIN + FL.CELL or cy[m].max() > G.Y_MAX - FL.CELL)
                    return rc, rs, box, float(cy[m].min()), float(cy[m].max())
                bc, bs, _, _, _ = run(np.full(my.size, n_mid), False)
                rows = []
                for f1 in FL.F_LIST:
                    nf = lambda yy, sl=f1 * unit: n_mid + sl * (yy - y_mid)
                    rc, rs, box, ylo, yhi = run(nf(my), True)
                    lo, hi = sorted((nf(ylo), nf(yhi)))
                    bad = box or lo < FL.RANGE[0] or hi > FL.RANGE[1]
                    rows.append((f1, rc / bc, rs / bs, bad))
                good = [r for r in rows if not r[3]]
                fc = crossing([r[0] for r in good], [r[1] for r in good]); fsub = crossing([r[0] for r in good], [r[2] for r in good])
                out.append("large box tau0 %4.1f  base %.4f / %.4f | F*_2D cell %.3f  sub %.3f  (diff %+.3f), %d usable F" % (
                    tau0, bc, bs, fc, fsub, fsub - fc, len(good)))
                for r in rows:
                    out.append("    F %6.3f  factor cell %.4f  sub %.4f  %s" % (r[0], r[1], r[2], "excluded" if r[3] else ""))
                print(out[-len(rows) - 1]); sys.stdout.flush()
            finally:
                FC.restore()
    finally:
        G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, G.CELL = saved

def main():
    out = []
    case_box(("1", "4"), 1.25, out)
    case_box(("2", "3"), 0.625, out)
    case_box(("2", "3"), 1.25, out)
    large_box(out)
    dest = os.path.join(REPO, "outputs", "fstar_readout")
    if not os.path.isdir(dest):
        os.makedirs(dest)
    open(os.path.join(dest, "fstar_readout.txt"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
