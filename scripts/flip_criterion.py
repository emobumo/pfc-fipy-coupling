# -*- coding: utf-8 -*-
"""
4.3 corollary 2 (V2 + V3): the straight-ray estimate and the flip criterion.

    powershell -File scripts\run_local.ps1 scripts\flip_criterion.py [--plot]

Straight-ray estimate. From a point P on the hole, shoot a ray along the
hole's normal (upward unit normal, vertical component c = cos 35 deg) and
solve for its length s:

    up:    int_0^s lambda(n(y_P + c s')) ds' + rho g c s = p0
    down:  int_0^s lambda(n(y_P - c s')) ds' - rho g c s = p0

up / down extent = max over P (51 points, mouth and toe included) -- the same
"largest distance normal to the hole axis" as gravity_control.up_down(); the
mid-point-only version is reported alongside. A ray that leaves the 60 x 60 m
box is stopped there and flagged (the porosity field is never extrapolated).

Linearized: up/down ~ 1 + c (gamma L_max - 2 Pi_g), gamma = |d ln lambda / dy|,
so the asymmetry flips upward when F = gamma p0 / (2 rho g) > 1 (porosity form:
n' / [n (1 - n)] > 2 rho g / p0).

V2  group A: uniform / gradient x gravity on / off, 2D reach (1.25 m) vs rays.
    group B: the 11 unclipped rows of reach_sensitivity_1.25.txt; each gets
    its own 2D geometric baseline (uniform, no gravity), factor = ratio/baseline;
    the rays' geometric baseline is 1 by construction.
V3  field family n(y) = n_mid + n' (y - y_mid), y_mid = hole mid-height,
    n_mid = the gradient field there, n' = F * 2 rho g n_mid (1 - n_mid) / p0
    with rho = 1830 (group 1). F = 2.634 reproduces the gradient field exactly.
    Groups: 1 v0.6; 2 tau0 60; 3 A x0.3; 4 rho 1450 (same fields, so each
    field's F is larger by 1830/1450). F* = where the factor crosses 1.

Solver-free. A digest of all 2D cost arrays is printed for the two-run check.
"""
from __future__ import print_function

import hashlib
import imp
import math
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.analysis.fill_diagnostics import reachable_domain

C = imp.load_source("inclined_hole_channel", os.path.join(REPO, "cases", "inclined_hole_channel.py"))
G = C.G
OUT = os.path.join(REPO, "outputs", "flip_criterion")
CELL = 1.25
G_ACC = 9.81
N_P = 51
DS = 0.01
DIGEST = hashlib.sha1()

BASE = {"tau0": G.TAU0, "A": G.A_CAL, "rho": G.RHO}
GROUPS = [("1 v0.6", 30.0, 1.0, 1830.0), ("2 tau0 60", 60.0, 1.0, 1830.0),
          ("3 A x0.3", 30.0, 0.3, 1830.0), ("4 rho 1450", 30.0, 1.0, 1450.0)]
F_LIST = (-1.0, 0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def set_params(tau0, a_factor, rho):
    G.TAU0, G.A_CAL, G.RHO = tau0, BASE["A"] * a_factor, rho


def restore():
    G.TAU0, G.A_CAL, G.RHO = BASE["tau0"], BASE["A"], BASE["rho"]


def lam(n):
    return 2.0 * G.TAU0 / math.sqrt(8.0 * G.A_CAL) * (1.0 - n) / n


def geometry():
    end, d, nrm = G.hole_geometry()
    up = (-nrm[0], -nrm[1])                   # upward normal
    pts = [(G.MOUTH[0] + t * (end[0] - G.MOUTH[0]), G.MOUTH[1] + t * (end[1] - G.MOUTH[1]))
           for t in np.linspace(0.0, 1.0, N_P)]
    return pts, up, nrm


def ray(p, direction, nfun, gravity, p0=G.P0, walls=True):
    """Length along `direction` at which the budget runs out; (s, clipped).
    walls=False ignores the left/right box sides (n depends on y only, so
    this extends no porosity data); the top and bottom always stop the ray."""
    c = direction[1]
    rg = G.RHO * G_ACC if gravity else 0.0
    s, acc = 0.0, 0.0
    lam_prev = lam(nfun(p[1]))
    while True:
        s1 = s + DS
        x, y = p[0] + s1 * direction[0], p[1] + s1 * direction[1]
        if not (G.Y_MIN <= y <= G.Y_MAX) or (walls and not (G.X_MIN <= x <= G.X_MAX)):
            return s, True
        lam1 = lam(nfun(y))
        acc1 = acc + 0.5 * (lam_prev + lam1) * DS
        total = acc1 + rg * c * s1
        if total >= p0:
            prev = acc + rg * c * s
            return s + (p0 - prev) / (total - prev) * DS, False
        s, acc, lam_prev = s1, acc1, lam1


def ray_ratio(nfun, gravity):
    pts, up, nrm = geometry()
    down = (nrm[0], nrm[1])
    ups = [ray(p, up, nfun, gravity) for p in pts]
    dns = [ray(p, down, nfun, gravity) for p in pts]
    iu = int(np.argmax([u[0] for u in ups]))
    idn = int(np.argmax([v[0] for v in dns]))
    su, sd = ups[iu][0], dns[idn][0]
    mid = N_P // 2
    mu, _ = ray(pts[mid], up, nfun, gravity, walls=False)
    md, _ = ray(pts[mid], down, nfun, gravity, walls=False)
    return {"ratio": su / sd, "mid": ups[mid][0] / dns[mid][0], "mid_free": mu / md,
            "up": su, "down": sd,
            # only the rays that SET the extents matter
            "clipped": bool(ups[iu][1] or dns[idn][1]),
            "n_clipped": sum(1 for u in ups if u[1]) + sum(1 for v in dns if v[1])}


def reach_ratio(phi, gravity):
    G.CELL = CELL
    st, mx, my, nx, ny, _, hc, _ = C.build_from_phi(phi)
    r = reachable_domain(st, hc, p0=G.P0, use_gravity=gravity)
    DIGEST.update(np.asarray(r["cost"], dtype=float).tobytes())
    hm = np.zeros(mx.size, dtype=bool)
    hm[hc] = True
    m = r["reachable"] & np.logical_not(hm)
    end, d, nrm = G.hole_geometry()
    perp = (mx[m] - G.MOUTH[0]) * nrm[0] + (my[m] - G.MOUTH[1]) * nrm[1]
    clipped = bool((mx[m].max() > G.X_MAX - CELL) or (my[m].max() > G.Y_MAX - CELL)
                   or (my[m].min() < G.Y_MIN + CELL))
    return {"ratio": float(-perp.min() / perp.max()), "clipped": clipped}, mx, my


def mesh_y():
    G.CELL = CELL
    _, x, y, _, _ = G.build_mesh_for_domain(G.X_MIN, G.X_MAX, G.Y_MIN, G.Y_MAX, CELL)
    return np.asarray(y, dtype=float)


def gradient_n(y):
    return G.PHI_BOTTOM + (G.PHI_TOP - G.PHI_BOTTOM) * (y / G.H)


def group_a(lines):
    my = mesh_y()
    lines += ["V2 group A (v0.6, %.2f m): 2D reach vs straight rays" % CELL, "",
              "%-10s %-8s | %8s | %8s %9s %9s | %s" % ("field", "gravity", "2D", "rays max", "mid(wall)", "mid(free)",
                                                        "max ray clipped / rays stopped by a side")]
    for kind, nfun in (("uniform", lambda y: G.PHI_UNIFORM), ("gradient", gradient_n)):
        phi = np.full(my.size, G.PHI_UNIFORM) if kind == "uniform" else gradient_n(my)
        for grav in (False, True):
            d2, _, _ = reach_ratio(phi, grav)
            rr = ray_ratio(nfun, grav)
            lines.append("%-10s %-8s | %8.3f | %8.3f %9.3f %9.3f | %s / %d of %d" % (
                kind, "on" if grav else "off", d2["ratio"], rr["ratio"], rr["mid"], rr["mid_free"],
                rr["clipped"], rr["n_clipped"], 2 * N_P))
    lines.append("")


def group_b(lines):
    path = os.path.join(REPO, "outputs", "reach_sensitivity", "reach_sensitivity_1.25.txt")
    rows = []
    for line in open(path):
        m = re.match(r"\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+\|\s+([\d.]+)(\*?)\s+([\d.]+)(\*?)", line)
        if m and not m.group(6) and not m.group(8):
            rows.append(tuple(float(m.group(i)) for i in (1, 2, 3, 4, 5, 7)))
    my = mesh_y()
    lines += ["V2 group B: the %d unclipped sensitivity rows; factor = ratio / geometric baseline" % len(rows), "",
              "%6s %5s %4s %5s | %8s | %8s %8s %8s | %8s %8s %8s" % (
                  "s", "tau0", "A x", "rho", "baseline", "uni 2D", "uni ray", "diff", "grad 2D", "grad ray", "diff")]
    pts = []
    for s, tau0, af, rho, uni, grad in rows:
        set_params(tau0, af, rho)
        try:
            base, _, _ = reach_ratio(np.full(my.size, G.PHI_UNIFORM), False)
            ru = ray_ratio(lambda y: G.PHI_UNIFORM, True)["ratio"]
            rgd = ray_ratio(gradient_n, True)["ratio"]
        finally:
            restore()
        fu, fg = uni / base["ratio"], grad / base["ratio"]
        pts.append((fu, ru, fg, rgd))
        lines.append("%6.3f %5.0f %4.1f %5.0f | %8.3f | %8.3f %8.3f %+8.3f | %8.3f %8.3f %+8.3f" % (
            s, tau0, af, rho, base["ratio"], fu, ru, ru - fu, fg, rgd, rgd - fg))
    lines.append("")
    return pts


def group_v3(lines):
    end, d, nrm = G.hole_geometry()
    y_mid = 0.5 * (G.MOUTH[1] + end[1])
    n_mid = float(gradient_n(y_mid))
    unit = 2.0 * 1830.0 * G_ACC * n_mid * (1 - n_mid) / G.P0     # n' per unit F (group 1)
    f_cur = ((G.PHI_TOP - G.PHI_BOTTOM) / G.H) / unit
    my = mesh_y()
    lines += ["V3 flip criterion: n(y) = n_mid + n'(y - y_mid), y_mid = %.3f m, n_mid = %.5f, "
              "n' = F1 x %.4e /m; the gradient field is F1 = %.3f" % (y_mid, n_mid, unit, f_cur), "",
              "%-11s %6s %7s %7s | %8s %8s | %8s %8s %8s | %s" % (
                  "group", "F1", "F(grp)", "L_max", "n(0)", "n(60)", "2D", "2D/base", "rays", "clipped")]
    results = {}
    for name, tau0, af, rho in GROUPS:
        set_params(tau0, af, rho)
        try:
            lmax = G.P0 / lam(n_mid)
            base, _, _ = reach_ratio(np.full(my.size, n_mid), False)
            rows = []
            for f1 in sorted(set(F_LIST + (round(f_cur, 3),))):
                slope = f1 * unit
                nf = lambda y, sl=slope: n_mid + sl * (y - y_mid)
                phi = nf(my)
                bad = bool(phi.min() <= 0.02 or phi.max() >= 0.9)
                f_g = f1 * 1830.0 / rho
                d2, _, _ = reach_ratio(phi, True)
                rr = ray_ratio(nf, True)
                rows.append((f1, f_g, d2["ratio"] / base["ratio"], rr["ratio"], d2["clipped"] or rr["clipped"] or bad))
                lines.append("%-11s %6.3f %7.3f %7.2f | %8.4f %8.4f | %8.3f %8.3f %8.3f | %s" % (
                    name, f1, f_g, lmax, nf(0.0), nf(60.0), d2["ratio"], d2["ratio"] / base["ratio"],
                    rr["ratio"], d2["clipped"] or rr["clipped"] or bad))
            # fine ray curve for the figure
            curve = []
            for f1 in np.linspace(-1.5, 3.5, 51):
                nf = lambda y, sl=f1 * unit: n_mid + sl * (y - y_mid)
                curve.append((f1 * 1830.0 / rho, ray_ratio(nf, True)["ratio"]))
            results[name] = {"rows": rows, "curve": curve, "lmax": lmax, "base": base["ratio"]}
        finally:
            restore()
    lines.append("")

    def crossing(xs, ys):
        for i in range(len(xs) - 1):
            if (ys[i] - 1.0) * (ys[i + 1] - 1.0) <= 0 and ys[i + 1] != ys[i]:
                return xs[i] + (1.0 - ys[i]) * (xs[i + 1] - xs[i]) / (ys[i + 1] - ys[i])
        return float("nan")

    lines.append("F* (factor = 1), on each group's own F axis; amplitude at F1 = 2:")
    for name, _, _, _ in GROUPS:
        rw = sorted(results[name]["rows"])
        fs = [r[1] for r in rw]
        f2 = crossing(fs, [r[2] for r in rw])
        fr = crossing(fs, [r[3] for r in rw])
        at2 = [r for r in rw if abs(r[0] - 2.0) < 1e-9][0]
        lm = results[name]["lmax"]
        lines.append("  %-11s F*_2D = %.3f  F*_rays = %.3f | L_max %.2f m | at F1=2: 2D-1 = %+.3f "
                     "((2D-1)/L_max = %.4f /m), rays-1 = %+.3f ((rays-1)/L_max = %.4f /m)"
                     % (name, f2, fr, lm, at2[2] - 1, (at2[2] - 1) / lm, at2[3] - 1, (at2[3] - 1) / lm))
    if "1 v0.6" in results:
        cur = [r for r in results["1 v0.6"]["rows"] if abs(r[0] - round(f_cur, 3)) < 1e-9][0]
        lines.append("  check: group 1 at the gradient field (F1 = %.3f): 2D factor %.3f (anchor 1.179 / baseline %.3f)"
                     % (f_cur, cur[2], results["1 v0.6"]["base"]))
    lines.append("")
    return results, f_cur


def plot(results, f_cur, pts):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    marks = ["o", "s", "^", "D"]
    names = {"1 v0.6": u"基准参数", "2 tau0 60": u"屈服应力加倍", "3 A x0.3": u"渗透率系数 ×0.3",
             "4 rho 1450": u"浆液密度 1450"}
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (name, _, _, _), col, mk in zip(GROUPS, cols, marks):
        r = results[name]
        cx, cy = zip(*r["curve"])
        ax.plot(cx, cy, color=col, lw=1.6)
        rw = r["rows"]
        ax.plot([q[1] for q in rw], [q[2] for q in rw], mk, color=col, ms=6, mfc="none", mew=1.4,
                label=names[name])
    ax.axvline(1.0, color="#555555", lw=0.8, ls="--")
    ax.axhline(1.0, color="#555555", lw=0.8, ls="--")
    ax.annotate(u"现梯度场", xy=(f_cur, [q[2] for q in results["1 v0.6"]["rows"] if abs(q[0] - round(f_cur, 3)) < 1e-9][0]),
                xytext=(f_cur - 0.3, 1.3), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel(u"翻转判据 F = γp$_0$/(2ρg)", fontsize=9)
    ax.set_ylabel(u"上/下扩散比（相对几何基线）", fontsize=9)
    ax.set_xlim(-1.3, 3.8)
    leg = ax.legend(fontsize=8, loc="upper left", numpoints=1, title=u"点：二维可达域（1.25 m）；线：直射线估计")
    leg.get_title().set_fontsize(8)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(u"孔隙率梯度的翻转判据", fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(OUT, "fig_flip_criterion.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([q[0] for q in pts], [q[1] for q in pts], "o", color="#2a78d6", mfc="none", mew=1.4, label=u"均质场")
    ax.plot([q[2] for q in pts], [q[3] for q in pts], "s", color="#eb6834", mfc="none", mew=1.4, label=u"梯度场")
    lo, hi = 0.85, 1.3
    ax.plot([lo, hi], [lo, hi], color="#555555", lw=0.8, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(u"二维可达域的上/下因子", fontsize=9)
    ax.set_ylabel(u"直射线估计的上/下比", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", numpoints=1)
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_title(u"直射线估计与二维可达域（敏感性各组）", fontsize=10)
    fig.tight_layout()
    p2 = os.path.join(OUT, "fig_ray_vs_2d.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    return p1, p2


def refine(argv):
    """V3 again at a finer cell for chosen groups (--refine 0.625 --groups 2,3)."""
    global CELL, GROUPS
    CELL = float(argv[argv.index("--refine") + 1])
    if "--groups" in argv:
        keep = argv[argv.index("--groups") + 1].split(",")
        GROUPS = [g for g in GROUPS if g[0].split()[0] in keep]
    lines = ["(refinement: cell %.3f m)" % CELL]
    group_v3(lines)
    lines.append("digest %s" % DIGEST.hexdigest())
    text = "\n".join(lines)
    open(os.path.join(OUT, "flip_criterion_refine_%.3f.txt" % CELL), "w").write(text + "\n")
    print(text)
    return 0


def main(argv):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    if "--refine" in argv:
        return refine(argv)
    lines = []
    group_a(lines)
    pts = group_b(lines)
    results, f_cur = group_v3(lines)
    lines.append("digest %s" % DIGEST.hexdigest())
    text = "\n".join(lines)
    open(os.path.join(OUT, "flip_criterion.txt"), "w").write(text + "\n")
    print(text)
    if "--plot" in argv:
        for p in plot(results, f_cur, pts):
            print("wrote", p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
