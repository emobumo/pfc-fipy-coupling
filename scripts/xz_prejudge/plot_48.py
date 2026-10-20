# -*- coding: utf-8 -*-
"""Sketches of figs. 4.8-1 and 4.8-4 (sequential part) for the configured end state.

    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\plot_48.py

Inputs: outputs/xz_prejudge<suffix>/step2b.npz (step2b.py), seq_reach.npz (seq_reach.py) and
data/xz_prejudge<suffix>/base_geometry_K1.25_v4.npz. Writes to figures/xz_step2b<suffix>/:
    fig_48_1_geometry   geometry: rubble (K 1.25), residual void, basal layer, original stope outline, drifts and the
                        8 holes; rubble surface for K 1.15 / 1.40 dashed
    fig_48_4_sequence   K 1.25, both scenarios: union of the single-hole reach domains against the stepwise reach of
                        the bottom-up and top-down orders (p0 = 2 MPa)
No case codes on the figures. Fig. 4.8-2 is plot.py --v4; the zoning panels of fig. 4.8-4 are fig_b3_zones_compare.
"""
from __future__ import print_function

import imp
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PL = imp.load_source("xz_plot", os.path.join(REPO, "scripts", "xz_prejudge", "plot.py"))
plt = PL.plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon

XC, CFG = PL.XC, PL.CFG
SRC = os.path.join(XC.out(CFG, "outputs", "xz_prejudge"), "step2b.npz")
SEQ = os.path.join(XC.out(CFG, "outputs", "xz_prejudge"), "seq_reach.npz")
BASE = os.path.join(XC.out(CFG, "data", "xz_prejudge"), "base_geometry_K1.25_v4.npz")
PL.OUT = XC.out(CFG, "figures", "xz_step2b")
CELL = PL.CELL


def top_line(cls, X, Z):
    """Rubble surface: top of the rubble cells in each column."""
    rub = cls == 0
    col = rub.any(axis=0)
    zc = Z[:, 0]
    zt = np.array([zc[rub[:, i]].max() + CELL / 2 if col[i] else np.nan for i in range(rub.shape[1])])
    return X[0], zt


def fig1(D, B):
    ki = list(D["ks"]).index(1.25)
    V = {"cls": D["cls"][ki], "X": D["X"], "Z": D["Z"], "drifts": D["drifts"], "holes": D["fan"][ki]}
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.imshow(V["cls"].astype(float), origin="lower", extent=PL.ext(V), cmap=ListedColormap(PL.COLORS),
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 6), interpolation="nearest")
    ax.contour(D["X"][0], D["Z"][:, 0], B["basal"].astype(float), levels=[0.5], colors=["#7a3b9e"], linewidths=0.8)
    org = CFG["stope"]["origin"]
    poly = [(x - org[0], y - org[1]) for x, y in XC.stope_polygon(CFG)]
    ax.add_patch(Polygon(poly, closed=True, fill=False, ec="k", lw=0.9, ls="dashdot"))
    for K, ls in ((1.15, ":"), (1.40, "--")):
        xs, zt = top_line(D["cls"][list(D["ks"]).index(K)], D["X"], D["Z"])
        ax.plot(xs, zt, ls, color="#1f4e9c", lw=1.0)
    PL.frame(ax, V)
    # drifts asked for but without a hole reaching the rubble (configuration drifts.requested), drawn dashed
    dc = CFG["drifts"]
    sn, cs = np.sin(np.radians(CFG["stope"]["dip_deg"])), np.cos(np.radians(CFG["stope"]["dip_deg"]))
    for d in range(len(D["drifts"]), dc.get("requested") or 0):
        zd = d * dc["spacing"]
        xd = zd * cs / sn + dc["dx"]
        ax.add_patch(plt.Rectangle((xd - dc["half"], zd - dc["half"]), 2 * dc["half"], 2 * dc["half"], fc="none",
                                   ec=PL.COLORS[5], ls="dashed", lw=0.8))
        for ang in (-30.0, 0.0):
            a_ = np.radians(ang)
            ex, ey = np.cos(a_) * -sn + np.sin(a_) * cs, np.cos(a_) * cs + np.sin(a_) * sn
            ax.plot([xd - dc["half"], xd - dc["half"] + 15 * ex], [zd, zd + 15 * ey], "--", color="#777777", lw=0.9)
        ax.text(xd + 1.6, zd - 1.2, u"巷道 %d（孔不入散体）" % (d + 1), fontsize=7, color="#555555")
    seen = set()
    for h, d in zip(D["fan"][ki], B["fan_drift"]):
        if int(d) not in seen:
            seen.add(int(d))
            ax.text(h[0] + 1.6, h[1] - 1.2, u"巷道 %d" % int(d), fontsize=7)
    hs = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in PL.COLORS]
    hs += [plt.Line2D([], [], color="#7a3b9e"), plt.Line2D([], [], color="k", ls="-.", lw=0.9),
           plt.Line2D([], [], color="#1f4e9c", ls=":"), plt.Line2D([], [], color="#1f4e9c", ls="--"),
           plt.Line2D([], [], color="k", lw=1.0)]
    ax.legend(hs, [u"散体（K = 1.25）", u"下盘", u"上盘", u"矿柱", u"残余空区（无流）", u"巷道（不透浆）",
                   u"下盘松散底层（s ≤ %g m）" % CFG["basal_s"], u"原采场轮廓", u"散体顶面 K = 1.15", u"散体顶面 K = 1.40",
                   u"钻孔（每巷道 −30°、0°）"], loc="upper left", frameon=False, fontsize=6.5, ncol=2)
    ax.set_xlabel(u"X/m（原采场下端底板角点为原点）"); ax.set_ylabel(u"Z/m")
    fig.tight_layout()
    PL.save(fig, "fig_48_1_geometry")


def fig4(D, Q):
    ki = list(D["ks"]).index(1.25)
    V = {"cls": D["cls"][ki], "X": D["X"], "Z": D["Z"], "drifts": D["drifts"], "holes": D["fan"][ki]}
    rub = V["cls"] == 0
    cols = ["#e34948", "#2a78d6", "#f2b84b"]
    fig, axs = plt.subplots(2, 3, figsize=(13.0, 5.6))
    for k in (1, 2):
        union = Q["K1.25_s%d_none" % k]
        for j, (key, name) in enumerate((("none", u"不计先后（单孔可达域并集）"), ("bottom-up", u"自下而上"),
                                         ("top-down", u"自上而下"))):
            ax = axs[k - 1, j]
            PL.background(ax, V)
            cov = Q["K1.25_s%d_%s" % (k, key)]
            img = np.where(rub, np.where(cov, 1.0, np.where(union, 2.0, 0.0)), np.nan)
            ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=PL.ext(V), cmap=ListedColormap(cols),
                      vmin=-0.5, vmax=2.5, interpolation="nearest")
            PL.frame(ax, V, u"情景%s，%s：%.1f%%" % (u"一" if k == 1 else u"二", name, 100.0 * cov[rub].mean()))
            if k == 2:
                ax.set_xlabel(u"X/m")
            if j == 0:
                ax.set_ylabel(u"Z/m")
    fig.legend([plt.Rectangle((0, 0), 1, 1, fc=c) for c in (cols[1], cols[2], cols[0])],
               [u"覆盖（$p_0$ = 2 MPa）", u"并集内、按顺序注浆后未覆盖", u"并集之外"], loc="lower center", ncol=3,
               frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.95, hspace=0.2, wspace=0.06)
    PL.save(fig, "fig_48_4_sequence")


def main():
    D = np.load(SRC)
    fig1(D, np.load(BASE))
    fig4(D, np.load(SEQ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
