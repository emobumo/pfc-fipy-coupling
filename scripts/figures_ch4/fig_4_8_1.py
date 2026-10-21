# -*- coding: utf-8 -*-
"""
Fig. 4.8-1 (draft): generalised dip section of the case stope, current end state.

    powershell -File scripts\\run_local.ps1 scripts\\figures_ch4\\fig_4_8_1.py

Rubble (K 1.25), footwall, hanging wall, pillar, residual void, drifts, the basal layer of the footwall, the original
stope outline, the outer caving outline (prejudge.npz "cav"), rubble surface for K 1.15 / 1.40, the 6 holes (each drift
-30 / 0 deg), the lower end of the rubble; the requested drift whose holes miss the rubble dashed. Data:
outputs/xz_prejudge<suffix>/step2b.npz, prejudge.npz; data/xz_prejudge<suffix>/base_geometry_K1.25_v4.npz.
"""
from __future__ import print_function

import os

import numpy as np

import xz48 as X
from common import plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Polygon

PL, CFG = X.PL, X.CFG


def main():
    D = np.load(os.path.join(X.PRE, "step2b.npz"))
    Q = np.load(os.path.join(X.PRE, "prejudge.npz"))
    B = np.load(os.path.join(X.XC.out(CFG, "data", "xz_prejudge"), "base_geometry_K1.25_v4.npz"))
    if Q["cav"].shape != D["X"].shape or not np.allclose(Q["X"], D["X"]):
        raise SystemExit("prejudge.npz and step2b.npz grids differ")
    ki = list(D["ks"]).index(1.25)
    V = {"cls": D["cls"][ki], "X": D["X"], "Z": D["Z"], "drifts": D["drifts"], "holes": D["fan"][ki]}
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.imshow(V["cls"].astype(float), origin="lower", extent=PL.ext(V), cmap=ListedColormap(PL.COLORS),
              norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], 6), interpolation="nearest")
    ax.contour(D["X"][0], D["Z"][:, 0], B["basal"].astype(float), levels=[0.5], colors=["#7a3b9e"], linewidths=0.8)
    ax.contour(D["X"][0], D["Z"][:, 0], Q["cav"].astype(float), levels=[0.5], colors=["#a0522d"], linewidths=1.0)
    org = CFG["stope"]["origin"]
    poly = [(x - org[0], y - org[1]) for x, y in X.XC.stope_polygon(CFG)]
    ax.add_patch(Polygon(poly, closed=True, fill=False, ec="k", lw=0.9, ls="dashdot"))
    for Kv, ls in ((1.15, ":"), (1.40, "--")):
        xs, zt = X.P48.top_line(D["cls"][list(D["ks"]).index(Kv)], D["X"], D["Z"])
        ax.plot(xs, zt, ls, color="#1f4e9c", lw=1.0)
    PL.frame(ax, V)
    X.draw_low(ax, dx=-10.0, dz=-2.2)
    half = 1.25                                    # drift half-width as drawn by PL.frame
    for d, (xd, zd) in enumerate(D["drifts"]):
        ax.text(xd + half + 0.5, zd, u"巷道 %d" % (d + 1), fontsize=7, va="center", ha="left")
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
        ax.text(xd + dc["half"] + 0.5, zd, u"巷道 %d（孔不入散体）" % (d + 1), fontsize=7, color="#555555", va="center")
    hs = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in PL.COLORS]
    hs += [plt.Line2D([], [], color="#7a3b9e"), plt.Line2D([], [], color="k", ls="-.", lw=0.9),
           plt.Line2D([], [], color="#a0522d"), plt.Line2D([], [], color="#1f4e9c", ls=":"),
           plt.Line2D([], [], color="#1f4e9c", ls="--"), plt.Line2D([], [], color="k", lw=1.0),
           plt.Line2D([], [], color=X.LOWC, lw=1.4)]
    ax.legend(hs, [u"散体（K = 1.25）", u"下盘", u"上盘", u"矿柱", u"残余空区", u"巷道（不透浆）", u"下盘松散底层",
                   u"原采场轮廓", u"冒落外轮廓", u"散体顶面 K = 1.15", u"散体顶面 K = 1.40", u"钻孔（每条巷道 −30°、0°）",
                   X.LOW_NAME], loc="upper left", frameon=False, fontsize=6.5, ncol=2)
    ax.set_xlabel(u"X/m（原采场下端底板角点为原点）"); ax.set_ylabel(u"Z/m")
    fig.tight_layout()
    X.K.save(fig, "fig_4_8_1")
    cav, rv = Q["cav"], (V["cls"] == 0) | (V["cls"] == 4)
    print("cav cells %d; rubble + void cells %d; rubble + void outside cav %d" % (cav.sum(), rv.sum(), (rv & ~cav).sum()))


if __name__ == "__main__":
    main()
