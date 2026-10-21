# -*- coding: utf-8 -*-
"""
Fig. 4.8-2 (draft): structure pre-judgement, five panels, K 1.25, current end state.

    powershell -File scripts\\run_local.ps1 scripts\\figures_ch4\\fig_4_8_2.py

(a)(b) porosity of the two scenarios; (c)-(e) scenario 1: p_req of the normal (0 deg) hole of drifts 1-3 with the
1 and 2 MPa reach-domain boundaries (scenario-2 p_req not shown, main line 2026-10-21). Data:
outputs/check_4_8<suffix>/prewrite.npz (n1, n2, cost = p_req of the normal holes, scenario 1 first). The lower end of
the rubble is marked on every panel.
"""
from __future__ import print_function

import os

import numpy as np

import xz48 as X
from common import plt
from matplotlib.colors import LinearSegmentedColormap

PL = X.PL
VIEW = (-31.0, 54.0, -4.0, 43.0)
HOLE_NAMES = [u"最低巷道法向孔", u"第二巷道法向孔", u"第三巷道法向孔"]


def panel(ax, D, title, hole=None):
    PL.background(ax, D)
    for d in D["drifts"]:
        ax.add_patch(plt.Rectangle((d[0] - 1.25, d[1] - 1.25), 2.5, 2.5, fc=PL.COLORS[5], ec="none"))
    if hole is not None:
        ax.plot([hole[0], hole[2]], [hole[1], hole[3]], "-", color="k", lw=1.2)
    X.draw_low(ax, dx=-9.0, dz=-2.0, lw=1.2)
    ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect("equal")
    ax.set_title(title)
    ax.tick_params(labelsize=7)


def main():
    D = dict(np.load(os.path.join(X.CHK, "prewrite.npz")))
    nh = len(D["holes"])
    if nh != 3 or D["cost"].shape[0] != 2 * nh:
        raise SystemExit("expected 3 normal holes x 2 scenarios in prewrite.npz")
    fig = plt.figure(figsize=(7.4, 8.4))
    w, h = 0.40, 0.27
    pos = [(0.06, 0.70), (0.50, 0.70), (0.06, 0.38), (0.50, 0.38), (0.06, 0.06)]
    axs = [fig.add_axes([x0, y0, w, h]) for x0, y0 in pos]
    rub = D["cls"] == 0
    for ax, key, tt in ((axs[0], "n1", u"(a) 情景一：沿倾向无趋势"),
                        (axs[1], "n2", u"(b) 情景二：孔隙率沿倾向向上增大（截断于 0.10～0.35）")):
        im_n = ax.imshow(np.ma.masked_where(~rub, D[key]), origin="lower", extent=PL.ext(D), cmap="YlGnBu", vmin=0.10,
                         vmax=0.35, interpolation="nearest")
        panel(ax, D, tt)
    cmap = LinearSegmentedColormap.from_list("preq", plt.get_cmap("Purples_r")(np.linspace(0.0, 0.92, 256)))
    cmap.set_over("white")
    for hi in range(nh):
        ax = axs[2 + hi]
        c = D["cost"][hi] / 1e6
        img = np.where(np.isfinite(c), c, 1e3)
        im_p = ax.imshow(np.ma.masked_where(~rub, img), origin="lower", extent=PL.ext(D), cmap=cmap, vmin=0, vmax=4,
                         interpolation="nearest")
        cc = np.where(np.isfinite(c), c, 99.0)
        ax.contour(D["X"][0], D["Z"][:, 0], cc, levels=[1.0], colors=["#1b9e77"], linewidths=1.0)
        ax.contour(D["X"][0], D["Z"][:, 0], cc, levels=[2.0], colors=["#e8743b"], linewidths=1.0)
        panel(ax, D, u"(%s) %s" % ("cde"[hi], HOLE_NAMES[hi]), D["holes"][hi])
        X.corner_note(ax)
    for ax in axs:
        ax.set_xlabel(u"X/m", fontsize=8)
    for ax in axs[0::2]:
        ax.set_ylabel(u"Z/m", fontsize=8)
    cb = fig.colorbar(im_n, cax=fig.add_axes([0.925, 0.72, 0.012, 0.23]))
    cb.set_label(u"孔隙率 n", fontsize=8)
    cb = fig.colorbar(im_p, cax=fig.add_axes([0.925, 0.10, 0.012, 0.50]), extend="max")
    cb.set_label(u"所需源压 $p_\\mathrm{req}$ / MPa（大于 4 MPa 显示为白色）", fontsize=8)
    fig.legend([plt.Line2D([], [], color="#1b9e77"), plt.Line2D([], [], color="#e8743b"),
                plt.Line2D([], [], color="k", lw=1.2), plt.Rectangle((0, 0), 1, 1, fc=PL.COLORS[5]),
                plt.Rectangle((0, 0), 1, 1, fc="#bdbdbd"), plt.Rectangle((0, 0), 1, 1, fc="#cfe3fa"),
                plt.Line2D([], [], color=X.LOWC, lw=1.2)],
               [u"$p_0$ = 1 MPa 可达域边界", u"$p_0$ = 2 MPa 可达域边界", u"钻孔", u"巷道", u"围岩与矿柱", u"残余空区",
                X.LOW_NAME], loc="center", bbox_to_anchor=(0.71, 0.19), ncol=1, frameon=False, fontsize=7.5)
    X.K.save(fig, "fig_4_8_2")


if __name__ == "__main__":
    main()
