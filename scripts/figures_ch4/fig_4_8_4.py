# -*- coding: utf-8 -*-
"""
Fig. 4.8-4 (draft, layout not settled): stepwise reach domains, K 1.25, both scenarios, p0 = 2 MPa.

    powershell -File scripts\\run_local.ps1 scripts\\figures_ch4\\fig_4_8_4.py

Union of the single-hole reach domains against the stepwise reach of the bottom-up and top-down orders (each hole's
reach domain is cemented once it has been grouted). Same panels as scripts/xz_prejudge/plot_48.py fig4, but
"covered" is green so that it differs from the reliable zone (blue) of the groutability zoning
(scripts/xz_prejudge/plot_step2b.py, fig_b3_zones_compare). Data: outputs/xz_prejudge<suffix>/step2b.npz,
seq_reach.npz.
"""
from __future__ import print_function

import os

import numpy as np

import xz48 as X
from common import plt, C3
from matplotlib.colors import ListedColormap

PL = X.PL
COVER = C3                                   # green; the zoning's reliable zone is #2a78d6
ZONE_RELIABLE = "#2a78d6"


def main():
    D = np.load(os.path.join(X.PRE, "step2b.npz"))
    Q = np.load(os.path.join(X.PRE, "seq_reach.npz"))
    if COVER.lower() == ZONE_RELIABLE:
        raise SystemExit("covered and reliable zone share a colour")
    ki = list(D["ks"]).index(1.25)
    V = {"cls": D["cls"][ki], "X": D["X"], "Z": D["Z"], "drifts": D["drifts"], "holes": D["fan"][ki]}
    rub = V["cls"] == 0
    cols = ["#e34948", COVER, "#f2b84b"]
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
            X.draw_low(ax, dx=-9.0, dz=-2.6, lw=1.2)
            if k == 2:
                ax.set_xlabel(u"X/m")
            if j == 0:
                ax.set_ylabel(u"Z/m")
            print("scenario %d %-9s covered %.4f" % (k, key, cov[rub].mean()))
    fig.legend([plt.Rectangle((0, 0), 1, 1, fc=c) for c in (cols[1], cols[2], cols[0])]
               + [plt.Line2D([], [], color=X.LOWC, lw=1.2)],
               [u"覆盖（$p_0$ = 2 MPa）", u"并集内、按顺序注浆后未覆盖", u"并集之外", X.LOW_NAME], loc="lower center",
               ncol=4, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.1, top=0.95, hspace=0.2, wspace=0.06)
    X.K.save(fig, "fig_4_8_4")


if __name__ == "__main__":
    main()
