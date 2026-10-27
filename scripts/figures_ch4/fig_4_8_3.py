# -*- coding: utf-8 -*-
"""
Fig. 4.8-3 (draft): groutability zoning and stepwise reach, eight panels.

    powershell -File scripts\\run_local.ps1 scripts\\figures_ch4\\fig_4_8_3.py

(a)-(f) union zoning of the 6 holes for K 1.15 / 1.25 / 1.40 (rows) x scenario 1 / 2 (columns): reliable (in the
        union at 1 MPa), conditional (in at 2 MPa only), hard (out at 2 MPa); area shares in each panel. Data:
        outputs/xz_prejudge<suffix>/step2b.npz "zones" (index scenario * 3 + K index), same as
        scripts/xz_prejudge/plot_step2b.py fig_b3_zones_compare (its hanging-side boundary is not drawn here).
(g)(h)  K 1.25 scenario 1, p0 = 2 MPa: stepwise reach of the bottom-up and top-down orders (each hole's reach domain
        cemented once grouted), seq_reach.npz; the order-free union is given in the in-figure note.
"Covered" is green and the reliable zone blue; "in the union but not covered" has its own colour (the zoning's amber
means conditional). Outside the 2 MPa union is the hard zone (seq_reach.txt checks union == zone <= 1), same red.
"""
from __future__ import print_function

import os

import numpy as np

import xz48 as X
from common import plt, C3
from matplotlib.colors import ListedColormap

PL = X.PL
VIEW = (-31.0, 54.0, -4.0, 43.0)
ZCOL = ["#2a78d6", "#f2b84b", "#e34948"]          # reliable, conditional, hard (as plot_step2b.py)
ZNAME = [u"可靠区（1 MPa 可达）", u"条件区（仅 2 MPa 可达）", u"难灌区（2 MPa 不可达）"]
COVER, LEFT = C3, "#9b7fd1"                        # covered; in the union, not covered after the sequence
SCEN = [u"一", u"二"]


def frame(ax, D, ki, title):
    for d in D["drifts"]:
        ax.add_patch(plt.Rectangle((d[0] - 1.25, d[1] - 1.25), 2.5, 2.5, fc=PL.COLORS[5], ec="none"))
    for h in D["fan"][ki]:
        ax.plot([h[0], h[2]], [h[1], h[3]], "-", color="k", lw=0.8)
    X.draw_low(ax, label=False, lw=1.1)
    ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3]); ax.set_aspect("equal")
    ax.set_title(title, fontsize=8.5)
    ax.tick_params(labelsize=6.5)


def main():
    D = np.load(os.path.join(X.PRE, "step2b.npz"))
    Q = np.load(os.path.join(X.PRE, "seq_reach.npz"))
    if COVER.lower() == ZCOL[0] or LEFT.lower() in ZCOL:
        raise SystemExit("stepwise-reach colours must differ from the zoning colours")
    ks = list(D["ks"])
    fig = plt.figure(figsize=(7.4, 10.6))
    w, h = 0.42, 0.175
    ys = [0.805, 0.605, 0.405, 0.095]
    xs = [0.07, 0.535]
    lab = iter("abcdefgh")
    for ki, K in enumerate(ks):
        for k in range(2):
            ax = fig.add_axes([xs[k], ys[ki], w, h])
            V = {"cls": D["cls"][ki]}
            V.update({"X": D["X"], "Z": D["Z"]})
            X.background(ax, V)
            z = D["zones"][k * 3 + ki]
            rub = z >= 0
            ax.imshow(np.ma.masked_where(~rub, z.astype(float)), origin="lower", extent=PL.ext(V),
                      cmap=ListedColormap(ZCOL), vmin=-0.5, vmax=2.5, interpolation="nearest")
            sh = [100.0 * (z[rub] == q).mean() for q in (0, 1, 2)]
            frame(ax, D, ki, u"(%s) K = %.2f，情景%s" % (next(lab), K, SCEN[k]))
            X.corner_note(ax, u"可靠 %.2f　条件 %.2f　难灌 %.2f" % tuple(s_ / 100.0 for s_ in sh))
            print("K %.2f scenario %d: reliable / conditional / hard %.4f / %.4f / %.4f" % ((K, k + 1) + tuple(
                s_ / 100.0 for s_ in sh)))
            if k == 0:
                ax.set_ylabel(u"Z/m", fontsize=8)
    ki = ks.index(1.25)
    V = {"cls": D["cls"][ki], "X": D["X"], "Z": D["Z"]}
    rub = V["cls"] == 0
    union = Q["K1.25_s1_none"]
    if not np.array_equal(union, rub & (D["zones"][ki] <= 1) & (D["zones"][ki] >= 0)):
        raise SystemExit("order-free union differs from the 2 MPa zoning union")
    for k, (key, name) in enumerate((("bottom-up", u"自下而上"), ("top-down", u"自上而下"))):
        ax = fig.add_axes([xs[k], ys[3], w, h])
        X.background(ax, V)
        cov = Q["K1.25_s1_%s" % key]
        img = np.where(rub, np.where(cov, 0.0, np.where(union, 1.0, 2.0)), np.nan)
        ax.imshow(np.ma.masked_invalid(img), origin="lower", extent=PL.ext(V),
                  cmap=ListedColormap([COVER, LEFT, ZCOL[2]]), vmin=-0.5, vmax=2.5, interpolation="nearest")
        frame(ax, D, ki, u"(%s) %s" % (next(lab), name))
        X.corner_note(ax, u"覆盖 %.3f" % cov[rub].mean())
        ax.set_xlabel(u"X/m", fontsize=8)
        if k == 0:
            ax.set_ylabel(u"Z/m", fontsize=8)
        print("K 1.25 scenario 1 %-9s covered %.4f" % (key, cov[rub].mean()))
    common = [plt.Line2D([], [], color="k", lw=0.8), plt.Rectangle((0, 0), 1, 1, fc=PL.COLORS[5]),
              plt.Rectangle((0, 0), 1, 1, fc=X.ROCKC), plt.Rectangle((0, 0), 1, 1, fc=X.VOIDC),
              plt.Line2D([], [], color=X.LOWC, lw=1.1)]
    cname = [u"钻孔（每条巷道 −30°、0°）", u"巷道", u"围岩与矿柱", u"残余空区", X.LOW_NAME]
    fig.legend([plt.Rectangle((0, 0), 1, 1, fc=c) for c in ZCOL] + common, ZNAME + cname, loc="center",
               bbox_to_anchor=(0.5, 0.348), ncol=4, frameon=False, fontsize=7)
    fig.text(0.5, 0.300, u"(g)(h) K = 1.25，情景一，$p_0$ = 2 MPa，每注完一孔即视其可达域为不透浆；"
             u"不计先后（6 孔可达域并集）覆盖 %.3f" % union[rub].mean(), ha="center", fontsize=7.5)
    fig.legend([plt.Rectangle((0, 0), 1, 1, fc=c) for c in (COVER, LEFT, ZCOL[2])],
               [u"覆盖", u"并集内、按顺序注浆后未覆盖", u"并集之外（难灌区）"], loc="center",
               bbox_to_anchor=(0.5, 0.022), ncol=3, frameon=False, fontsize=7)
    print("K 1.25 scenario 1 order-free union %.4f" % union[rub].mean())
    X.K.save(fig, "fig_4_8_3")


if __name__ == "__main__":
    main()
