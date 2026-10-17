# -*- coding: utf-8 -*-
"""Figures for docs/xz_prejudge_step1.md (from outputs/xz_prejudge/prejudge.npz written by prejudge.py).

    powershell -File scripts\\run_local.ps1 scripts\\xz_prejudge\\plot.py

Writes figures/xz_prejudge/fig_xz_{1..5}.png (300 dpi) and .svg. Coordinates X, Z relative to the footwall
corner of the original stope's lower end.
"""
from __future__ import print_function

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "outputs", "xz_prejudge", "prejudge.npz")
OUT = os.path.join(REPO, "figures", "xz_prejudge")
matplotlib.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans SC", "SimHei"],
                            "axes.unicode_minus": False, "font.size": 9, "axes.titlesize": 9,
                            "legend.fontsize": 7, "savefig.dpi": 300, "svg.fonttype": "none"})
CELL = 0.5
P0S = (1.0, 2.0, 3.0, 5.0)
PC = ("#1b9e77", "#2a78d6", "#e8743b", "#7a3b9e")


def save(fig, name):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    fig.savefig(os.path.join(OUT, name + ".svg"))
    plt.close(fig)
    print("wrote", name)


def ext(D):
    X, Z = D["X"], D["Z"]
    return [X.min() - CELL / 2, X.max() + CELL / 2, Z.min() - CELL / 2, Z.max() + CELL / 2]


def base(ax, D, title):
    cls = D["cls"]
    rock = np.ma.masked_where(~((cls >= 1) & (cls <= 3)), np.ones(cls.shape))
    void = np.ma.masked_where(cls != 4, np.ones(cls.shape))
    ax.imshow(rock, origin="lower", extent=ext(D), cmap=ListedColormap(["#bdbdbd"]), interpolation="nearest")
    ax.imshow(void, origin="lower", extent=ext(D), cmap=ListedColormap(["#cfe3fa"]), interpolation="nearest")
    for h in D["holes"]:
        ax.plot([h[0], h[2]], [h[1], h[3]], "-", color="k", lw=1.2)
        ax.plot(h[0], h[1], "o", color="k", ms=2.5)
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.tick_params(labelsize=7)


def fig1(D):
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    cls = D["cls"].astype(float)
    cmap = ListedColormap(["#f2a65a", "#8a8a8a", "#c9c9c9", "#d9c58f", "#cfe3fa"])
    ax.imshow(cls, origin="lower", extent=ext(D), cmap=cmap, norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5),
              interpolation="nearest")
    ax.contour(D["X"][0], D["Z"][:, 0], D["basal"].astype(float), levels=[0.5], colors=["#a0522d"], linewidths=0.9)
    ax.contour(D["X"][0], D["Z"][:, 0], D["hang"].astype(float), levels=[0.5], colors=["#2a78d6"], linewidths=0.9)
    for i, h in enumerate(D["holes"]):
        ax.plot([h[0], h[2]], [h[1], h[3]], "-", color="k", lw=1.4)
        ax.plot(h[0], h[1], "o", color="k", ms=3)
        ax.text(h[0] + 1, h[1] - 2.5, u"孔 %d" % (i + 1), fontsize=7)
    ax.set_aspect("equal")
    hs = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in ("#f2a65a", "#8a8a8a", "#c9c9c9", "#d9c58f", "#cfe3fa")]
    hs += [plt.Line2D([], [], color="#a0522d"), plt.Line2D([], [], color="#2a78d6"), plt.Line2D([], [], color="k")]
    ax.legend(hs, [u"散体", u"下盘", u"上盘", u"矿柱", u"残余空区（无流）", u"下盘松散底层（s ≤ 3 m）",
                   u"上盘侧区域（s > %.1f m）" % float(D["s13"]), u"钻孔（孔口管 2 m 不出浆）"],
              loc="upper left", frameon=False, fontsize=6.5)
    ax.set_xlabel(u"X/m（原采场下端底板角点为原点）"); ax.set_ylabel(u"Z/m")
    fig.tight_layout()
    save(fig, "fig_xz_1_masks")


def fig2(D):
    fig, axs = plt.subplots(2, 1, figsize=(7.6, 7.0))
    for ax, key, tt in ((axs[0], "n1", u"(a) 情景一：沿倾向无趋势"), (axs[1], "n2", u"(b) 情景二：沿倾向上松下密（截断 0.10～0.35）")):
        n = np.where(D["cls"] == 0, D[key], np.nan)
        base(ax, D, tt)
        im = ax.imshow(np.ma.masked_invalid(n), origin="lower", extent=ext(D), cmap="YlGnBu", vmin=0.10, vmax=0.35,
                       interpolation="nearest")
        ax.set_ylabel(u"Z/m")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label(u"映射孔隙率 n", fontsize=8)
    axs[1].set_xlabel(u"X/m")
    fig.tight_layout()
    save(fig, "fig_xz_2_porosity")


def fig3(D):
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(D["ns"], D["zs"], "o", color="#888888", ms=3.5, label=u"PFC 颗粒尺度（1.5 m 薄层）")
    ax.plot(D["sm"], D["zs"], "-", color="#2a78d6", lw=1.2, label=u"五点滑动平均")
    ax.plot(D["mapped"], D["zs"], "s-", color="#e8743b", ms=3.5, lw=1.2, label=u"映射值 $n_s(s)$（0.13～0.29）")
    for z, a in zip(D["zs"], D["ar"]):
        if a < 40:
            ax.text(0.36, z, u"%.0f m²" % a, fontsize=6, va="center", color="#52514e")
    ax.axhline(14.0, color="#888888", ls="--", lw=0.8)
    ax.text(0.30, 14.4, u"原顶板 s = 14 m", fontsize=6.5, color="#52514e")
    ax.set_xlabel(u"孔隙率（PFC 为二维面积孔隙率）"); ax.set_ylabel(u"距下盘接触面的法向距离 s/m")
    ax.set_xlim(0.10, 0.40)
    ax.legend(frameon=False, loc="center right", fontsize=7, numpoints=1)
    ax.text(0.355, -3.2, u"右侧数字：面积 < 40 m² 的薄层", fontsize=6, color="#52514e", ha="center")
    fig.tight_layout()
    save(fig, "fig_xz_3_mapping")


def fig4(D):
    fig, axs = plt.subplots(2, 1, figsize=(7.6, 7.0))
    k = 1e6 / (2 * 1830.0 * 9.81)
    for ax, key, tt in ((axs[0], "gam1", u"(a) 情景一"), (axs[1], "gam2", u"(b) 情景二")):
        g = np.where(D["cls"] == 0, D[key] * k, np.nan)
        base(ax, D, tt + u"：局部 F / $p_0$（每 MPa），3 m 窗口")
        im = ax.imshow(np.ma.masked_invalid(g), origin="lower", extent=ext(D), cmap="RdBu_r", vmin=-4, vmax=4,
                       interpolation="nearest")
        ax.set_ylabel(u"Z/m")
        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.set_label(u"F/$p_0$（1/MPa）；$p_0$ = 5 MPa 时乘 5", fontsize=7)
    axs[1].set_xlabel(u"X/m")
    fig.tight_layout()
    save(fig, "fig_xz_4_F")


def fig5(D):
    fig, axs = plt.subplots(3, 2, figsize=(10.0, 8.4))
    C = D["cost"]
    for k in range(2):
        for hi in range(3):
            ax = axs[hi, k]
            c = np.where(D["cls"] == 0, C[k * 3 + hi] / 1e6, np.nan)
            base(ax, D, u"情景%s，孔 %d" % (u"一" if k == 0 else u"二", hi + 1))
            im = ax.imshow(np.ma.masked_invalid(np.minimum(c, 8.0)), origin="lower", extent=ext(D), cmap="Purples_r",
                           vmin=0, vmax=8, interpolation="nearest")
            cc = np.where(np.isfinite(c), c, 99.0)
            ax.contour(D["X"][0], D["Z"][:, 0], cc, levels=list(P0S), colors=list(PC), linewidths=0.9)
            if hi == 2:
                ax.set_xlabel(u"X/m")
            if k == 0:
                ax.set_ylabel(u"Z/m")
    hs = [plt.Line2D([], [], color=c) for c in PC]
    fig.legend(hs, [u"$p_\mathrm{req}$ = %g MPa（$p_0$ = %g MPa 的可达域边界）" % (p, p) for p in P0S], loc="lower center", ncol=4,
               frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.06, right=0.9, bottom=0.08, top=0.96, hspace=0.25, wspace=0.08)
    cax = fig.add_axes([0.92, 0.25, 0.012, 0.5])
    cb = fig.colorbar(im, cax=cax); cb.set_label(u"所需源压 $p_\mathrm{req}$/MPa（截至 8）；$p_\mathrm{req}$/$p_0$ 即除以 $p_0$", fontsize=7)
    save(fig, "fig_xz_5_preq")


def main():
    D = np.load(SRC)
    fig1(D)
    fig2(D)
    fig3(D)
    fig4(D)
    fig5(D)
    return 0


if __name__ == "__main__":
    sys.exit(main())
